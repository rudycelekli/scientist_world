"""Data preparation shared by the SFT and offline-GRPO trainers.

Turns the Scientist World traces dataset (one row per finished evaluation run)
into the two training sets:

  - SFT examples (``sft_examples``): task specification -> final idea. A run
    with cleaned reasoning in out/reasoning.jsonl gets it as a <think> block in
    front of the idea.
  - GRPO samples (``grpo_dataset``): the same completions, each with an
    advantage, the z-score of its reward inside its (paper, mode) group.

Both modes (innovate, improve) are used. The prompt holds the task
specification only (problem, direction, constraints, targets); the completion
is the run's consolidated final idea.

    python data_prep.py [--data PARQUET]    # print set sizes and one sample
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("innovate", "improve")

# Where the traces dataset comes from; _load_rows gives the resolution order.
_PARQUET = (Path(__file__).resolve().parent.parent
            / "evaluation" / "dataset_builder" / "out"
            / "scientist_world_traces.parquet")
_HF_REPO = "zlab-princeton/scientist-world-traces"
_HF_REPO_ENV = "SW_TRACES_REPO"

# Prepared files written by the *_batch.py scripts.
_OUT = Path(__file__).resolve().parent / "out"
_REASONING = _OUT / "reasoning.jsonl"
_REPHRASED = _OUT / "rephrased.jsonl"
_CRAFT_JUDGE = _OUT / "craft_judge.jsonl"

SYSTEM = (
    "You are an AI research scientist. You are given a research task: a problem, "
    "an allowed direction, constraints, and target metrics. Propose a single, "
    "novel, concrete method that addresses the task within the constraints. "
    "Describe the method so it could be implemented directly."
)

# Paper-id lists behind the held-out comparison. TOP10 is a subset of TOP30,
# and HELDOUT_PAPERS is the first ten ids, in sorted order, of TOP30 - TOP10
# (what baseline_untrained.pick_papers selects). The held-out papers are
# excluded from training, so comparing the untrained and trained models on
# them measures generalization, not memory.
TOP10 = {'2502.06806', '2502.11881', '2503.17338', '2505.16017', '2506.00772',
         '2507.01439', '2504.14815', '2502.06523', '2506.06409', '2503.04422'}
TOP30 = {'2502.06806', '2503.06808', '2507.10602', '2502.10563', '2501.01741',
         '2503.10197', '2502.06523', '2506.07747', '2504.14815', '2503.17338',
         '2506.04452', '2502.11881', '2507.02856', '2505.11883', '2508.13791',
         '2505.21825', '2506.05188', '2503.04422', '2506.06409', '2506.06842',
         '2506.00772', '2504.13752', '2507.22733', '2505.16017', '2503.21907',
         '2508.01495', '2508.01249', '2509.12939', '2507.01439', '2511.13588'}
HELDOUT_PAPERS = {
    '2501.01741', '2502.10563', '2503.06808', '2503.10197', '2503.21907',
    '2504.13752', '2505.11883', '2505.21825', '2506.04452', '2506.05188',
}


def add_data_argument(parser) -> None:
    """Add the ``--data`` option shared by every script that reads the traces."""
    parser.add_argument(
        "--data", default=None, metavar="PARQUET",
        help="local parquet file of the traces dataset, or 'hub' to force the "
             "Hub dataset. Default: the dataset builder's output "
             "(../evaluation/dataset_builder/out/scientist_world_traces.parquet) "
             f"if it exists, otherwise the Hub dataset ${_HF_REPO_ENV} "
             f"(default {_HF_REPO}).")


_ANNOUNCED = set()


def _announce(origin: str) -> None:
    """Say once per process where the traces were read from."""
    if origin not in _ANNOUNCED:
        _ANNOUNCED.add(origin)
        print(f"traces dataset: {origin}", flush=True)


def _load_rows(source: str | None = None):
    """Rows of the traces dataset, one mapping per finished run.

    ``source`` is the path of a local parquet file, ``"hub"`` for the Hub
    dataset, or None for the default: the dataset builder's local output if it
    exists, otherwise the Hub dataset named by $SW_TRACES_REPO.
    """
    if source == "hub":
        path = None
    elif source:
        path = Path(source).expanduser()
        if not path.is_file():
            raise SystemExit(f"traces parquet not found: {path}")
    else:
        path = _PARQUET if _PARQUET.exists() else None
    if path is None:
        return _load_hub(os.environ.get(_HF_REPO_ENV) or _HF_REPO)
    from datasets import load_dataset
    _announce(str(path))
    return load_dataset("parquet", data_files=str(path), split="train")


def _load_hub(repo: str):
    """The Hub dataset as one table.

    It is published with one split per mode. The splits are concatenated and
    ordered by (task_id, mode): papers then come in the same order as in the
    single-file parquet, which matters because ``grpo_dataset`` truncates in
    row order. Rows of one (paper, mode) group keep their relative order.
    """
    from datasets import concatenate_datasets, load_dataset
    splits = load_dataset(repo)
    missing = [m for m in MODES if m not in splits]
    if missing:
        raise SystemExit(
            f"Hub dataset {repo} has no split named {' or '.join(missing)} "
            f"(found: {', '.join(sorted(map(str, splits)))}). One split per mode is "
            f"expected; set ${_HF_REPO_ENV} or pass --data.")
    _announce(f"{repo} (Hub, splits {' + '.join(MODES)})")
    ds = concatenate_datasets([splits[m] for m in MODES])
    keys = list(zip(ds["task_id"], ds["mode"]))
    return ds.select(sorted(range(len(keys)), key=keys.__getitem__))


def build_prompt(row) -> str:
    """The task specification shown to the model (problem, direction,
    constraints, targets). It contains nothing about the paper's method or
    results."""
    parts = []
    for label, key in (("# Problem", "problem"), ("# Direction", "direction"),
                       ("# Constraints", "constraint"), ("# Targets", "target")):
        v = row.get(key)
        if v:
            parts.append(f"{label}\n{v.strip()}")
    return "\n\n".join(parts)


def conversation(row, reasoning: str | None = None) -> list[dict]:
    """Chat form: system + task -> final_idea. When ``reasoning`` is given, the
    assistant turn is a <think> reasoning block followed by the final idea;
    otherwise it is the final idea alone."""
    idea = row["final_idea"].strip()
    if reasoning:
        answer = f"<think>\n{reasoning.strip()}\n</think>\n\n{idea}"
    else:
        answer = idea
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(row)},
        {"role": "assistant", "content": answer},
    ]


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _prepared(path: Path, field: str, script: str, keep=bool, hint: str = "") -> dict:
    """run_id -> ``field`` over the usable records of a prepared file in out/.

    A missing or empty file is an error: the callers would otherwise build a
    different training set without saying so.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it with `python {script}` "
            f"(README.md, 'Data preparation').{hint}")
    out = {}
    for ln in path.read_text().splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if keep(d.get(field)):
            out[d["run_id"]] = d[field]
    if not out:
        raise SystemExit(
            f"{path} has no record with a usable '{field}'. Rerun "
            f"`python {script}`.{hint}")
    return out


def _load_reasoning() -> dict:
    """run_id -> cleaned reasoning (clean_reasoning_batch.py)."""
    return _prepared(_REASONING, "reasoning", "clean_reasoning_batch.py")


def _load_rephrased() -> dict:
    """run_id -> reworded completion (rephrase_batch.py)."""
    return _prepared(_REPHRASED, "rephrased_completion", "rephrase_batch.py")


def sft_examples(source=None, modes=MODES,
                 n_single_backfill=None, exclude_papers=HELDOUT_PAPERS,
                 include_rephrased=False):
    """SFT examples: task specification -> final idea.

    - A run with an entry in out/reasoning.jsonl (written for runs with at
      least two idea versions) gives a reasoning example: the assistant turn
      is <think> cleaned reasoning </think> followed by the final idea. These
      runs are used whether or not they have a score.
    - The other runs give a plain final-idea example. Only those with a finite
      test_score are eligible; they are taken in decreasing test_score order,
      the first ``n_single_backfill`` of them, or all of them when it is None.
    - With ``include_rephrased``, every record of out/rephrased.jsonl adds one
      more example for its run: the same prompt with the reworded completion
      (run_id suffixed "-aug", ``augmented`` set).

    Rows of ``exclude_papers`` and rows without a final idea are skipped.
    Returns a list of {run_id, task_id, mode, conversation, prompt, completion,
    has_reasoning, test_score}.
    """
    ds = _load_rows(source)
    reasoning = _load_reasoning()
    exclude = set(exclude_papers or ())
    multi, single = [], []
    for r in ds:
        if (r["mode"] not in modes or not r["final_idea"]
                or r["task_id"] in exclude):
            continue
        (multi if r["run_id"] in reasoning else single).append(r)

    out = []
    for r in multi:
        rz = reasoning[r["run_id"]]
        out.append({
            "run_id": r["run_id"],
            "task_id": r["task_id"], "mode": r["mode"],
            "conversation": conversation(r, reasoning=rz),
            "prompt": build_prompt(r),
            "completion": r["final_idea"].strip(),
            "has_reasoning": True,
            "test_score": r["test_score"],
        })
    single_scored = [r for r in single if _finite(r.get("test_score"))]
    single_scored.sort(key=lambda r: r["test_score"], reverse=True)
    limit = len(single_scored) if n_single_backfill is None else n_single_backfill
    for r in single_scored[:limit]:
        out.append({
            "run_id": r["run_id"],
            "task_id": r["task_id"], "mode": r["mode"],
            "conversation": conversation(r),
            "prompt": build_prompt(r),
            "completion": r["final_idea"].strip(),
            "has_reasoning": False,
            "test_score": r["test_score"],
        })
    if include_rephrased:
        reph = _load_rephrased()
        rowmap = {r["run_id"]: r for r in ds}
        for run_id, comp in reph.items():
            r = rowmap.get(run_id)
            if not r or r["task_id"] in exclude:
                continue
            out.append({
                "run_id": run_id + "-aug",
                "task_id": r["task_id"], "mode": r["mode"],
                "conversation": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_prompt(r)},
                    {"role": "assistant", "content": comp},
                ],
                "prompt": build_prompt(r),
                "completion": comp,
                "has_reasoning": "<think>" in comp,
                "test_score": r["test_score"],
                "augmented": True,
            })
    return out


def _znorm_group(members: list[dict], reward_key: str):
    """Yield members with group-relative advantage = z-score of the reward.
    Drops degenerate groups (std == 0 or < 2 members). Non-finite rewards
    are excluded first."""
    vals = [m[reward_key] for m in members
            if isinstance(m.get(reward_key), (int, float))
            and math.isfinite(m[reward_key])]
    members = [m for m in members
               if isinstance(m.get(reward_key), (int, float))
               and math.isfinite(m[reward_key])]
    if len(members) < 2:
        return []
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals)
    if std == 0:
        return []
    for m in members:
        m = dict(m)
        m["advantage"] = (m[reward_key] - mean) / std
        yield m


_REASONING_CACHE = None


def _reasoning_for(run_id):
    """Cached lookup of a run's cleaned reasoning, so that GRPO completions
    have the same <think> format as the SFT examples the policy was trained
    on."""
    global _REASONING_CACHE
    if _REASONING_CACHE is None:
        _REASONING_CACHE = _load_reasoning()
    return _REASONING_CACHE.get(run_id)


def _source_cross_model(source, modes, exclude=()):
    """GROUP = (paper, mode); reward = measured test_score."""
    exclude = set(exclude or ())
    ds = _load_rows(source)
    bykey = defaultdict(list)
    for r in ds:
        if r["mode"] in modes and r["task_id"] not in exclude and r["final_idea"] and r["test_score"] is not None:
            bykey[(r["task_id"], r["mode"])].append({
                "group_id": f"{r['task_id']}:{r['mode']}",
                "task_id": r["task_id"], "mode": r["mode"],
                "conversation": conversation(r, reasoning=_reasoning_for(r["run_id"])),
                "completion": r["final_idea"].strip(),
                "reward": r["test_score"],
            })
    out = []
    for members in bykey.values():
        out.extend(_znorm_group(members, "reward"))
    return out


def _source_generalization(source, modes, exclude=(), lam=1.0):
    """GROUP = (paper, mode); reward = test_score - lam * max(0, train - test).

    Only the overfit gap (train above test) is penalized; scoring at or above
    train on the test split is not. Needs both train_score and test_score."""
    exclude = set(exclude or ())
    ds = _load_rows(source)
    bykey = defaultdict(list)
    for r in ds:
        if (r["mode"] in modes and r["task_id"] not in exclude and r["final_idea"]
                and _finite(r.get("test_score")) and _finite(r.get("train_score"))):
            gap = max(0.0, r["train_score"] - r["test_score"])
            bykey[(r["task_id"], r["mode"])].append({
                "group_id": f"{r['task_id']}:{r['mode']}",
                "task_id": r["task_id"], "mode": r["mode"],
                "conversation": conversation(r, reasoning=_reasoning_for(r["run_id"])),
                "completion": r["final_idea"].strip(),
                "reward": r["test_score"] - lam * gap,
            })
    out = []
    for members in bykey.values():
        out.extend(_znorm_group(members, "reward"))
    return out


def _source_judge(source, modes, exclude=(), axis="overall"):
    """GROUP = (paper, mode); reward = the craft judge's score on ``axis``
    (novelty, soundness or overall) from out/craft_judge.jsonl. The judge sees
    the task and the idea only, never the outcome."""
    exclude = set(exclude or ())
    scores = _prepared(
        _CRAFT_JUDGE, axis, "judge_craft_batch.py", keep=_finite,
        hint=" The judge sources are optional and were not used for the "
             "reported results; without them pass "
             "--sources cross_model,generalization.")
    ds = _load_rows(source)
    bykey = defaultdict(list)
    for r in ds:
        if r["mode"] in modes and r["task_id"] not in exclude and r["final_idea"] and r["run_id"] in scores:
            bykey[(r["task_id"], r["mode"])].append({
                "group_id": f"{r['task_id']}:{r['mode']}",
                "task_id": r["task_id"], "mode": r["mode"],
                "conversation": conversation(r, reasoning=_reasoning_for(r["run_id"])),
                "completion": r["final_idea"].strip(),
                "reward": scores[r["run_id"]],
            })
    out = []
    for members in bykey.values():
        out.extend(_znorm_group(members, "reward"))
    return out


def grpo_dataset(source=None, sources=("cross_model",),
                 modes=MODES,
                 weights=None, exclude_papers=HELDOUT_PAPERS,
                 max_samples=None):
    """Mixed GRPO set. Combines one or more reward sources; each (paper, mode)
    group is z-normalized on its own, so that different rewards share a scale.
    A per-source ``weight`` scales the advantage. Returns a list of {group_id,
    source, task_id, mode, conversation, completion, reward, advantage} with
    the source weight already applied to the advantage.

    sources:
      - "cross_model": reward = measured test_score.
      - "generalization": reward = test_score minus the overfit gap.
      - "judge", "judge_novelty": reward = the craft judge's overall / novelty
        score (optional, needs out/craft_judge.jsonl).

    The sources are concatenated in the order given, and ``max_samples`` keeps
    the first ``max_samples`` entries of that list: later sources are cut
    first, in dataset row order.
    """
    weights = weights or {}
    out = []
    for s in sources:
        w = float(weights.get(s, 1.0))
        if s == "cross_model":
            members = _source_cross_model(source, modes, exclude_papers)
        elif s == "generalization":
            members = _source_generalization(source, modes, exclude_papers)
        elif s == "judge":
            members = _source_judge(source, modes, exclude_papers)
        elif s == "judge_novelty":
            members = _source_judge(source, modes, exclude_papers, axis="novelty")
        else:
            raise ValueError(f"unknown GRPO source: {s}")
        for m in members:
            m["source"] = s
            m["advantage"] = m["advantage"] * w
            out.append(m)
    if max_samples is not None:
        out = out[:max_samples]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Print the sizes of the default SFT set and of the "
                    "cross-model GRPO set, and one SFT sample.")
    add_data_argument(ap)
    args = ap.parse_args()

    sft = sft_examples(source=args.data)
    grpo = grpo_dataset(source=args.data, sources=("cross_model",))
    print(f"SFT examples (held-out papers excluded, no rephrased examples): "
          f"{len(sft)} | by mode {dict(Counter(e['mode'] for e in sft))}")
    print(f"GRPO samples (cross_model, held-out papers excluded): {len(grpo)} "
          f"across {len(set(e['group_id'] for e in grpo))} groups | "
          f"by mode {dict(Counter(e['mode'] for e in grpo))}")
    advs = [e["advantage"] for e in grpo]
    if advs:
        print(f"  advantage: min {min(advs):.2f} max {max(advs):.2f} "
              f"mean {statistics.fmean(advs):.3f} (should be ~0)")
    if sft:
        print("\n--- sample SFT prompt (truncated) ---")
        print(sft[0]["prompt"][:400])
        print("--- sample completion (truncated) ---")
        print(sft[0]["completion"][:300])


if __name__ == "__main__":
    main()
