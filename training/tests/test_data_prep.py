"""data_prep: prompt format, grouping and z-normalization, held-out exclusion,
the SFT selection rules, the prepared-file checks and the dataset loading
order."""
from __future__ import annotations

import argparse
import math
import sys
import types

import pytest

from conftest import make_row, write_jsonl

HELD = "2501.01741"      # one of the held-out papers
OTHER = "2401.00001"     # papers that are not held out
OTHER2 = "2401.00002"


@pytest.fixture
def reasoning(prep):
    """A reasoning file with one cleaned trace (run "m1") and one failure."""
    write_jsonl(prep._REASONING, [{"run_id": "m1", "reasoning": "why"},
                                  {"run_id": "bad", "error": "clean_failed"}])


# ---------------------------------------------------------------- paper lists

def test_heldout_is_first_ten_of_top30_minus_top10(prep):
    assert len(prep.TOP10) == 10 and len(prep.TOP30) == 30
    assert prep.TOP10 < prep.TOP30
    assert sorted(prep.HELDOUT_PAPERS) == sorted(prep.TOP30 - prep.TOP10)[:10]
    assert HELD in prep.HELDOUT_PAPERS and OTHER not in prep.HELDOUT_PAPERS


# ------------------------------------------------------- prompt, conversation

def test_prompt_and_conversation(prep):
    r = make_row(OTHER, "innovate", "r1", idea="  the idea \n")
    r["direction"] = None                       # empty fields are left out
    prompt = prep.build_prompt(r)
    assert prompt == (f"# Problem\nproblem of {OTHER}\n\n"
                      "# Constraints\nconstraint\n\n# Targets\ntarget")
    conv = prep.conversation(r)
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    assert conv[0]["content"] == prep.SYSTEM
    assert conv[1]["content"] == prompt
    assert conv[2]["content"] == "the idea"
    with_think = prep.conversation(r, reasoning=" because \n")
    assert with_think[2]["content"] == "<think>\nbecause\n</think>\n\nthe idea"


# ------------------------------------------------------------ z-normalization

def test_znorm_group_is_population_zscore(prep):
    members = [{"reward": 1.0}, {"reward": 2.0}, {"reward": 3.0}]
    out = list(prep._znorm_group(members, "reward"))
    std = math.sqrt(2 / 3)
    assert [m["advantage"] for m in out] == pytest.approx([-1 / std, 0.0, 1 / std])
    assert "advantage" not in members[0]        # the inputs are not mutated


@pytest.mark.parametrize("rewards", [
    [],                                         # empty group
    [5.0],                                      # a single member
    [2.0, 2.0, 2.0],                            # zero spread
    [1.0, None, float("nan")],                  # one finite reward only
])
def test_znorm_group_drops_degenerate_groups(prep, rewards):
    members = [{"reward": x} for x in rewards]
    assert list(prep._znorm_group(members, "reward")) == []


def test_znorm_group_ignores_non_finite_rewards(prep):
    members = [{"reward": 0.0}, {"reward": float("inf")}, {"reward": None},
               {"reward": 2.0}]
    out = list(prep._znorm_group(members, "reward"))
    assert [m["reward"] for m in out] == [0.0, 2.0]
    assert [m["advantage"] for m in out] == pytest.approx([-1.0, 1.0])


# ---------------------------------------------------------------- GRPO groups

def test_cross_model_groups_by_paper_and_mode(prep, use_rows, reasoning):
    use_rows([
        make_row(OTHER, "innovate", "m1", test=1.0),
        make_row(OTHER, "improve", "b1", test=10.0),
        make_row(OTHER, "innovate", "a2", test=3.0),
        make_row(OTHER, "improve", "b2", test=30.0),
        make_row(OTHER2, "innovate", "c1", test=7.0),           # alone in its group
        make_row(OTHER, "innovate", "a3", test=None),           # no score
        make_row(OTHER, "innovate", "a4", test=2.0, idea=""),   # no final idea
    ])
    out = prep.grpo_dataset(sources=("cross_model",))
    assert [(e["group_id"], e["reward"]) for e in out] == [
        (f"{OTHER}:innovate", 1.0), (f"{OTHER}:innovate", 3.0),
        (f"{OTHER}:improve", 10.0), (f"{OTHER}:improve", 30.0)]
    # each (paper, mode) group is normalized on its own
    assert [e["advantage"] for e in out] == pytest.approx([-1.0, 1.0, -1.0, 1.0])
    assert {e["source"] for e in out} == {"cross_model"}
    # a run with cleaned reasoning gets it as the <think> block
    assert out[0]["conversation"][2]["content"] == "<think>\nwhy\n</think>\n\nidea"
    assert out[1]["conversation"][2]["content"] == "idea"


def test_generalization_penalizes_only_the_overfit_gap(prep, use_rows, reasoning):
    use_rows([
        make_row(OTHER, "innovate", "a1", test=1.0, train=3.0),   # 1 - (3 - 1)
        make_row(OTHER, "innovate", "a2", test=2.0, train=1.0),   # no penalty
        make_row(OTHER, "innovate", "a3", test=5.0, train=None),  # needs both scores
    ])
    out = prep.grpo_dataset(sources=("generalization",))
    assert [e["reward"] for e in out] == [-1.0, 2.0]
    assert [e["advantage"] for e in out] == pytest.approx([-1.0, 1.0])


def test_grpo_dataset_weights_sources_and_truncates_in_order(prep, use_rows, reasoning):
    use_rows([make_row(OTHER, "innovate", "a1", test=1.0, train=1.0),
              make_row(OTHER, "innovate", "a2", test=3.0, train=3.0)])
    sources = ("cross_model", "generalization")
    out = prep.grpo_dataset(sources=sources, weights={"generalization": 0.5})
    assert [e["source"] for e in out] == ["cross_model"] * 2 + ["generalization"] * 2
    assert [e["advantage"] for e in out] == pytest.approx([-1.0, 1.0, -0.5, 0.5])
    # max_samples keeps a prefix: the later source is cut first
    cut = prep.grpo_dataset(sources=sources, max_samples=3)
    assert [e["source"] for e in cut] == ["cross_model", "cross_model", "generalization"]


def test_grpo_dataset_excludes_heldout_papers(prep, use_rows, reasoning):
    use_rows([make_row(HELD, "innovate", "h1", test=1.0),
              make_row(HELD, "innovate", "h2", test=2.0),
              make_row(OTHER, "innovate", "a1", test=1.0),
              make_row(OTHER, "innovate", "a2", test=2.0)])
    assert {e["task_id"] for e in prep.grpo_dataset()} == {OTHER}
    assert {e["task_id"] for e in prep.grpo_dataset(exclude_papers=None)} == {HELD, OTHER}


def test_grpo_dataset_rejects_unknown_source(prep, use_rows, reasoning):
    use_rows([])
    with pytest.raises(ValueError, match="unknown GRPO source: revision"):
        prep.grpo_dataset(sources=("revision",))


def test_judge_sources_read_the_craft_scores(prep, use_rows, reasoning):
    write_jsonl(prep._CRAFT_JUDGE, [
        {"run_id": "a1", "overall": 40, "novelty": 90},
        {"run_id": "a2", "overall": 60, "novelty": 10},
        {"run_id": "a3", "error": "judge_failed"},
    ])
    use_rows([make_row(OTHER, "innovate", r) for r in ("a1", "a2", "a3")])
    out = prep.grpo_dataset(sources=("judge", "judge_novelty"))
    assert [(e["source"], e["reward"]) for e in out] == [
        ("judge", 40), ("judge", 60), ("judge_novelty", 90), ("judge_novelty", 10)]


def test_judge_source_without_scores_names_the_published_sources(prep, use_rows, reasoning):
    use_rows([])
    with pytest.raises(SystemExit) as exc:
        prep.grpo_dataset(sources=("judge",))
    assert "craft_judge.jsonl not found" in str(exc.value)
    assert "judge_craft_batch.py" in str(exc.value)
    assert "--sources cross_model,generalization" in str(exc.value)


# --------------------------------------------------------------- SFT examples

def test_sft_examples_selection(prep, use_rows, reasoning):
    use_rows([
        make_row(OTHER, "innovate", "m1", test=None, versions=3),  # has reasoning
        make_row(OTHER, "improve", "s1", test=0.5),
        make_row(OTHER2, "innovate", "s2", test=2.0),
        make_row(OTHER2, "improve", "s3", test=None),              # no reasoning, no score
        make_row(OTHER2, "improve", "s4", test=float("nan")),      # non-finite score
        make_row(OTHER, "innovate", "s5", test=1.0, idea=None),    # no final idea
        make_row(HELD, "innovate", "h1", test=9.0),                # held-out paper
    ])
    exs = prep.sft_examples()
    # reasoning examples first (kept even without a score), then by score
    assert [e["run_id"] for e in exs] == ["m1", "s2", "s1"]
    assert [e["has_reasoning"] for e in exs] == [True, False, False]
    assert exs[0]["conversation"][2]["content"] == "<think>\nwhy\n</think>\n\nidea"
    assert exs[1]["conversation"][2]["content"] == exs[1]["completion"] == "idea"
    assert exs[1]["prompt"] == exs[1]["conversation"][1]["content"]

    assert [e["run_id"] for e in prep.sft_examples(n_single_backfill=1)] == ["m1", "s2"]
    assert [e["run_id"] for e in prep.sft_examples(modes=("improve",))] == ["s1"]
    # the held-out papers enter only when the exclusion is switched off
    everything = prep.sft_examples(exclude_papers=None)
    assert [e["run_id"] for e in everything] == ["m1", "h1", "s2", "s1"]


def test_sft_examples_adds_rephrased_copies(prep, use_rows, reasoning):
    use_rows([make_row(OTHER, "innovate", "m1", test=1.0, versions=2),
              make_row(HELD, "innovate", "h1", test=9.0)])
    write_jsonl(prep._REPHRASED, [
        {"run_id": "m1", "task_id": OTHER,
         "rephrased_completion": "<think>\nso\n</think>\n\nreworded"},
        {"run_id": "h1", "task_id": HELD, "rephrased_completion": "held out"},
        {"run_id": "gone", "task_id": OTHER, "rephrased_completion": "no such run"},
        {"run_id": "m9", "task_id": OTHER, "error": "rephrase_failed"},
    ])
    assert [e["run_id"] for e in prep.sft_examples()] == ["m1"]
    exs = prep.sft_examples(include_rephrased=True)
    assert [e["run_id"] for e in exs] == ["m1", "m1-aug"]
    original, copy = exs
    assert copy["augmented"] is True and copy["has_reasoning"] is True
    assert copy["conversation"][:2] == original["conversation"][:2]
    assert copy["conversation"][2]["content"] == copy["completion"]
    assert copy["completion"] == "<think>\nso\n</think>\n\nreworded"


# ------------------------------------------------------------- prepared files

def test_missing_reasoning_file_is_an_error(prep, use_rows):
    use_rows([make_row(OTHER, "innovate", "a1", test=1.0),
              make_row(OTHER, "innovate", "a2", test=2.0)])
    for build in (prep.sft_examples, prep.grpo_dataset):
        with pytest.raises(SystemExit) as exc:
            build()
        assert "reasoning.jsonl not found" in str(exc.value)
        assert "clean_reasoning_batch.py" in str(exc.value)


def test_missing_rephrased_file_is_an_error_only_when_used(prep, use_rows, reasoning):
    use_rows([make_row(OTHER, "innovate", "m1", versions=2)])
    assert len(prep.sft_examples()) == 1
    with pytest.raises(SystemExit) as exc:
        prep.sft_examples(include_rephrased=True)
    assert "rephrased.jsonl not found" in str(exc.value)
    assert "rephrase_batch.py" in str(exc.value)


def test_prepared_file_without_usable_record_is_an_error(prep, use_rows):
    prep._REASONING.write_text(
        '{"run_id": "a1", "error": "clean_failed"}\nnot json\n\n')
    use_rows([])
    with pytest.raises(SystemExit, match="no record with a usable 'reasoning'"):
        prep.sft_examples()


# ------------------------------------------------------------ dataset loading

class FakeTable:
    """The part of the `datasets.Dataset` interface that data_prep uses."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        return [r[column] for r in self.rows]

    def select(self, indices):
        return FakeTable(self.rows[i] for i in indices)


@pytest.fixture
def fake_datasets(monkeypatch):
    """Install a stub `datasets` module. Returns (calls, hub): the log of
    load_dataset calls and the mapping repo id -> splits it serves."""
    calls, hub = [], {}
    mod = types.ModuleType("datasets")

    def load_dataset(path, data_files=None, split=None):
        calls.append((path, data_files, split))
        if path == "parquet":
            return FakeTable([{"task_id": "local", "mode": "innovate"}])
        return hub[path]

    mod.load_dataset = load_dataset
    mod.concatenate_datasets = lambda parts: FakeTable(r for p in parts for r in p)
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return calls, hub


def test_load_rows_explicit_parquet(prep, fake_datasets, tmp_path, capsys):
    calls, _ = fake_datasets
    with pytest.raises(SystemExit, match="traces parquet not found"):
        prep._load_rows(str(tmp_path / "nope.parquet"))
    assert calls == []
    path = tmp_path / "mine.parquet"
    path.write_bytes(b"")
    assert [r["task_id"] for r in prep._load_rows(str(path))] == ["local"]
    assert calls == [("parquet", str(path), "train")]
    assert str(path) in capsys.readouterr().out          # the origin is reported


def test_load_rows_prefers_the_local_build(prep, fake_datasets, monkeypatch, tmp_path):
    calls, hub = fake_datasets
    local = tmp_path / "built.parquet"
    local.write_bytes(b"")
    monkeypatch.setattr(prep, "_PARQUET", local)
    monkeypatch.setenv("SW_TRACES_REPO", "someone/traces")
    hub["someone/traces"] = {"innovate": FakeTable([]), "improve": FakeTable([])}
    prep._load_rows()
    assert calls == [("parquet", str(local), "train")]
    prep._load_rows("hub")                               # "hub" forces the Hub dataset
    assert calls[-1] == ("someone/traces", None, None)


def test_hub_fallback_loads_both_modes(prep, fake_datasets, capsys):
    calls, hub = fake_datasets
    hub[prep._HF_REPO] = {
        "innovate": FakeTable([
            {"task_id": "p2", "mode": "innovate", "run_id": "n1"},
            {"task_id": "p1", "mode": "innovate", "run_id": "n2"},
            {"task_id": "p2", "mode": "innovate", "run_id": "n3"}]),
        "improve": FakeTable([
            {"task_id": "p1", "mode": "improve", "run_id": "i1"},
            {"task_id": "p2", "mode": "improve", "run_id": "i2"},
            {"task_id": "p1", "mode": "improve", "run_id": "i3"}]),
    }
    rows = list(prep._load_rows())
    assert calls == [(prep._HF_REPO, None, None)]
    # both splits; papers in id order; inside a (paper, mode) group the rows
    # keep their original order
    assert [r["run_id"] for r in rows] == ["i1", "i3", "n2", "i2", "n1", "n3"]
    assert prep._HF_REPO in capsys.readouterr().out


def test_hub_repo_override_and_missing_split(prep, fake_datasets, monkeypatch):
    calls, hub = fake_datasets
    monkeypatch.setenv("SW_TRACES_REPO", "someone/traces")
    hub["someone/traces"] = {"train": FakeTable([]), "improve": FakeTable([])}
    with pytest.raises(SystemExit) as exc:
        prep._load_rows()
    assert calls == [("someone/traces", None, None)]
    assert "no split named innovate" in str(exc.value)
    assert "improve, train" in str(exc.value)


def test_data_argument(prep):
    ap = argparse.ArgumentParser()
    prep.add_data_argument(ap)
    assert ap.parse_args([]).data is None
    assert ap.parse_args(["--data", "x.parquet"]).data == "x.parquet"
    assert "SW_TRACES_REPO" in ap.format_help()
