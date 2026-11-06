"""Stage 2: offline GRPO from the SFT checkpoint, via Tinker.

The ideas are already scored, so nothing is sampled online. They are grouped
by (paper, mode); the group-relative advantage ((reward - group mean) / group
std, computed in data_prep) says how much better or worse an idea is than its
group's average.

The update is a clipped policy gradient (PPO), warm-started from the SFT state:

  1. Build a cross_entropy datum per idea (model_input, target_tokens, and the
     loss weights that mark the assistant tokens).
  2. Compute reference log-probs: a no-gradient `forward` pass of the SFT model
     over each completion. These fix the denominator of the PPO ratio.
  3. Build a PPO datum: logprobs = reference, advantages = the idea's scalar
     advantage, both zero on the prompt. No clip thresholds are passed, so the
     Tinker server default applies.
  4. forward_backward(loss_fn="ppo") + optim_step. The clip keeps every update
     inside a trust region around the SFT policy; a cross-entropy loss with
     signed per-token weights has no such bound on negative-advantage tokens.

    export TINKER_API_KEY=...
    python grpo_offline.py --init-state tinker://.../sworld-sft-1000-heldout \
        --sources cross_model,generalization \
        --weights cross_model=1.0,generalization=0.5 --max-samples 1500

The defaults are not the published configuration; README.md gives the full
published command.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

from data_prep import add_data_argument, grpo_dataset
from sft import get_renderer


def _td(arr) -> tinker.TensorData:
    return tinker.TensorData.from_numpy(np.asarray(arr, dtype=np.float32))


def build_ppo_datum(ce_datum, ref_logprobs, adv: float):
    """Turn a cross_entropy datum + reference log-probs + scalar advantage into a
    PPO datum. The datum's loss weights from conversation_to_datum are used as
    the completion mask: the reference log-probs and the advantage are both
    multiplied by them, which makes them zero on the prompt. The code assumes
    the weights are 1 on the assistant tokens; whether they are depends on the
    installed tinker_cookbook (README.md, 'Version sensitivity')."""
    mask = np.asarray(ce_datum.loss_fn_inputs["weights"].tolist(), dtype=np.float32)
    ref = np.asarray(ref_logprobs, dtype=np.float32) * mask     # 0 on prompt
    advs = mask * float(adv)                                    # scalar adv on completion, 0 on prompt
    return tinker.Datum(
        model_input=ce_datum.model_input,
        loss_fn_inputs={
            "target_tokens": ce_datum.loss_fn_inputs["target_tokens"],
            "logprobs": _td(ref),
            "advantages": _td(advs),
        },
    )


def _completion_nll(loss_fn_outputs, ce_data_batch):
    """Mean NLL over completion tokens only (uses each datum's completion mask)."""
    tot_ll, tot_n = 0.0, 0.0
    for o, ce in zip(loss_fn_outputs, ce_data_batch):
        if "logprobs" not in o:
            continue
        lp = np.asarray(o["logprobs"].tolist(), dtype=np.float64)
        m = np.asarray(ce.loss_fn_inputs["weights"].tolist(), dtype=np.float64)
        tot_ll += float((lp * m).sum())
        tot_n += float(m.sum())
    return -(tot_ll / tot_n) if tot_n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-state", required=True, help="tinker:// SFT state from sft.py")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--renderer", default="qwen3")
    ap.add_argument("--save-name", default="sworld-grpo")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--loss-fn", default="ppo", choices=["ppo", "cispo", "importance_sampling"])
    ap.add_argument("--sources", default="cross_model,generalization,judge,judge_novelty",
                    help="comma-separated reward sources to mix (the published "
                         "run used cross_model,generalization; the judge "
                         "sources need out/craft_judge.jsonl)")
    ap.add_argument("--modes", default="innovate,improve")
    ap.add_argument("--weights",
                    default="cross_model=1.0,generalization=0.5,judge=0.5,judge_novelty=0.5",
                    help="per-source scale on the advantage")
    ap.add_argument("--max-samples", type=int, default=2500,
                    help="keep the first N samples of the mixed set (the "
                         "published run used 1500)")
    ap.add_argument("--nll-abort-mult", type=float, default=2.0,
                    help="abort+save if batch completion NLL exceeds this x the SFT baseline")
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, run only this many examples for 1 step and exit (cheap validation)")
    ap.add_argument("--shuffle-rewards", action="store_true",
                    help="control run: permute the advantages among the members "
                         "of each (group, source). Every group keeps its exact "
                         "set of advantages, so the reward distribution, its "
                         "scale and the number of positive and negative updates "
                         "are unchanged; only the assignment of rewards to "
                         "ideas is destroyed.")
    ap.add_argument("--shuffle-seed", type=int, default=7)
    add_data_argument(ap)
    args = ap.parse_args()

    weights = {}
    for kv in args.weights.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            weights[k.strip()] = float(v)

    # The samples are built before the Tinker client is created, so a missing
    # input stops the run before anything is started on the service.
    groups = grpo_dataset(
        source=args.data,
        sources=tuple(args.sources.split(",")),
        modes=tuple(args.modes.split(",")),
        weights=weights, max_samples=args.max_samples)
    if args.shuffle_rewards:
        by_group = {}
        for i, e in enumerate(groups):
            by_group.setdefault((e.get("group_id"), e.get("source")), []).append(i)
        rng, moved, singles = random.Random(args.shuffle_seed), 0, 0
        for key, idxs in sorted(by_group.items(), key=lambda kv: str(kv[0])):
            if len(idxs) < 2:
                singles += 1
                continue
            advs = [groups[i]["advantage"] for i in idxs]
            shuffled = advs[:]
            # a permutation that moves something: with n>=2 identical values it
            # may be impossible, and then leaving it alone is correct anyway
            for _ in range(20):
                rng.shuffle(shuffled)
                if any(a != b for a, b in zip(advs, shuffled)):
                    break
            for i, a in zip(idxs, shuffled):
                groups[i]["advantage"] = a
            moved += sum(1 for a, b in zip(advs, shuffled) if a != b)
        print(f"SHUFFLED REWARDS (seed {args.shuffle_seed}): {moved} of "
              f"{len(groups)} advantages reassigned within their own group; "
              f"{singles} single-member groups unchanged. Per-group multiset of "
              f"advantages is identical to the unshuffled run.", flush=True)

    if args.smoke:
        groups = groups[:args.smoke]

    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(args.init_state)
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer(tokenizer, args.renderer, args.base_model)

    # 1) cross_entropy datums (model_input, target_tokens, completion mask=weights)
    ce_data = [conversation_to_datum(e["conversation"], renderer,
                                     max_length=args.max_length,
                                     train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
               for e in groups]

    # 2) reference log-probs: no-gradient forward pass of the SFT model
    print(f"Computing reference log-probs over {len(ce_data)} completions "
          f"(SFT forward, no grad)...", flush=True)
    ref_lps = []
    for i in range(0, len(ce_data), args.batch_size):
        b = ce_data[i:i + args.batch_size]
        out = tc.forward(b, "cross_entropy").result()
        for o in out.loss_fn_outputs:
            ref_lps.append(o["logprobs"].tolist())
    sft_nll = _completion_nll(
        [{"logprobs": _td(lp)} for lp in ref_lps], ce_data)
    print(f"SFT baseline completion NLL = {sft_nll:.4f}", flush=True)

    # 3) PPO datums
    ppo_data = [build_ppo_datum(ce_data[j], ref_lps[j], groups[j]["advantage"])
                for j in range(len(ce_data))]

    if args.smoke:
        print(f"SMOKE: 1 {args.loss_fn} step on {len(ppo_data)} examples", flush=True)
        fb = tc.forward_backward(ppo_data, args.loss_fn).result()
        print("loss_fn_outputs[0] keys:", list(fb.loss_fn_outputs[0].keys()), flush=True)
        tc.optim_step(tinker.AdamParams(
            learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2,
            grad_clip_norm=args.grad_clip_norm)).result()
        nll = _completion_nll(fb.loss_fn_outputs, ce_data)
        print(f"SMOKE ok: post-step completion NLL={nll:.4f} (SFT {sft_nll:.4f})", flush=True)
        return

    n_steps = -(-len(ppo_data) // args.batch_size) * args.epochs
    n_papers = len(set(e["task_id"] for e in groups))
    print(f"Offline GRPO ({args.loss_fn}, server-default clip) on "
          f"{len(ppo_data)} samples ({n_papers} papers) | "
          f"warm-start {args.init_state} | lr={args.lr} epochs={args.epochs} "
          f"batch={args.batch_size} -> {n_steps} steps", flush=True)

    abort_at = args.nll_abort_mult * sft_nll
    step = 0
    for epoch in range(args.epochs):
        idx = list(range(len(ppo_data)))
        random.Random(2025 + epoch).shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            bidx = idx[i:i + args.batch_size]
            batch = [ppo_data[j] for j in bidx]
            fb = tc.forward_backward(batch, args.loss_fn).result()
            tc.optim_step(tinker.AdamParams(
                learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2,
                grad_clip_norm=args.grad_clip_norm)).result()
            step += 1
            nll = _completion_nll(fb.loss_fn_outputs, [ce_data[j] for j in bidx])
            if step % 10 == 0 or step == 1:
                print(f"  step {step}/{n_steps}  completion_nll={nll:.4f} "
                      f"(SFT {sft_nll:.4f})", flush=True)
            if np.isfinite(nll) and nll > abort_at:
                print(f"!! completion NLL {nll:.4f} > {abort_at:.4f} (={args.nll_abort_mult}x "
                      f"SFT): aborting to avoid divergence, saving current state", flush=True)
                path = tc.save_state(name=args.save_name + "-aborted", overwrite=True).result().path
                print(f"SAVED (aborted) state: {path}")
                return
        print(f"epoch {epoch+1}/{args.epochs} done  steps={step}", flush=True)

    path = tc.save_state(name=args.save_name, overwrite=True).result().path
    print(f"\nSAVED GRPO state: {path}")


if __name__ == "__main__":
    main()
