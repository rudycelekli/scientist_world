"""Stage 1: LoRA SFT of Qwen3-8B on task-specification -> idea examples, via Tinker.

The loss is cross-entropy on the last assistant message only (the <think>
reasoning block plus the final idea); the system and user turns are masked.
A resumable Tinker state is saved after every epoch and at the end, and its
tinker:// path is printed: grpo_offline.py and baseline_untrained.py take it
as --init-state.

    export TINKER_API_KEY=...
    python sft.py --exclude-heldout --epochs 3 --lr 1e-4 --batch-size 8 --val-frac 0

The defaults are not the published configuration; README.md gives the full
published command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

from data_prep import HELDOUT_PAPERS, add_data_argument, sft_examples

_OUT = Path(__file__).resolve().parent / "out"


def get_renderer(tokenizer, name: str, base_model: str):
    """Resolve a renderer by name, falling back to the Qwen renderers."""
    for n in [name, "qwen3", "qwen3_5", "role_colon"]:
        try:
            renderer = renderers.get_renderer(n, tokenizer, model_name=base_model)
        except Exception:
            continue
        if n != name:
            print(f"WARNING: renderer '{name}' could not be resolved, "
                  f"using '{n}' instead", flush=True)
        return renderer
    raise SystemExit(f"could not resolve a renderer (tried {name}, qwen3, ...)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1,
                    help="passes over the training set (the published run used 3)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--renderer", default="qwen3")
    ap.add_argument("--save-name", default="sworld-innovate-sft")
    ap.add_argument("--exclude-heldout", action="store_true",
                    help="the published data selection: drop the ten held-out "
                         "papers, use every scored run that has no reasoning "
                         "record, and add the reworded examples of "
                         "out/rephrased.jsonl. Without it the held-out papers "
                         "are trained on.")
    ap.add_argument("--n-backfill", type=int, default=37,
                    help="without --exclude-heldout: how many of the "
                         "best-scoring runs without a reasoning record to add "
                         "to the reasoning examples")
    ap.add_argument("--mixture",
                    help="JSONL file of {\"conversation\": [...]} records to "
                         "train on instead of the set built from the traces, "
                         "for example a frozen copy written by export_data.py. "
                         "A file that contains a held-out paper (by task_id) "
                         "is rejected.")
    ap.add_argument("--smoke", type=int, default=0,
                    help="train on the first N examples only, to check the "
                         "loop before spending a full run")
    ap.add_argument("--val-frac", type=float, default=0.03,
                    help="fraction of PAPERS held out for validation loss, so a "
                         "paper the model trained on cannot appear in its own "
                         "validation number. 0 disables (the published run "
                         "had no validation split).")
    ap.add_argument("--train-on", choices=["all", "think"], default="all",
                    help="'all' trains on the whole assistant message. 'think' "
                         "zeroes the loss after the last </think>, so only the "
                         "reasoning block is trained; examples without a "
                         "</think> are dropped.")
    ap.add_argument("--log-dir", default=str(_OUT / "runs"),
                    help="per-run curve.jsonl + manifest.json land here, "
                         "under <save-name>/")
    add_data_argument(ap)
    args = ap.parse_args()

    # The examples are selected before the Tinker client is created, so a
    # missing input stops the run before anything is started on the service.
    if args.mixture:
        exs = [json.loads(l) for l in
               Path(args.mixture).read_text().splitlines() if l.strip()]
        leaked = {e.get("task_id") for e in exs} & HELDOUT_PAPERS
        if leaked:
            raise SystemExit(f"held-out papers in the mixture: {sorted(leaked)}")
    elif args.exclude_heldout:
        exs = sft_examples(source=args.data, exclude_papers=HELDOUT_PAPERS,
                           n_single_backfill=None, include_rephrased=True)
    else:
        exs = sft_examples(source=args.data, exclude_papers=None,
                           n_single_backfill=args.n_backfill,
                           include_rephrased=False)
    if args.smoke:
        exs = exs[:args.smoke]

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer(tokenizer, args.renderer, args.base_model)

    # Validation split: held out by PAPER, not by example, so a paper the model
    # trained on cannot show up in its own validation loss.
    val = []
    if args.val_frac > 0 and any(e.get("task_id") for e in exs):
        papers = sorted({e["task_id"] for e in exs if e.get("task_id")})
        random.Random(99).shuffle(papers)
        held = set(papers[:max(1, int(len(papers) * args.val_frac))])
        val = [e for e in exs if e.get("task_id") in held]
        exs = [e for e in exs if e.get("task_id") not in held]

    _END = tokenizer.encode("</think>", add_special_tokens=False)

    def _think_only(d):
        """Zero the mask past the last </think>. Leaves the prompt masked as it
        already is, so what remains is exactly the reasoning."""
        ids = d.model_input.to_ints()
        m = np.asarray(d.loss_fn_inputs["weights"].tolist(), dtype=np.float32)
        cut = None
        for i in range(len(ids) - len(_END), -1, -1):
            if ids[i:i + len(_END)] == _END:
                cut = i + len(_END)
                break
        if cut is None:
            return None                  # malformed: drop rather than guess
        m[cut:] = 0.0
        tot = m.sum()
        if tot <= 0:
            return None
        d.loss_fn_inputs["weights"] = tinker.TensorData.from_numpy(m / tot)
        return d

    def _datums(rows):
        out = []
        for e in rows:
            d = conversation_to_datum(e["conversation"], renderer,
                                      max_length=args.max_length,
                                      train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            if args.train_on == "think":
                d = _think_only(d)
                if d is None:
                    continue
            out.append(d)
        return out

    data, val_data = _datums(exs), _datums(val)

    def _mask(d):
        """The datum's loss weights as a 0/1 completion mask. tinker_cookbook
        can return them normalized to sum to 1 per example; they are binarized
        here so a long answer counts for more tokens than a short one when
        losses are pooled across a batch."""
        m = np.asarray(d.loss_fn_inputs["weights"].tolist(), dtype=np.float64)
        return (m > 0).astype(np.float64)

    def _nll(outs, batch):
        """Completion-only NLL: prompt tokens are masked out, as in training."""
        ll = n = 0.0
        for o, d in zip(outs, batch):
            if "logprobs" not in o:
                continue
            lp = np.asarray(o["logprobs"].tolist(), dtype=np.float64)
            m = _mask(d)[:lp.size]
            ll += -float((lp[:m.size] * m).sum())
            n += float(m.sum())
        return ll, n

    def _val_nll():
        if not val_data:
            return None
        ll = n = 0.0
        for i in range(0, len(val_data), args.batch_size):
            b = val_data[i:i + args.batch_size]
            out = tc.forward(b, "cross_entropy").result()
            a, c = _nll(out.loss_fn_outputs, b)
            ll += a
            n += c
        return ll / max(1.0, n)

    run_dir = Path(args.log_dir) / args.save_name
    run_dir.mkdir(parents=True, exist_ok=True)
    curve = (run_dir / "curve.jsonl").open("a")

    def log(rec):
        curve.write(json.dumps(rec) + "\n")
        curve.flush()

    manifest = {
        "save_name": args.save_name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args), "base_model": args.base_model,
        "mixture": args.mixture,
        "mixture_sha256": (hashlib.sha256(Path(args.mixture).read_bytes()).hexdigest()[:16]
                           if args.mixture else None),
        "n_train": len(data), "n_val": len(val_data),
        "val_papers": sorted({e["task_id"] for e in val if e.get("task_id")}),
        "train_papers": len({e["task_id"] for e in exs if e.get("task_id")}),
        "train_tokens": int(sum(_mask(d).sum() for d in data)),
        "checkpoints": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"SFT on {len(data)} examples ({manifest['train_papers']} papers, "
          f"{manifest['train_tokens']:,} completion tokens) | val {len(val_data)} "
          f"from {len(manifest['val_papers'])} validation papers | "
          f"{args.base_model} rank={args.rank} lr={args.lr} "
          f"epochs={args.epochs} bs={args.batch_size}", flush=True)
    print(f"logging -> {run_dir}", flush=True)

    v0 = _val_nll()
    if v0 is not None:
        print(f"val_nll before training = {v0:.4f}", flush=True)
        log({"step": 0, "epoch": 0, "val_nll": v0})

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        idx = list(range(len(data)))
        random.Random(1234 + epoch).shuffle(idx)
        ep_loss, ep_tok = 0.0, 0
        for i in range(0, len(idx), args.batch_size):
            batch = [data[j] for j in idx[i:i + args.batch_size]]
            fb = tc.forward_backward(batch, "cross_entropy").result()
            tc.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
            b_ll, b_n = _nll(fb.loss_fn_outputs, batch)
            ep_loss += b_ll
            ep_tok += b_n
            step += 1
            log({"step": step, "epoch": epoch + 1,
                 "batch_nll": b_ll / max(1.0, b_n),
                 "running_nll": ep_loss / max(1.0, ep_tok),
                 "tokens": int(b_n), "elapsed_s": round(time.time() - t0, 1)})
            if step % 25 == 0 or step == 1:
                el = time.time() - t0
                done = step / (args.epochs * -(-len(data) // args.batch_size))
                print(f"  step {step}  nll={ep_loss / max(1, ep_tok):.4f}  "
                      f"{el / step:.1f}s/step  {100 * done:.0f}%  "
                      f"eta {(el / max(done, 1e-9) - el) / 3600:.1f}h", flush=True)

        v = _val_nll()
        # A state is saved after every epoch, so an earlier epoch can be picked
        # from the curve without rerunning.
        p = tc.save_state(name=f"{args.save_name}-ep{epoch + 1}",
                          overwrite=True).result().path
        manifest["checkpoints"][f"epoch{epoch + 1}"] = p
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"epoch {epoch + 1}/{args.epochs}  train_nll="
              f"{ep_loss / max(1, ep_tok):.4f}"
              + (f"  val_nll={v:.4f}" if v is not None else "")
              + f"  steps={step}  -> {p}", flush=True)
        log({"step": step, "epoch": epoch + 1,
             "train_nll_epoch": ep_loss / max(1.0, ep_tok), "val_nll": v,
             "checkpoint": p})

    path = tc.save_state(name=args.save_name, overwrite=True).result().path
    manifest["checkpoints"]["final"] = path
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["wall_hours"] = round((time.time() - t0) / 3600, 2)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    curve.close()
    print(f"\nSAVED SFT state: {path}", flush=True)
    print(f"curve + manifest: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
