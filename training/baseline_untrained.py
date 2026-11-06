"""Sample one idea per held-out paper from the base model or from a trained
state, for the comparison of the training stages.

Without --init-state the untrained Qwen3-8B (thinking) is sampled through a
fresh LoRA client, whose zero adapter leaves the base model unchanged. With
--init-state tinker://... the SFT or GRPO state is sampled instead. The prompt
has the same format as in SFT (system + task specification). One record per
paper, {task_id, mode, response, thinking, idea}, is written to out/<--out>;
evaluation/idea_impl_eval.py reads the three files named below.

    export TINKER_API_KEY=...
    python baseline_untrained.py                                    # out/baseline_untrained.jsonl
    python baseline_untrained.py --init-state tinker://.../<sft>  --out sft_trained_gen.jsonl
    python baseline_untrained.py --init-state tinker://.../<grpo> --out grpo_trained_gen.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tinker
from tinker_cookbook.renderers import Message

from data_prep import (HELDOUT_PAPERS, SYSTEM, TOP10, TOP30, _load_rows,
                       add_data_argument, build_prompt)
from sft import get_renderer

_OUT = Path(__file__).resolve().parent / "out"


def pick_papers(n=10, source=None):
    """The first ``n`` papers, by id, of TOP30 - TOP10 that have an innovate
    row with a final idea. That row supplies the task specification."""
    ds = _load_rows(source)
    seen = {}
    for r in ds:
        if (r["mode"] == "innovate" and r["final_idea"]
                and r["task_id"] in (TOP30 - TOP10) and r["task_id"] not in seen):
            seen[r["task_id"]] = r
    return [seen[p] for p in sorted(seen)[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--renderer", default="qwen3")
    ap.add_argument("--n-papers", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--init-state", default=None,
                    help="tinker:// SFT or GRPO state to sample from; omit = untrained base")
    ap.add_argument("--out", default="baseline_untrained.jsonl",
                    help="output filename under out/")
    add_data_argument(ap)
    args = ap.parse_args()

    # The papers are chosen before the Tinker client is created, so a dataset
    # problem stops the run before anything is started on the service.
    papers = pick_papers(args.n_papers, args.data)
    print(f"papers: {[r['task_id'] for r in papers]}")
    trained_on = sorted({r["task_id"] for r in papers} - HELDOUT_PAPERS)
    if trained_on:
        print(f"WARNING: {trained_on} are not in HELDOUT_PAPERS: the training "
              f"sets do not exclude them", flush=True)

    sc = tinker.ServiceClient()
    if args.init_state:
        tc = sc.create_training_client_from_state(args.init_state)
        sampler_name = "sft-gen"
    else:
        tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
        sampler_name = "untrained-baseline"
    tokenizer = tc.get_tokenizer()
    renderer = get_renderer(tokenizer, args.renderer, args.base_model)
    sampling = tc.save_weights_and_get_sampling_client(name=sampler_name)

    params = tinker.SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.8,
        stop=renderer.get_stop_sequences())

    _OUT.mkdir(exist_ok=True)
    out_path = _OUT / args.out
    with out_path.open("w") as f:
        for r in papers:
            msgs = [Message(role="system", content=SYSTEM),
                    Message(role="user", content=build_prompt(r))]
            prompt = renderer.build_generation_prompt(msgs)
            resp = sample_text(sampling, prompt, params, tokenizer, renderer)
            think, idea = split_think(resp)
            f.write(json.dumps({
                "task_id": r["task_id"], "mode": "innovate",
                "response": resp, "thinking": think, "idea": idea,
            }) + "\n")
            f.flush()
            print(f"  {r['task_id']}: response {len(resp)} chars "
                  f"(think {len(think or '')}, idea {len(idea or '')})", flush=True)
    print(f"wrote {out_path}")


def sample_text(sampling, prompt, params, tokenizer, renderer):
    res = sampling.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
    seq = res.sequences[0]
    toks = getattr(seq, "tokens", None)
    if toks is not None:
        return tokenizer.decode(toks)
    return getattr(seq, "text", "")


def split_think(text):
    if "<think>" in text and "</think>" in text:
        think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        idea = text.split("</think>", 1)[1].strip()
        return think, idea
    return None, text.strip()


if __name__ == "__main__":
    main()
