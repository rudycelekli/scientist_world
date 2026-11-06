#!/usr/bin/env bash
# The training run reported in the paper, one step per invocation:
#
#   ./reproduce.sh prep      codex passes -> out/reasoning.jsonl, out/rephrased.jsonl
#   ./reproduce.sh export    optional, no Tinker: write the two training sets to out/
#   ./reproduce.sh sft       stage 1; prints "SAVED SFT state: tinker://..."
#   ./reproduce.sh grpo      stage 2; needs SFT_STATE; prints "SAVED GRPO state: ..."
#   ./reproduce.sh sample    one idea per held-out paper from base, SFT and GRPO;
#                            needs SFT_STATE and GRPO_STATE
#
# Each trainer prints the tinker:// URI of the state it saves. Put the two URIs
# in SFT_STATE and GRPO_STATE (environment, or replace the placeholders below)
# before the steps that need them.
#
# Every flag that affects the result is written out, because several script
# defaults differ from the published configuration. README.md
# ("Reproducibility") says what a rerun reproduces.
#
# Optional: DATA=/path/to/traces.parquet makes every step read that file
# instead of the default dataset location.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
SFT_STATE="${SFT_STATE:-tinker://REPLACE-WITH-SFT-STATE/sworld-sft-1000-heldout}"
GRPO_STATE="${GRPO_STATE:-tinker://REPLACE-WITH-GRPO-STATE/sworld-grpo-1500-real}"

need_tinker() {
    : "${TINKER_API_KEY:?set TINKER_API_KEY first}"
}

need_state() {
    case "$2" in
    *REPLACE-WITH-*)
        echo "set $1 to the tinker:// URI printed by the step that saved it" >&2
        exit 2
        ;;
    esac
}

case "${1:-}" in
prep)
    "$PYTHON" clean_reasoning_batch.py ${DATA:+--data "$DATA"} \
        --workers 6 --model gpt-5.5 --effort medium
    "$PYTHON" rephrase_batch.py ${DATA:+--data "$DATA"} \
        --n 75 --workers 6 --model gpt-5.5 --effort medium
    ;;
export)
    "$PYTHON" export_data.py ${DATA:+--data "$DATA"}
    ;;
sft)
    need_tinker
    "$PYTHON" sft.py ${DATA:+--data "$DATA"} \
        --base-model Qwen/Qwen3-8B --rank 32 --renderer qwen3 --max-length 8192 \
        --exclude-heldout --val-frac 0 --train-on all \
        --epochs 3 --lr 1e-4 --batch-size 8 \
        --save-name sworld-sft-1000-heldout
    ;;
grpo)
    need_tinker
    need_state SFT_STATE "$SFT_STATE"
    "$PYTHON" grpo_offline.py ${DATA:+--data "$DATA"} \
        --init-state "$SFT_STATE" \
        --base-model Qwen/Qwen3-8B --renderer qwen3 --max-length 8192 \
        --sources cross_model,generalization \
        --weights cross_model=1.0,generalization=0.5 \
        --modes innovate,improve --max-samples 1500 \
        --loss-fn ppo --lr 1e-5 --epochs 1 --batch-size 16 \
        --beta1 0.9 --beta2 0.95 --grad-clip-norm 1.0 --nll-abort-mult 2.0 \
        --save-name sworld-grpo-1500-real
    ;;
sample)
    need_tinker
    need_state SFT_STATE "$SFT_STATE"
    need_state GRPO_STATE "$GRPO_STATE"
    "$PYTHON" baseline_untrained.py ${DATA:+--data "$DATA"} \
        --base-model Qwen/Qwen3-8B --rank 32 --renderer qwen3 \
        --n-papers 10 --temperature 0.7 --max-tokens 4096 \
        --out baseline_untrained.jsonl
    "$PYTHON" baseline_untrained.py ${DATA:+--data "$DATA"} \
        --init-state "$SFT_STATE" \
        --base-model Qwen/Qwen3-8B --renderer qwen3 \
        --n-papers 10 --temperature 0.7 --max-tokens 4096 \
        --out sft_trained_gen.jsonl
    "$PYTHON" baseline_untrained.py ${DATA:+--data "$DATA"} \
        --init-state "$GRPO_STATE" \
        --base-model Qwen/Qwen3-8B --renderer qwen3 \
        --n-papers 10 --temperature 0.7 --max-tokens 4096 \
        --out grpo_trained_gen.jsonl
    ;;
*)
    echo "usage: $0 {prep|export|sft|grpo|sample}" >&2
    exit 2
    ;;
esac
