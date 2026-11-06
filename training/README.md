# Training

Code for the training study of Scientist World: a small model is trained on
the traces of finished evaluation runs to propose research ideas, and the ideas
it proposes for ten held-out papers are then implemented by a fixed implementer
(`evaluation/idea_impl_eval.py`).

- Base model: `Qwen/Qwen3-8B` with the thinking renderer (`qwen3`), LoRA rank
  32, trained and sampled through the Tinker API.
- Stage 1, SFT (`sft.py`): prompt = task specification, target = a `<think>`
  block of cleaned reasoning followed by the run's final idea, cross-entropy on
  the assistant message only.
- Stage 2, offline GRPO (`grpo_offline.py`): already-scored ideas, grouped by
  (paper, mode), rewarded by their measured scores, clipped PPO loss with
  reference log-probabilities from the SFT state.
- Sampling (`baseline_untrained.py`): one idea per held-out paper from the
  base model, the SFT state and the GRPO state.

The scripts are plain files, not a package. Run them from this directory
(`python sft.py ...`); all outputs go to `out/` next to the scripts.

## Files

| File | Purpose |
|---|---|
| `data_prep.py` | Loads the traces dataset, builds the SFT examples and the GRPO samples, holds the held-out paper list. `python data_prep.py` prints set sizes and one sample. |
| `clean_reasoning.py`, `clean_reasoning_batch.py` | Rewrite each multi-version run's chain of idea versions as a forward reasoning trace without experiment results. Writes `out/reasoning.jsonl`. |
| `rephrase.py`, `rephrase_batch.py` | Reword the best-scoring SFT completions as extra SFT examples. Writes `out/rephrased.jsonl`. |
| `sft.py` | Stage 1. Saves a Tinker state and prints its `tinker://` path. |
| `grpo_offline.py` | Stage 2, warm-started from the SFT state. Saves a Tinker state. |
| `baseline_untrained.py` | Samples one idea per held-out paper from the base model or from a saved state. |
| `export_data.py` | Writes the two training sets of the published configuration to JSONL, without Tinker. |
| `reproduce.sh` | The published commands, one step per invocation. |
| `judge_craft.py`, `judge_craft_batch.py` | Optional craft judge. Built but not used for the reported results (see below). |
| `tests/` | Tests that need neither Tinker, nor `codex`, nor the dataset. |

## Requirements

- Python 3.12 or newer, and `pip install -r requirements.txt` (`tinker`,
  `tinker_cookbook`, `datasets`, `numpy`, pinned to the versions the code was
  last run with; see "Version sensitivity").
- A Tinker API key (from https://tinker-console.thinkingmachines.ai) in
  `TINKER_API_KEY`, for `sft.py`, `grpo_offline.py` and
  `baseline_untrained.py`. Training and sampling run on the Tinker service
  (documentation: https://tinker-docs.thinkingmachines.ai/); the scripts use no
  local GPU.
- The `codex` CLI on `PATH`, logged in, for the two rewriting passes (and the
  optional judge). Each pass runs
  `codex exec --skip-git-repo-check --sandbox read-only --model <model> -c reasoning_effort=<effort> --output-schema <file> -`
  with the prompt on stdin. The published run used model `gpt-5.5` and effort
  `medium`; these are the defaults.
- The traces dataset (next section). If it is read from the Hugging Face Hub
  and is not public for your account, authenticate first (for example with
  `HF_TOKEN`).

Environment variables:

| Variable | Read by | Meaning |
|---|---|---|
| `TINKER_API_KEY` | the Tinker client | required for training and sampling |
| `SW_TRACES_REPO` | `data_prep.py` | Hub id of the traces dataset, default `zlab-princeton/scientist-world-traces` |
| `SW_CODEX_MODEL` | the `codex` passes | default of `--model`, itself defaulting to `gpt-5.5` |
| `SW_CODEX_EFFORT` | the `codex` passes | default of `--effort`, itself defaulting to `medium` |
| `PYTHON`, `SFT_STATE`, `GRPO_STATE`, `DATA` | `reproduce.sh` | interpreter, the two `tinker://` state paths, optional local parquet |

## Data

The input is the Scientist World traces dataset built by
`evaluation/dataset_builder`: one row per finished evaluation run. Every script
that reads it takes `--data`, and resolves the dataset in this order:

1. `--data PATH`: a local parquet file.
2. `../evaluation/dataset_builder/out/scientist_world_traces.parquet`, the
   dataset builder's output, if it exists.
3. The Hub dataset named by `SW_TRACES_REPO`, default
   `zlab-princeton/scientist-world-traces`. Its two splits, `innovate` and
   `improve`, are both loaded and merged. `--data hub` forces this choice.

Each script prints the source it used (`traces dataset: ...`).

Columns read:

| Column | Use |
|---|---|
| `task_id` | arXiv id of the paper; grouping and held-out exclusion |
| `mode` | `innovate` or `improve`; both are used |
| `run_id` | key of the prepared files in `out/` |
| `problem`, `direction`, `constraint`, `target` | the prompt (task specification) |
| `final_idea` | the completion; rows without it are skipped |
| `idea_versions`, `n_idea_versions` | input of the reasoning rewrite |
| `test_score`, `train_score` | SFT ordering and GRPO rewards |

The `thinking_trace` column is not used: the `<think>` text is the rewrite of
`idea_versions`. The dataset holds grader-verified runs only; no filter on
`score_locked` is applied.

Row order matters for `--max-samples` in `grpo_offline.py`, which keeps a
prefix of the sample list. (It also sets the order in which examples enter the
per-epoch shuffle and breaks ties between equal scores.) The dataset builder
writes its parquet paper by paper, in id order. The Hub splits are merged in
(`task_id`, `mode`) order, which gives the same paper order and keeps the order
of rows inside each (paper, mode) group, but not necessarily the order of the
two modes of one paper.

## Data preparation

Two passes write the prepared files that the trainers need. Both are
resumable: a rerun skips the runs already done and retries the failures.

```bash
python clean_reasoning_batch.py     # -> out/reasoning.jsonl
python rephrase_batch.py            # -> out/rephrased.jsonl  (needs reasoning.jsonl)
```

| File | Records | Written by | Needed by |
|---|---|---|---|
| `out/reasoning.jsonl` | `{run_id, task_id, mode, reasoning}` | `clean_reasoning_batch.py`, for every run with a final idea and at least two idea versions | SFT and GRPO sets (the `<think>` block) |
| `out/rephrased.jsonl` | `{run_id, task_id, rephrased_completion}` | `rephrase_batch.py`, for the `--n` (75) best-scoring examples of the held-out-excluded SFT set | `sft.py --exclude-heldout`, `export_data.py` |
| `out/craft_judge.jsonl` | `{run_id, task_id, mode, novelty, soundness, overall, reason}` | `judge_craft_batch.py` (optional) | GRPO sources `judge`, `judge_novelty` |

A failed call is recorded as `{run_id, ..., error}` and ignored by the readers.
A prepared file that is missing, or that holds no usable record, stops the
script that needs it with a message naming the pass to run; the trainers build
their sets before they contact Tinker, so this happens before anything is
started on the service. An incomplete file is not detected: a run without a
record is treated as a run without reasoning, so check the `ok=` / `fail=`
counts that the passes print.

`python export_data.py` then writes the two training sets of the published
configuration to `out/sft_train.jsonl` and `out/grpo_train.jsonl` and prints
their sizes. It needs no Tinker access and shows what your dataset snapshot
selects before any training is paid for.

## How the training sets are built

Prompt (both stages): a fixed system message plus a user message with the
sections `# Problem`, `# Direction`, `# Constraints`, `# Targets`, taken from
the columns above. Empty sections are left out. The prompt does not include
the mode. The dataset builder reads these four columns from the paper's task
specification, once per paper, so the innovate and improve runs of one paper
share the same prompt.

SFT examples (`data_prep.sft_examples`):

- A run with a record in `out/reasoning.jsonl` gives a reasoning example: the
  target is `<think>`, the cleaned reasoning, `</think>`, then the final idea.
  These runs are used whether or not they have a score.
- Any other run gives a plain final-idea example, and only if it has a finite
  `test_score`. They are taken in decreasing `test_score` order.
- With the reworded examples switched on, each record of `out/rephrased.jsonl`
  adds one more example for its run: same prompt, reworded target.

`sft.py` selects in one of two ways:

- `--exclude-heldout` (the published selection): the ten held-out papers are
  dropped, every scored non-reasoning run is used, and the reworded examples
  are added.
- default: all papers are used, including the held-out ones, with the
  `--n-backfill` (37) best-scoring non-reasoning runs and no reworded examples.

`--val-frac` (default 0.03) moves that fraction of the papers (at least one
paper) to a validation split that is only evaluated; `--val-frac 0` trains on
everything selected.
The loss is cross-entropy on the last assistant message, examples are
truncated to `--max-length` tokens, the learning rate is constant, and the
order is reshuffled every epoch with a fixed seed. A state is saved after each
epoch (`<save-name>-ep<k>`) and at the end (`<save-name>`); the loss curve and
a manifest with the arguments and checkpoint paths go to
`out/runs/<save-name>/`.

GRPO samples (`data_prep.grpo_dataset`):

| Source | Reward | Eligible rows |
|---|---|---|
| `cross_model` | `test_score` | final idea and a `test_score` |
| `generalization` | `test_score - max(0, train_score - test_score)` | final idea, finite `test_score` and `train_score` |
| `judge`, `judge_novelty` (optional) | craft judge `overall` / `novelty` | final idea and a judge record |

- Groups are (paper, mode). Inside a group the advantage of a sample is the
  z-score of its reward (population standard deviation). Groups with fewer
  than two finite rewards or with zero spread are dropped.
- The advantage is multiplied by the weight of its source (`--weights`).
- The completion has the same format as in SFT (with the `<think>` block when
  the run has cleaned reasoning).
- The held-out papers are always excluded.
- The sources are concatenated in the order of `--sources`, so an idea that is
  eligible for two sources appears once per source, and `--max-samples` keeps
  the first N entries of that list. It is a prefix, not a random sample: the
  last source is cut first, in dataset row order.

`grpo_offline.py` computes reference log-probabilities with a forward pass of
the initial (SFT) state, builds one PPO datum per sample (reference
log-probabilities and the scalar advantage on the completion tokens) and
trains with Tinker's `ppo` loss. No clip thresholds are passed, so the Tinker
server default applies. If the completion NLL of a batch exceeds
`--nll-abort-mult` times the NLL of the initial state, the run stops and saves
`<save-name>-aborted`.

## Published commands

Several defaults differ from the published configuration, so every flag that
affects the result is written out.
`./reproduce.sh {prep|export|sft|grpo|sample}` runs the same commands.

```bash
# 1. Data preparation (codex CLI, no Tinker)
python clean_reasoning_batch.py --workers 6 --model gpt-5.5 --effort medium
python rephrase_batch.py --n 75 --workers 6 --model gpt-5.5 --effort medium

# 2. Stage 1: SFT. Prints "SAVED SFT state: tinker://.../sworld-sft-1000-heldout"
python sft.py \
    --base-model Qwen/Qwen3-8B --rank 32 --renderer qwen3 --max-length 8192 \
    --exclude-heldout --val-frac 0 --train-on all \
    --epochs 3 --lr 1e-4 --batch-size 8 \
    --save-name sworld-sft-1000-heldout

# 3. Stage 2: offline GRPO from the SFT state.
#    Prints "SAVED GRPO state: tinker://.../sworld-grpo-1500-real"
python grpo_offline.py --init-state "$SFT_STATE" \
    --base-model Qwen/Qwen3-8B --renderer qwen3 --max-length 8192 \
    --sources cross_model,generalization \
    --weights cross_model=1.0,generalization=0.5 \
    --modes innovate,improve --max-samples 1500 \
    --loss-fn ppo --lr 1e-5 --epochs 1 --batch-size 16 \
    --beta1 0.9 --beta2 0.95 --grad-clip-norm 1.0 --nll-abort-mult 2.0 \
    --save-name sworld-grpo-1500-real

# 4. One idea per held-out paper from the base model, the SFT state and the GRPO state
python baseline_untrained.py \
    --base-model Qwen/Qwen3-8B --rank 32 --renderer qwen3 \
    --n-papers 10 --temperature 0.7 --max-tokens 4096 \
    --out baseline_untrained.jsonl
python baseline_untrained.py --init-state "$SFT_STATE" \
    --base-model Qwen/Qwen3-8B --renderer qwen3 \
    --n-papers 10 --temperature 0.7 --max-tokens 4096 \
    --out sft_trained_gen.jsonl
python baseline_untrained.py --init-state "$GRPO_STATE" \
    --base-model Qwen/Qwen3-8B --renderer qwen3 \
    --n-papers 10 --temperature 0.7 --max-tokens 4096 \
    --out grpo_trained_gen.jsonl
```

`$SFT_STATE` and `$GRPO_STATE` are the `tinker://` paths printed by steps 2
and 3. Sampling uses top-p 0.8, which is fixed in `baseline_untrained.py`.

Defaults that differ from the published values:

| Script | Flag | Default | Published |
|---|---|---|---|
| `sft.py` | `--epochs` | 1 | 3 |
| `sft.py` | `--exclude-heldout` | off: all papers, held-out ones included | on |
| `sft.py` | `--val-frac` | 0.03 | 0 |
| `sft.py` | `--save-name` | `sworld-innovate-sft` | `sworld-sft-1000-heldout` |
| `grpo_offline.py` | `--sources` | `cross_model,generalization,judge,judge_novelty` | `cross_model,generalization` |
| `grpo_offline.py` | `--weights` | `cross_model=1.0,generalization=0.5,judge=0.5,judge_novelty=0.5` | `cross_model=1.0,generalization=0.5` |
| `grpo_offline.py` | `--max-samples` | 2500 | 1500 |
| `grpo_offline.py` | `--save-name` | `sworld-grpo` | `sworld-grpo-1500-real` |

With its default `--sources`, `grpo_offline.py` needs `out/craft_judge.jsonl`
and stops with a message if it is missing.

Other switches, not used for the reported results:

- `sft.py --mixture FILE` trains on a JSONL file of `{"conversation": [...]}`
  records instead of the set built from the traces, for example a frozen copy
  written by `export_data.py`. A file containing a held-out paper is rejected.
- `sft.py --train-on think` restricts the loss to the `<think>` block.
- `sft.py --smoke N` trains on the first N examples only, and
  `grpo_offline.py --smoke N` takes a single step on the first N samples and
  exits. Both are cheap checks of the loop and still call Tinker.
- `grpo_offline.py --shuffle-rewards` (with `--shuffle-seed`) is a control: it
  permutes the advantages among the members of each group, so only the
  assignment of rewards to ideas is destroyed.

## Held-out papers

The ten held-out papers (`data_prep.HELDOUT_PAPERS`):

```
2501.01741  2502.10563  2503.06808  2503.10197  2503.21907
2504.13752  2505.11883  2505.21825  2506.04452  2506.05188
```

What the code shows about how they were chosen: `data_prep.py` holds two fixed
id lists, `TOP30` (30 papers) and `TOP10` (10 of them). `baseline_untrained.py`
samples the first `--n-papers` ids, in sorted order, of `TOP30` minus `TOP10`
that have an innovate row with a final idea in the dataset, and
`HELDOUT_PAPERS` is the first ten ids of that sorted difference (a test checks
this). The code does not record how `TOP30` and `TOP10` were selected.

Only these ten papers are excluded from training: by `sft.py --exclude-heldout`
and always by the GRPO set. The papers of `TOP10` and the other ten papers of
`TOP30` minus `TOP10` are not excluded. With `--n-papers` above 10, or with a
dataset in which a held-out paper has no innovate row with a final idea,
`baseline_untrained.py` would therefore sample a paper that the training sets
may contain; it prints a warning when that happens.

## Hand-off to the implementer

`baseline_untrained.py` writes one record per paper,
`{task_id, mode, response, thinking, idea}`: `response` is the decoded sample,
`thinking` the text between `<think>` and `</think>`, `idea` the text after
it. `evaluation/idea_impl_eval.py` reads three files by name and uses `task_id`
and `idea`:

| Stage | File |
|---|---|
| `base` | `out/baseline_untrained.jsonl` |
| `sft` | `out/sft_trained_gen.jsonl` |
| `grpo` | `out/grpo_trained_gen.jsonl` |

It then has one fixed implementer carry out each idea in the paper's innovate
environment and score it. That harness, its requirements and its flags are
documented in `evaluation/`.

## Reproducibility

The repository contains the full pipeline, the published configuration
(`reproduce.sh`) and the data selection logic. Rerunning it trains the same
method on your snapshot of the traces, not the same checkpoint:

- The set sizes are not fixed in code. With `--exclude-heldout` every eligible
  run is used, and the GRPO set is a prefix of the eligible samples. The traces
  dataset keeps growing, so a later snapshot selects more SFT examples than the
  1,000 of the published run and a different prefix of 1,500 GRPO samples.
  `python export_data.py` prints the sizes and composition for your snapshot
  before any training is paid for.
- `out/reasoning.jsonl` and `out/rephrased.jsonl` are written by a language
  model and differ between rebuilds.
- The frozen training sets and the trained Tinker states of the published run
  are not part of this repository; `reproduce.sh` has placeholders for the two
  state paths.
- Sampling is stochastic (temperature 0.7, no seed), so the sampled ideas
  differ between runs.

## Version sensitivity

The trainers take per-token loss weights from
`tinker_cookbook.supervised.data.conversation_to_datum` and do not pass its
`reduction` argument, so they follow the installed version's default. In
`tinker_cookbook` 0.5.7 the default is `reduction="mean"`, which normalizes the
weights of each example to sum to 1. This default sets how examples of
different length are weighted in a batch (`sft.py`) and the scale of the
per-token terms handed to the PPO loss (`build_ppo_datum` in
`grpo_offline.py`). `requirements.txt` pins the versions the code was last run
with; check this default before moving to another version.

## Optional: craft judge

`judge_craft.py` and `judge_craft_batch.py` score every final idea for novelty,
soundness and overall quality from the task and the idea text alone, without
the outcome, and write `out/craft_judge.jsonl`. `data_prep.grpo_dataset` can
use the scores as the reward sources `judge` and `judge_novelty`. This
component was built but not used for the reported results, whose GRPO rewards
are the two measured sources only. It is still part of the default `--sources`
of `grpo_offline.py`, which is why the published command sets `--sources`
explicitly.

## Tests

```bash
python -m pytest tests
```

The tests need only `pytest`. They replace the dataset with in-memory rows,
the `datasets` package with a stub and the `codex` CLI with a fake, so they
run without network, without Tinker and without the heavy dependencies. They
cover the grouping and z-normalization, the held-out exclusion, the SFT
selection rules, the prepared-file checks, the dataset loading order, the
`codex` invocation, `export_data.py`, and the flags used in this file and in
`reproduce.sh`. Of `sft.py`, `grpo_offline.py` and `baseline_untrained.py` only
the argument parsing and the helpers that do not touch Tinker are tested; the
training and sampling loops need the service.

## Operational notes

- The rewriting passes send task specifications and idea texts from the
  dataset to the model provider behind the `codex` CLI, one call per run, with
  up to `--workers` calls in parallel. `codex` runs with a read-only sandbox in
  the system temporary directory.
- `TINKER_API_KEY` is read from the environment by the Tinker client. The
  scripts write their outputs under `out/` (or the `--log-dir` / `--out-dir`
  you pass). The passes also create a short-lived schema file in the system
  temporary directory, and `datasets` keeps its own cache.
- `sft.py` and `grpo_offline.py` save their states with `overwrite=True`: a
  rerun with the same `--save-name` replaces the earlier state.
