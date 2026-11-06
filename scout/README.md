# Scout

Scout is the paper-selection pipeline of Scientist World. It builds a pool
of recent arXiv papers, converts them to Markdown, screens them with two cheap
LLM passes, runs a structured analysis on the papers that pass, and exports a
ranked, category-balanced list of papers. That list is submitted to Swarm,
whose builders turn each paper into an environment.

The component is a set of standalone Python scripts that share one data
directory. There is no daemon and no configuration file: every setting is a
command-line option.

## How it fits the system

Scout hands two things to Swarm:

- the ordered list of selected arXiv IDs, sent to the Swarm dashboard by
  `submit_to_swarm.py`;
- the data directory. Swarm reads each paper's Markdown, images, PDF and title
  from it, so Swarm's `scout_data_dir` setting (or the environment
  variable `SCOUT_DATA_DIR`) must point at the directory used here.

Nothing in Scout imports another component, and no other component
imports Scout.

## Requirements

- Python 3.10 or newer. The scripts are run by path; nothing is installed as a
  package.
- Stages 1 and 2: network access to arXiv.
- Stage 3: a GPU machine with the MinerU command line tool (`mineru`).
- Stages 4 and 5: the `openai` Python package and an OpenAI-compatible
  chat-completions endpoint. The endpoint should support JSON-schema structured
  output (`response_format: json_schema`); without it the scripts fall back to
  parsing free text.
- Stage 7: the Claude Code CLI (`claude`), installed and authenticated, and
  network access for the model API and web search.
- Stage 9: a running Swarm dashboard.

```bash
pip install -r requirements.txt                       # openai, for stages 4 and 5
pip install "mineru[all]>=2.7.6" "transformers<5"     # stage 3 only, on the GPU machine
```

## Data directory

Every script takes `--data-dir`. When the option is absent the scripts use
`$SCOUT_DATA_DIR`, and if that is unset, `./data` relative to the current
directory. The path is made absolute before use. Exporting
`SCOUT_DATA_DIR` once serves all scripts and Swarm.

```
<data-dir>/
  paper_index.json                      {arxiv_id: metadata}, written by stage 1
  arxiv_listings/<date>/<category>.json list of arXiv IDs for one day and category
  arxiv_pdfs/<id>.pdf
  arxiv_mds/<id>/<id>.md                Markdown of the paper
  arxiv_mds/<id>/images/                figures extracted by MinerU
  arxiv_mds/<id>/<id>.pdf               symlink to arxiv_pdfs/<id>.pdf
  arxiv_prefilter/scores.jsonl          one quality score per line (stage 4)
  arxiv_categories/categories.jsonl     one paper type per line (stage 5)
  arxiv_analysis/<id>/result.json       structured analysis (stage 7)
  arxiv_analysis/<id>/.done             marker: the analysis passed validation
  arxiv_analysis/<id>/analysis.txt      final text of the agent
  arxiv_analysis/<id>/raw_output.jsonl  raw output of the agent CLI
  arxiv_analysis/<id>/*.jsonl           the agent's copy of the reference sheets
```

`.lock` files and `locks/` directories under `arxiv_mds/`, `arxiv_prefilter/`
and `arxiv_categories/` are transient claims of running processes.

`paper_index.json` holds, per paper, the title, abstract, authors, dates,
primary category, all categories and links. `analyze_papers.py --tag NAME`
writes to `arxiv_analysis_NAME/` instead of `arxiv_analysis/`.

## The pipeline

Commands are run from this directory. `$DATA` is the data directory
(`export DATA=/path/to/data`); `HOST`, `MODEL` and `MODEL_ID` are placeholders.

| # | Stage | Script | Command | Input | Output |
|---|---|---|---|---|---|
| 1 | Candidate pool | `fetch_daily_arxiv.py` | `python fetch_daily_arxiv.py --date 2025-06-09 --category cs.CL cs.CV cs.AI cs.LG --data-dir $DATA` | arXiv Atom API | `arxiv_listings/<date>/<category>.json`, `paper_index.json` |
| 2 | PDFs | `download_arxiv_pdfs.py` | `python download_arxiv_pdfs.py --date 2025-06-09 --category cs.CL cs.CV cs.AI cs.LG --data-dir $DATA` | listings of that date | `arxiv_pdfs/<id>.pdf` |
| 3 | PDF to Markdown | `convert_pdf2md.py` | `python convert_pdf2md.py --all --batch-size 80 --data-dir $DATA` | `arxiv_pdfs/` | `arxiv_mds/<id>/` |
| 4 | Quality prefilter | `prefilter_papers.py` | `python prefilter_papers.py --all --api-url http://HOST:8000/v1 --model MODEL --workers 64 --data-dir $DATA` | first 15,000 characters of each Markdown | `arxiv_prefilter/scores.jsonl` |
| 5 | Type prefilter | `categorize_papers.py` | `python categorize_papers.py --all --api-url http://HOST:8000/v1 --model MODEL --workers 64 --data-dir $DATA` | first 15,000 characters of each Markdown | `arxiv_categories/categories.jsonl` |
| 6 | Hand-off to analysis | `select_for_analysis.py` | `python select_for_analysis.py --top-fraction 0.25 --min-score 75 --exclude-type theoretical --out $DATA/analysis_ids.json --data-dir $DATA` | `scores.jsonl`, `categories.jsonl` | JSON list of arXiv IDs |
| 7 | Structured analysis | `analyze_papers.py` | `python analyze_papers.py --ids-file $DATA/analysis_ids.json --provider claude --model MODEL_ID --effort medium --workers 8 --data-dir $DATA` | ID list, `arxiv_mds/`, `arxiv_pdfs/`, `reference_sheets/` | `arxiv_analysis/<id>/result.json` and `.done` |
| 8 | Filter, rank, balance | `export_selection_snapshot.py` | `python export_selection_snapshot.py --count 1500 --temperature 1.2 --ids-out $DATA/selection_ids.json --stats-out $DATA/selection_stats.json --data-dir $DATA` | every finished `arxiv_analysis/<id>/` | ordered ID list, statistics file |
| 9 | Submit to Swarm | `submit_to_swarm.py` | `python submit_to_swarm.py --ids-json $DATA/selection_ids.json --count 300 --data-dir $DATA --swarm-url http://localhost:39100 --submit` | ID list, `arxiv_analysis/` (for topics) | papers queued in Swarm |

Stages 1 to 5 and 7 skip work whose output already exists, so they can be
interrupted and run again. Stages 6 and 8 rewrite their output files, and stage
9 skips papers that Swarm already knows.

### Notes per stage

**1 and 2, pool and PDFs.** One call handles one day. The arXiv `cat:` query
also returns cross-listed papers, so a listing is not limited to papers whose
primary category is the requested one. To respect arXiv's rate limits the
download waits three seconds after each PDF and the listing fetch three
seconds between result pages; pause between days as well when looping over a
date range (GNU `date`):

```bash
d=2025-01-01
while [ "$d" != 2026-03-02 ]; do
    python fetch_daily_arxiv.py   --date "$d" --category cs.CL cs.CV cs.AI cs.LG --data-dir "$DATA"
    python download_arxiv_pdfs.py --date "$d" --category cs.CL cs.CV cs.AI cs.LG --data-dir "$DATA"
    sleep 15
    d=$(date -I -d "$d + 1 day")
done
```

A PDF is downloaded to a temporary file and renamed when complete. Withdrawn
papers return HTTP 404 and are skipped by the later stages.

**3, conversion.** `--all` converts every PDF that has no Markdown yet and can
be started on several machines at once: a lock file per paper prevents double
work. `--shard K/N` splits the papers deterministically instead. `--date` with
`--category`, or `--ids`, convert a subset. On machines without network access
the MinerU models must be downloaded beforehand; the paper's runs also set
`YOLO_OFFLINE=true` there.

**4 and 5, prefilters.** Both scripts accept several `--api-url` values and use
them round-robin, and both support `--shard K/N`, `--ids` and `--date`. The API
key is read from `--api-key`, else from `$SCOUT_API_KEY`, else the literal
`unused` is sent, which suits servers without authentication. The quality
prefilter returns an integer from 0 to 100; the type prefilter returns one of
`metric_method`, `benchmark_dataset`, `analysis_findings`, `theoretical`,
`survey_position`. `python merge_prefilter_scores.py --data-dir $DATA` copies
the scores into `paper_index.json` as `prefilter_score`; this is an optional
annotation that no later stage reads.

**6, hand-off.** `select_for_analysis.py` ranks all scored papers by quality
score (ties by arXiv ID), keeps the top fraction, drops papers below the
minimum score, and drops the excluded types. No filter is applied unless its
option is given; papers without a type label are kept. See "Numbers from the
paper's run" for what the paper did at this step.

**7, analysis.** One agent run per paper. The agent reads the Markdown (and the
PDF if needed), consults the reference sheets and the web, and fills in
`result.json`, a 13-field entry whose structure is `prompts/example.json`. The
script validates the structure and the 0 to 10 score ranges, resumes the agent
session with the error list when validation fails (`--max-retries`), and
repeats whole passes over unfinished papers (`--pass-retries`). `--model` is
passed through to the Claude Code CLI; without it the CLI uses its own default
model.
`--env KEY=VALUE` sets an environment variable for the agent process, for
example `ANTHROPIC_BASE_URL` to point Claude Code at an Anthropic-compatible
endpoint; `--env=-KEY` removes one. The agent process inherits the calling
environment, so export API keys there instead of passing them with `--env`.
`--api-url URL [URL ...]` spreads the runs over several backends in turn.
`--date` with `--category` analyses one day's listing instead of an ID list.
Papers that lack a PDF or a Markdown folder are skipped.

**8, selection.** See "Selection rule". The shipped `top1500.json` was made
with these options; the command in the table writes to other file names so
that it does not overwrite it.

**9, submission.** See "Submitting to Swarm".

`select_papers.py` holds the ranking and balancing functions used by stage 8.
Its own command line applies the same filters, as options, to the listings of
one date (`python select_papers.py --date 2026-03-01 --data-dir $DATA`, with
`--submit` to queue the result in Swarm). It was not used for the paper's
selection.

## Selection rule

This is the rule implemented by `export_selection_snapshot.py` with the helper
functions of `select_papers.py`. All scores are integers from 0 to 10 taken
from `result.json`.

**Filters.** A paper with a finished analysis becomes a candidate when it
passes all of the following, checked in this order. A rejected paper is counted
under the first check it fails.

1. Quality: `paper_quality.score >= 5`.
2. Reproduction difficulty: `r <= 6`, where `r` is the weighted mean of the
   overall score (weight 5.0) and the four sub-scores `coding_difficulty`,
   `compute_difficulty`, `speed_difficulty`, `data_model_difficulty` (weight
   1.25 each). With all five present,
   `r = (5 * overall + 1.25 * (coding + compute + speed + data_model)) / 10`.
3. Sub-score cap: no sub-score above 5. (A second rule, at most two sub-scores
   of 7 or more, cannot trigger while the cap is 5.)
4. Paper type: `paper_type.type` is `metric_method`.
5. Paid resources: `paid_resources.required` is false. Compute is not a paid
   resource.
6. Problem novelty: `problem_novelty.score >= 5`. A paper without this score
   passes.

These thresholds are constants in `build_candidates()`; they equal the defaults
of the `select_papers.py` options.

**Ranking.** With quality `q` and difficulty `r`, the rank score is the
geometric mean of normalised quality and easiness:

```
score = (q / 10) ** 0.5 * (1 - r / 10) ** 0.5
```

Candidates are sorted by score, highest first, ties by arXiv ID.

**Category balancing.** Each candidate is assigned one of 18 buckets: its
`taxonomy.primary.category` (or, if empty, `area`, `method`, `domain`) is
looked up in `category_merge_map.json`, first among the exact aliases, then
against the ordered regular-expression rules, and otherwise falls into
`Others`. For a selection of `N` papers with temperature `T`:

- a bucket holding `n` candidates gets the weight `n ** (1 / T)` and the raw
  quota `N * weight / sum(weights)`;
- its quota is the raw quota rounded down, never more than `n`; the slots left
  over are handed out one per bucket, largest fractional part first;
- each bucket contributes its best-ranked papers up to its quota, and the
  selection is sorted by rank score.

`T = 1` allocates in proportion to bucket size, larger `T` shifts slots to
small buckets, and `T <= 0` disables balancing, as does an `N` that is not
smaller than the number of candidates. The paper used `N = 1500` and
`T = 1.2`.

## Numbers from the paper's run

| Stage | Papers |
|---|---|
| 1. Candidate pool: 425 days (2025-01-01 to 2026-03-01), categories cs.CL, cs.CV, cs.AI, cs.LG, cross-lists included | 124,816 |
| 2. PDFs downloaded | 124,270 |
| 4. Quality scores (median 68) | 123,976 |
| 5. Type labels (94,161 of them `metric_method`) | 124,018 |
| 6. Top 25 percent by quality score | 30,994 |
| 7. Analysed | 21,355 |
| 8. Candidates after the filters | 8,987 |
| 8. Selected (`N = 1500`, `T = 1.2`, 18 buckets) | 1,500 |
| Environments built by Swarm | about 1,000 |

Rejections at stage 8, from `top1500_stats.json`: quality 217, reproduction
difficulty 1,106, sub-score cap 5,831, paper type 3,056, paid resources 1,574,
problem novelty 583. These and the 8,987 candidates add up to the 21,354
analyses that existed when the snapshot was exported; one more paper was
analysed afterwards (see "Selection snapshot"). The other counts come from the
authors' data directory, which is not part of this repository.

**The hand-off at stage 6.** For the paper, the list given to the analysis was
the top 25 percent of all scored papers by quality score. It was assembled by
hand; `select_for_analysis.py` was written afterwards to make the step
explicit. The 21,355 papers that were analysed all have a quality score of at
least 75, none of them is typed `theoretical` by the type prefilter, and all
but one appear in the cs.CV, cs.CL or cs.AI listings: papers reached only
through the cs.LG listing were not analysed. About 2,800 further papers meet
these three conditions and were not analysed either, and no rule for that
remainder could be recovered. The command in the pipeline table applies the
recoverable part of the rule (top quarter, score of at least 75, not
`theoretical`), not the listing restriction, so it does not regenerate the
exact analysed set.

**Models.** The scripts are not tied to a model; the paper's run used:

- stages 4 and 5: MiniMax M2.5 (model ID `MiniMax-M2.5`) behind an
  OpenAI-compatible endpoint. Eight of the quality scores were entered by hand.
- stage 7: MiniMax M2.7 (model ID `MiniMax-M2.7-highspeed`, effort `medium`),
  run by the Claude Code CLI pointed at the vendor's Anthropic-compatible
  endpoint. The invocation had this shape:

```bash
export ANTHROPIC_AUTH_TOKEN=...        # API key of the endpoint
python analyze_papers.py --ids-file $DATA/analysis_ids.json --provider claude \
    --model MiniMax-M2.7-highspeed --effort medium \
    --env ANTHROPIC_BASE_URL=https://ANTHROPIC_COMPATIBLE_ENDPOINT \
    --env ANTHROPIC_MODEL=MiniMax-M2.7-highspeed \
    --workers 8 --data-dir $DATA
```

In a sample of 500 papers the analysis took a median of 11 minutes and 19 agent
turns per paper.

## Selection snapshot

- `top1500.json` is the list that was submitted to Swarm: 1,501 arXiv IDs.
  Entries 1 to 1,500 are the output of stage 8 with `--count 1500` and
  `--temperature 1.2`, in rank order. The last entry, `2512.23675`, was
  appended by hand. That paper was analysed separately after the export and
  does not pass the filters (its coding and compute sub-scores are 6, above
  the cap of 5), but it was submitted and built, and it is part of the final
  environment set.
- `top1500_stats.json` is the statistics file of that export: the temperature,
  the number of finished analyses, candidates and selected papers, the
  rejections per filter (`skip`), and for each bucket the number of candidates
  (`before`) and of selected papers (`after`).

The environments that were built are the papers of this list that Swarm
completed.

## Submitting to Swarm

```bash
# Dry run: show the first 300 papers of the list and their topics
python submit_to_swarm.py --ids-json top1500.json --count 300 --data-dir $DATA

# Submit them
python submit_to_swarm.py --ids-json top1500.json --count 300 --data-dir $DATA --submit
```

`--swarm-url` defaults to `http://localhost:39100`, the default
`dashboard_port` of Swarm. Each paper is sent as `POST <swarm-url>/api/papers`
with its arXiv ID, the workflow name (`--workflow`, default `standard`) and a
list of topics derived from the analysis: `paper-type-...`, `domain-...`,
`area-...`, `category-...`, `method-...`, the same for secondary taxonomy
entries, and `selection-category-...` for the balancing bucket.

`--count N` takes the first `N` IDs of the list (default 10). Papers that
Swarm already knows are skipped, so raising `--count` step by step submits the
list in batches; this is how the paper's list was submitted. If the dashboard
cannot be reached the script stops before sending anything.

Swarm needs the Markdown of every submitted paper in its `scout_data_dir`.

Other modes: `--ids ID [ID ...]` submits explicit papers, and `--date` (with
`--category`, `--min-score`, `--max-score`, `--paper-type`, `--no-cost`)
submits papers of one day's listing ordered by overall reproduction score,
easiest first.

## Reference sheets

`reference_sheets/` holds five JSONL registries (models, API prices,
benchmarks and datasets, taxonomy domains, taxonomy areas) that the analysis
agent consults and extends. They are LLM-generated and unverified; see
`reference_sheets/README.md`. They are not needed to re-run the selection.

## What can be reproduced

The per-paper outputs of the paper's run (index, PDFs, Markdown, prefilter
scores, type labels and the 21,355 `result.json` files) are not in this
repository. Consequently:

- `top1500.json` is a deterministic function of those `result.json` files and
  `category_merge_map.json`. Stage 8 run on them reproduces entries 1 to 1,500
  exactly; without them the list can be neither regenerated nor checked.
- Stages 4, 5 and 7 are LLM calls. A fresh run gives different scores and
  therefore a different, comparable selection.
- The set of analysed papers cannot be regenerated exactly (see the hand-off
  note above).
- Stage 1 can be repeated at any time; the pool differs only by later changes
  on arXiv, such as withdrawals.

## Checks

There is no test suite. `python select_for_analysis.py --self-test` checks the
hand-off rule on synthetic rows. `submit_to_swarm.py` and `select_papers.py`
print what they would submit unless `--submit` is given.

## Security and operational notes

- The analysis agent is not sandboxed. Claude Code is started with the tools
  Read, Write, Edit, Glob, Grep, WebSearch and WebFetch and no path
  restriction; the sentence in the prompt that confines it to the listed paths
  is advisory. Its input is text derived from untrusted PDFs, and the model may
  be served by a third party, so a prompt injection could make the agent read
  local files and send them out. Run stage 7 under an account, container or
  machine that holds nothing the agent must not read.
- Command-line arguments are visible to other users of a shared machine. Put
  API keys in the environment (`SCOUT_API_KEY` for the prefilters, the
  agent CLI's own variables for the analysis), not in `--api-key` or `--env`.
- Stage 7 appends new rows to the tracked files in `reference_sheets/`, so the
  checkout is modified after a run.
- Each result folder keeps its own copy of the five reference sheets (about
  3.5 MB per paper at their current size). Delete the `*.jsonl` copies of
  finished papers if space matters.
- Run one `analyze_papers.py` process per data directory and use `--workers`
  for parallelism: a second process would delete the unfinished result folders
  of the first.
- The Swarm dashboard API is not authenticated; `--swarm-url` should point at a
  dashboard on a trusted network.

## Known limitations

- `fetch_daily_arxiv.py` uses plain HTTP, has no request timeout and retries
  only HTTP 429. It writes a day's listings before it saves the index, so a
  crash in between leaves listings whose papers are missing from
  `paper_index.json`; delete that day's listing files and run the day again.
- `download_arxiv_pdfs.py` stops on network errors other than HTTP errors; run
  it again to continue. Neither script can fetch a single paper by ID.
- The selection thresholds are constants in `export_selection_snapshot.py`
  that duplicate the `select_papers.py` defaults, and `select_papers.py`
  itself works on one listing date only.
- `convert_pdf2md.py` depends on the command line and output layout of MinerU
  2.7; newer major versions have not been checked.
- Only rows that an agent adds to a reference sheet are merged back; its edits
  to existing rows are not.
