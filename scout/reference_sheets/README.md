# Reference sheets

Lookup registries for the analysis agent of `analyze_papers.py`. Each file is
JSONL: one JSON object per line.

| File | Rows | Row identity | Other fields |
|---|---|---|---|
| `model_sheet.jsonl` | 2,340 | `name` | type, organization, parameter count, file sizes, context length, licence, access, notes |
| `benchmark_dataset_registry.jsonl` | 4,498 | `name` | type, domain, number of samples, storage, evaluation time, format, metric, licence, access, notes |
| `area_registry.jsonl` | 324 | `name` | domain |
| `domain_registry.jsonl` | 45 | `name` | |
| `api_price_sheet.jsonl` | 26 | `provider` and `model` | type, prices per million tokens, context window |

## These files are unverified seed data

Every row was written by an LLM agent while it analysed a paper. Nobody has
checked the rows against their sources. Model sizes, licences, prices, storage
figures and URLs can be wrong or out of date, and names are not normalised
beyond exact-match deduplication. Use the sheets as a starting point for an
analysis run, not as a reference.

The sheets contain public identifiers only. They are not needed to re-run the
selection (stage 8 of the pipeline).

## How the analysis uses them

- Before an agent starts, the five sheets are copied into the paper's result
  folder. The agent looks entries up in its copy and may append rows for
  models, prices, benchmarks, datasets, domains or areas it could not find.
- When the analysis of a paper passes validation, the rows the agent added are
  appended to the files in this directory. A row is new when no existing row
  has the same identity (see the table). Changes an agent made to existing
  rows are not merged back.
- An analysis run therefore modifies these tracked files. Commit the growth or
  discard it (`git checkout -- reference_sheets`) as you prefer.
