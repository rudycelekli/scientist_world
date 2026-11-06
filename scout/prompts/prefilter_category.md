Read the paper excerpt below and classify the paper into exactly one primary `paper_type`.

Use these labels only:
- `metric_method`
- `benchmark_dataset`
- `analysis_findings`
- `theoretical`
- `survey_position`

Choose the paper's main contribution:
- `metric_method`: introduces or improves a method and evaluates it with quantitative results
- `benchmark_dataset`: introduces a benchmark, dataset, or evaluation suite as the main contribution
- `analysis_findings`: the main contribution is empirical insight, characterization, or analysis
- `theoretical`: the main contribution is proofs, bounds, complexity analysis, or formal guarantees
- `survey_position`: the paper mainly surveys, organizes, or argues about existing work

Use the excerpt only. It is acceptable to make a rough first-pass judgment from the abstract and early sections.

Output exactly one JSON object in this format:

```json
{"paper_type": "metric_method"}
```

The `paper_type` value must be exactly one of:
- `metric_method`
- `benchmark_dataset`
- `analysis_findings`
- `theoretical`
- `survey_position`

Paper excerpt:
