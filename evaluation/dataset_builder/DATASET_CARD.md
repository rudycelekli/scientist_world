---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
tags:
- ai-scientist
- research-agents
- agent-traces
- reasoning
- scientific-discovery
pretty_name: Scientist World Traces
size_categories:
- 1K<n<10K
---

# Scientist World Traces

<p align="center">
  <a href="https://ai-scientist.world">
    <img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90_Website-ai--scientist.world-2F5496">
  </a>
  <a href="https://arxiv.org/abs/2611.08478">
    <img alt="Paper abstract" src="https://img.shields.io/badge/arXiv-Abstract-b31b1b?logo=arxiv&logoColor=white">
  </a>
  <a href="https://arxiv.org/pdf/2611.08478">
    <img alt="Paper PDF" src="https://img.shields.io/badge/arXiv-PDF-b31b1b?logo=arxiv&logoColor=white">
  </a>
  <a href="https://github.com/zlab-princeton/scientist_world">
    <img alt="Scientist World repository" src="https://img.shields.io/badge/GitHub-Scientist_World-181717?logo=github&logoColor=white">
  </a>
  <a href="https://github.com/zlab-princeton/scientist_world_cli">
    <img alt="Command line" src="https://img.shields.io/badge/GitHub-Command_line-181717?logo=github&logoColor=white">
  </a>
</p>

Verified research attempts by AI scientists in [Scientist World](https://ai-scientist.world). In each row a frontier model receives a research problem taken from a published paper, proposes and implements a method, and is scored against the paper's own method.

**Every row is grader-verified.** An independent grader re-ran the frozen method of each run. Runs that did not pass are not in the dataset, and the scores are the grader's measurements, not the numbers the agent reported.

| | `innovate` | `improve` | Total |
|---|---|---|---|
| Runs | 1,517 | 564 | 2,081 |
| Papers | 374 | 81 | 378 |
| Runs with FNI | | | 1,948 on 342 papers |
| Models | | | 19 |

## Load

```python
from datasets import load_dataset

ds = load_dataset("zlab-princeton/scientist-world-traces")
run = ds["innovate"][0]
print(run["paper_title"], run["model"], run["fni"])
```

The two splits are the two modes.

| Mode | The model gets | The model must |
|---|---|---|
| `innovate` | Problem and evaluator only | Design a method of its own |
| `improve` | Problem, evaluator, the paper's method and its scores | Beat the paper's method |

## Columns

**Run**

| Column | Meaning |
|---|---|
| `task_id`, `paper_title`, `paper_topics` | arXiv id, title and topic tags of the paper |
| `mode` | `innovate` or `improve` |
| `model`, `scaffold` | Model id with effort tier, and the agent CLI that drove it |
| `run_id`, `branch_url` | Run id, and the branch that holds the run's code |
| `gpu_hours`, `cost_usd`, `turn_count`, `output_tokens`, `reasoning_tokens` | What the run used |

**Task given to the model**

`problem`, `direction`, `constraint`, `target`, `experiments`, `overview`, `evaluation`. The paper's method, its numbers and the held-out evaluator are not included.

**Idea and reasoning**

| Column | Meaning |
|---|---|
| `idea` | The first proposal |
| `idea_versions`, `n_idea_versions` | Every version in order; later versions are changes to earlier ones |
| `final_idea` | The method that was scored, as one self-contained text |
| `reflection`, `has_reflection` | The model's closing report of what it did |
| `thinking_trace`, `thinking_segments`, `n_trace_segments` | The model's reasoning in turn order |
| `has_real_cot` | `True` when the trace is logged chain-of-thought, `False` when it is narrated reasoning only |

**Scores**

| Column | Meaning |
|---|---|
| `fni` | Frontier-Normalized Improvement, the leaderboard number: mean of the two splits |
| `fni_train`, `fni_test` | FNI on the visible and on the held-out split |
| `train_score`, `test_score` | The same two values divided by 100 |
| `train_score_reason`, `test_score_reason` | Why a score is missing |
| `score_locked`, `n_models_cohort` | Whether the paper's coefficients are frozen, and the size of the reference cohort |
| `train_score_raw`, `test_score_raw` | The full metric tables the grader measured, as JSON |
| `grader_profile` | The grader that verified the run |

## FNI

```
FNI = 100 * sum over cells of  coefficient * (signed gain over the paper's method)
```

A cell is one experiment, method and metric. 0 matches the paper's method, a positive value improves on it, and 100 is the typical gain of a frozen reference cohort of frontier models. FNI is comparable across papers. The [command line](https://github.com/zlab-princeton/scientist_world_cli) computes the same number with `sw evaluate`.

## Notes

- **Missing FNI.** 133 runs have none: their paper has no coefficients, too small a cohort, or only tied cells. The raw metric tables are still there.
- **Outliers.** A few papers have cells where the reference cohort barely differs, so FNI explodes: 22 runs on 10 papers exceed 1,000 in absolute value. Clip or use medians when you aggregate.
- **Missing text.** `final_idea` is empty for 224 runs and `thinking_trace` for 515. Only 160 runs carry logged chain-of-thought (`has_real_cot`).
- **Redaction.** Credentials and the names and paths of the machines the runs used are removed from all text.

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The task descriptions are derived from published papers, which stay under the licenses their authors chose.

## Citation

```bibtex
@article{scientist-world,
  title   = {Scientist World: Grow a Research World for AI-Scientists},
  author  = {Lu, Taiming and Konan, Sachin and Liu, Zhuang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2611.08478},
}
```
