# Scientist World

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
  <a href="https://github.com/orgs/ScientistWorld/repositories">
    <img alt="Environments" src="https://img.shields.io/badge/GitHub-Environments-181717?logo=github&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/zlab-princeton/scientist-world-traces">
    <img alt="Traces dataset" src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Traces_dataset-FFD21E">
  </a>
  <a href="https://github.com/zlab-princeton/scientist_world_cli">
    <img alt="Command line" src="https://img.shields.io/badge/Command_line-sciworld-1f6feb?logo=gnubash&logoColor=white">
  </a>
</p>

**Grow a Research World for AI-Scientists.** Scientist World turns published papers into executable research environments. In each one, an AI scientist is handed the paper's problem, runs real experiments, and is scored by how far its method improves on the paper's own.

This repository holds the system behind the paper: the pipeline that selects papers and builds environments from them, the compute layer the agents run on, the evaluation harness, and the training study.

## Just want to run an AI scientist?

You do not need this repository. The environments are used through the `sciworld` command line, which lives in [`cli/`](https://github.com/zlab-princeton/scientist_world_cli) (a submodule) and installs on its own:

```bash
pip install sciworld

sw init --data-dir ~/sciworld-data --backend venv
sw list
sw research 2502.13759 --mode innovate --agent claude-code
sw status
sw evaluate 2502.13759
```

See its [quick start](https://github.com/zlab-princeton/scientist_world_cli/blob/main/QUICKSTART.md).

## How the system fits together

```
                      papers (arXiv)
                            │
                     ┌──────▼──────┐
                     │    Scout    │  select papers worth building
                     └──────┬──────┘
                            │ ranked list
        ┌───────────────────▼────────────────────┐
        │                 Swarm                  │  build one environment per paper
        │  reproducer → designer → packager →    │  with builder agents driven
        │  verifier, through the Worker scaffold │  through Worker
        └───────────────────┬────────────────────┘
                            │ environments (one GitHub repository each)
              ┌─────────────┴─────────────┐
       ┌──────▼──────┐             ┌──────▼──────┐
       │ sciworld    │             │ Evaluation  │  run scientists at scale,
       │ CLI (cli/)  │             │             │  grade, normalize (FNI)
       └─────────────┘             └──────┬──────┘
                                          │ traces
                                   ┌──────▼──────┐
                                   │  Training   │  train a scientist on the traces
                                   └─────────────┘

   Every experiment that Swarm or Evaluation runs goes through
   Nexus (a budget per experiment) to Hive (a pool of SLURM nodes).
```

| Directory | Component | What it does |
|---|---|---|
| [`scout/`](scout) | Scout | Builds a pool of arXiv papers, screens and analyses them, and exports a ranked, category-balanced list to build. |
| [`infrastructure/swarm/`](infrastructure/swarm) | Swarm | Coordinates the builder agents that turn a paper into an environment: reproduce the result, design the task, package it, verify it. |
| [`infrastructure/worker/`](infrastructure/worker) | Worker | The agent scaffold (Claude Code, Codex) that Swarm drives, with the builder prompts and the workspace template. |
| [`infrastructure/nexus/`](infrastructure/nexus) | Nexus | The experiment boundary: a GPU-hour budget and a concurrency cap per project, enforced on the server side. |
| [`infrastructure/hive/`](infrastructure/hive) | Hive | An interactive compute pool: SLURM nodes joined into one Ray cluster that runs jobs on request. |
| [`evaluation/`](evaluation) | Evaluation | Runs a scientist on an environment under a fixed budget, grades it on held-out data, and normalizes scores across papers. Also builds the traces dataset. |
| [`training/`](training) | Training | Supervised fine-tuning and offline GRPO of a small model on the traces. |
| [`cli/`](https://github.com/zlab-princeton/scientist_world_cli) | `sciworld` | The command line for using environments. Evaluation uses it to build workspaces. |

Each directory has its own README with requirements, configuration and commands. The components are separate Python packages; install only what you need.

## Environments

An environment is a GitHub repository of the [ScientistWorld](https://github.com/orgs/ScientistWorld/repositories) organization, named after the paper's arXiv id. It holds the briefing a scientist reads, the data, a baseline, the paper's method, an evaluator with a visible and a held-out split, and the scores the method reaches.

A scientist works in one of two modes. In **improve** it starts from the paper's method and tries to beat it. In **innovate** it sees only the problem and designs a method of its own. Its score is **FNI** (Frontier-Normalized Improvement): 0 matches the paper's reproduced method, a positive value improves on it, and 100 is the typical gain of a frozen reference cohort of frontier models.

## Getting started

```bash
git clone --recurse-submodules https://github.com/zlab-princeton/scientist_world
cd scientist_world
```

What you install depends on what you want to do.

| Goal | Install | Read |
|---|---|---|
| Use environments | `pip install -e cli` | [`cli/README.md`](https://github.com/zlab-princeton/scientist_world_cli#readme) |
| Select papers | `pip install -r scout/requirements.txt` | [`scout/README.md`](scout/README.md) |
| Build environments | `pip install -e infrastructure/hive -e infrastructure/nexus -e infrastructure/swarm` | [`infrastructure/swarm/README.md`](infrastructure/swarm/README.md) |
| Run the evaluation | the three above, then `pip install -e cli -e evaluation` | [`evaluation/README.md`](evaluation/README.md) |
| Train on the traces | `pip install -r training/requirements.txt` | [`training/README.md`](training/README.md) |

Building environments and running the evaluation need a Linux cluster with SLURM and Apptainer, Python 3.10 or newer, and an agent CLI (Claude Code or Codex) with its own login. Scout, the command line and the training scripts run on a single machine.

The services (Hive, Nexus, Swarm, the Evaluation dashboard) have no authentication. They listen on `127.0.0.1` by default; expose them only on a network you trust.

## Tests

Every component has a test suite that needs no cluster, no agent and no network. Run each suite in its own `pytest` invocation:

```bash
(cd infrastructure/hive  && pytest tests)
(cd infrastructure/nexus && pytest tests)
python -m pytest infrastructure/swarm/tests      # from the repository root
python -m pytest infrastructure/worker/tests     # from the repository root
(cd evaluation && pytest tests -m "not slow")
(cd training   && pytest tests)
(cd cli        && pytest)
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The environments are separate, and that license does not cover them. Each environment is built from a published paper, and what it takes from the paper (text, code, data, model weights) stays under the license its authors chose. Check the license of the paper, of its code and of its datasets before you use or redistribute an environment. The same holds for the traces dataset, which is derived from the environments, and for the agent CLIs, models and datasets that the components download or call: each follows its own terms.

## Citation

```bibtex
@article{scientist-world,
  title   = {Scientist World: Grow a Research World for AI-Scientists},
  author  = {Lu, Taiming and Konan, Sachin and Liu, Zhuang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2611.08478},
}
```
