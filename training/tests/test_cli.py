"""Command-line surface of every script, and the parts of the Tinker scripts
that can run without the service.

`tinker`, `tinker_cookbook` and `numpy` are replaced by stub modules and the
Tinker client by a small fake, so this covers argument parsing, the defaults,
the data selection each trainer asks data_prep for, the renderer fallback, the
choice of the papers to sample and the records the sampler writes. The training
loops themselves are not run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types

import pytest

import export_data
from conftest import make_row

TINKER_SCRIPTS = ("sft", "grpo_offline", "baseline_untrained")
CLI_SCRIPTS = TINKER_SCRIPTS + ("data_prep", "export_data", "clean_reasoning_batch",
                                "rephrase_batch", "judge_craft_batch")


class Stop(Exception):
    """Raised by a fake to end main() once the call under test was recorded."""


class FakeService:
    """Stands in for tinker.ServiceClient; records how clients are created."""

    created = []

    def create_lora_training_client(self, base_model, rank):
        self.created.append(("base", base_model, rank))
        return FakeClient()

    def create_training_client_from_state(self, state):
        self.created.append(("state", state))
        return FakeClient()


class FakeClient:
    def get_tokenizer(self):
        return types.SimpleNamespace(decode=lambda tokens: tokens[0])

    def save_weights_and_get_sampling_client(self, name):
        FakeService.created.append(("sampler", name))
        return FakeSampler()


class FakeSampler:
    def sample(self, prompt, num_samples, sampling_params):
        """Answer every prompt with <think>plan for it</think> idea for it."""
        text = f"<think>plan for {prompt}</think> idea for {prompt}"
        sequence = types.SimpleNamespace(tokens=[text])
        return types.SimpleNamespace(
            result=lambda: types.SimpleNamespace(sequences=[sequence]))


@pytest.fixture
def scripts(monkeypatch):
    """name -> module for every script with a command line, imported against
    stub `tinker`, `tinker_cookbook` and `numpy` modules."""
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    stub("numpy")
    stub("tinker")
    renderers = stub("tinker_cookbook.renderers", TrainOnWhat=None, Message=dict,
                     get_renderer=None)
    stub("tinker_cookbook", renderers=renderers)
    stub("tinker_cookbook.supervised")
    stub("tinker_cookbook.supervised.data", conversation_to_datum=None)
    for name in TINKER_SCRIPTS:
        monkeypatch.delitem(sys.modules, name, raising=False)
    modules = {name: importlib.import_module(name) for name in CLI_SCRIPTS}
    yield modules
    for name in TINKER_SCRIPTS:          # do not leave modules bound to the stubs
        sys.modules.pop(name, None)


@pytest.fixture
def fake_tinker(scripts, monkeypatch):
    """Give the stub `tinker` module a fake service and the scripts a fake
    renderer. Returns the list of clients the service was asked for."""
    FakeService.created = []
    renderer = types.SimpleNamespace(
        get_stop_sequences=lambda: ["<stop>"],
        # the "prompt" is the line under "# Problem" of the user message
        build_generation_prompt=lambda msgs: msgs[1]["content"].splitlines()[1])
    for name in TINKER_SCRIPTS:
        module = scripts[name]
        monkeypatch.setattr(module.tinker, "ServiceClient", FakeService, raising=False)
        monkeypatch.setattr(module.tinker, "SamplingParams", dict, raising=False)
        monkeypatch.setattr(module, "get_renderer", lambda *a: renderer)
    return FakeService.created


def _recorder(captured):
    def record(**kwargs):
        captured.update(kwargs)
        raise Stop
    return record


# ------------------------------------------------------------ argument parsing

@pytest.mark.parametrize("name", CLI_SCRIPTS)
def test_help_renders(name, scripts, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [name + ".py", "--help"])
    with pytest.raises(SystemExit) as exc:
        scripts[name].main()
    assert exc.value.code == 0
    assert "--data" in capsys.readouterr().out


def test_result_affecting_defaults_are_unchanged(scripts, monkeypatch):
    """The defaults are deliberately left as they were (README.md lists the
    ones that differ from the published configuration)."""
    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def capture(self, *a, **kw):
        captured.update(vars(orig(self, *a, **kw)))
        raise Stop

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)

    monkeypatch.setattr(sys, "argv", ["sft.py"])
    with pytest.raises(Stop):
        scripts["sft"].main()
    assert {k: captured[k] for k in ("base_model", "rank", "lr", "epochs", "batch_size",
                                     "max_length", "renderer", "exclude_heldout",
                                     "n_backfill", "val_frac", "train_on", "data")} == {
        "base_model": "Qwen/Qwen3-8B", "rank": 32, "lr": 1e-4, "epochs": 1,
        "batch_size": 8, "max_length": 8192, "renderer": "qwen3",
        "exclude_heldout": False, "n_backfill": 37, "val_frac": 0.03,
        "train_on": "all", "data": None}

    captured.clear()
    monkeypatch.setattr(sys, "argv", ["grpo_offline.py", "--init-state", "tinker://x"])
    with pytest.raises(Stop):
        scripts["grpo_offline"].main()
    assert {k: captured[k] for k in ("lr", "epochs", "batch_size", "max_length",
                                     "loss_fn", "sources", "weights", "modes",
                                     "max_samples", "beta1", "beta2",
                                     "grad_clip_norm", "nll_abort_mult",
                                     "shuffle_rewards")} == {
        "lr": 1e-5, "epochs": 1, "batch_size": 16, "max_length": 8192,
        "loss_fn": "ppo",
        "sources": "cross_model,generalization,judge,judge_novelty",
        "weights": "cross_model=1.0,generalization=0.5,judge=0.5,judge_novelty=0.5",
        "modes": "innovate,improve", "max_samples": 2500, "beta1": 0.9,
        "beta2": 0.95, "grad_clip_norm": 1.0, "nll_abort_mult": 2.0,
        "shuffle_rewards": False}

    captured.clear()
    monkeypatch.setattr(sys, "argv", ["baseline_untrained.py"])
    with pytest.raises(Stop):
        scripts["baseline_untrained"].main()
    assert {k: captured[k] for k in ("n_papers", "temperature", "max_tokens",
                                     "init_state", "out")} == {
        "n_papers": 10, "temperature": 0.7, "max_tokens": 4096,
        "init_state": None, "out": "baseline_untrained.jsonl"}


# ------------------------------------------------- data selection of the trainers

def test_sft_selects_the_published_set_only_with_exclude_heldout(
        scripts, fake_tinker, prep, monkeypatch, tmp_path):
    sft, captured = scripts["sft"], {}
    monkeypatch.setattr(sft, "sft_examples", _recorder(captured))

    monkeypatch.setattr(sys, "argv", ["sft.py", "--exclude-heldout", "--data", "x.parquet"])
    with pytest.raises(Stop):
        sft.main()
    assert captured == {"source": "x.parquet", "exclude_papers": prep.HELDOUT_PAPERS,
                        "n_single_backfill": None, "include_rephrased": True}

    captured.clear()
    monkeypatch.setattr(sys, "argv", ["sft.py"])
    with pytest.raises(Stop):
        sft.main()
    assert captured == {"source": None, "exclude_papers": None,
                        "n_single_backfill": 37, "include_rephrased": False}

    # a --mixture file may not contain a held-out paper
    mixture = tmp_path / "mixture.jsonl"
    mixture.write_text('{"conversation": [], "task_id": "2501.01741"}\n')
    monkeypatch.setattr(sys, "argv", ["sft.py", "--mixture", str(mixture)])
    with pytest.raises(SystemExit, match="held-out papers in the mixture"):
        sft.main()

    # the examples are selected before the service is contacted
    assert fake_tinker == []


def test_grpo_offline_passes_the_published_selection(scripts, fake_tinker, monkeypatch):
    grpo, captured = scripts["grpo_offline"], {}
    monkeypatch.setattr(grpo, "grpo_dataset", _recorder(captured))
    monkeypatch.setattr(sys, "argv", [
        "grpo_offline.py", "--init-state", "tinker://sft",
        "--sources", "cross_model,generalization",
        "--weights", "cross_model=1.0,generalization=0.5", "--max-samples", "1500"])
    with pytest.raises(Stop):
        grpo.main()
    assert fake_tinker == []         # selected before the service is contacted
    # the same selection as export_data.py writes
    assert captured == {"source": None, "sources": export_data.GRPO_SOURCES,
                        "modes": ("innovate", "improve"),
                        "weights": export_data.GRPO_WEIGHTS,
                        "max_samples": export_data.GRPO_MAX_SAMPLES}


# ----------------------------------------------------------- renderer fallback

def test_get_renderer_reports_a_fallback(scripts, monkeypatch, capsys):
    sft = scripts["sft"]

    def fake(name, tokenizer, model_name=None):
        if name not in ("qwen3", "role_colon"):
            raise KeyError(name)
        return f"renderer:{name}"

    monkeypatch.setattr(sft.renderers, "get_renderer", fake)
    assert sft.get_renderer(None, "qwen3", "model") == "renderer:qwen3"
    assert "WARNING" not in capsys.readouterr().out
    assert sft.get_renderer(None, "qwen-typo", "model") == "renderer:qwen3"
    assert "'qwen-typo' could not be resolved" in capsys.readouterr().out

    def broken(name, tokenizer, model_name=None):
        raise KeyError(name)

    monkeypatch.setattr(sft.renderers, "get_renderer", broken)
    with pytest.raises(SystemExit, match="could not resolve a renderer"):
        sft.get_renderer(None, "qwen3", "model")


# -------------------------------------------------------------------- sampling

def test_pick_papers_selects_the_heldout_papers(scripts, prep, monkeypatch):
    sampler = scripts["baseline_untrained"]
    papers = sorted(prep.TOP30) + ["2401.00001"]
    rows = ([make_row(p, "improve", f"{p}-improve") for p in papers]
            + [make_row(p, "innovate", f"{p}-first") for p in reversed(papers)]
            + [make_row(p, "innovate", f"{p}-second") for p in papers])
    monkeypatch.setattr(sampler, "_load_rows", lambda source=None: rows)

    picked = sampler.pick_papers(10)
    assert [r["task_id"] for r in picked] == sorted(prep.HELDOUT_PAPERS)
    # the task specification comes from the paper's first innovate row
    assert all(r["run_id"].endswith("-first") for r in picked)

    # Existing behavior, pinned here because it matters: a held-out paper
    # without an innovate row is replaced by the next paper of TOP30 - TOP10,
    # which is not excluded from training. main() warns about it.
    gone = sorted(prep.HELDOUT_PAPERS)[0]
    rows = [r for r in rows if not (r["task_id"] == gone and r["mode"] == "innovate")]
    picked = {r["task_id"] for r in sampler.pick_papers(10)}
    assert picked - prep.HELDOUT_PAPERS == {sorted(prep.TOP30 - prep.TOP10)[10]}


def test_sampler_writes_the_handoff_records(scripts, fake_tinker, prep, monkeypatch,
                                            tmp_path, capsys):
    sampler = scripts["baseline_untrained"]
    candidates = sorted(prep.TOP30 - prep.TOP10)
    rows = [make_row(p, "innovate", f"{p}-run") for p in candidates[1:]]
    monkeypatch.setattr(sampler, "_load_rows", lambda source=None: rows)
    monkeypatch.setattr(sampler, "_OUT", tmp_path)

    monkeypatch.setattr(sys, "argv", ["baseline_untrained.py", "--init-state",
                                      "tinker://grpo", "--out", "ideas.jsonl"])
    sampler.main()

    assert fake_tinker == [("state", "tinker://grpo"), ("sampler", "sft-gen")]
    records = [json.loads(ln) for ln in (tmp_path / "ideas.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in records] == candidates[1:11]
    for r in records:
        # evaluation/idea_impl_eval.py reads task_id and idea
        assert set(r) == {"task_id", "mode", "response", "thinking", "idea"}
        assert r["mode"] == "innovate"
        assert r["thinking"] == f"plan for problem of {r['task_id']}"
        assert r["idea"] == f"idea for problem of {r['task_id']}"
    # the first held-out paper has no row, so a paper that is not held out was
    # sampled in its place, and main() says so
    out = capsys.readouterr().out
    assert f"WARNING: ['{candidates[10]}'] are not in HELDOUT_PAPERS" in out


def test_sampler_uses_the_base_model_without_init_state(scripts, fake_tinker, prep,
                                                        monkeypatch, tmp_path, capsys):
    sampler = scripts["baseline_untrained"]
    rows = [make_row(p, "innovate", f"{p}-run") for p in sorted(prep.HELDOUT_PAPERS)]
    monkeypatch.setattr(sampler, "_load_rows", lambda source=None: rows)
    monkeypatch.setattr(sampler, "_OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["baseline_untrained.py"])
    sampler.main()

    assert fake_tinker == [("base", "Qwen/Qwen3-8B", 32), ("sampler", "untrained-baseline")]
    written = (tmp_path / "baseline_untrained.jsonl").read_text().splitlines()
    assert [json.loads(ln)["task_id"] for ln in written] == sorted(prep.HELDOUT_PAPERS)
    assert "WARNING" not in capsys.readouterr().out


def test_split_think(scripts):
    split_think = scripts["baseline_untrained"].split_think
    assert split_think("<think>\n why \n</think>\n\n the idea ") == ("why", "the idea")
    assert split_think(" no reasoning ") == (None, "no reasoning")
