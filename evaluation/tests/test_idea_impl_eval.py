"""The idea-implementation ablation driver (``idea_impl_eval.py``).

``run_one`` is stubbed: no agent is launched.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from evaluation import config as _config
from evaluation import runner

_SCRIPT = Path(__file__).resolve().parents[1] / "idea_impl_eval.py"
_spec = importlib.util.spec_from_file_location("idea_impl_eval", _SCRIPT)
impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(impl)


def _ideas(ideas_dir: Path, stage: str, ideas: dict[str, str]) -> None:
    ideas_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"task_id": p, "idea": i}) for p, i in ideas.items()]
    (ideas_dir / impl._STAGE_FILES[stage]).write_text("\n".join(lines) + "\n\n")


@pytest.fixture
def ideas_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ideas"
    _ideas(d, "base", {"p1": "base idea 1", "p2": "base idea 2", "p3": "  "})
    _ideas(d, "sft", {"p1": "sft idea 1", "p2": "sft idea 2", "p3": "sft idea 3"})
    _ideas(d, "grpo", {"p2": "grpo idea 2", "p1": "grpo idea 1"})
    return d


def test_profile_is_required(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        impl._parse_args([])
    assert exc.value.code == 2
    assert "--profile" in capsys.readouterr().err


def test_defaults_and_stage_validation(capsys) -> None:
    args = impl._parse_args(["--profile", "implementer"])
    assert args.stages == ["base", "sft", "grpo"]
    assert Path(args.ideas_dir) == _SCRIPT.parent.parent / "training" / "out"
    assert args.out_dir is None and args.concurrency == 3 and args.smoke is False
    with pytest.raises(SystemExit):
        impl._parse_args(["--profile", "x", "--stages", "base,bogus"])
    assert "bogus" in capsys.readouterr().err


def test_worklist_keeps_papers_with_an_idea_in_every_stage(ideas_dir: Path) -> None:
    work = impl._worklist(["base", "sft", "grpo"], None, ideas_dir)
    # p3 has an empty base idea, so it is left out
    assert [(s, p) for s, p, _ in work] == [
        ("base", "p1"), ("sft", "p1"), ("grpo", "p1"),
        ("base", "p2"), ("sft", "p2"), ("grpo", "p2")]
    assert work[1][2] == "sft idea 1"
    assert [(s, p) for s, p, _ in impl._worklist(["sft"], ["p3", "p1"], ideas_dir)] == [
        ("sft", "p1"), ("sft", "p3")]


def test_cell_runs_as_a_variant_and_records_a_relative_path(
    eval_config, tmp_path: Path, monkeypatch,
) -> None:
    data_dir = eval_config()
    calls: list[dict] = []

    def fake_run_one(paper, profile, mode="innovate", *, variant="", injected_idea=None):
        calls.append({"paper": paper, "profile": profile, "mode": mode,
                      "variant": variant, "idea": injected_idea})
        env_dir = data_dir / "envs" / paper / f"{profile}-{mode}{variant}-eval"
        return runner.RunResult(
            run_id="r", paper_id=paper, model=profile, mode=mode, status="completed",
            attempt=1, env_dir=str(env_dir), train_score={"s": 1}, test_score=None,
            error=None)

    monkeypatch.setattr(impl, "run_one", fake_run_one)
    out_dir = tmp_path / "results"
    rec = impl._run_cell("implementer", "sft", "p1", "the idea", out_dir)

    assert calls == [{"paper": "p1", "profile": "implementer", "mode": "innovate",
                      "variant": "-impl-sft", "idea": "the idea"}]
    assert rec["status"] == "completed" and rec["train_score"] == {"s": 1}
    assert rec["env_dir"] == "envs/p1/implementer-innovate-impl-sft-eval"
    assert json.loads((out_dir / "p1_sft.json").read_text()) == rec
    assert str(data_dir) not in (out_dir / "p1_sft.json").read_text()


def test_cell_with_work_on_disk_is_skipped(eval_config, tmp_path: Path, monkeypatch) -> None:
    data_dir = eval_config()
    env_dir = data_dir / "envs" / "p1" / "implementer-innovate-impl-base-eval"
    (env_dir / "scoring").mkdir(parents=True)
    (env_dir / "scoring" / "scores_train.json").write_text("{}")

    def boom(*a, **k):
        raise AssertionError("run_one must not be called for a cell with work on disk")

    monkeypatch.setattr(impl, "run_one", boom)
    rec = impl._run_cell("implementer", "base", "p1", "idea", tmp_path / "results")
    assert rec["status"] == "skipped_in_flight"
    assert rec["env_dir"] == "envs/p1/implementer-innovate-impl-base-eval"


def test_a_failing_cell_does_not_stop_the_batch(eval_config, tmp_path: Path, monkeypatch) -> None:
    eval_config()

    def failing(*a, **k):
        raise RuntimeError("scaffold failed")

    monkeypatch.setattr(impl, "run_one", failing)
    rec = impl._run_cell("implementer", "grpo", "p1", "idea", tmp_path / "results")
    assert rec["status"] == "driver_error" and "scaffold failed" in rec["error"]


def test_main_smoke_runs_one_grpo_cell(eval_config, ideas_dir: Path, monkeypatch, capsys) -> None:
    data_dir = eval_config()
    cells: list[tuple] = []

    def fake_cell(profile, stage, paper, idea, out_dir):
        cells.append((profile, stage, paper, idea, out_dir))
        return {"paper_id": paper, "stage": stage, "status": "completed"}

    monkeypatch.setattr(impl, "_run_cell", fake_cell)
    impl.main(["--profile", "implementer", "--ideas-dir", str(ideas_dir), "--smoke"])

    out_dir = data_dir / "idea_impl_results"
    assert cells == [("implementer", "grpo", "p1", "grpo idea 1", out_dir)]
    assert json.loads((out_dir / "SUMMARY.json").read_text()) == [
        {"paper_id": "p1", "stage": "grpo", "status": "completed"}]
    assert "1/1 completed" in capsys.readouterr().out


def test_main_runs_every_cell_and_honours_out_dir(
    eval_config, ideas_dir: Path, tmp_path: Path, monkeypatch, capsys,
) -> None:
    eval_config()
    cells: list[tuple] = []
    monkeypatch.setattr(impl, "_run_cell", lambda profile, stage, paper, idea, out_dir:
                        cells.append((stage, paper)) or {"status": "failed"})
    out_dir = tmp_path / "elsewhere"
    impl.main(["--profile", "implementer", "--ideas-dir", str(ideas_dir),
               "--stages", "base,sft", "--papers", "p2", "--concurrency", "1",
               "--out-dir", str(out_dir)])
    assert cells == [("base", "p2"), ("sft", "p2")]
    assert (out_dir / "SUMMARY.json").exists()
    assert "0/2 completed" in capsys.readouterr().out


def test_main_without_any_cell_exits(eval_config, ideas_dir: Path) -> None:
    eval_config()
    with pytest.raises(SystemExit, match="no \\(stage, paper\\) cell"):
        impl.main(["--profile", "implementer", "--ideas-dir", str(ideas_dir),
                   "--papers", "unknown"])


def test_default_out_dir_needs_data_dir(ideas_dir: Path) -> None:
    with pytest.raises(_config.ConfigError, match="data_dir"):
        impl.main(["--profile", "implementer", "--ideas-dir", str(ideas_dir)])
