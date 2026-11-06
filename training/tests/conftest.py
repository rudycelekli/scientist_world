"""Shared fixtures.

The tests need no third-party package besides pytest and reach no service: the
traces dataset is replaced by in-memory rows, `datasets` by a stub module, and
the `codex` CLI by a fake `subprocess.run`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING))

import data_prep  # noqa: E402


def make_row(task_id, mode, run_id, test=None, train=None, idea="idea",
             versions=1):
    """One row of the traces dataset, with the columns the training code reads."""
    return {
        "task_id": task_id, "mode": mode, "run_id": run_id,
        "problem": f"problem of {task_id}", "direction": "direction",
        "constraint": "constraint", "target": "target",
        "final_idea": idea, "test_score": test, "train_score": train,
        "idea_versions": [f"version {i}" for i in range(1, versions + 1)],
        "n_idea_versions": versions,
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def prep(tmp_path, monkeypatch):
    """data_prep with its prepared files under a temp dir, no local parquet,
    no cached state and no environment override."""
    monkeypatch.setattr(data_prep, "_REASONING", tmp_path / "reasoning.jsonl")
    monkeypatch.setattr(data_prep, "_REPHRASED", tmp_path / "rephrased.jsonl")
    monkeypatch.setattr(data_prep, "_CRAFT_JUDGE", tmp_path / "craft_judge.jsonl")
    monkeypatch.setattr(data_prep, "_PARQUET", tmp_path / "absent.parquet")
    monkeypatch.setattr(data_prep, "_REASONING_CACHE", None)
    monkeypatch.setattr(data_prep, "_ANNOUNCED", set())
    monkeypatch.delenv("SW_TRACES_REPO", raising=False)
    return data_prep


@pytest.fixture
def use_rows(prep, monkeypatch):
    """Serve the given rows as the traces dataset."""
    def _use(rows):
        monkeypatch.setattr(prep, "_load_rows", lambda source=None: list(rows))
    return _use
