"""export_data writes the two sets that the published commands select."""
from __future__ import annotations

import json
import sys

import pytest

import export_data
from conftest import make_row, write_jsonl

HELD = "2501.01741"
OTHER = "2401.00001"


def _records(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()]


@pytest.fixture
def exported(prep, use_rows, monkeypatch, tmp_path):
    """Run export_data on a small dataset; returns the output directory."""
    write_jsonl(prep._REASONING, [{"run_id": "m1", "reasoning": "why"}])
    write_jsonl(prep._REPHRASED, [{"run_id": "m1", "rephrased_completion": "reworded"}])
    use_rows([
        make_row(OTHER, "innovate", "m1", test=1.0, train=1.0, versions=2),
        make_row(OTHER, "innovate", "s1", test=3.0, train=7.0),
        make_row(HELD, "innovate", "h1", test=1.0, train=1.0),
        make_row(HELD, "innovate", "h2", test=2.0, train=2.0),
    ])
    out_dir = tmp_path / "exported"
    monkeypatch.setattr(sys, "argv", ["export_data.py", "--out-dir", str(out_dir)])
    return out_dir


def test_export_writes_the_published_selection(exported):
    export_data.main()

    sft = _records(exported / "sft_train.jsonl")
    assert [e["run_id"] for e in sft] == ["m1", "s1", "m1-aug"]
    assert [e["augmented"] for e in sft] == [False, False, True]
    assert [e["has_reasoning"] for e in sft] == [True, False, False]
    assert set(sft[0]) == {"conversation", "run_id", "task_id", "mode",
                           "has_reasoning", "augmented", "test_score"}
    assert [m["role"] for m in sft[0]["conversation"]] == ["system", "user", "assistant"]

    grpo = _records(exported / "grpo_train.jsonl")
    assert [(e["source"], e["reward"]) for e in grpo] == [
        ("cross_model", 1.0), ("cross_model", 3.0),
        ("generalization", 1.0), ("generalization", -1.0)]
    # generalization advantages carry the published weight of 0.5
    assert [e["advantage"] for e in grpo] == pytest.approx([-1.0, 1.0, 0.5, -0.5])
    assert set(grpo[0]) == {"conversation", "advantage", "reward", "source",
                            "group_id", "task_id", "mode"}

    assert {e["task_id"] for e in sft + grpo} == {OTHER}     # held-out papers excluded


def test_export_applies_the_sample_cap(exported, monkeypatch):
    monkeypatch.setattr(export_data, "GRPO_MAX_SAMPLES", 3)
    export_data.main()
    grpo = _records(exported / "grpo_train.jsonl")
    assert [e["source"] for e in grpo] == ["cross_model", "cross_model", "generalization"]
