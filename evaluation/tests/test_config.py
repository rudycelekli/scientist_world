"""Configuration loading, defaults and pre-flight checks."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from evaluation import config as _config

_EXAMPLE = Path(_config.__file__).with_name("config.example.yaml")


# ── Defaults ─────────────────────────────────────────────────────────────


def test_defaults_without_a_config_file() -> None:
    assert not _config.CONFIG_PATH.exists()
    cfg = _config.load()
    assert cfg["host"] == "127.0.0.1"
    assert cfg["publish"] == {"enabled": False, "github_org": ""}
    assert cfg["verifier"]["enabled"] is False
    assert cfg["verifier"]["grader_profile"] == ""
    assert cfg["profiles"] == {}
    assert cfg["data_dir"] == ""
    assert cfg["nexus"]["url"] == "http://localhost:39000"
    assert cfg["nexus"]["resources"] == {
        "gpu": [], "gpu_test": [], "cpu": [], "cpu_test": []}
    assert cfg["dashboard"] == {"paper_groups": {}, "model_order": []}


def test_load_returns_independent_copies() -> None:
    cfg = _config.load()
    cfg["nexus"]["resources"]["gpu"].append("mutated")
    cfg["dashboard"]["model_order"].append("mutated")
    again = _config.load()
    assert again["nexus"]["resources"]["gpu"] == []
    assert again["dashboard"]["model_order"] == []
    assert _config.DEFAULT_CONFIG["nexus"]["resources"]["gpu"] == []


def test_blocks_merge_key_by_key_with_defaults(eval_config) -> None:
    eval_config(
        nexus={"max_gpus": 4, "resources": {"gpu": ["a"]}},
        verifier={"enabled": True},
        publish={"github_org": "example-org"},
        dashboard={"model_order": ["m1"]},
    )
    cfg = _config.load()
    assert cfg["nexus"]["max_gpus"] == 4
    assert cfg["nexus"]["url"] == "http://localhost:39000"
    assert cfg["nexus"]["gpu_budget_hours"] == 30.0
    assert cfg["nexus"]["resources"] == {
        "gpu": ["a"], "gpu_test": [], "cpu": [], "cpu_test": []}
    assert cfg["verifier"]["enabled"] is True
    assert cfg["verifier"]["max_turns"] == 12
    assert cfg["publish"] == {"enabled": False, "github_org": "example-org"}
    assert cfg["dashboard"] == {"paper_groups": {}, "model_order": ["m1"]}


def test_edits_to_the_file_are_picked_up(eval_config) -> None:
    eval_config(max_turns=7)
    assert _config.load()["max_turns"] == 7
    # A second write goes straight to disk, bypassing the fixture's cache
    # reset: load() must notice the change by itself.
    _config.CONFIG_PATH.write_text(yaml.safe_dump({"max_turns": 12345}))
    assert _config.load()["max_turns"] == 12345


def test_non_mapping_config_is_an_error() -> None:
    _config.CONFIG_PATH.write_text("- just\n- a list\n")
    with pytest.raises(_config.ConfigError):
        _config.load()


def test_save_round_trips(eval_config) -> None:
    eval_config()
    cfg = _config.load()
    cfg["max_concurrent"] = 3
    _config.save(cfg)
    assert _config.load()["max_concurrent"] == 3


# ── Required values ──────────────────────────────────────────────────────


def test_data_dir_unset_is_a_clear_error() -> None:
    with pytest.raises(_config.ConfigError, match="data_dir is not set"):
        _config.data_dir()
    with pytest.raises(_config.ConfigError):
        _config.envs_dir()


def test_data_dir_and_envs_dir(eval_config, tmp_path: Path) -> None:
    data_dir = eval_config()
    assert _config.data_dir() == data_dir
    assert _config.envs_dir() == data_dir / "envs"


def test_check_run_config_lists_every_problem() -> None:
    with pytest.raises(_config.ConfigError) as exc:
        _config.check_run_config()
    msg = str(exc.value)
    for needle in ("data_dir", "agent_sif", "nexus.resources.gpu ",
                   "nexus.resources.gpu_test "):
        assert needle in msg, f"{needle!r} missing from: {msg}"
    assert "grader_profile" not in msg  # verifier is off by default


def test_check_run_config_requires_a_grader_when_verifier_is_on(eval_config) -> None:
    eval_config(
        agent_sif="/x/agent.sif",
        nexus={"resources": {"gpu": ["g"], "gpu_test": ["g"]}},
        verifier={"enabled": True},
    )
    with pytest.raises(_config.ConfigError, match="grader_profile"):
        _config.check_run_config()


def test_check_run_config_passes_when_complete(eval_config) -> None:
    eval_config(
        agent_sif="/x/agent.sif",
        nexus={"resources": {"gpu": ["g"], "gpu_test": ["g"]}},
        verifier={"enabled": True, "grader_profile": "grader"},
    )
    _config.check_run_config()


# ── Nexus resource selection ─────────────────────────────────────────────


def test_nexus_pick_by_category(eval_config) -> None:
    eval_config(nexus={"resources": {
        "gpu": ["g1", "g2"], "gpu_test": ["gt"], "cpu": ["c"], "cpu_test": ["ct"]}})
    assert _config.nexus_pick(1) == "g1,g2"
    assert _config.nexus_pick(2, test=True) == "gt"
    assert _config.nexus_pick(0) == "c"
    assert _config.nexus_pick(0, test=True) == "ct"


def test_nexus_pick_empty_category_is_an_error(eval_config) -> None:
    eval_config(nexus={"resources": {"gpu": ["g1"]}})
    with pytest.raises(_config.ConfigError, match="nexus.resources.cpu"):
        _config.nexus_pick(0)
    with pytest.raises(_config.ConfigError, match="gpu_test"):
        _config.nexus_pick(1, test=True)


def test_nexus_all_resources_is_an_ordered_union(eval_config) -> None:
    eval_config(nexus={"resources": {
        "gpu": ["a", "b"], "gpu_test": ["b", "t"], "cpu": ["c"], "cpu_test": ["c"]}})
    assert _config.nexus_all_resources() == ["a", "b", "t", "c"]


# ── The example file documents what the code reads ───────────────────────


def _documented_keys(text: str) -> set[str]:
    """Every ``key:`` in the example, active or commented out."""
    return set(re.findall(r"^[#\s]*([A-Za-z_][A-Za-z0-9_]*):", text, flags=re.M))


def test_example_config_parses_and_uses_only_known_keys() -> None:
    example = yaml.safe_load(_EXAMPLE.read_text())
    assert isinstance(example, dict)
    assert set(example) <= set(_config.DEFAULT_CONFIG), (
        f"unknown keys in the example: {set(example) - set(_config.DEFAULT_CONFIG)}")
    for block in _config._MERGED_BLOCKS:
        assert set(example[block]) <= set(_config.DEFAULT_CONFIG[block]), block
    assert set(example["nexus"]["resources"]) == set(
        _config.DEFAULT_CONFIG["nexus"]["resources"])


def test_example_config_documents_every_key() -> None:
    documented = _documented_keys(_EXAMPLE.read_text())
    expected = set(_config.DEFAULT_CONFIG)
    for block in _config._MERGED_BLOCKS:
        expected |= set(_config.DEFAULT_CONFIG[block])
    expected |= set(_config.DEFAULT_CONFIG["nexus"]["resources"])
    assert not expected - documented, (
        f"keys missing from config.example.yaml: {sorted(expected - documented)}")


def test_example_config_defaults_match_the_code() -> None:
    """Apart from the placeholders, the example states the real defaults."""
    example = yaml.safe_load(_EXAMPLE.read_text())
    placeholders = {"data_dir", "agent_sif"}
    for key, default in _config.DEFAULT_CONFIG.items():
        if key in placeholders or key == "nexus":
            continue
        assert example[key] == default, key
    nexus = dict(example["nexus"])
    nexus.pop("resources")  # example resource names, documented as such
    default_nexus = dict(_config.DEFAULT_CONFIG["nexus"])
    default_nexus.pop("resources")
    assert nexus == default_nexus


def test_example_config_is_safe_by_default() -> None:
    example = yaml.safe_load(_EXAMPLE.read_text())
    assert example["host"] == "127.0.0.1"
    assert example["publish"]["enabled"] is False
    assert example["profiles"] == {}
    text = _EXAMPLE.read_text()
    assert "example.com" in text  # the gateway example uses a placeholder host
