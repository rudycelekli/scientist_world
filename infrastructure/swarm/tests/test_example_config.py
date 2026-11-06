"""The shipped config.example.yaml must load with the real loader."""

import re
import shutil
from pathlib import Path

import yaml

from swarm.cli import _validate_config
from swarm.config import (
    MILESTONES,
    SUPPORTED_SCAFFOLDS,
    WORKFLOW_ROLES,
    _TOP_LEVEL_KEYS,
    NexusConnectionConfig,
    WorkerConfig,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "swarm" / "config"
EXAMPLE = CONFIG_DIR / "config.example.yaml"

# (role, GPU-hour budget, min_milestone, revert_to_iteration) of the paper's
# nine-iteration build workflow.
STANDARD_WORKFLOW = [
    ("reproducer", 30, "core_claim", 1),
    ("reproducer", 15, "core_claim", 1),
    ("reproducer", 10, "core_claim", 2),
    ("designer_stage_1", 1, "", None),
    ("designer_stage_2", 1, "", None),
    ("designer_stage_3", 5, "core_claim", 1),
    ("packager", 1, "", None),
    ("verifier", 3, "core_claim", 1),
    ("verifier", 3, "core_claim", 2),
]


def _load_example(tmp_swarm_dir):
    shutil.copy(EXAMPLE, tmp_swarm_dir / "config.yaml")
    return load_config()


def test_example_config_loads_and_validates(tmp_swarm_dir, monkeypatch):
    monkeypatch.delenv("NEXUS_URL", raising=False)
    monkeypatch.delenv("SCOUT_DATA_DIR", raising=False)
    config = _load_example(tmp_swarm_dir)

    assert _validate_config(config) is None
    assert config.dashboard_host == "127.0.0.1"
    assert config.dashboard_port == 39100
    assert config.nexus.url == "http://localhost:39000"
    assert config.nexus.autostart_host == "127.0.0.1"
    assert config.github_org
    assert config.worker.container_sif
    assert config.worker.scaffold in SUPPORTED_SCAFFOLDS
    assert config.worker.model
    assert config.target_milestone in MILESTONES
    for category in ("gpu", "gpu_test", "cpu", "cpu_test"):
        assert getattr(config.nexus.resources, category), category


def test_example_standard_workflow_is_the_nine_iteration_build(tmp_swarm_dir):
    config = _load_example(tmp_swarm_dir)

    assert config.default_workflow == "standard"
    steps = config.workflows["standard"]
    assert config.workflow_meta["standard"].max_revert_attempts == 2
    assert [
        (s.role, s.gpu_budget_hours, s.min_milestone, s.revert_to_iteration)
        for s in steps
    ] == STANDARD_WORKFLOW
    assert all(s.role in WORKFLOW_ROLES for s in steps)
    assert all(s.milestone_cap == "full" for s in steps)


def test_example_profile_references_exist():
    raw = yaml.safe_load(EXAMPLE.read_text())
    profiles = raw["profiles"]

    assert raw["worker"]["profile"] in profiles
    for workflow in raw["workflows"].values():
        for step in workflow["steps"]:
            assert step["profile"] in profiles
            for fallback in step.get("fallback_profiles", []):
                assert fallback in profiles
    for name, profile in profiles.items():
        assert profile["scaffold"] in SUPPORTED_SCAFFOLDS, name
        assert profile["model"], name
    assert {p["scaffold"] for p in profiles.values()} == set(SUPPORTED_SCAFFOLDS)


def test_example_uses_only_keys_the_loader_reads():
    raw = yaml.safe_load(EXAMPLE.read_text())
    sections = {"profiles", "worker", "workflows", "nexus"}

    assert set(raw) <= set(_TOP_LEVEL_KEYS) | sections
    worker_fields = set(WorkerConfig.__dataclass_fields__)
    assert set(raw["worker"]) <= worker_fields
    for profile in raw["profiles"].values():
        assert set(profile) <= worker_fields - {"profile"}
    assert set(raw["nexus"]) <= set(NexusConnectionConfig.__dataclass_fields__)
    step_keys = {
        "profile", "role", "gpu_budget_hours", "milestone_cap",
        "min_milestone", "revert_to_iteration", "fallback_profiles",
    }
    for workflow in raw["workflows"].values():
        assert set(workflow) <= {"max_revert_attempts", "steps"}
        for step in workflow["steps"]:
            assert set(step) <= step_keys


def test_example_documents_every_key_the_loader_reads():
    text = EXAMPLE.read_text()
    documented = set(re.findall(r"[a-z_]+", text))

    missing = [k for k in _TOP_LEVEL_KEYS if k not in documented]
    assert not missing, missing
    worker_fields = set(WorkerConfig.__dataclass_fields__)
    assert not worker_fields - documented, worker_fields - documented
    nexus_fields = set(NexusConnectionConfig.__dataclass_fields__)
    assert not nexus_fields - documented, nexus_fields - documented


def test_example_has_no_site_specific_values():
    text = EXAMPLE.read_text()
    raw = yaml.safe_load(text)

    for key in ("scout_data_dir", "workspace_base_dir", "shared_datasets_path",
                "shared_models_path", "hf_cache_path", "container_cache_dir",
                "uv_cache_dir", "apptainer_cache_dir"):
        assert raw[key].startswith("/path/to/"), key
    assert raw["worker"]["container_sif"].startswith("/path/to/")
    # Secrets appear only as <placeholders>, and only in commented examples.
    assert "env_vars" not in str(raw)
    assert "api_keys" not in str(raw)


def test_only_example_files_ship_in_config_dir():
    # config.yaml and excluded_papers.yaml are local, gitignored files.
    assert EXAMPLE.is_file()
    assert (CONFIG_DIR / "excluded_papers.example.yaml").is_file()
    gitignore = (CONFIG_DIR.parent.parent / ".gitignore").read_text().splitlines()
    assert "swarm/config/config.yaml" in gitignore
    assert "swarm/config/excluded_papers.yaml" in gitignore
    assert ".swarm/" in gitignore
    assert ".swarm_*/" in gitignore
