"""Shared fixtures.

Every test runs against an isolated configuration directory, so the
suite never reads a ``config.yaml`` that happens to exist next to the
package and never depends on ``$EVALUATION_DIR``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evaluation import config as _config


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch) -> None:
    cfg_dir = tmp_path / "evaluation-config"
    cfg_dir.mkdir()
    monkeypatch.setattr(_config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(_config, "CONFIG_PATH", cfg_dir / "config.yaml")
    _config._PARSED_CACHE.clear()
    yield
    _config._PARSED_CACHE.clear()


@pytest.fixture
def eval_config(tmp_path: Path):
    """Return ``write(**keys)``: write the config file and return ``data_dir``.

    ``data_dir`` defaults to ``<tmp_path>/data`` (created). Keys given in
    later calls are added to the ones already written.
    """
    data_dir = tmp_path / "data"
    state: dict = {"data_dir": str(data_dir)}

    def write(**keys) -> Path:
        state.update(keys)
        Path(state["data_dir"]).mkdir(parents=True, exist_ok=True)
        _config.CONFIG_PATH.write_text(yaml.safe_dump(state, sort_keys=False))
        _config._PARSED_CACHE.clear()
        return Path(state["data_dir"])

    return write
