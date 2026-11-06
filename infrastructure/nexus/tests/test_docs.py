"""The README must list what the code offers: config keys, commands, routes."""

import dataclasses
from pathlib import Path

from nexus import agent_cli, cli
from nexus.config import NexusConfig

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def _subcommands(parser):
    """Names of the subcommands of an argparse parser."""
    return sorted(parser._subparsers._group_actions[0].choices)


def test_readme_documents_every_config_key():
    for section, values in dataclasses.asdict(NexusConfig()).items():
        for key in values:
            assert f"`{section}.{key}`" in README


def test_readme_documents_every_admin_command():
    for name in _subcommands(cli.build_parser()):
        assert f"`nexus {name}" in README


def test_readme_documents_every_agent_command():
    for name in _subcommands(agent_cli.build_parser()):
        assert f"`nx {name}" in README


def test_readme_documents_every_environment_variable():
    for name in ("NEXUS_DIR", "HIVE_URL", "NEXUS_URL", "NEXUS_PROJECT_ID",
                 "NEXUS_LOG_STATE"):
        assert f"`{name}`" in README


def test_readme_documents_every_route(daemon):
    for rule in daemon.app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        path = rule.rule.replace("<project_id>", "<id>")
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            assert f"`{method} {path}`" in README
