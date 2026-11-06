"""The documented commands. reproduce.sh and README.md may only use flags that
the scripts define, and reproduce.sh must carry the published configuration.
Nothing is executed: the files are parsed as text and the scripts with `ast`.
"""
from __future__ import annotations

import ast
import re
import shlex
import shutil
import subprocess

import pytest

import data_prep
import export_data
from conftest import TRAINING

SCRIPTS = sorted(p.name for p in TRAINING.glob("*.py"))

# Flags of other programs that README.md quotes (the `codex exec` argv).
FOREIGN_FLAGS = {"--skip-git-repo-check", "--sandbox", "--output-schema"}


def defined_flags(script: str) -> set:
    """The long options a script's argument parser defines."""
    flags = set()
    for node in ast.walk(ast.parse((TRAINING / script).read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name == "add_argument":
            flags.update(a.value for a in node.args
                         if isinstance(a, ast.Constant)
                         and isinstance(a.value, str) and a.value.startswith("--"))
        elif name == "add_data_argument":
            flags.add("--data")
    return flags


def literal_defaults(script: str) -> dict:
    """flag -> default, for the options whose default is a literal."""
    defaults = {}
    for node in ast.walk(ast.parse((TRAINING / script).read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[flags[0]] = kw.value.value
    return defaults


def commands(text: str) -> list:
    """(script, {flag: value or None}) for each `python <script>.py ...` line."""
    found = []
    for line in text.replace("\\\n", " ").splitlines():
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            continue
        for i, tok in enumerate(toks):
            if i and tok in SCRIPTS and "python" in toks[i - 1].lower():
                rest, args = toks[i + 1:], {}
                for j, t in enumerate(rest):
                    if t.startswith("--"):
                        nxt = rest[j + 1] if j + 1 < len(rest) else None
                        args[t] = None if nxt is None or nxt.startswith("--") else nxt
                found.append((tok, args))
                break
    return found


def code_blocks(markdown: str) -> str:
    return "\n".join(re.findall(r"```[a-z]*\n(.*?)```", markdown, flags=re.S))


def test_every_script_is_parsed():
    assert "sft.py" in SCRIPTS and "grpo_offline.py" in SCRIPTS
    assert {"--exclude-heldout", "--val-frac", "--data"} <= defined_flags("sft.py")
    assert "--data" in defined_flags("data_prep.py")


@pytest.mark.parametrize("doc", ["reproduce.sh", "README.md"])
def test_documented_commands_use_existing_flags(doc):
    text = (TRAINING / doc).read_text()
    found = commands(code_blocks(text) if doc.endswith(".md") else text)
    assert found, f"no command found in {doc}"
    for script, args in found:
        unknown = set(args) - defined_flags(script)
        assert not unknown, f"{doc}: {script} does not define {sorted(unknown)}"


def test_readme_mentions_only_existing_flags():
    known = set().union(*(defined_flags(s) for s in SCRIPTS)) | FOREIGN_FLAGS
    text = (TRAINING / "README.md").read_text()
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", text))
    assert mentioned - known == set()


def test_readme_names_the_default_dataset_and_its_override():
    text = (TRAINING / "README.md").read_text()
    assert data_prep._HF_REPO in text
    assert data_prep._HF_REPO_ENV in text


def test_reproduce_sh_is_valid_bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    proc = subprocess.run([bash, "-n", str(TRAINING / "reproduce.sh")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# The flags whose default is not the published value (README.md lists them).
# --out differs only because each trained stage writes its own file.
DEFAULTS_THAT_DIFFER = {
    ("sft.py", "--epochs"), ("sft.py", "--val-frac"), ("sft.py", "--save-name"),
    ("grpo_offline.py", "--sources"), ("grpo_offline.py", "--weights"),
    ("grpo_offline.py", "--max-samples"), ("grpo_offline.py", "--save-name"),
    ("baseline_untrained.py", "--out"),
}


def test_defaults_that_differ_from_the_published_configuration():
    differing = set()
    for script, args in commands((TRAINING / "reproduce.sh").read_text()):
        defaults = literal_defaults(script)
        for flag, value in args.items():
            default = defaults.get(flag)
            if value is None or default is None:
                continue
            if isinstance(default, (int, float)):
                same = float(value) == float(default)
            else:
                same = value == default
            if not same:
                differing.add((script, flag))
    assert differing == DEFAULTS_THAT_DIFFER

    readme = (TRAINING / "README.md").read_text()
    for script, flag in DEFAULTS_THAT_DIFFER - {("baseline_untrained.py", "--out")}:
        assert re.search(rf"\| `{re.escape(script)}` \| `{flag}` \|", readme), (script, flag)
    # the store_true flag of the published selection is off by default
    assert re.search(r"\| `sft.py` \| `--exclude-heldout` \| off", readme)


def test_reproduce_sh_carries_the_published_configuration():
    by_script = {}
    for script, args in commands((TRAINING / "reproduce.sh").read_text()):
        by_script.setdefault(script, []).append(args)

    for script in ("clean_reasoning_batch.py", "rephrase_batch.py"):
        (prep,) = by_script[script]
        assert (prep["--model"], prep["--effort"]) == ("gpt-5.5", "medium")
    assert by_script["rephrase_batch.py"][0]["--n"] == "75"

    (sft,) = by_script["sft.py"]
    assert "--exclude-heldout" in sft
    assert {k: sft[k] for k in ("--base-model", "--rank", "--lr", "--epochs",
                                "--batch-size", "--max-length", "--val-frac",
                                "--train-on", "--renderer")} == {
        "--base-model": "Qwen/Qwen3-8B", "--rank": "32", "--lr": "1e-4",
        "--epochs": "3", "--batch-size": "8", "--max-length": "8192",
        "--val-frac": "0", "--train-on": "all", "--renderer": "qwen3"}

    (grpo,) = by_script["grpo_offline.py"]
    assert "--init-state" in grpo
    assert {k: grpo[k] for k in ("--loss-fn", "--lr", "--epochs", "--batch-size",
                                 "--max-length", "--modes")} == {
        "--loss-fn": "ppo", "--lr": "1e-5", "--epochs": "1", "--batch-size": "16",
        "--max-length": "8192", "--modes": "innovate,improve"}
    # the same data selection as export_data.py
    assert grpo["--sources"] == ",".join(export_data.GRPO_SOURCES)
    assert grpo["--weights"] == ",".join(
        f"{k}={v}" for k, v in export_data.GRPO_WEIGHTS.items())
    assert int(grpo["--max-samples"]) == export_data.GRPO_MAX_SAMPLES == 1500

    # the save name of each trainer is the state the next step starts from
    text = (TRAINING / "reproduce.sh").read_text()
    assert f"/{sft['--save-name']}}}" in text and f"/{grpo['--save-name']}}}" in text

    samples = by_script["baseline_untrained.py"]
    assert [a["--out"] for a in samples] == [
        "baseline_untrained.jsonl", "sft_trained_gen.jsonl", "grpo_trained_gen.jsonl"]
    assert ["--init-state" in a for a in samples] == [False, True, True]
    for a in samples:
        assert (a["--n-papers"], a["--temperature"], a["--max-tokens"]) == (
            "10", "0.7", "4096")
