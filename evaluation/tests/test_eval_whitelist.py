"""The scaffold whitelist must not amputate the train evaluator.

``eval/train/evaluate.py`` is the only way an agent can score itself
during the main phase. Several environments factor their scoring code
into a module that sits beside ``eval/train/`` and import it after
putting ``eval/`` on ``sys.path``. Keeping only ``eval/train/`` deletes
that module, so the evaluator dies on ``ModuleNotFoundError`` and the
agent is pushed into rebuilding the scorer — and its labels — by hand.

What must stay true: the closure keeps what the train evaluator imports,
and never keeps ``eval/test/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.runner import _enforce_workspace_whitelist, _eval_keep_names


def _mkeval(root: Path, files: dict[str, str]) -> Path:
    eval_dir = root / "eval"
    for rel, body in files.items():
        p = eval_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return eval_dir


def test_keeps_sibling_module_the_train_evaluator_imports(tmp_path: Path) -> None:
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
            "from split_common import run_split_evaluation\n"
        ),
        "split_common.py": "def run_split_evaluation(split, argv):\n    return 0\n",
        "test/evaluate.py": "SECRET = 1\n",
    })
    assert _eval_keep_names(eval_dir) == {"train", "split_common.py"}


def test_follows_the_import_chain(tmp_path: Path) -> None:
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": "from split_common import go\n",
        "split_common.py": "from metrics import f1\n",
        "metrics.py": "def f1():\n    return 0\n",
        "unused.py": "x = 1\n",
    })
    assert _eval_keep_names(eval_dir) == {"train", "split_common.py", "metrics.py"}


def test_dotted_package_form_keeps_init(tmp_path: Path) -> None:
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": "from eval.metrics import f1\n",
        "__init__.py": "",
        "metrics.py": "def f1():\n    return 0\n",
    })
    assert _eval_keep_names(eval_dir) == {"train", "__init__.py", "metrics.py"}


@pytest.mark.parametrize("statement", ["from eval import metrics", "import eval.metrics"])
def test_other_spellings_of_a_package_import(tmp_path: Path, statement: str) -> None:
    """Both forms occur in real environments. Missing them removed a module
    the train evaluator needs."""
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": statement + "\n",
        "__init__.py": "",
        "metrics.py": "def f1():\n    return 0\n",
        "unused.py": "x = 1\n",
    })
    assert _eval_keep_names(eval_dir) == {"train", "__init__.py", "metrics.py"}


def test_test_dir_is_never_kept_even_if_imported(tmp_path: Path) -> None:
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": "from test import answers\n",
        "test/__init__.py": "answers = [1, 2, 3]\n",
    })
    assert "test" not in _eval_keep_names(eval_dir)


def test_whitelist_removes_test_and_keeps_the_import(tmp_path: Path) -> None:
    env = tmp_path / "env"
    _mkeval(env, {
        "train/evaluate.py": "from split_common import go\n",
        "split_common.py": "def go():\n    return 0\n",
        "test/evaluate.py": "SECRET = 1\n",
        "official_evaluate.py": "SECRET = 2\n",
    })
    _enforce_workspace_whitelist(env, "innovate")
    assert (env / "eval" / "train" / "evaluate.py").exists()
    assert (env / "eval" / "split_common.py").exists()
    assert not (env / "eval" / "test").exists()
    assert not (env / "eval" / "official_evaluate.py").exists()


def test_third_party_imports_do_not_resurrect_anything(tmp_path: Path) -> None:
    eval_dir = _mkeval(tmp_path, {
        "train/evaluate.py": "import json\nimport numpy as np\nfrom pathlib import Path\n",
        "evaluate.py": "SECRET = 1\n",
    })
    assert _eval_keep_names(eval_dir) == {"train"}
