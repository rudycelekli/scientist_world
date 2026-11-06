"""Syntax validation for GPU workload scripts in tests/workloads/.

These scripts are self-contained Python programs submitted to the hive cluster
via `hive submit`. This test suite verifies they are syntactically correct.

Usage with a live cluster:
    bash tests/live/submit_workloads.sh   # submit every workload
    hive submit "python tests/workloads/matmul_bench.py"
    hive submit "torchrun --nproc_per_node=2 tests/workloads/multi_gpu_ddp.py" --gpus 2
"""

import subprocess
import sys
from pathlib import Path

import pytest

WORKLOADS_DIR = Path(__file__).parent / "workloads"

# Every .py file in workloads/ except __init__.py
WORKLOAD_SCRIPTS = sorted(
    p.name for p in WORKLOADS_DIR.glob("*.py") if p.name != "__init__.py"
)


def _check_syntax(path: Path):
    result = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open('{path}').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Syntax error in {path.name}: {result.stderr}"


class TestWorkloadSyntax:
    """Verify all workload scripts are syntactically valid Python."""

    @pytest.mark.parametrize("script", WORKLOAD_SCRIPTS)
    def test_syntax(self, script):
        _check_syntax(WORKLOADS_DIR / script)


class TestWorkloadStructure:
    """Verify workload scripts follow expected conventions."""

    @pytest.mark.parametrize("script", WORKLOAD_SCRIPTS)
    def test_has_main_guard(self, script):
        content = (WORKLOADS_DIR / script).read_text()
        assert 'if __name__ == "__main__"' in content, (
            f"{script} missing if __name__ == '__main__' guard"
        )

    @pytest.mark.parametrize("script", WORKLOAD_SCRIPTS)
    def test_has_docstring(self, script):
        content = (WORKLOADS_DIR / script).read_text()
        assert content.startswith('"""'), f"{script} missing module docstring"

    def test_workload_count(self):
        """Ensure we have the expected number of workload scripts."""
        assert len(WORKLOAD_SCRIPTS) >= 12, (
            f"Expected at least 12 workload scripts, found {len(WORKLOAD_SCRIPTS)}: "
            f"{WORKLOAD_SCRIPTS}"
        )
