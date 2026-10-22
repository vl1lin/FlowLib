"""Тест примеров `docs/examples`: каждый скрипт запускается и завершается с кодом 0."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _ROOT / "docs" / "examples"
_EXAMPLES = sorted(_EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))


def test_examples_are_found() -> None:
    """Каталог примеров не пуст (защита от тихого пропуска параметризации)."""
    assert len(_EXAMPLES) >= 7


@pytest.mark.parametrize("script", _EXAMPLES, ids=lambda path: path.name)
def test_example_runs(script: Path) -> None:
    """Пример завершается кодом 0, печатает результат и не пишет traceback в stderr."""
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
    assert "Traceback" not in completed.stderr
