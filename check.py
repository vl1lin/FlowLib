"""Единая проверка проекта: ruff check --fix, ruff format, pyright, pytest -q.

Запуск: uv run python check.py. Завершается кодом первой неуспешной проверки.
"""

import subprocess
import sys

STEPS: list[list[str]] = [
    ["ruff", "check", "--fix", "."],
    ["ruff", "format", "."],
    ["pyright"],
    ["pytest", "-q"],
]


def main() -> int:
    """Запускает проверки по очереди; возвращает код возврата первой упавшей или 0."""
    for step in STEPS:
        print(f"\n>>> {' '.join(step)}", flush=True)
        code = subprocess.run(["uv", "run", *step], check=False).returncode
        if code != 0:
            print(f"\nFAILED: {' '.join(step)} (код {code})")
            return code
    print("\nВсе проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
