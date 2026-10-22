"""Ошибки и предупреждения: входные данные, несходимость, область применимости.

Запуск: uv run python docs/examples/05_validation_and_errors.py
"""

import warnings

from _fluids import AIR_WATER, make_state, setup_console
from pydantic import ValidationError

from correlations import ConvergenceError, OutOfRangeWarning, get_correlation


def main() -> None:
    """Вызывает каждую из ошибок и предупреждений библиотеки и показывает, как их обрабатывать."""
    print("1. Некорректный вход -> pydantic.ValidationError")
    try:
        make_state(AIR_WATER, 0.3, 2.0, 0.0508, 90.0, rho_g_kg_m3=1200.0)
    except ValidationError as error:
        print(f"  {error.errors()[0]['msg']}")

    print("\n2. Несходимость -> ConvergenceError (Ek >= 1: очень высокая скорость газа)")
    state = make_state(AIR_WATER, 0.5, 100.0, 0.0508, 90.0)
    try:
        get_correlation("beggs_brill").calc(state)
    except ConvergenceError as error:
        print(f"  {error.message}")
        print(f"  параметры задачи: {error.params}")

    print("\n3. Выход за область применимости -> OutOfRangeWarning")
    inclined = make_state(AIR_WATER, 0.1, 1.0, 0.0508, 20.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_correlation("hasan_kabir").calc(inclined)
    for item in caught:
        if issubclass(item.category, OutOfRangeWarning):
            print(f"  {item.message}")

    print("\n4. Превратить предупреждение в ошибку (строгий режим)")
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutOfRangeWarning)
        try:
            get_correlation("hasan_kabir").calc(inclined)
        except OutOfRangeWarning as error:
            print(f"  OutOfRangeWarning как исключение: {error}")


if __name__ == "__main__":
    setup_console()
    main()
