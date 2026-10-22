"""Флаги корреляций и моделей режима: описания, передача, проверка.

Запуск: uv run python docs/examples/04_options_and_describe.py
"""

from _fluids import AIR_WATER, make_state, setup_console
from pydantic import ValidationError

from correlations import (
    describe_correlation,
    describe_flow_model,
    get_correlation,
    list_correlations,
    list_flow_models,
)


def main() -> None:
    """Печатает доступные модели и флаги, считает с разными флагами, показывает ошибки флагов."""
    print("Корреляции:", ", ".join(list_correlations()))
    print("Модели режима:", ", ".join(list_flow_models()))

    print("\nФлаги beggs_brill:")
    for flag, text in describe_correlation("beggs_brill").items():
        print(f"  {flag}: {text}")

    print("\nФлаги barnea:")
    for flag, text in describe_flow_model("barnea").items():
        print(f"  {flag}: {text}")

    state = make_state(AIR_WATER, v_sl_m_s=0.3, v_sg_m_s=2.0, d_m=0.0508, theta_deg=90.0)
    base = get_correlation("beggs_brill").calc(state)
    payne = get_correlation("beggs_brill", payne_correction=True).calc(state)
    print("\nBeggs–Brill, поправка Payne:")
    print(f"  без поправки: H_L={base.holdup:.3f}")
    print(f"  с поправкой:  H_L={payne.holdup:.3f}")

    print("\nПроверка флагов:")
    for bad_flags in ({"no_such_flag": True}, {"payne_correction": "yes"}):
        try:
            get_correlation("beggs_brill", **bad_flags)
        except ValidationError as error:
            print(f"  {bad_flags}: ValidationError (ошибок: {error.error_count()})")
    try:
        get_correlation("no_such_model")
    except ValueError as error:
        print(f"  неизвестное имя: ValueError: {error}")


if __name__ == "__main__":
    setup_console()
    main()
