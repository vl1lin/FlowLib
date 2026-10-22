"""Сравнение всех корреляций на одном состоянии потока.

Запуск: uv run python docs/examples/02_compare_correlations.py
"""

from _fluids import OIL_GAS, make_state, setup_console

from correlations import ConvergenceError, get_correlation, list_correlations


def main() -> None:
    """Считает вертикальный подъём нефть–газ всеми корреляциями и печатает таблицу."""
    state = make_state(OIL_GAS, v_sl_m_s=0.3, v_sg_m_s=2.0, d_m=0.1, theta_deg=90.0)

    print(f"{'корреляция':<16}{'режим':<18}{'H_L':>7}{'-dp/dL, Па/м':>16}")
    for name in list_correlations():
        try:
            result = get_correlation(name).calc(state)
        except ConvergenceError as error:
            print(f"{name:<16}не сошёлся: {error}")
            continue
        print(f"{name:<16}{result.pattern.name:<18}{result.holdup:>7.3f}{result.dpdl_pa_m:>16.1f}")


if __name__ == "__main__":
    setup_console()
    main()
