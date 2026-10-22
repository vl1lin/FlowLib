"""Сканирование по скорости газа и углу наклона: таблица градиента, режима и H_L.

Графики в библиотеку не входят: таблицу удобно сохранить и построить графики во внешнем проекте.

Запуск: uv run python docs/examples/07_sweep_table.py
"""

import warnings

from _fluids import AIR_WATER, make_state, setup_console

from correlations import ConvergenceError, OutOfRangeWarning, get_correlation

V_SG_M_S = (0.1, 0.5, 1.0, 3.0, 10.0, 30.0)
ANGLES_DEG = (-30.0, 0.0, 30.0, 90.0)
V_SL_M_S = 0.3
DIAMETER_M = 0.0508
CELL_WIDTH = 18


def main() -> None:
    """Считает Beggs–Brill на сетке «угол × v_sg» и печатает градиент, режим и H_L."""
    model = get_correlation("beggs_brill")
    print(f"Beggs–Brill, воздух–вода, v_sl = {V_SL_M_S} м/с, d = {DIAMETER_M} м")
    print("Ячейка: -dp/dL (Па/м) / режим (3 буквы) / H_L; «—» — расчёт не сошёлся.\n")

    print(f"{'θ, ° / v_sg':>12}" + "".join(f"{v:>{CELL_WIDTH}}" for v in V_SG_M_S))
    for angle in ANGLES_DEG:
        cells: list[str] = []
        for v_sg in V_SG_M_S:
            state = make_state(AIR_WATER, V_SL_M_S, v_sg, DIAMETER_M, angle)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OutOfRangeWarning)
                    result = model.calc(state)
            except ConvergenceError:
                cells.append("—")
                continue
            cells.append(f"{result.dpdl_pa_m:.0f}/{result.pattern.name[:3]}/{result.holdup:.2f}")
        print(f"{angle:>12.0f}" + "".join(f"{cell:>{CELL_WIDTH}}" for cell in cells))


if __name__ == "__main__":
    setup_console()
    main()
