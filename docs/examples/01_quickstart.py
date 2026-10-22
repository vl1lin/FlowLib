"""Быстрый старт: одно состояние потока и расчёт одной корреляцией.

Запуск: uv run python docs/examples/01_quickstart.py
"""

from _fluids import AIR_WATER, make_state, setup_console

from correlations import get_correlation


def main() -> None:
    """Считает вертикальный восходящий поток воздух–вода по Beggs–Brill и печатает результат."""
    state = make_state(AIR_WATER, v_sl_m_s=0.3, v_sg_m_s=2.0, d_m=0.0508, theta_deg=90.0)

    model = get_correlation("beggs_brill")
    result = model.calc(state)

    print(f"Режим потока:      {result.pattern.name}")
    print(f"Истинное H_L:      {result.holdup:.3f}")
    print(f"-dp/dL, Па/м:      {result.dpdl_pa_m:.1f}")
    print(f"  гравитация:      {result.dpdl_gravity_pa_m:.1f}")
    print(f"  трение:          {result.dpdl_friction_pa_m:.1f}")
    print(f"  ускорение:       {result.dpdl_acceleration_pa_m:.1f}")


if __name__ == "__main__":
    setup_console()
    main()
