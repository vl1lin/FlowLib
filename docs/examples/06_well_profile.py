"""Профиль давления в вертикальной скважине: расчёт участками снизу вверх.

Свойства флюида и приведённые скорости заданы константами (PVT и сжимаемость газа в примере не
моделируются), давление в `FlowState` обновляется на каждом участке.

Запуск: uv run python docs/examples/06_well_profile.py
"""

from _fluids import OIL_GAS, make_state, setup_console

from correlations import get_correlation

BOTTOM_PRESSURE_PA = 20.0e6
WELL_LENGTH_M = 2000.0
SEGMENT_LENGTH_M = 200.0
DIAMETER_M = 0.0762


def main() -> None:
    """Идёт по скважине снизу вверх, на каждом участке вычитает градиент давления на длину."""
    model = get_correlation("mukherjee_brill")
    pressure_pa = BOTTOM_PRESSURE_PA
    depth_m = WELL_LENGTH_M

    print(f"{'глубина, м':>11}{'p, МПа':>9}{'режим':>16}{'H_L':>8}{'-dp/dL, Па/м':>15}")
    n_segments = round(WELL_LENGTH_M / SEGMENT_LENGTH_M)
    for _ in range(n_segments):
        state = make_state(
            OIL_GAS, v_sl_m_s=0.3, v_sg_m_s=1.5, d_m=DIAMETER_M, theta_deg=90.0, p_pa=pressure_pa
        )
        result = model.calc(state)
        print(
            f"{depth_m:>11.0f}{pressure_pa / 1e6:>9.2f}{result.pattern.name:>16}"
            f"{result.holdup:>8.3f}{result.dpdl_pa_m:>15.1f}"
        )
        # Восходящий поток: давление падает на градиент × длину участка.
        pressure_pa -= result.dpdl_pa_m * SEGMENT_LENGTH_M
        depth_m -= SEGMENT_LENGTH_M

    print(f"{depth_m:>11.0f}{pressure_pa / 1e6:>9.2f}  (устье)")
    print(f"Перепад давления: {(BOTTOM_PRESSURE_PA - pressure_pa) / 1e6:.2f} МПа")


if __name__ == "__main__":
    setup_console()
    main()
