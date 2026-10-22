"""Постоянные свойства флюидов для примеров (СИ).

Свойства заданы константами для наглядности; подготовка свойств по давлению и температуре
(PVT) остаётся за вызывающим кодом и в примерах не моделируется.
"""

import sys
from typing import Any

from correlations import FlowState

# Воздух–вода, 25 °C, 0.1 МПа.
AIR_WATER: dict[str, float] = {
    "rho_l_kg_m3": 997.0,
    "rho_g_kg_m3": 1.18,
    "mu_l_pa_s": 0.89e-3,
    "mu_g_pa_s": 1.8e-5,
    "sigma_n_m": 0.072,
    "p_pa": 0.1e6,
}

# Нефть–газ в скважине, упрощённо: 10 МПа, ~60 °C (условные значения для иллюстрации).
OIL_GAS: dict[str, float] = {
    "rho_l_kg_m3": 800.0,
    "rho_g_kg_m3": 80.0,
    "mu_l_pa_s": 2.0e-3,
    "mu_g_pa_s": 1.5e-5,
    "sigma_n_m": 0.025,
    "p_pa": 10.0e6,
}


def make_state(
    fluid: dict[str, float],
    v_sl_m_s: float,
    v_sg_m_s: float,
    d_m: float,
    theta_deg: float,
    **override: Any,
) -> FlowState:
    """Строит `FlowState` из свойств флюида, скоростей, диаметра и угла.

    Параметры:
        fluid: свойства флюида (`AIR_WATER` или `OIL_GAS`), СИ.
        v_sl_m_s: приведённая скорость жидкости, м/с.
        v_sg_m_s: приведённая скорость газа, м/с.
        d_m: внутренний диаметр трубы, м.
        theta_deg: угол от горизонтали, град (> 0 — вверх).
        **override: переопределение любых полей `FlowState` (например, `p_pa`, `eps_m`).

    Возвращает:
        Неизменяемое состояние потока `FlowState`.
    """
    return FlowState(
        **{
            **fluid,
            "v_sl_m_s": v_sl_m_s,
            "v_sg_m_s": v_sg_m_s,
            "d_m": d_m,
            "theta_deg": theta_deg,
            **override,
        }
    )


def setup_console() -> None:
    """Переключает вывод на UTF-8 (консоль Windows по умолчанию cp1251 не знает «×», «θ»)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
