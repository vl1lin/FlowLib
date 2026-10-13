"""Общие фикстуры тестов: наборы входных данных R1, R2, AW из ТЗ (СИ).

# reference: ТЗ, раздел «Тесты и эталонные значения», «Наборы входных данных».
"""

from collections.abc import Callable
from typing import Any

import pytest

from correlations.core.state import FlowState

BAR_TO_PA = 1.0e5


@pytest.fixture
def r1_kwargs() -> dict[str, Any]:
    """R1: Пример 3.2, вариант Примеров 4.7–4.8 (BB, MB). Аргументы `FlowState`."""
    return {
        "v_sl_m_s": 1.21,
        "v_sg_m_s": 1.176,
        "rho_l_kg_m3": 762.64,
        "rho_g_kg_m3": 94.19,
        "mu_l_pa_s": 0.97e-3,
        "mu_g_pa_s": 0.016e-3,
        "sigma_n_m": 8.41e-3,
        "d_m": 0.1524,
        "eps_m": 18.288e-6,
        "theta_deg": 90.0,
        "p_pa": 117.13 * BAR_TO_PA,
    }


@pytest.fixture
def r2_kwargs(r1_kwargs: dict[str, Any]) -> dict[str, Any]:
    """R2: вариант Примеров 4.9–4.10 (Ansari, HK). Аргументы `FlowState`."""
    return {
        **r1_kwargs,
        "v_sl_m_s": 1.208,
        "v_sg_m_s": 1.173,
        "rho_l_kg_m3": 761.7,
        "rho_g_kg_m3": 94.1,
        "eps_m": 1.83e-5,
    }


@pytest.fixture
def aw_state(aw_fluid_kwargs: dict[str, Any]) -> Callable[..., FlowState]:
    """Фабрика `FlowState` для воздуха–воды (AW): `aw_state(v_sl, v_sg, d_m, theta_deg, **override)`."""

    def make(
        v_sl_m_s: float, v_sg_m_s: float, d_m: float, theta_deg: float = 0.0, **override: Any
    ) -> FlowState:
        return FlowState(
            **{
                **aw_fluid_kwargs,
                "v_sl_m_s": v_sl_m_s,
                "v_sg_m_s": v_sg_m_s,
                "d_m": d_m,
                "theta_deg": theta_deg,
                **override,
            }
        )

    return make


@pytest.fixture
def aw_fluid_kwargs() -> dict[str, Any]:
    """AW: воздух–вода, 25 °C, 0.1 МПа (без скоростей, диаметра и угла)."""
    return {
        "rho_l_kg_m3": 997.0,
        "rho_g_kg_m3": 1.18,
        "mu_l_pa_s": 0.89e-3,
        "mu_g_pa_s": 1.8e-5,
        "sigma_n_m": 0.072,
        "p_pa": 0.1e6,
    }
