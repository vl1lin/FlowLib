"""Регрессионные тесты устойчивости: сквозные расчёты, ранее завершавшиеся `ConvergenceError`.

Решение пользователя: шероховатость ε/d выше предела 0.05 ограничивается (OutOfRangeWarning),
расчёт не падает. Ek ≥ 1 по решению пользователя остаётся `ConvergenceError` (формула источника не
определена).
"""

import math

import pytest

from correlations import FlowState, OutOfRangeWarning, get_correlation


def assert_valid(result_holdup: float, dpdl: float) -> None:
    assert math.isfinite(dpdl)
    assert 0.0 <= result_holdup <= 1.0


def test_mukherjee_brill_stratified_with_extreme_relative_roughness() -> None:
    """Расслоённая ветвь: ε/D_E газовой фазы > 3 (газ почти исчез) раньше давала «нет смены знака»."""
    # reference: случайный прогон проверки готовности (seed 1), ConvergenceError в colebrook
    state = FlowState(
        v_sl_m_s=0.016279875671034123,
        v_sg_m_s=0.01770059744875851,
        rho_l_kg_m3=656.6139792867085,
        rho_g_kg_m3=184.12987973023243,
        mu_l_pa_s=0.0887274330420863,
        mu_g_pa_s=1.693846441844447e-05,
        sigma_n_m=0.05908498127609239,
        d_m=0.048298480885528426,
        eps_m=1e-05,
        theta_deg=-60.0,
        p_pa=30848298.8698833,
    )
    with pytest.warns(OutOfRangeWarning, match="ε/d"):
        result = get_correlation("mukherjee_brill").calc(state)
    assert_valid(result.holdup, result.dpdl_pa_m)
