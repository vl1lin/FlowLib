"""Тесты `core/bubbles.py` — скорости всплытия (пересечения №4, №5 ТЗ).

Ожидаемый API (`from correlations.core import bubbles`):
    harmathy_rise_velocity(rho_l, rho_g, sigma, *, coef=1.53) -> float
        v∞ = coef·[g·σ·Δρ/ρ_L²]^0.25  (Harmathy 1960; HK 1988 ур. 3; Barnea [3]; Ansari (3))
    taylor_rise_velocity(d_m, rho_l, rho_g, *, coef) -> float
        v∞T = coef·√(g·d·Δρ/ρ_L)      (HK ур. 5, 28; Ansari 0.35; затрубье HK (30) — d = d_c − d_t)
    inverse_viscosity_number(d_m, rho_l, rho_g, mu_l) -> float     N_f, HK (24)
    eotvos_number(d_m, rho_l, rho_g, sigma) -> float               N_Eo, HK (25)
    wallis_taylor_coefficient(n_f, n_eo) -> float                  C2, HK (23), (26a–c), с исправлениями
Константы: TAYLOR_COEFFICIENT_HK = 0.345, TAYLOR_COEFFICIENT_ANSARI = 0.35.
"""

from typing import Any

import pytest

from correlations.core import bubbles


def test_harmathy_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v∞ = 0.1510 м/с (rtol 1e-3)
    v = bubbles.harmathy_rise_velocity(
        r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"], r2_kwargs["sigma_n_m"]
    )
    assert v == pytest.approx(0.1510, rel=1e-3)


def test_harmathy_coefficient_is_parameter(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, п. 4 «Пересечений»: HK ур. 17 — 1.50 вместо 1.53
    args = (r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"], r2_kwargs["sigma_n_m"])
    default = bubbles.harmathy_rise_velocity(*args)
    custom = bubbles.harmathy_rise_velocity(*args, coef=1.50)
    assert custom / default == pytest.approx(1.50 / 1.53, rel=1e-12)


def test_harmathy_air_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента по HK ур. 3 для набора AW: 0.2495 м/с
    v = bubbles.harmathy_rise_velocity(
        aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"], aw_fluid_kwargs["sigma_n_m"]
    )
    assert v == pytest.approx(0.2495, rel=1e-3)


def test_taylor_constants() -> None:
    # reference: ТЗ п. 5; HK ур. 27 (0.345), Ansari (0.35)
    assert bubbles.TAYLOR_COEFFICIENT_HK == 0.345
    assert bubbles.TAYLOR_COEFFICIENT_ANSARI == 0.35


def test_taylor_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v∞T (0.345) = 0.3949 м/с (rtol 1e-3)
    v = bubbles.taylor_rise_velocity(
        r2_kwargs["d_m"],
        r2_kwargs["rho_l_kg_m3"],
        r2_kwargs["rho_g_kg_m3"],
        coef=bubbles.TAYLOR_COEFFICIENT_HK,
    )
    assert v == pytest.approx(0.3949, rel=1e-3)


def test_taylor_scales_with_coefficient_and_sqrt_of_diameter(r2_kwargs: dict[str, Any]) -> None:
    rho_l, rho_g = r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"]
    hk = bubbles.taylor_rise_velocity(0.1, rho_l, rho_g, coef=0.345)
    ansari = bubbles.taylor_rise_velocity(0.1, rho_l, rho_g, coef=0.35)
    assert ansari / hk == pytest.approx(0.35 / 0.345, rel=1e-12)
    wide = bubbles.taylor_rise_velocity(0.4, rho_l, rho_g, coef=0.345)
    assert wide / hk == pytest.approx(2.0, rel=1e-12)  # √(0.4/0.1)


def test_bubbly_to_slug_diameter_effect_air_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    """HK: v∞T пересекает v∞ при d ≈ 4.6 см по тексту статьи (по расчёту для AW и 0.35 — ≈5.2 см);
    в трубе меньшего диаметра v∞T < v∞, большего — v∞T > v∞."""
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    v_inf = bubbles.harmathy_rise_velocity(rho_l, rho_g, aw_fluid_kwargs["sigma_n_m"])
    small = bubbles.taylor_rise_velocity(0.0254, rho_l, rho_g, coef=0.35)
    large = bubbles.taylor_rise_velocity(0.0762, rho_l, rho_g, coef=0.35)
    assert small < v_inf < large


def test_inverse_viscosity_number_water_with_1_cp(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки»: HK, AW d = 0.0508: N_f = 35 000 (пересчёт 35 700), rtol 3%.
    # ВНИМАНИЕ: 35 700 получается при μ_L = 1.0e-3 Па·с; набор AW в ТЗ имеет μ_L = 0.89e-3 и даёт
    # ≈ 40 150 (расхождение 12–15%). Тест использует μ_L = 1.0e-3 — расхождение ТЗ вынесено в отчёт.
    n_f = bubbles.inverse_viscosity_number(
        0.0508, aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"], 1.0e-3
    )
    assert n_f == pytest.approx(35_000, rel=3e-2)


def test_inverse_viscosity_number_is_inverse_to_viscosity(aw_fluid_kwargs: dict[str, Any]) -> None:
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    a = bubbles.inverse_viscosity_number(0.0508, rho_l, rho_g, 1.0e-3)
    b = bubbles.inverse_viscosity_number(0.0508, rho_l, rho_g, 2.0e-3)
    assert a / b == pytest.approx(2.0, rel=1e-12)


def test_eotvos_number_definition(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: определение N_Eo = g·d²·Δρ/σ (расчёт агента), AW, d = 0.0508: 350.1
    n_eo = bubbles.eotvos_number(
        0.0508,
        aw_fluid_kwargs["rho_l_kg_m3"],
        aw_fluid_kwargs["rho_g_kg_m3"],
        aw_fluid_kwargs["sigma_n_m"],
    )
    assert n_eo == pytest.approx(350.1, rel=1e-3)


def test_wallis_c2_reduces_to_0_345_for_large_numbers() -> None:
    # reference: HK ур. 27: при N_f > 300 и N_Eo > 100 C2 = 0.345
    assert bubbles.wallis_taylor_coefficient(35_700.0, 322.0) == pytest.approx(0.345, rel=1e-12)


def test_wallis_c2_is_positive_and_below_0_345_for_small_numbers() -> None:
    """Исправленная форма exp((3.37 − N_Eo)/m) даёт 0 < C2 < 0.345 (напечатанная давала C2 < 0)."""
    c2 = bubbles.wallis_taylor_coefficient(20.0, 5.0)
    assert 0.0 < c2 < 0.345


@pytest.mark.parametrize(("n_f", "m_expected"), [(250.0, 10.0), (18.0, 25.0)])
def test_wallis_m_is_continuous_at_regime_boundaries(n_f: float, m_expected: float) -> None:
    """ТЗ, «Опечатки», HK (26b): 69·N_f^-0.35 сшивается с (26a) и (26c): ≈ 10 и ≈ 25."""
    assert 69.0 * n_f**-0.35 == pytest.approx(m_expected, rel=1e-2)
    below = bubbles.wallis_taylor_coefficient(n_f * (1 - 1e-9), 40.0)
    above = bubbles.wallis_taylor_coefficient(n_f * (1 + 1e-9), 40.0)
    # сшивка приближённая (m = 69·N_f^-0.35 даёт 9.99 и 25.08, а не ровно 10 и 25) — допуск 5e-3
    assert below == pytest.approx(above, rel=5e-3)
