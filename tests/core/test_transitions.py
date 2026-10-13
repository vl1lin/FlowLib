"""Тесты `core/transitions.py` — общие критерии переходов (пересечения №6–№10 ТЗ).

Ожидаемый API (`from correlations.core import transitions`):
    bubble_slug_drift_flux(v_sl, v_rise, *, c0, alpha=0.25) -> float
        v_sg = (α·C0·v_sl + α·v_rise)/(1 − α·C0)   — HK ур. 1–2 (v_g = C0·v_m + v∞)
    bubble_slug_slip(v_sl, v_rise, *, alpha=0.25, sin_theta=1.0) -> float
        v_sg = α/(1−α)·v_sl + α·v_rise·sinθ        — Barnea [4], Taitel 1980, Ansari (2)
    min_bubble_diameter(rho_l, rho_g, sigma, *, coef) -> float
        d_min = coef·√(Δρ·σ/(ρ_L²·g)); coef = 19 (Barnea [1]) или 19.01 (Ansari)
    dispersed_bubble_barnea(v_sl, v_sg, rho_l, rho_g, sigma, d, f_darcy_mixture, *, theta_rad)
        -> DispersedBubbleBoundary(d_max_m, d_deformation_m, d_migration_m, d_critical_m,
                                   alpha_ns)                    + свойство ratio = d_c/d_max
        Barnea [5]–[7]; f_M задаётся как коэффициент Darcy (Fanning·4); ratio ≥ 1 -> dispersed
    dispersed_bubble_mixture_velocity_taitel1980(d, rho_l, rho_g, sigma, mu_l) -> float
        v_M по HK ур. 6 (с ρ_L, Erratum 1989)
    max_packing_velocity_ratio(alpha) -> float                   v_sl/v_sg = (1−α)/α  (Barnea [8])
    exceeds_max_packing(v_sl, v_sg, alpha) -> bool               v_sg/(v_sl+v_sg) > α
    annular_turner_velocity(rho_l, rho_g, sigma) -> float        v_sg = 3.1·[σ·g·Δρ/ρ_g²]^0.25

Примечание: ТЗ называет переход «пузырьковый → пробковый» одной функцией; формулы Barnea/Taitel
(скольжение относительно жидкости) и HK (дрейф относительно смеси) не сводятся к одной, поэтому
реализуются двумя функциями с общим α.
"""

import math
from typing import Any

import pytest

from correlations.core import bubbles, friction, transitions
from correlations.core.constants import GRAVITY_M_S2


# --- №6: переход пузырьковый -> пробковый -----------------------------------------------
def test_drift_flux_hk_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v_gs,BS = 0.5721 м/с (rtol 1e-3)
    v_inf = bubbles.harmathy_rise_velocity(
        r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"], r2_kwargs["sigma_n_m"]
    )
    v = transitions.bubble_slug_drift_flux(r2_kwargs["v_sl_m_s"], v_inf, c0=1.2)
    assert v == pytest.approx(0.5721, rel=1e-3)


def test_drift_flux_reproduces_hk_equation_2_coefficients() -> None:
    # reference: HK ур. 2: v_gs = 0.429·v_Ls + 0.357·v∞ (α = 0.25, C0 = 1.2)
    v_sl, v_inf = 2.0, 0.3
    v = transitions.bubble_slug_drift_flux(v_sl, v_inf, c0=1.2)
    assert v == pytest.approx(0.429 * v_sl + 0.357 * v_inf, rel=2e-3)


def test_drift_flux_stagnant_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки», HK (4): AW, v_sl = 0: 0.088 м/с (пересчёт 0.0891), rtol 3%
    v_inf = bubbles.harmathy_rise_velocity(
        aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"], aw_fluid_kwargs["sigma_n_m"]
    )
    assert transitions.bubble_slug_drift_flux(0.0, v_inf, c0=1.2) == pytest.approx(0.088, rel=3e-2)


def test_slip_form_matches_barnea_equation_4() -> None:
    # reference: Barnea [4] при α = 0.25: v_sg = v_sl/3 + 0.25·v0·sinβ; Ansari (2): 0.333·v_sl
    v = transitions.bubble_slug_slip(1.0, 0.3, sin_theta=1.0)
    assert v == pytest.approx(1.0 / 3.0 + 0.25 * 0.3, rel=1e-12)


def test_slip_form_sin_theta_scales_only_rise_term() -> None:
    full = transitions.bubble_slug_slip(1.0, 0.3, sin_theta=1.0)
    half = transitions.bubble_slug_slip(1.0, 0.3, sin_theta=0.5)
    assert full - half == pytest.approx(0.25 * 0.3 * 0.5, rel=1e-12)


def test_slip_form_default_has_no_inclination_dependence() -> None:
    """Ansari (2) без sinθ — умолчание sin_theta = 1."""
    assert transitions.bubble_slug_slip(1.0, 0.3) == transitions.bubble_slug_slip(
        1.0, 0.3, sin_theta=1.0
    )


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.4])
def test_slip_form_equals_gas_void_fraction_alpha(alpha: float) -> None:
    """На границе v_g = v_L + v_rise при v_g = v_sg/α, v_L = v_sl/(1−α) (вывод Barnea [4])."""
    v_sl, v_rise = 1.3, 0.25
    v_sg = transitions.bubble_slug_slip(v_sl, v_rise, alpha=alpha)
    assert v_sg / alpha - v_sl / (1.0 - alpha) == pytest.approx(v_rise, rel=1e-12)


# --- №7: минимальный диаметр ------------------------------------------------------------
def test_min_diameter_ansari_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», Ansari: d_min = 0.01888 м (rtol 1e-3), коэффициент 19.01
    d_min = transitions.min_bubble_diameter(
        r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"], r2_kwargs["sigma_n_m"], coef=19.01
    )
    assert d_min == pytest.approx(0.01888, rel=1e-3)


def test_min_diameter_barnea_air_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки»: Barnea [1], AW: D_crit = 0.0515 м (rtol 1e-3)
    d_crit = transitions.min_bubble_diameter(
        aw_fluid_kwargs["rho_l_kg_m3"],
        aw_fluid_kwargs["rho_g_kg_m3"],
        aw_fluid_kwargs["sigma_n_m"],
        coef=19.0,
    )
    assert d_crit == pytest.approx(0.0515, rel=1e-3)
    assert d_crit > 0.0254  # d = 2.54 см < D_crit -> пузырькового режима нет


def test_min_diameter_coefficient_is_parameter() -> None:
    a = transitions.min_bubble_diameter(1000.0, 1.0, 0.07, coef=19.0)
    b = transitions.min_bubble_diameter(1000.0, 1.0, 0.07, coef=19.01)
    assert b / a == pytest.approx(19.01 / 19.0, rel=1e-12)


# --- №8: дисперсно-пузырьковый режим ----------------------------------------------------
def aw_barnea_inputs(aw: dict[str, Any]) -> dict[str, Any]:
    return {
        "rho_l": aw["rho_l_kg_m3"],
        "rho_g": aw["rho_g_kg_m3"],
        "sigma": aw["sigma_n_m"],
        "d": 0.051,
    }


def ansari_equation_4(
    v_sl: float, v_sg: float, rho_l: float, rho_g: float, sigma: float, d: float, f_moody: float
) -> tuple[float, float]:
    """Левая и правая части Ansari (4) — независимая запись формулы (f — Moody)."""
    v_m = v_sl + v_sg
    lhs = (
        2.0
        * (0.4 * sigma / ((rho_l - rho_g) * GRAVITY_M_S2)) ** 0.5
        * (rho_l / sigma) ** 0.6
        * (f_moody / (2.0 * d)) ** 0.4
        * v_m**1.2
    )
    rhs = 0.725 + 4.15 * (v_sg / v_m) ** 0.5
    return lhs, rhs


@pytest.mark.parametrize(("v_sl", "v_sg"), [(3.0, 0.5), (6.0, 1.0), (1.5, 0.3)])
def test_barnea_5_and_ansari_4_give_the_same_boundary(
    aw_fluid_kwargs: dict[str, Any], v_sl: float, v_sg: float
) -> None:
    """ТЗ, «Контрольные точки»: Barnea [5] и Ansari (4) — одна граница при f_Moody = 4·f_Fanning,
    d_C = d_CD (rtol 1e-9). Здесь θ = 90° (cosβ = 0 → d_CB = ∞ → d_C = d_CD)."""
    a = aw_barnea_inputs(aw_fluid_kwargs)
    re = (a["rho_l"] * (v_sl + v_sg) * a["d"]) / aw_fluid_kwargs["mu_l_pa_s"]
    f_moody = friction.moody(re, 0.0)
    boundary = transitions.dispersed_bubble_barnea(
        v_sl, v_sg, a["rho_l"], a["rho_g"], a["sigma"], a["d"], f_moody, theta_rad=math.pi / 2
    )
    lhs, rhs = ansari_equation_4(v_sl, v_sg, a["rho_l"], a["rho_g"], a["sigma"], a["d"], f_moody)
    assert boundary.d_critical_m == boundary.d_deformation_m
    assert boundary.ratio == pytest.approx(lhs / rhs, rel=1e-9)


def test_barnea_5_fanning_input_convention(aw_fluid_kwargs: dict[str, Any]) -> None:
    """Moody/Fanning: f_M в [5] — Fanning; ядро принимает Darcy. 2·f_F/D ≡ f_D/(2D)."""
    a = aw_barnea_inputs(aw_fluid_kwargs)
    v_sl, v_sg = 3.0, 0.5
    f_fanning = friction.fanning_taitel_dukler(1.5e5, laminar=False)
    boundary = transitions.dispersed_bubble_barnea(
        v_sl,
        v_sg,
        a["rho_l"],
        a["rho_g"],
        a["sigma"],
        a["d"],
        friction.fanning_to_darcy(f_fanning),
        theta_rad=math.pi / 2,
    )
    v_m = v_sl + v_sg
    # reference: Barnea [5] (с исправлением показателей 3/5, −2/5; ТЗ «Опечатки»), f_M — Fanning
    d_max = (
        (0.725 + 4.15 * (v_sg / v_m) ** 0.5)
        * (a["sigma"] / a["rho_l"]) ** 0.6
        * (2.0 * f_fanning / a["d"] * v_m**3) ** -0.4
    )
    assert boundary.d_max_m == pytest.approx(d_max, rel=1e-12)


def test_barnea_critical_diameters_formulas(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: Barnea [6], [7] (расчёт агента по формулам статьи; f_M — Fanning = f_D/4)
    a = aw_barnea_inputs(aw_fluid_kwargs)
    v_sl, v_sg, f_darcy, theta = 3.0, 0.5, 0.02, math.radians(30.0)
    boundary = transitions.dispersed_bubble_barnea(
        v_sl, v_sg, a["rho_l"], a["rho_g"], a["sigma"], a["d"], f_darcy, theta_rad=theta
    )
    drho = a["rho_l"] - a["rho_g"]
    d_cd = 2.0 * (0.4 * a["sigma"] / (drho * GRAVITY_M_S2)) ** 0.5
    d_cb = (
        3.0
        / 8.0
        * a["rho_l"]
        / drho
        * (f_darcy / 4.0)
        * (v_sl + v_sg) ** 2
        / (GRAVITY_M_S2 * math.cos(theta))
    )
    assert boundary.d_deformation_m == pytest.approx(d_cd, rel=1e-12)
    assert boundary.d_migration_m == pytest.approx(d_cb, rel=1e-12)
    assert boundary.d_critical_m == pytest.approx(min(d_cd, d_cb), rel=1e-12)


def test_barnea_vertical_pipe_has_no_migration_limit(aw_fluid_kwargs: dict[str, Any]) -> None:
    """ТЗ: при cosβ = 0 d_C = d_CD (миграция к верхней стенке невозможна)."""
    a = aw_barnea_inputs(aw_fluid_kwargs)
    boundary = transitions.dispersed_bubble_barnea(
        3.0, 0.5, a["rho_l"], a["rho_g"], a["sigma"], a["d"], 0.02, theta_rad=math.pi / 2
    )
    assert math.isinf(boundary.d_migration_m)
    assert boundary.d_critical_m == boundary.d_deformation_m


def test_barnea_alpha_ns_and_ratio_definition(aw_fluid_kwargs: dict[str, Any]) -> None:
    a = aw_barnea_inputs(aw_fluid_kwargs)
    boundary = transitions.dispersed_bubble_barnea(
        3.0, 1.0, a["rho_l"], a["rho_g"], a["sigma"], a["d"], 0.02, theta_rad=math.pi / 2
    )
    assert boundary.alpha_ns == pytest.approx(0.25, rel=1e-12)
    assert boundary.ratio == pytest.approx(boundary.d_critical_m / boundary.d_max_m, rel=1e-12)


def test_barnea_high_mixture_velocity_is_dispersed_low_is_not(
    aw_fluid_kwargs: dict[str, Any],
) -> None:
    a = aw_barnea_inputs(aw_fluid_kwargs)

    def ratio(v_sl: float, v_sg: float) -> float:
        re = a["rho_l"] * (v_sl + v_sg) * a["d"] / aw_fluid_kwargs["mu_l_pa_s"]
        return transitions.dispersed_bubble_barnea(
            v_sl,
            v_sg,
            a["rho_l"],
            a["rho_g"],
            a["sigma"],
            a["d"],
            friction.moody(re, 0.0),
            theta_rad=math.pi / 2,
        ).ratio

    assert ratio(6.0, 1.0) >= 1.0  # Barnea Fig. 2 (d = 2.54 см, β = 90°): DB при больших U_LS
    assert ratio(0.1, 0.1) < 1.0


def test_taitel1980_mixture_velocity_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v_M,DB = 4.405 м/с (rtol 1e-3)
    v = transitions.dispersed_bubble_mixture_velocity_taitel1980(
        r2_kwargs["d_m"],
        r2_kwargs["rho_l_kg_m3"],
        r2_kwargs["rho_g_kg_m3"],
        r2_kwargs["sigma_n_m"],
        r2_kwargs["mu_l_pa_s"],
    )
    assert v == pytest.approx(4.405, rel=1e-3)


def test_taitel1980_mixture_velocity_air_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки», HK (6): AW, d = 0.0508: v_M = 3.6 м/с (пересчёт 3.54), rtol 3%
    v = transitions.dispersed_bubble_mixture_velocity_taitel1980(
        0.0508,
        aw_fluid_kwargs["rho_l_kg_m3"],
        aw_fluid_kwargs["rho_g_kg_m3"],
        aw_fluid_kwargs["sigma_n_m"],
        aw_fluid_kwargs["mu_l_pa_s"],
    )
    assert v == pytest.approx(3.6, rel=3e-2)


def test_taitel1980_uses_liquid_viscosity_not_mixture() -> None:
    """ТЗ, «Опечатки», HK (6): множитель (ρ_L/μ_L)^0.08, а не ρ_M. Вязкость ×2 -> v_M × 2^(−0.08/1.12)."""
    a = transitions.dispersed_bubble_mixture_velocity_taitel1980(0.1, 1000.0, 1.0, 0.07, 1.0e-3)
    b = transitions.dispersed_bubble_mixture_velocity_taitel1980(0.1, 1000.0, 1.0, 0.07, 2.0e-3)
    assert b / a == pytest.approx(2.0 ** (-0.08 / 1.12), rel=1e-12)


# --- №9: плотная упаковка ---------------------------------------------------------------
def test_max_packing_ansari_5() -> None:
    # reference: Ansari (5): α = 0.76 -> v_sg = 3.17·v_sl; в статье округлено (точно 0.76/0.24 = 3.1667),
    # поэтому допуск 2e-3 (округление до 3 знаков даёт 1.1e-3)
    ratio = transitions.max_packing_velocity_ratio(0.76)
    assert 1.0 / ratio == pytest.approx(3.17, rel=2e-3)
    assert 1.0 / ratio == pytest.approx(0.76 / 0.24, rel=1e-12)


def test_max_packing_barnea_8() -> None:
    # reference: Barnea [8]: α = 0.52 -> U_LS = U_GS·(1−α)/α
    assert transitions.max_packing_velocity_ratio(0.52) == pytest.approx(0.48 / 0.52, rel=1e-12)


def test_exceeds_max_packing() -> None:
    assert transitions.exceeds_max_packing(0.4, 0.6, 0.52)  # α_ns = 0.6
    assert not transitions.exceeds_max_packing(0.6, 0.4, 0.52)  # α_ns = 0.4


# --- №10: Тёрнер -----------------------------------------------------------------------
def test_turner_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны»: Ansari v_sg(6) и HK v_gs,A = 0.8706 м/с (rtol 1e-3)
    v = transitions.annular_turner_velocity(
        r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"], r2_kwargs["sigma_n_m"]
    )
    assert v == pytest.approx(0.8706, rel=1e-3)


def test_turner_air_water(aw_fluid_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки», HK (13): AW ≈ 15 м/с (пересчёт 14.70), rtol 5%
    v = transitions.annular_turner_velocity(
        aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"], aw_fluid_kwargs["sigma_n_m"]
    )
    assert v == pytest.approx(15.0, rel=5e-2)


def test_dispersion_factor_is_ansari_4_right_hand_side(aw_fluid_kwargs: Any) -> None:
    """Ansari (4): правая часть 0.725 + 4.15·(v_sg/v_m)^0.5; левая = d_C/d_max · правая."""
    boundary = transitions.dispersed_bubble_barnea(
        3.0, 1.0, 997.0, 1.18, 0.072, 0.051, 0.02, theta_rad=math.pi / 2
    )
    assert boundary.dispersion_factor == pytest.approx(0.725 + 4.15 * math.sqrt(0.25), rel=1e-12)
