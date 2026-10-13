"""Тесты `core/annular.py` — кольцевой режим (пересечения №11–№13 ТЗ).

Ожидаемый API (`from correlations.core import annular`):
    critical_velocity(v_sg, mu_g, sigma, rho_g, rho_l) -> float
        v_crit = 1e4·v_sg·μ_g/σ·√(ρ_g/ρ_L)  (Ansari (72), HK (34); безразмерная, уже ×1e4)
    entrainment_wallis(v_crit) -> float          F_E = max(0, 1 − exp[−0.125(v_crit − 1.5)])  Ansari (71)
    entrainment_steen_wallis(v_crit) -> float    E по HK (35)/(36), ограничение 0…1
    core_liquid_fraction(f_e, v_sl, v_sg) -> float   λ_LC = F_E·v_sl/(v_sg + F_E·v_sl)  Ansari (70)
    wallis_z(delta_over_d) -> float              Z = 1 + 300·δ/d                 Ansari (84), HK (38)
    wallis_z_from_holdup(alpha_l) -> float       Z = 1 + 75·α_L                  Barnea [15], HK (39)
    whalley_hewitt_z(delta_over_d, rho_l, rho_g) -> float   Z = 1 + 24(ρ_L/ρ_g)^(1/3)·δ  Ansari (85)
    film_holdup_from_delta(delta) -> float       α = 4δ(1 − δ)                    Ansari (11)
    delta_from_film_holdup(alpha_l) -> float     обратная функция (корень < 0.5)
    film_equation_residual(alpha_l, x_sq, y, z) -> float
        Y − Z/(α(1−α)^2.5) + X²/α³   (Ansari (96) при α = 4δ(1−δ); Barnea [15] с исправлением)
    solve_film_holdup(x_sq, y, z_of_alpha) -> float | None    наименьший корень на (0, 1)
    film_instability_y(alpha_l, x_sq) -> float   (2 − 1.5α)/(α³(1 − 1.5α))·X²   Barnea [16], Ansari (8)
    film_blockage_ansari(film_holdup, lam_lc, delta) -> float   H_LF + λ_LC·(1−2δ)²   Ansari (12)
    film_blocked_barnea(alpha_l, r_sm=0.48) -> bool            α_L/R_sm ≥ 0.5     Barnea [17]
"""

import math
from typing import Any

import pytest

from correlations.core import annular, friction, gradient, mixture

# reference: ТЗ, «Элементные эталоны», Ansari R2: F_E / λ_LC / ρ_C; X_M² / Y_M / δ (96)
R2_F_E = 0.5475
R2_LAM_LC = 0.3605
R2_RHO_C = 334.8
R2_X_M_SQ = 0.2214
R2_Y_M = 71.13
R2_DELTA = 0.1311
R2_BLOCKAGE = 0.652


def r2_v_crit(r2: dict[str, Any]) -> float:
    return annular.critical_velocity(
        r2["v_sg_m_s"], r2["mu_g_pa_s"], r2["sigma_n_m"], r2["rho_g_kg_m3"], r2["rho_l_kg_m3"]
    )


# --- №13: унос -------------------------------------------------------------------------
def test_critical_velocity_r2(r2_kwargs: dict[str, Any]) -> None:
    # reference: определение (72) (расчёт агента, R2): 7.844
    assert r2_v_crit(r2_kwargs) == pytest.approx(7.844, rel=1e-3)


def test_entrainment_wallis_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    assert annular.entrainment_wallis(r2_v_crit(r2_kwargs)) == pytest.approx(R2_F_E, rel=1e-3)


def test_entrainment_wallis_clipped_at_zero_below_threshold() -> None:
    # reference: ТЗ, Ansari «ограничение ≥ 0»: при v_crit < 1.5 экспонента даёт отрицательное F_E
    assert annular.entrainment_wallis(1.0) == 0.0
    assert annular.entrainment_wallis(1.5) == 0.0


def test_entrainment_steen_wallis_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: E = 0.5666 (provisional, rtol 1e-3)
    assert annular.entrainment_steen_wallis(r2_v_crit(r2_kwargs)) == pytest.approx(0.5666, rel=1e-3)


def test_entrainment_steen_wallis_low_branch() -> None:
    # reference: HK (35): E = 0.0055·v^2.86 при v < 4 (расчёт агента: v = 3 -> 0.1271)
    assert annular.entrainment_steen_wallis(3.0) == pytest.approx(0.0055 * 3.0**2.86, rel=1e-12)


def test_entrainment_steen_wallis_high_branch_and_clip() -> None:
    # reference: HK (36): E = 0.857·lg v − 0.20 при v > 4; E ≤ 1 (HK, текст после (36))
    assert annular.entrainment_steen_wallis(8.0) == pytest.approx(
        0.857 * math.log10(8.0) - 0.20, rel=1e-12
    )
    assert annular.entrainment_steen_wallis(1.0e3) == 1.0


def test_entrainment_is_bounded_and_monotonic() -> None:
    values = [annular.entrainment_wallis(v) for v in (0.0, 2.0, 5.0, 10.0, 50.0)]
    assert all(0.0 <= x <= 1.0 for x in values)
    assert values == sorted(values)


def test_core_liquid_fraction_and_density_r2(r2_kwargs: dict[str, Any]) -> None:
    lam = annular.core_liquid_fraction(R2_F_E, r2_kwargs["v_sl_m_s"], r2_kwargs["v_sg_m_s"])
    assert lam == pytest.approx(R2_LAM_LC, rel=1e-3)
    # ρ_C (69) — линейное смешение по λ_LC (повторно используется mixture.slip)
    rho_c = mixture.slip(
        lam,
        r2_kwargs["rho_l_kg_m3"],
        r2_kwargs["rho_g_kg_m3"],
        r2_kwargs["mu_l_pa_s"],
        r2_kwargs["mu_g_pa_s"],
    ).rho_s_kg_m3
    assert rho_c == pytest.approx(R2_RHO_C, rel=1e-3)


@pytest.mark.parametrize(("f_e", "expected"), [(0.0, 0.0), (1.0, 0.5)])
def test_core_liquid_fraction_limits(f_e: float, expected: float) -> None:
    assert annular.core_liquid_fraction(f_e, 1.0, 1.0) == expected


# --- №12: фактор межфазного трения -------------------------------------------------------
def test_wallis_z() -> None:
    # reference: Ansari (84) / HK (38): Z = 1 + 300·δ; δ = 0.1311 (R2) -> 40.33
    assert annular.wallis_z(R2_DELTA) == pytest.approx(1.0 + 300.0 * R2_DELTA, rel=1e-12)


def test_wallis_z_from_holdup() -> None:
    # reference: Barnea [15], HK (39): Z = 1 + 75·α_L
    assert annular.wallis_z_from_holdup(0.4) == pytest.approx(31.0, rel=1e-12)


def test_wallis_two_forms_agree_for_thin_film() -> None:
    """ТЗ п. 12: 1 + 300·δ ≈ 1 + 75·α_L при α_L = 4δ(1−δ) ≈ 4δ."""
    delta = 0.005
    alpha = annular.film_holdup_from_delta(delta)
    assert annular.wallis_z_from_holdup(alpha) == pytest.approx(annular.wallis_z(delta), rel=5e-3)


def test_whalley_hewitt_z_uses_liquid_to_gas_density_ratio(r2_kwargs: dict[str, Any]) -> None:
    # reference: Ansari (85) с исправлением (ρ_L/ρ_g)^(1/3) (ТЗ, «Опечатки»)
    z = annular.whalley_hewitt_z(R2_DELTA, r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"])
    ratio = r2_kwargs["rho_l_kg_m3"] / r2_kwargs["rho_g_kg_m3"]
    assert z == pytest.approx(1.0 + 24.0 * ratio ** (1.0 / 3.0) * R2_DELTA, rel=1e-12)
    assert z > 1.0


# --- геометрия плёнки -------------------------------------------------------------------
@pytest.mark.parametrize("delta", [0.01, 0.1311, 0.3, 0.5])
def test_film_holdup_delta_roundtrip(delta: float) -> None:
    alpha = annular.film_holdup_from_delta(delta)
    assert annular.delta_from_film_holdup(alpha) == pytest.approx(delta, rel=1e-12)


def test_film_holdup_from_delta_values() -> None:
    # reference: Ansari (11): α = 4δ(1−δ); δ = 0.1311 -> H_LF = 0.4557 (ТЗ: 0.652 = 0.4557 + 0.1963)
    assert annular.film_holdup_from_delta(R2_DELTA) == pytest.approx(0.4557, rel=1e-3)
    assert annular.film_holdup_from_delta(0.5) == pytest.approx(1.0, rel=1e-12)


# --- №11: устойчивость и перемыкание -----------------------------------------------------
def test_film_equation_residual_zero_on_solution() -> None:
    alpha, x_sq, z = 0.3, 2.0, annular.wallis_z_from_holdup(0.3)
    y = z / (alpha * (1 - alpha) ** 2.5) - x_sq / alpha**3
    assert annular.film_equation_residual(alpha, x_sq, y, z) == pytest.approx(0.0, abs=1e-9)


def test_solve_film_holdup_roundtrip() -> None:
    alpha0, x_sq = 0.3, 2.0
    z0 = annular.wallis_z_from_holdup(alpha0)
    y = z0 / (alpha0 * (1 - alpha0) ** 2.5) - x_sq / alpha0**3
    alpha = annular.solve_film_holdup(x_sq, y, annular.wallis_z_from_holdup)
    assert alpha == pytest.approx(alpha0, rel=1e-8)


def test_solve_film_thickness_r2_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    """Ansari (96) для R2: δ = 0.1311 (rtol 1e-3); Z — Whalley–Hewitt, так как F_E = 0.5475 < 0.9."""
    rho_l, rho_g = r2_kwargs["rho_l_kg_m3"], r2_kwargs["rho_g_kg_m3"]

    def z_of_alpha(alpha: float) -> float:
        return annular.whalley_hewitt_z(annular.delta_from_film_holdup(alpha), rho_l, rho_g)

    alpha = annular.solve_film_holdup(R2_X_M_SQ, R2_Y_M, z_of_alpha)
    assert alpha is not None
    assert annular.delta_from_film_holdup(alpha) == pytest.approx(R2_DELTA, rel=1e-3)


def test_solve_film_holdup_returns_none_when_no_root() -> None:
    # Z = 0 -> Y − 0 + X²/α³ не меняет знак при Y > 0, X² > 0
    assert annular.solve_film_holdup(1.0, 1.0, lambda _alpha: 0.0) is None


def test_film_instability_y_values() -> None:
    # reference: Barnea [16] / Ansari (8), расчёт агента: α = 0.4, X² = 1 -> 1.4/(0.064·0.4) = 54.6875
    assert annular.film_instability_y(0.4, 1.0) == pytest.approx(54.6875, rel=1e-12)
    # α = 0.5, X² = 2 -> 2·1.25/(0.125·0.25) = 80
    assert annular.film_instability_y(0.5, 2.0) == pytest.approx(80.0, rel=1e-12)


def test_film_instability_y_is_linear_in_x_squared() -> None:
    assert annular.film_instability_y(0.3, 4.0) == pytest.approx(
        4.0 * annular.film_instability_y(0.3, 1.0), rel=1e-12
    )


def test_film_blockage_ansari_r2_matches_tz() -> None:
    # reference: ТЗ, «Элементные эталоны», Ansari: H_LF + λ_LC·A_C/A = 0.652 (> 0.12 -> не кольцевой)
    h_lf = annular.film_holdup_from_delta(R2_DELTA)
    value = annular.film_blockage_ansari(h_lf, R2_LAM_LC, R2_DELTA)
    assert value == pytest.approx(R2_BLOCKAGE, rel=1e-3)
    assert value > 0.12


def test_film_blockage_without_entrainment_reduces_to_holdup() -> None:
    assert annular.film_blockage_ansari(0.1, 0.0, 0.03) == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize(
    ("alpha", "blocked"), [(0.20, False), (0.23, False), (0.25, True), (0.4, True)]
)
def test_film_blocked_barnea(alpha: float, blocked: bool) -> None:
    # reference: Barnea [17]: α_L/R_sm ≥ 0.5 при R_sm = 0.48 -> граница α_L = 0.24
    assert annular.film_blocked_barnea(alpha) is blocked


def test_film_blocked_barnea_custom_r_sm() -> None:
    assert annular.film_blocked_barnea(0.3, r_sm=0.8) is False
    assert annular.film_blocked_barnea(0.41, r_sm=0.8) is True


# --- Moody/Fanning: HK кольцевой f_c ------------------------------------------------------
def test_hk_annular_fanning_friction_gradient_equals_core_darcy_form(
    r2_kwargs: dict[str, Any],
) -> None:
    """HK (32)–(33): трение = 2·f_c·ρ_c·v_g²/d с f_c — Fanning (Blasius·[1 + 75(1 − f_g)]).

    Ядро считает через Darcy: f_D·ρ·v²/(2d), f_D = 4·f_F. Градиенты обязаны совпасть.
    """
    rho_c, v_g, d = R2_RHO_C, 1.5, r2_kwargs["d_m"]
    re_g = 3.0e5
    f_g = 0.674  # reference: ТЗ, HK: f_g (кольцевой) = 0.6740 (provisional)
    f_c_fanning = friction.fanning_blasius(re_g) * annular.wallis_z_from_holdup(1.0 - f_g)
    hk_form = 2.0 * f_c_fanning * rho_c * v_g**2 / d
    core_form = gradient.friction_gradient(friction.fanning_to_darcy(f_c_fanning), rho_c, v_g, d)
    assert core_form == pytest.approx(hk_form, rel=1e-12)
