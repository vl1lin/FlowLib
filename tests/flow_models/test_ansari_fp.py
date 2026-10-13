"""Тесты `flow_models/ansari_fp.py` — модуль режима и решение плёнки Ansari 1994 (IFlowModel).

Ожидаемый API (`from correlations.flow_models.ansari_fp import ...`):
    AnsariFlowOptions(bridging_with_delta_min=True, bubble_slug_sin_theta=False, ff_over_fsl_unity=False)
    AnsariFlowModel(options=None); .name == "ansari_fp"; .options; .predict(state)
    AnnularFilm (frozen dataclass): f_e, lam_lc, rho_c_kg_m3, mu_c_pa_s, v_sc_m_s, f_sc, dp_sc_pa_m, f_sl,
        dp_sl_pa_m, y_m, f_f_over_f_sl, x_m_sq, z, delta, delta_min (float | None), h_lf
    solve_annular_film(state, *, ff_over_fsl_unity=False) -> AnnularFilm | None   — решение (96); None, если корня нет
    film_blockage(film, *, use_delta_min) -> float                                  — H_LF + λ_LC·(1 − 2δ)², (12)
Порядок проверок (ТЗ, «Ansari et al. (1994)», порядок по Примеру 4.9; Q2):
    (а) DB: левая часть (4) ≥ 0.725 + 4.15·(v_sg/v_m)^0.5 и v_sg < 3.17·v_sl (5) → DISPERSED_BUBBLE;
    (б) ANNULAR: v_sg > (6) и (12) не выполнено: H_LF + λ_LC·(1 − 2δ)² ≤ 0.12, где δ = δ_min из (8) при
        bridging_with_delta_min (Q1; наименьший корень) или δ из (96);
    (в) BUBBLE: d > d_min (19.01) и v_sg < v_sl/3 + 0.25·v_s·sinθ (2) — sinθ только при bubble_slug_sin_theta;
    (г) иначе SLUG (churn входит в slug).
details: db_left, db_right, v_sg_annular_m_s, d_min_m, v_sg_bubble_slug_m_s; при v_sg > (6) и найденной плёнке:
    film_delta, film_delta_min, film_blockage.
Источник: Ansari A.M., Sylvester N.D., Sarica C., Shoham O., Brill J.P., 1994, SPE Production & Facilities 9(2),
145–152, ур. (2)–(15), (69)–(72), (74)–(85), (94)–(102). Опечатки (ТЗ): (85) (ρ_L/ρ_g)^(1/3).
Допущения (вне ТЗ, TODO(question 1) ТЗ): критерий «δ < δ_min» отдельно не применяется (статья проверяет только (12));
при отсутствии корня (8) в (12) берётся δ из (96); при отсутствии корня (96) режим не кольцевой;
корни ищутся на сетке по ln δ (δ от 1e-9 до 0.5).
"""

import itertools
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core import annular, friction, gradient, mixture
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.ansari_fp import (
    AnsariFlowModel,
    AnsariFlowOptions,
    film_blockage,
    solve_annular_film,
)

D = 0.0254
G = 9.81


def test_name_and_default_flags() -> None:
    model = AnsariFlowModel()
    assert model.name == "ansari_fp"
    # reference: ТЗ, флаги Ansari: bridging_with_delta_min = True, остальные — False (= оригинал)
    assert model.options.bridging_with_delta_min is True
    assert model.options.bubble_slug_sin_theta is False
    assert model.options.ff_over_fsl_unity is False


# --- R2: эталоны ТЗ ---
def test_r2_pattern_is_slug(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, главные находки: для точки Примера 3.2 L_c > L_TB и режим SLUG; (12) не выполнено → не кольцевой
    assert AnsariFlowModel().predict(FlowState(**r2_kwargs)).pattern is FlowPattern.SLUG


def test_r2_transition_values(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», Ansari: DB: левая/правая часть (4) = 1.162 / 3.638;
    # v_sg кольцевой (6) / d_min = 0.8706 м/с / 0.01888 м (rtol 1e-3)
    d = AnsariFlowModel().predict(FlowState(**r2_kwargs)).details
    assert d["db_left"] == pytest.approx(1.162, rel=1e-3)
    assert d["db_right"] == pytest.approx(3.638, rel=1e-3)
    assert d["v_sg_annular_m_s"] == pytest.approx(0.8706, rel=1e-3)
    assert d["d_min_m"] == pytest.approx(0.01888, rel=1e-3)


def test_dispersed_bubble_requires_left_greater_than_right(r2_kwargs: dict[str, Any]) -> None:
    d = AnsariFlowModel().predict(FlowState(**r2_kwargs)).details
    assert d["db_left"] < d["db_right"]  # R2 — не дисперсно-пузырьковый


def test_r2_film_elements_with_ff_unity(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», Ansari (флаги Примера 4.9: ff_over_fsl_unity = True):
    # F_E / λ_LC / ρ_C = 0.5475 / 0.3605 / 334.8; X_M² / Y_M / δ (96) = 0.2214 / 71.13 / 0.1311 (rtol 1e-3)
    film = solve_annular_film(FlowState(**r2_kwargs), ff_over_fsl_unity=True)
    assert film is not None
    assert film.f_e == pytest.approx(0.5475, rel=1e-3)
    assert film.lam_lc == pytest.approx(0.3605, rel=1e-3)
    assert film.rho_c_kg_m3 == pytest.approx(334.8, rel=1e-3)
    assert film.x_m_sq == pytest.approx(0.2214, rel=1e-3)
    assert film.y_m == pytest.approx(71.13, rel=1e-3)
    assert film.delta == pytest.approx(0.1311, rel=1e-3)
    assert film.f_f_over_f_sl == 1.0


def test_r2_film_blockage_matches_tz(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ: H_LF + λ_LC·A_C/A = 0.652 (> 0.12 → не кольцевой), δ из (96), rtol 1e-3
    film = solve_annular_film(FlowState(**r2_kwargs), ff_over_fsl_unity=True)
    assert film is not None
    value = film_blockage(film, use_delta_min=False)
    assert value == pytest.approx(0.652, rel=1e-3)
    assert value > 0.12


def test_r2_delta_min_smallest_root_of_equation_8(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, вопрос 1: для R2 (8) даёт корни δ_min 0.052 и 0.208 — берётся наименьший
    film = solve_annular_film(FlowState(**r2_kwargs), ff_over_fsl_unity=True)
    assert film is not None
    assert film.delta_min == pytest.approx(0.052, rel=1e-2)


def test_r2_blockage_is_above_limit_for_either_delta(r2_kwargs: dict[str, Any]) -> None:
    """ТЗ, вопрос 1: на R2 режим SLUG при любом выборе δ в критерии (12)."""
    film = solve_annular_film(FlowState(**r2_kwargs), ff_over_fsl_unity=True)
    assert film is not None
    assert film_blockage(film, use_delta_min=True) > 0.12
    assert film_blockage(film, use_delta_min=False) > 0.12


def test_film_without_ff_unity_uses_film_friction_ratio(r2_kwargs: dict[str, Any]) -> None:
    """f_F/f_SL по Re плёнки (74), d_HF (76): X_M² = (1 − F_E)²·(f_F/f_SL)·(dp/dL)_SL/(dp/dL)_SC (9)."""
    state = FlowState(**r2_kwargs)
    film = solve_annular_film(state, ff_over_fsl_unity=False)
    assert film is not None
    alpha = 4.0 * film.delta * (1.0 - film.delta)
    v_f = state.v_sl_m_s * (1.0 - film.f_e) / alpha  # (75)
    d_hf = alpha * state.d_m  # (76)
    re_f = state.rho_l_kg_m3 * v_f * d_hf / state.mu_l_pa_s  # (74)
    f_f = friction.moody(re_f, state.eps_m / d_hf)
    assert film.f_f_over_f_sl == pytest.approx(f_f / film.f_sl, rel=1e-9)
    assert film.x_m_sq == pytest.approx(
        (1.0 - film.f_e) ** 2 * film.f_f_over_f_sl * film.dp_sl_pa_m / film.dp_sc_pa_m, rel=1e-9
    )
    unity = solve_annular_film(state, ff_over_fsl_unity=True)
    assert unity is not None
    assert film.delta != pytest.approx(unity.delta, rel=1e-4)


def test_film_solves_equation_96(r2_kwargs: dict[str, Any]) -> None:
    """(96): Y_M − Z/(4δ(1−δ)[1−4δ(1−δ)]^2.5) + X_M²/[4δ(1−δ)]³ = 0 на найденном δ."""
    film = solve_annular_film(FlowState(**r2_kwargs), ff_over_fsl_unity=True)
    assert film is not None
    a = 4.0 * film.delta * (1.0 - film.delta)
    residual = film.y_m - film.z / (a * (1.0 - a) ** 2.5) + film.x_m_sq / a**3
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_film_core_quantities(r2_kwargs: dict[str, Any]) -> None:
    """Ядро: v_SC = F_E·v_SL + v_Sg (89); μ_C — линейное смешение (90); Moody-трение по Re_SC (88)."""
    state = FlowState(**r2_kwargs)
    film = solve_annular_film(state, ff_over_fsl_unity=True)
    assert film is not None
    v_sc = film.f_e * state.v_sl_m_s + state.v_sg_m_s
    assert film.v_sc_m_s == pytest.approx(v_sc, rel=1e-12)
    mix = mixture.slip(
        film.lam_lc, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s
    )
    assert film.mu_c_pa_s == pytest.approx(mix.mu_s_pa_s, rel=1e-12)
    re_sc = film.rho_c_kg_m3 * v_sc * state.d_m / film.mu_c_pa_s
    assert film.f_sc == pytest.approx(friction.moody(re_sc, state.eps_m / state.d_m), rel=1e-12)
    assert film.dp_sc_pa_m == pytest.approx(
        gradient.friction_gradient(film.f_sc, film.rho_c_kg_m3, v_sc, state.d_m), rel=1e-12
    )


def test_film_z_depends_on_entrainment(aw_state: Callable[..., FlowState]) -> None:
    """(84)/(85): Z = 1 + 300δ при F_E > 0.9 (Wallis), иначе Whalley–Hewitt с (ρ_L/ρ_g)^(1/3)."""
    low = aw_state(0.001, 17.7828, D, 90.0)
    high = aw_state(0.001, 316.2278, D, 90.0)
    film_low = solve_annular_film(low)
    film_high = solve_annular_film(high)
    assert film_low is not None and film_high is not None
    assert film_low.f_e < 0.9 < film_high.f_e
    assert film_low.z == pytest.approx(
        annular.whalley_hewitt_z(film_low.delta, low.rho_l_kg_m3, low.rho_g_kg_m3), rel=1e-12
    )
    assert film_high.z == pytest.approx(annular.wallis_z(film_high.delta), rel=1e-12)


def test_film_thin_for_high_gas_rate(aw_state: Callable[..., FlowState]) -> None:
    """Очень большой расход газа — очень тонкая плёнка (δ ~ 1e-5): корень ищется на сетке по ln δ."""
    film = solve_annular_film(aw_state(0.001, 316.2278, D, 90.0))
    assert film is not None
    assert 1e-7 < film.delta < 1e-3


# --- ветви порядка (а)–(г): AW, p = 0.1 МПа, θ = 90° ---
# reference: расчёт агента по Ansari (2), (4)–(6), (8)–(12), (96); точки с малой чувствительностью к ±10% скоростей
BRANCH_POINTS = [
    (0.0254, 1.7783, 0.01, FlowPattern.DISPERSED_BUBBLE),  # (а)
    (0.0254, 0.001, 17.7828, FlowPattern.ANNULAR),  # (б)
    (0.0254, 0.001, 0.01, FlowPattern.SLUG),  # (г): d < d_min
    (0.1524, 0.001, 0.01, FlowPattern.BUBBLE),  # (в)
    (0.1524, 0.001, 0.1, FlowPattern.SLUG),  # d > d_min, но v_sg выше границы (2)
    (0.1524, 3.1623, 0.01, FlowPattern.DISPERSED_BUBBLE),
    (0.0508, 0.001, 17.7828, FlowPattern.ANNULAR),
]


@pytest.mark.parametrize(("d", "v_sl", "v_sg", "expected"), BRANCH_POINTS)
def test_regime_branches(
    aw_state: Callable[..., FlowState], d: float, v_sl: float, v_sg: float, expected: FlowPattern
) -> None:
    assert AnsariFlowModel().predict(aw_state(v_sl, v_sg, d, 90.0)).pattern is expected


def test_annular_requires_gas_velocity_above_turner(aw_state: Callable[..., FlowState]) -> None:
    """(6): при v_sg ниже 14.7 м/с (AW) кольцевой режим невозможен независимо от плёнки."""
    result = AnsariFlowModel().predict(aw_state(0.001, 10.0, D, 90.0))
    assert result.pattern is not FlowPattern.ANNULAR
    assert "film_delta" not in result.details


def test_annular_details_report_film(aw_state: Callable[..., FlowState]) -> None:
    d = AnsariFlowModel().predict(aw_state(0.001, 17.7828, D, 90.0)).details
    assert {"film_delta", "film_delta_min", "film_blockage"} <= set(d)
    assert d["film_blockage"] <= 0.12


def test_bubble_requires_d_greater_than_d_min(aw_state: Callable[..., FlowState]) -> None:
    """d = 2.54 см < d_min = 5.2 см (19.01): пузырькового режима нет даже при v_sg ниже границы (2)."""
    result = AnsariFlowModel().predict(aw_state(0.001, 0.01, D, 90.0))
    assert result.details["d_min_m"] > D
    assert result.pattern is FlowPattern.SLUG


def test_dispersed_bubble_not_allowed_above_packing_limit(
    aw_state: Callable[..., FlowState],
) -> None:
    """(5): при v_sg ≥ 3.17·v_sl (α_ns ≥ 0.76) дисперсно-пузырькового режима нет, даже если (4) выполнено."""
    result = AnsariFlowModel().predict(aw_state(0.5, 3.0, D, 90.0))
    assert result.pattern is not FlowPattern.DISPERSED_BUBBLE


# --- флаги ---
def test_flag_bridging_with_delta_min_changes_regime(aw_state: Callable[..., FlowState]) -> None:
    """(12): δ_min (текст статьи) vs δ из (96) (Пример 4.9): на (0.17783, 33.246) режимы разные."""
    state = aw_state(0.17783, 33.246, D, 90.0)
    paper = AnsariFlowModel(AnsariFlowOptions(bridging_with_delta_min=True)).predict(state)
    example = AnsariFlowModel(AnsariFlowOptions(bridging_with_delta_min=False)).predict(state)
    assert paper.pattern is FlowPattern.SLUG
    assert example.pattern is FlowPattern.ANNULAR


def test_flag_bubble_slug_sin_theta_changes_regime(aw_state: Callable[..., FlowState]) -> None:
    """(2) с sinθ (как у Barnea [4]): θ = 80°, граница снижается на 0.25·v_s·(1 − sin 80°)."""
    state = aw_state(0.001, 0.06224, 0.1524, 80.0)
    assert AnsariFlowModel().predict(state).pattern is FlowPattern.BUBBLE
    on = AnsariFlowModel(AnsariFlowOptions(bubble_slug_sin_theta=True))
    assert on.predict(state).pattern is FlowPattern.SLUG


def test_bubble_boundary_value(aw_state: Callable[..., FlowState]) -> None:
    # reference: Ansari (2): v_sg = 0.25·v_s + 0.333·v_sl; v_s = 1.53·[g·σ·Δρ/ρ_L²]^0.25 = 0.2495 м/с (AW)
    d = AnsariFlowModel().predict(aw_state(0.5, 0.01, 0.1524, 90.0)).details
    assert d["v_sg_bubble_slug_m_s"] == pytest.approx(0.25 * 0.2495 + 0.5 / 3.0, rel=1e-3)


def test_flag_ff_over_fsl_unity_changes_film(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 17.7828, D, 90.0)
    plain = solve_annular_film(state)
    unity = solve_annular_film(state, ff_over_fsl_unity=True)
    assert plain is not None and unity is not None
    assert unity.f_f_over_f_sl == 1.0
    assert plain.f_f_over_f_sl != pytest.approx(1.0, rel=1e-3)
    assert plain.delta != pytest.approx(unity.delta, rel=1e-4)


def test_flow_model_ff_flag_reaches_details(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 17.7828, D, 90.0)
    plain = AnsariFlowModel().predict(state).details
    unity = AnsariFlowModel(AnsariFlowOptions(ff_over_fsl_unity=True)).predict(state).details
    assert plain["film_delta"] != pytest.approx(unity["film_delta"], rel=1e-4)


# --- устойчивость ---
@pytest.mark.parametrize("theta", [-90.0, -30.0, 0.0, 30.0, 90.0])
def test_regime_defined_for_all_angles(aw_state: Callable[..., FlowState], theta: float) -> None:
    for v_sl, v_sg in itertools.product((0.001, 0.1, 2.0), (0.001, 0.5, 15.0, 316.0)):
        assert isinstance(
            AnsariFlowModel().predict(aw_state(v_sl, v_sg, 0.0508, theta)).pattern, FlowPattern
        )


@pytest.mark.parametrize("theta", [-90.0, 0.0, 90.0])
def test_film_is_none_or_physical_on_grid(aw_state: Callable[..., FlowState], theta: float) -> None:
    """`solve_annular_film` возвращает None (нет корня (96)) либо физичную плёнку 0 < δ < 0.5."""
    for v_sl, v_sg in itertools.product((0.001, 0.1, 2.0), (1.0, 15.0, 316.0)):
        film = solve_annular_film(aw_state(v_sl, v_sg, 0.0508, theta))
        assert film is None or 0.0 < film.delta < 0.5


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    model = AnsariFlowModel()
    assert model.predict(aw_state(1.0, 0.0, D, 90.0)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert model.predict(aw_state(0.0, 1.0, D, 90.0)).pattern is FlowPattern.SINGLE_PHASE_GAS
