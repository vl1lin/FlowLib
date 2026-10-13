"""Тесты `flow_models/barnea.py` — унифицированная модель режимов Barnea (IFlowModel).

Ожидаемый API (`from correlations.flow_models.barnea import BarneaModel, BarneaOptions`):
    BarneaOptions(fm_noslip_reynolds=False, elongated_bubble_horizontal_only=True,
                  use_moody_friction=True, split_churn=True, friction_ratio_as_published=True)
    published_friction_ratio(d_tilde_l, u_tilde_l, re_l) -> float   # (D̃_L·Ũ_L)^-n, n = 0.2 / 1
    BarneaModel(options=None); .name == "barnea"; .options; .predict(state) -> FlowPatternResult
Логика — Fig. 6 (ТЗ, «Barnea (1987)»), θ = β > 0 вверх, обработка в порядке:
    1. дисперсно-пузырьковый (d_max < d_C, α_ns < 0.52) → DISPERSED_BUBBLE;
    2. при cosβ ≠ 0: равновесный уровень; [9] < 1 и [14] не выполнено → STRATIFIED_SMOOTH/WAVY ([20] или [23]);
    3. иначе уравнение плёнки [15] (наименьший корень); [16] или [17] не выполнены → ANNULAR;
    4. пузырьковый возможен (D > D_crit [1], |β| ≥ β_crit [2]) и v_sg ниже [4] → BUBBLE;
    5. иначе (пузырьковый невозможен) α_s = 0 и β = 0 → ELONGATED_BUBBLE;
    6. α_s ≤ 0.52 → SLUG, иначе CHURN.
details: d_max_m, d_critical_m, alpha_ns, alpha_s, h_l_over_d (если считался уровень),
         alpha_l_film, film_unstable, film_blocked (если считалась плёнка).
Допущения (TODO(question N) ТЗ): [2] — d = d_CD, γ = 1.3, C_L = 0.8 (Q7); при нескольких корнях
берётся наименьший (Q9).
Флаг friction_ratio_as_published (по умолчанию True): f_LS/f_L в [14] — по статье (D̃_L·Ũ_L)^-n,
n = 0.2 (турбулентное течение жидкости), n = 1 (ламинарное) — подтверждено пользователем по тексту
статьи. False — отношение коэффициентов трения f(Re_LS)/f(Re_L) (прежняя реализация агента).
Допущение: режим течения (n) выбирается по Re слоя жидкости Re_L < 2000 (`friction.RE_LAMINAR_MAX`).
details["f_ls_over_f_l"] — значение отношения, использованное в [14] (если [9] < 1).
"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from correlations.core import friction, gradient, stratified, transitions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.barnea import BarneaModel, BarneaOptions, published_friction_ratio

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "barnea_maps.json"


def load_map_cases() -> list[dict[str, Any]]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"]


def predict(
    aw_state: Callable[..., FlowState],
    v_sl: float,
    v_sg: float,
    d: float,
    theta: float,
    options: BarneaOptions | None = None,
    **override: Any,
) -> Any:
    return BarneaModel(options).predict(aw_state(v_sl, v_sg, d, theta, **override))


def test_name_and_default_flags() -> None:
    model = BarneaModel()
    assert model.name == "barnea"
    # reference: ТЗ, флаги Barnea и значения по умолчанию (= оригинал статьи)
    assert model.options.fm_noslip_reynolds is False
    assert model.options.elongated_bubble_horizontal_only is True
    assert model.options.use_moody_friction is True
    assert model.options.split_churn is True
    # reference: пользователь, решение по [14]: расчёт по алгоритму статьи
    assert model.options.friction_ratio_as_published is True


# --- 12 ориентировочных точек карт (Fig. 1–2): reference/barnea_maps.json, provisional ---
@pytest.mark.parametrize("case", load_map_cases(), ids=lambda c: str(c["inputs"]))
def test_barnea_map_points_from_tz(
    aw_state: Callable[..., FlowState], case: dict[str, Any]
) -> None:
    """Точки помечены provisional: при падении точку не сдвигаем и допуск не ослабляем — в отчёт."""
    assert case["provisional"] is True
    options = BarneaOptions.model_validate(case["flags"])
    inputs = case["inputs"]
    result = BarneaModel(options).predict(
        aw_state(inputs["v_sl_m_s"], inputs["v_sg_m_s"], inputs["d_m"], inputs["theta_deg"])
    )
    assert result.pattern is FlowPattern(case["expected"])


def test_reference_cases_have_required_fields() -> None:
    for case in load_map_cases():
        assert {"source", "flags", "inputs", "expected", "provisional"} <= set(case)


# --- ветви Fig. 6: (d, β°, v_sl, v_sg, режим) ---
# reference: расчёт агента по Fig. 6 и [1]–[23] (ТЗ), воздух–вода; точки устойчивы к ±10% скоростей
BRANCH_POINTS = [
    (0.0254, 0.0, 3.16228, 0.01, FlowPattern.DISPERSED_BUBBLE),  # шаг 1
    (
        0.0254,
        0.0,
        0.001,
        0.01,
        FlowPattern.STRATIFIED_SMOOTH,
    ),  # шаг 2: [9] < 1, [14], [20], [23] нет
    (0.0254, 0.0, 0.001, 10.0, FlowPattern.STRATIFIED_WAVY),  # [20] (ветровые волны)
    # [23] (гравитационные волны); прежняя точка (0.051, -80°, 0.001, 0.01) при friction_ratio_as_published=True
    # выполняет [14] и уходит из расслоённого режима — она сохранена в test_flag_friction_ratio_legacy_points
    (0.051, -60.0, 0.01, 0.01, FlowPattern.STRATIFIED_WAVY),
    (0.0254, 0.0, 0.001, 31.62278, FlowPattern.ANNULAR),  # [9] ≥ 1 → шаг 3
    (0.051, -80.0, 0.56234, 0.01, FlowPattern.ANNULAR),  # [14] выполнено → шаг 3
    (0.0254, 90.0, 0.001, 17.7828, FlowPattern.ANNULAR),  # cosβ = 0: стратифицированного нет
    (0.051, -80.0, 1.0, 1.0, FlowPattern.SLUG),  # [14] → шаг 3 → [17] → шаг 4–6
    (0.1524, 90.0, 0.001, 0.01, FlowPattern.BUBBLE),  # пузырьковый возможен, ниже [4]
    (0.1524, 60.0, 0.001, 0.01, FlowPattern.BUBBLE),
    (0.1524, 90.0, 0.001, 0.1, FlowPattern.SLUG),  # возможен, выше [4], α_s ≤ 0.52
    (0.1524, 90.0, 0.001, 10.0, FlowPattern.CHURN),  # возможен, выше [4], α_s > 0.52
    (0.0254, 0.0, 0.17783, 0.31623, FlowPattern.ELONGATED_BUBBLE),  # невозможен, α_s = 0, β = 0
    (0.0254, 0.0, 0.31623, 0.01778, FlowPattern.ELONGATED_BUBBLE),
    (0.0254, 90.0, 0.001, 0.01, FlowPattern.SLUG),  # α_s = 0, но β ≠ 0 → SLUG (флаг по умолчанию)
    (0.0254, 90.0, 0.001, 1.0, FlowPattern.SLUG),
    (0.0254, 90.0, 0.001, 5.6234, FlowPattern.CHURN),  # α_s > 0.52
    (0.0254, 90.0, 0.001, 10.0, FlowPattern.CHURN),  # плёнка неустойчива ([16]), не перемкнута
    (0.0254, 90.0, 0.17783, 3.16228, FlowPattern.SLUG),  # перемыкание ([17]), устойчива
]


@pytest.mark.parametrize(("d", "theta", "v_sl", "v_sg", "expected"), BRANCH_POINTS)
def test_flow_chart_branches(
    aw_state: Callable[..., FlowState],
    d: float,
    theta: float,
    v_sl: float,
    v_sg: float,
    expected: FlowPattern,
) -> None:
    assert predict(aw_state, v_sl, v_sg, d, theta).pattern is expected


def test_film_criteria_details_distinguish_16_and_17(aw_state: Callable[..., FlowState]) -> None:
    unstable_only = predict(aw_state, 0.001, 10.0, 0.0254, 90.0).details
    blocked_only = predict(aw_state, 0.17783, 3.16228, 0.0254, 90.0).details
    assert unstable_only["film_unstable"] == 1.0
    assert unstable_only["film_blocked"] == 0.0
    assert blocked_only["film_unstable"] == 0.0
    assert blocked_only["film_blocked"] == 1.0


def test_film_blocked_matches_barnea_17_threshold(aw_state: Callable[..., FlowState]) -> None:
    """[17]: перемыкание при α_L ≥ 0.24 (R_sm = 0.48)."""
    details = predict(aw_state, 0.17783, 3.16228, 0.0254, 90.0).details
    assert details["alpha_l_film"] >= 0.24


def test_angle_sign_convention_up_vs_down(aw_state: Callable[..., FlowState]) -> None:
    """Знак угла (θ > 0 вверх): точки Fig. 1 из ТЗ — подъём +30° даёт SLUG, спуск −30° — расслоённый."""
    up = predict(aw_state, 0.1, 1.0, 0.051, 30.0).pattern
    down = predict(aw_state, 0.01, 1.0, 0.051, -30.0).pattern
    assert up is FlowPattern.SLUG
    assert down is FlowPattern.STRATIFIED_WAVY
    # то же течение при зеркальном угле расслаивается/перемежается иначе
    assert predict(aw_state, 0.01, 1.0, 0.051, 30.0).pattern is not FlowPattern.STRATIFIED_WAVY


def test_vertical_pipe_has_no_stratified_flow(aw_state: Callable[..., FlowState]) -> None:
    """ТЗ: при |β| = 90° расслоённый режим невозможен (cosβ = 0) — ни при каких скоростях."""
    for v_sl, v_sg in [(0.001, 0.01), (0.001, 1.0), (0.01, 0.01), (0.1, 5.0)]:
        for theta in (90.0, -90.0):
            pattern = predict(aw_state, v_sl, v_sg, 0.0254, theta).pattern
            assert pattern not in (FlowPattern.STRATIFIED_SMOOTH, FlowPattern.STRATIFIED_WAVY)


# --- details ---
def test_details_stratified_level_satisfies_equilibrium(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 0.01, 0.0254, 0.0)
    result = BarneaModel().predict(state)
    h = result.details["h_l_over_d"]
    f_l = friction.moody(state.rho_l_kg_m3 * 0.001 * 0.0254 / state.mu_l_pa_s, 0.0)
    f_g = friction.moody(state.rho_g_kg_m3 * 0.01 * 0.0254 / state.mu_g_pa_s, 0.0)
    grad_l = gradient.friction_gradient(f_l, state.rho_l_kg_m3, 0.001, 0.0254)
    grad_g = gradient.friction_gradient(f_g, state.rho_g_kg_m3, 0.01, 0.0254)
    x_sq = stratified.x_squared(grad_l, grad_g)
    assert stratified.equilibrium_residual(h, x_sq, 0.0) == pytest.approx(0.0, abs=1e-8)


def test_details_alpha_s_matches_barnea_18(aw_state: Callable[..., FlowState]) -> None:
    """[18]: α_s = 0.058·[d_C·(2f_M/D·U_M³)^(2/5)·(ρ_L/σ)^(3/5) − 0.725]²; через d_max из [5]."""
    state = aw_state(0.001, 1.0, 0.0254, 90.0)
    result = BarneaModel().predict(state)
    v_m = 0.001 + 1.0
    f_m = friction.moody(state.rho_l_kg_m3 * v_m * 0.0254 / state.mu_l_pa_s, 0.0)
    boundary = transitions.dispersed_bubble_barnea(
        0.001,
        1.0,
        state.rho_l_kg_m3,
        state.rho_g_kg_m3,
        state.sigma_n_m,
        0.0254,
        f_m,
        theta_rad=math.pi / 2,
    )
    bracket = boundary.ratio * (0.725 + 4.15 * math.sqrt(boundary.alpha_ns)) - 0.725
    assert result.details["alpha_s"] == pytest.approx(0.058 * max(bracket, 0.0) ** 2, rel=1e-9)
    assert result.details["alpha_ns"] == pytest.approx(boundary.alpha_ns, rel=1e-12)


def test_alpha_s_is_zero_when_bracket_negative(aw_state: Callable[..., FlowState]) -> None:
    """ТЗ: внутренняя скобка [18] < 0 → α_s = 0."""
    assert predict(aw_state, 0.001, 0.01, 0.0254, 90.0).details["alpha_s"] == 0.0


# --- флаги: каждый меняет результат хотя бы в одной точке ---
def test_flag_split_churn(aw_state: Callable[..., FlowState]) -> None:
    point = (0.001, 5.6234, 0.0254, 90.0)
    assert predict(aw_state, *point).pattern is FlowPattern.CHURN
    off = BarneaOptions(split_churn=False)
    assert predict(aw_state, *point, options=off).pattern is FlowPattern.SLUG


def test_flag_elongated_bubble_horizontal_only(aw_state: Callable[..., FlowState]) -> None:
    point = (0.001, 0.01, 0.0254, 90.0)
    assert predict(aw_state, *point).pattern is FlowPattern.SLUG
    off = BarneaOptions(elongated_bubble_horizontal_only=False)
    assert predict(aw_state, *point, options=off).pattern is FlowPattern.ELONGATED_BUBBLE


def test_flag_use_moody_friction_rough_pipe(aw_state: Callable[..., FlowState]) -> None:
    """Шероховатая труба (ε = 1 мм): Colebrook (по умолчанию) и гладкие Fanning-формулы TD дают разное."""
    point = (0.001, 10.0, 0.0254, 90.0)
    assert predict(aw_state, *point, eps_m=1.0e-3).pattern is FlowPattern.ANNULAR
    off = BarneaOptions(use_moody_friction=False)
    assert predict(aw_state, *point, options=off, eps_m=1.0e-3).pattern is FlowPattern.CHURN


def test_flag_fm_noslip_reynolds(aw_state: Callable[..., FlowState]) -> None:
    """f_M по no-slip Re вместо Re = ρ_L·v_m·D/μ_L (полоса вблизи границы SLUG/CHURN)."""
    point = (0.01, 3.843667, 0.0254, 30.0)
    assert predict(aw_state, *point).pattern is FlowPattern.SLUG
    on = BarneaOptions(fm_noslip_reynolds=True)
    assert predict(aw_state, *point, options=on).pattern is FlowPattern.CHURN


# --- [14]: f_LS/f_L по статье, флаг friction_ratio_as_published ---
# reference: Barnea 1987, [14] и текст статьи: f_LS/f_L = (D̃_L·Ũ_L)^-n, n = 0.2 (турбулентное течение
# жидкости), n = 1 (ламинарное); n по Re слоя жидкости (допущение, RE_LAMINAR_MAX = 2000)
@pytest.mark.parametrize("h", [0.1, 0.3, 0.5, 0.7])
def test_published_friction_ratio_turbulent(h: float) -> None:
    geo = stratified.geometry(h)
    ratio = published_friction_ratio(geo.d_l, geo.u_l, re_l=1.0e5)
    assert ratio == pytest.approx((geo.d_l * geo.u_l) ** -0.2, rel=1e-12)


@pytest.mark.parametrize("h", [0.1, 0.3, 0.5, 0.7])
def test_published_friction_ratio_laminar(h: float) -> None:
    geo = stratified.geometry(h)
    ratio = published_friction_ratio(geo.d_l, geo.u_l, re_l=500.0)
    assert ratio == pytest.approx((geo.d_l * geo.u_l) ** -1.0, rel=1e-12)


def test_published_friction_ratio_switches_at_re_laminar_max() -> None:
    geo = stratified.geometry(0.3)
    below = published_friction_ratio(geo.d_l, geo.u_l, re_l=friction.RE_LAMINAR_MAX - 1.0)
    at = published_friction_ratio(geo.d_l, geo.u_l, re_l=friction.RE_LAMINAR_MAX)
    assert below == pytest.approx((geo.d_l * geo.u_l) ** -1.0, rel=1e-12)
    assert at == pytest.approx((geo.d_l * geo.u_l) ** -0.2, rel=1e-12)


def _layer_reynolds(state: FlowState, h: float) -> float:
    geo = stratified.geometry(h)
    return state.rho_l_kg_m3 * state.v_sl_m_s * geo.u_l * geo.d_l * state.d_m / state.mu_l_pa_s


def test_details_ratio_turbulent_layer_follows_article(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.2, 0.01, 0.051, -80.0)
    details = BarneaModel().predict(state).details
    h = details["h_l_over_d"]
    geo = stratified.geometry(h)
    assert _layer_reynolds(state, h) > friction.RE_LAMINAR_MAX
    assert details["f_ls_over_f_l"] == pytest.approx((geo.d_l * geo.u_l) ** -0.2, rel=1e-9)


def test_details_ratio_laminar_layer_follows_article(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 0.01, 0.0254, 0.0)
    details = BarneaModel().predict(state).details
    h = details["h_l_over_d"]
    geo = stratified.geometry(h)
    assert _layer_reynolds(state, h) < friction.RE_LAMINAR_MAX
    assert details["f_ls_over_f_l"] == pytest.approx((geo.d_l * geo.u_l) ** -1.0, rel=1e-9)


def test_details_ratio_flag_off_is_reynolds_ratio(aw_state: Callable[..., FlowState]) -> None:
    """False: f(Re_LS)/f(Re_L) по Moody (прежняя реализация)."""
    state = aw_state(0.2, 0.01, 0.051, -80.0)
    off = BarneaOptions(friction_ratio_as_published=False)
    details = BarneaModel(off).predict(state).details
    h = details["h_l_over_d"]
    rel = state.eps_m / state.d_m
    re_sl = state.rho_l_kg_m3 * state.v_sl_m_s * state.d_m / state.mu_l_pa_s
    expected = friction.moody(re_sl, rel) / friction.moody(_layer_reynolds(state, h), rel)
    assert details["f_ls_over_f_l"] == pytest.approx(expected, rel=1e-9)


def test_flag_friction_ratio_as_published_changes_pattern(
    aw_state: Callable[..., FlowState],
) -> None:
    """(d = 5.1 см, β = −80°, v_sl = 0.2, v_sg = 0.01): z/limit = 1.6 по статье — [14] выполнено,
    f(Re) — не выполнено (расслоённый режим). Запас по границе > 10%."""
    point = (0.2, 0.01, 0.051, -80.0)
    on = predict(aw_state, *point)
    off = predict(aw_state, *point, options=BarneaOptions(friction_ratio_as_published=False))
    assert off.pattern is FlowPattern.STRATIFIED_WAVY
    assert on.pattern not in (FlowPattern.STRATIFIED_SMOOTH, FlowPattern.STRATIFIED_WAVY)


def test_flag_friction_ratio_legacy_points(aw_state: Callable[..., FlowState]) -> None:
    """Прежняя точка ветви [23] (ламинарная плёнка, Re_L ≈ 680): с f(Re) — STRATIFIED_WAVY, по статье —
    [14] выполнено (z = 6.4e-6 > предела 2.5e-6)."""
    point = (0.001, 0.01, 0.051, -80.0)
    off = BarneaOptions(friction_ratio_as_published=False)
    assert predict(aw_state, *point, options=off).pattern is FlowPattern.STRATIFIED_WAVY
    assert predict(aw_state, *point).pattern not in (
        FlowPattern.STRATIFIED_SMOOTH,
        FlowPattern.STRATIFIED_WAVY,
    )


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    assert predict(aw_state, 1.0, 0.0, 0.0254, 30.0).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert predict(aw_state, 0.0, 1.0, 0.0254, 30.0).pattern is FlowPattern.SINGLE_PHASE_GAS


def test_bubbly_flow_impossible_in_small_pipe(aw_state: Callable[..., FlowState]) -> None:
    """[1]: d = 2.54 см < D_crit = 5.15 см — пузырькового режима нет ни при каких v_sl, v_sg на вертикали."""
    d_crit = transitions.min_bubble_diameter(997.0, 1.18, 0.072, coef=19.0)
    assert d_crit > 0.0254
    for v_sl, v_sg in [(0.001, 0.01), (0.01, 0.01), (0.1, 0.05), (0.3, 0.1)]:
        pattern = predict(aw_state, v_sl, v_sg, 0.0254, 90.0).pattern
        assert pattern is not FlowPattern.BUBBLE
