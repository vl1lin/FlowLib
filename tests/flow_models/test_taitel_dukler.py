"""Тесты `flow_models/taitel_dukler.py` — модель Taitel–Dukler 1976 (IFlowModel).

Ожидаемый API (`from correlations.flow_models.taitel_dukler import TaitelDuklerModel, TaitelDuklerOptions`):
    TaitelDuklerOptions(laminar_switch=True, use_moody_friction=False)
    TaitelDuklerModel(options=None); .name == "taitel_dukler"; .options; .predict(state)
Схема (ТЗ, «Taitel & Dukler (1976)»); θ > 0 вверх (внутри α = −θ):
    1. X², Y из градиентов однофазных потоков (Fanning TD → Darcy ×4), уравнение (7) → h̃_L (наименьший корень);
    2. (25) F²·[ũ_G²·S̃_i/(C2²·Ã_G)] < 1, C2 = 1 − h̃ → расслоённый: (30) → STRATIFIED_WAVY, иначе SMOOTH;
    3. иначе h̃ < 0.5 → ANNULAR; иначе (36) T² ≥ 8Ã_G/(S̃_i·ũ_L²·(ũ_L·D̃_L)^-n) → DISPERSED_BUBBLE, иначе INTERMITTENT.
details: h_l_over_d, holdup_stratified (Ã_L/Ã), x, y, f, k, t, n_liquid, m_gas.
Допущения (TODO(question N) ТЗ): laminar_switch — ламинарность фазы по Re с фактической скоростью и
гидравлическим диаметром (6) при критическом Re = 2000 (TD порог не задаёт); перебор комбинаций режимов
«турб.–турб., турб.–лам., лам.–турб., лам.–лам.», берётся первая согласованная, иначе турб.–турб.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core import friction, gradient, stratified
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.taitel_dukler import TaitelDuklerModel, TaitelDuklerOptions

D = 0.0254

# reference: расчёт агента по TD (7), (25), (30), (36), воздух–вода, d = 2.54 см; устойчивы к ±10% скоростей
POINTS = [
    # горизонталь
    (0.0, 0.001, 0.01, FlowPattern.STRATIFIED_SMOOTH),
    (0.0, 0.001, 10.0, FlowPattern.STRATIFIED_WAVY),
    (0.0, 0.001, 31.62278, FlowPattern.ANNULAR),
    (0.0, 0.17783, 0.1, FlowPattern.INTERMITTENT),
    (0.0, 3.16228, 0.01, FlowPattern.DISPERSED_BUBBLE),
    # лёгкий подъём: расслоённый поток сокращается
    (1.0, 0.001, 0.01, FlowPattern.INTERMITTENT),
    (1.0, 0.001, 10.0, FlowPattern.STRATIFIED_WAVY),
    (1.0, 0.001, 31.62278, FlowPattern.ANNULAR),
    (1.0, 3.16228, 0.01, FlowPattern.DISPERSED_BUBBLE),
    # лёгкий спуск
    (-1.0, 0.001, 0.01, FlowPattern.STRATIFIED_SMOOTH),
    (-1.0, 0.001, 10.0, FlowPattern.STRATIFIED_WAVY),
    (-1.0, 0.56234, 0.56234, FlowPattern.INTERMITTENT),
    (-1.0, 3.16228, 0.01, FlowPattern.DISPERSED_BUBBLE),
]


def test_name_and_default_flags() -> None:
    model = TaitelDuklerModel()
    assert model.name == "taitel_dukler"
    assert model.options.laminar_switch is True  # reference: ТЗ, флаги TD
    assert model.options.use_moody_friction is False


@pytest.mark.parametrize(("theta", "v_sl", "v_sg", "expected"), POINTS)
def test_regime_branches(
    aw_state: Callable[..., FlowState],
    theta: float,
    v_sl: float,
    v_sg: float,
    expected: FlowPattern,
) -> None:
    assert TaitelDuklerModel().predict(aw_state(v_sl, v_sg, D, theta)).pattern is expected


def test_all_five_regimes_covered() -> None:
    assert {p[3] for p in POINTS} == {
        FlowPattern.STRATIFIED_SMOOTH,
        FlowPattern.STRATIFIED_WAVY,
        FlowPattern.ANNULAR,
        FlowPattern.INTERMITTENT,
        FlowPattern.DISPERSED_BUBBLE,
    }


def test_sign_convention_uphill_shrinks_stratified_flow(aw_state: Callable[..., FlowState]) -> None:
    """У TD α > 0 — спуск; в библиотеке θ > 0 — подъём. Подъём на 1° убирает расслоение, спуск — нет."""
    up = TaitelDuklerModel().predict(aw_state(0.001, 0.01, D, 1.0)).pattern
    down = TaitelDuklerModel().predict(aw_state(0.001, 0.01, D, -1.0)).pattern
    assert up is FlowPattern.INTERMITTENT
    assert down is FlowPattern.STRATIFIED_SMOOTH


def test_details_level_satisfies_equation_7(aw_state: Callable[..., FlowState]) -> None:
    result = TaitelDuklerModel().predict(aw_state(0.001, 10.0, D, 1.0))
    d = result.details
    residual = stratified.equilibrium_residual(
        d["h_l_over_d"], d["x"] ** 2, d["y"], n=d["n_liquid"], m=d["m_gas"]
    )
    assert residual == pytest.approx(0.0, abs=1e-8)


def test_details_holdup_stratified_is_area_fraction(aw_state: Callable[..., FlowState]) -> None:
    d = TaitelDuklerModel().predict(aw_state(0.001, 0.01, D, 0.0)).details
    expected = stratified.geometry(d["h_l_over_d"]).a_l / (math.pi / 4.0)
    assert d["holdup_stratified"] == pytest.approx(expected, rel=1e-12)


def test_details_k_and_t(
    aw_state: Callable[..., FlowState], aw_fluid_kwargs: dict[str, Any]
) -> None:
    """K = F·√Re_LS (31); T² = (dp/dx)_LS/(Δρ·g·cosα) (37). Точка (0.17783, 0.1) — турбулентная жидкость."""
    v_sl, v_sg, theta = 0.17783, 0.1, 0.0
    d = TaitelDuklerModel().predict(aw_state(v_sl, v_sg, D, theta)).details
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    re_sl = rho_l * v_sl * D / aw_fluid_kwargs["mu_l_pa_s"]
    froude = stratified.modified_froude(v_sg, rho_l, rho_g, D, theta)
    assert d["f"] == pytest.approx(froude, rel=1e-12)
    assert d["k"] == pytest.approx(froude * math.sqrt(re_sl), rel=1e-12)
    grad_l = gradient.friction_gradient(
        friction.fanning_to_darcy(friction.fanning_taitel_dukler(re_sl, laminar=False)),
        rho_l,
        v_sl,
        D,
    )
    assert d["t"] == pytest.approx(math.sqrt(grad_l / ((rho_l - rho_g) * 9.81)), rel=1e-12)


def test_details_x_matches_core_definition(
    aw_state: Callable[..., FlowState], aw_fluid_kwargs: dict[str, Any]
) -> None:
    """X² = (dp/dx)_LS/(dp/dx)_GS, Fanning TD · 4 (Moody/Fanning). Турбулентные обе фазы."""
    v_sl, v_sg = 0.17783, 0.1
    d = TaitelDuklerModel().predict(aw_state(v_sl, v_sg, D, 0.0)).details
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    re_sl = rho_l * v_sl * D / aw_fluid_kwargs["mu_l_pa_s"]
    re_sg = rho_g * v_sg * D / aw_fluid_kwargs["mu_g_pa_s"]
    fl = friction.fanning_taitel_dukler(re_sl, laminar=False)
    fg = friction.fanning_taitel_dukler(re_sg, laminar=re_sg < 2000.0)
    x_sq = (4.0 / D * fl * rho_l * v_sl**2 / 2.0) / (4.0 / D * fg * rho_g * v_sg**2 / 2.0)
    assert d["x"] ** 2 == pytest.approx(x_sq, rel=1e-9)


# --- флаги ---
def test_flag_laminar_switch_viscous_liquid(aw_state: Callable[..., FlowState]) -> None:
    """Вязкая жидкость (μ_L = 0.2 Па·с): учёт ламинарного режима по фактическому Re меняет результат."""
    point = (0.001, 10.0, D, 0.0)
    on = TaitelDuklerModel(TaitelDuklerOptions(laminar_switch=True))
    off = TaitelDuklerModel(TaitelDuklerOptions(laminar_switch=False))
    assert on.predict(aw_state(*point, mu_l_pa_s=0.2)).pattern is FlowPattern.ANNULAR
    assert off.predict(aw_state(*point, mu_l_pa_s=0.2)).pattern is FlowPattern.STRATIFIED_SMOOTH


def test_laminar_switch_does_not_matter_for_turbulent_flow(
    aw_state: Callable[..., FlowState],
) -> None:
    point = (0.17783, 0.1, D, 0.0)
    on = TaitelDuklerModel(TaitelDuklerOptions(laminar_switch=True)).predict(aw_state(*point))
    off = TaitelDuklerModel(TaitelDuklerOptions(laminar_switch=False)).predict(aw_state(*point))
    assert on.pattern is off.pattern is FlowPattern.INTERMITTENT


def test_flag_use_moody_friction_rough_pipe(aw_state: Callable[..., FlowState]) -> None:
    """Шероховатая труба (ε = 2 мм): Colebrook вместо гладких 0.046·Re^-0.2 меняет результат."""
    point = (0.1, 0.0178, D, 0.0)
    smooth = TaitelDuklerOptions(laminar_switch=False, use_moody_friction=False)
    rough = TaitelDuklerOptions(laminar_switch=False, use_moody_friction=True)
    assert TaitelDuklerModel(smooth).predict(aw_state(*point, eps_m=2.0e-3)).pattern is (
        FlowPattern.INTERMITTENT
    )
    assert TaitelDuklerModel(rough).predict(aw_state(*point, eps_m=2.0e-3)).pattern is (
        FlowPattern.STRATIFIED_SMOOTH
    )


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    model = TaitelDuklerModel()
    assert model.predict(aw_state(1.0, 0.0, D, 0.0)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert model.predict(aw_state(0.0, 1.0, D, 0.0)).pattern is FlowPattern.SINGLE_PHASE_GAS
