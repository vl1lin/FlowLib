"""Тесты `models/mukherjee_brill.py` — корреляция Mukherjee–Brill (ICorrelation).

Ожидаемый API (`from correlations.models.mukherjee_brill import ...`):
    MukherjeeBrillOptions(include_acceleration=True, stratified_combined_momentum=False,
                          wall_shear_as_published=False, annular_fr_log_interp=False)
    MukherjeeBrill(options=None, flow_model=None); .name == "mukherjee_brill"; .options; .flow_model;
        .calc(state) -> CorrelationResult        (flow_model по умолчанию — MukherjeeBrillFlowModel())
    liquid_holdup(n_lv, n_gv, n_l, theta_rad, *, stratified) -> float          ур. (10), Table 1
    annular_friction_ratio(h_r, *, log_interpolation=False) -> float          Table 2, вне 0.01…10 → 1.0
    equivalent_diameters(delta, d_m) -> tuple[float, float]                    (D_EL, D_Eg), ур. (24)–(25)
details: lambda_l, n_lv, n_gv, n_l, rho_ns_kg_m3, mu_ns_pa_s, re_ns, f_ns, rho_s_kg_m3, ek, h_l;
    ANNULAR: + h_r, f_r; STRATIFIED: + delta, h_l_over_d, d_el_m, d_eg_m, f_l, f_g, tau_wl_pa, tau_wg_pa.
Источник: Mukherjee H., Brill J.P., 1985, JERT 107(4), 549–554, ур. (1)–(27), Table 1–2.
Исправления (ТЗ, «Опечатки»): N_L с g; (10) — общепринятая форма скобок; (4.135) монографии: sin²θ, не sin³θ.
Допущения (вне ТЗ, TODO(question N)): в стратифицированной ветви относительная шероховатость фаз ε/D_E;
модели режима принимаются только с шаблонами BUBBLE/SLUG/ANNULAR/STRATIFIED (иначе UnsupportedPatternError).
"""

import math
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core import dimensionless, friction, mixture, stratified
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.mukherjee_brill_fp import MukherjeeBrillFlowModel
from correlations.models.mukherjee_brill import (
    MukherjeeBrill,
    MukherjeeBrillOptions,
    annular_friction_ratio,
    equivalent_diameters,
    liquid_holdup,
)

# reference: ТЗ, «Сквозные эталоны» MB: R1, include_acceleration = False
EX_4_8 = MukherjeeBrillOptions(include_acceleration=False)
D = 0.1


def test_default_flags() -> None:
    options = MukherjeeBrill().options
    assert options.include_acceleration is True
    assert options.stratified_combined_momentum is False
    assert options.wall_shear_as_published is False
    assert options.annular_fr_log_interp is False


def test_default_flow_model_is_mukherjee_brill_flow_patterns() -> None:
    assert isinstance(MukherjeeBrill().flow_model, MukherjeeBrillFlowModel)


# --- R1: сквозной эталон ---
def test_r1_end_to_end(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Сквозные эталоны», MB R1, accel = False: SLUG, H_L = 0.5603, 4734 Па/м
    # (книга 4727), rtol 0.5%
    result = MukherjeeBrill(EX_4_8).calc(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.SLUG
    assert result.holdup == pytest.approx(0.5603, rel=5e-3)
    assert result.dpdl_pa_m == pytest.approx(4734, rel=5e-3)
    assert result.dpdl_acceleration_pa_m == 0.0


def test_r1_components(r1_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R1): гравитация 4598.2, трение 135.71 (f_ns Colebrook, ρ_s), Ek = 1.123e-4
    result = MukherjeeBrill(EX_4_8).calc(FlowState(**r1_kwargs))
    assert result.dpdl_gravity_pa_m == pytest.approx(4598.2, rel=1e-3)
    assert result.dpdl_friction_pa_m == pytest.approx(135.71, rel=2e-3)
    assert result.details["ek"] == pytest.approx(1.1229e-4, rel=1e-3)


def test_r1_elements(r1_kwargs: dict[str, Any]) -> None:
    d = MukherjeeBrill().calc(FlowState(**r1_kwargs)).details
    # reference: ТЗ, «Элементные эталоны», MB: N_L / N_gv = 0.01176 / 11.53
    assert d["n_l"] == pytest.approx(0.01176, rel=1e-3)
    assert d["n_gv"] == pytest.approx(11.53, rel=1e-3)
    assert d["re_ns"] == pytest.approx(3.152e5, rel=1e-3)


def test_r1_default_includes_acceleration(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    with_acc = MukherjeeBrill().calc(state)
    without = MukherjeeBrill(EX_4_8).calc(state)
    ek = with_acc.details["ek"]
    base = without.dpdl_pa_m
    assert with_acc.dpdl_pa_m == pytest.approx(base / (1.0 - ek), rel=1e-12)
    assert with_acc.dpdl_acceleration_pa_m > 0.0


def test_bubble_slug_friction_uses_slip_density_and_moody(r1_kwargs: dict[str, Any]) -> None:
    """ур. (14): трение f_ns·ρ_s·v_m²/(2d), f_ns — Moody по Re_ns; гравитация ρ_s·g·sinθ."""
    state = FlowState(**r1_kwargs)
    result = MukherjeeBrill(EX_4_8).calc(state)
    d = result.details
    v_m = state.v_sl_m_s + state.v_sg_m_s
    f_ns = friction.moody(d["re_ns"], state.eps_m / state.d_m)
    assert d["f_ns"] == pytest.approx(f_ns, rel=1e-12)
    assert result.dpdl_friction_pa_m == pytest.approx(
        f_ns * d["rho_s_kg_m3"] * v_m**2 / (2.0 * state.d_m), rel=1e-12
    )
    assert result.dpdl_gravity_pa_m == pytest.approx(d["rho_s_kg_m3"] * 9.81, rel=1e-12)


# --- ур. (10), Table 1 ---
C_UPHILL = (-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657)
C_DOWN_STRATIFIED = (-1.330282, 4.808139, 4.171584, 56.262268, 0.079951, 0.504887)
C_DOWN_OTHER = (-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952)


def holdup_by_formula(
    coef: tuple[float, ...], n_lv: float, n_gv: float, n_l: float, sin_theta: float, power: int = 2
) -> float:
    """Ур. (10) с параметрами Table 1; `power` — степень sinθ при C3 (по ТЗ — 2)."""
    c1, c2, c3, c4, c5, c6 = coef
    return math.exp(
        (c1 + c2 * sin_theta + c3 * sin_theta**power + c4 * n_l**2) * n_gv**c5 / n_lv**c6
    )


def test_liquid_holdup_r1_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    nums = dimensionless.duns_ros(
        r1_kwargs["v_sl_m_s"],
        r1_kwargs["v_sg_m_s"],
        r1_kwargs["rho_l_kg_m3"],
        r1_kwargs["sigma_n_m"],
        r1_kwargs["mu_l_pa_s"],
    )
    h = liquid_holdup(nums.n_lv, nums.n_gv, nums.n_l, math.pi / 2, stratified=False)
    assert h == pytest.approx(0.5603, rel=1e-3)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 90.0])
def test_liquid_holdup_uphill_and_horizontal_set(theta_deg: float) -> None:
    n_lv, n_gv, n_l = 3.0, 8.0, 0.01
    value = liquid_holdup(n_lv, n_gv, n_l, math.radians(theta_deg), stratified=False)
    expected = holdup_by_formula(C_UPHILL, n_lv, n_gv, n_l, math.sin(math.radians(theta_deg)))
    assert value == pytest.approx(expected, rel=1e-12)
    # «All» — для подъёма флаг stratified не влияет
    assert value == liquid_holdup(n_lv, n_gv, n_l, math.radians(theta_deg), stratified=True)


def test_liquid_holdup_downhill_stratified_set() -> None:
    n_lv, n_gv, n_l, theta = 3.0, 8.0, 0.01, math.radians(-30.0)
    value = liquid_holdup(n_lv, n_gv, n_l, theta, stratified=True)
    assert value == pytest.approx(
        holdup_by_formula(C_DOWN_STRATIFIED, n_lv, n_gv, n_l, math.sin(theta)), rel=1e-12
    )


def test_liquid_holdup_downhill_other_set() -> None:
    n_lv, n_gv, n_l, theta = 3.0, 8.0, 0.01, math.radians(-30.0)
    value = liquid_holdup(n_lv, n_gv, n_l, theta, stratified=False)
    assert value == pytest.approx(
        holdup_by_formula(C_DOWN_OTHER, n_lv, n_gv, n_l, math.sin(theta)), rel=1e-12
    )
    assert value != pytest.approx(liquid_holdup(n_lv, n_gv, n_l, theta, stratified=True), rel=1e-3)


def test_liquid_holdup_c3_multiplies_sin_squared_not_cubed() -> None:
    """ТЗ, «Опечатки», монография (4.135): sin³θ неверно — MB (10): sin²θ. θ = 30°."""
    n_lv, n_gv, n_l, theta = 3.0, 8.0, 0.01, math.radians(30.0)
    value = liquid_holdup(n_lv, n_gv, n_l, theta, stratified=False)
    squared = holdup_by_formula(C_UPHILL, n_lv, n_gv, n_l, math.sin(theta), power=2)
    cubed = holdup_by_formula(C_UPHILL, n_lv, n_gv, n_l, math.sin(theta), power=3)
    assert value == pytest.approx(squared, rel=1e-12)
    assert abs(squared - cubed) / squared > 1e-3


def test_liquid_holdup_in_unit_interval() -> None:
    for theta in (-90.0, -30.0, 0.0, 45.0, 90.0):
        for stratified_flag in (False, True):
            value = liquid_holdup(1.0, 5.0, 0.01, math.radians(theta), stratified=stratified_flag)
            assert 0.0 <= value <= 1.0


# --- Table 2 ---
TABLE_2 = [
    (0.01, 1.00),
    (0.20, 0.98),
    (0.30, 1.20),
    (0.40, 1.25),
    (0.50, 1.30),
    (0.70, 1.25),
    (1.00, 1.00),
    (10.00, 1.00),
]


@pytest.mark.parametrize(("h_r", "f_r"), TABLE_2)
def test_annular_friction_ratio_table_nodes(h_r: float, f_r: float) -> None:
    # reference: MB 1985, Table 2
    assert annular_friction_ratio(h_r) == pytest.approx(f_r, rel=1e-12)
    assert annular_friction_ratio(h_r, log_interpolation=True) == pytest.approx(f_r, rel=1e-12)


def test_annular_friction_ratio_linear_interpolation() -> None:
    assert annular_friction_ratio(0.25) == pytest.approx(1.09, rel=1e-12)  # (0.98 + 1.20)/2
    assert annular_friction_ratio(0.6) == pytest.approx(1.275, rel=1e-12)  # (1.30 + 1.25)/2


def test_annular_friction_ratio_log_interpolation() -> None:
    """Флаг annular_fr_log_interp: интерполяция по lg H_R (таблица снята с графика в лог-шкале)."""
    t = (math.log10(0.25) - math.log10(0.2)) / (math.log10(0.3) - math.log10(0.2))
    expected = 0.98 + t * (1.20 - 0.98)
    assert annular_friction_ratio(0.25, log_interpolation=True) == pytest.approx(
        expected, rel=1e-12
    )
    assert annular_friction_ratio(0.25, log_interpolation=True) != pytest.approx(1.09, rel=1e-3)


@pytest.mark.parametrize("h_r", [0.001, 0.0099, 10.5, 1.0e3, math.inf])
def test_annular_friction_ratio_outside_table_is_one(h_r: float) -> None:
    # reference: ТЗ: вне 0.01…10 → 1.0
    assert annular_friction_ratio(h_r) == 1.0
    assert annular_friction_ratio(h_r, log_interpolation=True) == 1.0


def test_annular_friction_ratio_flat_between_one_and_ten() -> None:
    assert annular_friction_ratio(3.0) == pytest.approx(1.0, rel=1e-12)


# --- ANNULAR ---
ANNULAR_POINTS = [
    (45.0, 0.01, 15.0),
    (0.0, 0.05, 20.0),
]  # H_R ≈ 0.539 и 0.484 (между узлами Table 2)


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), ANNULAR_POINTS)
def test_annular_gradient_definition(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    """ТЗ: H_R = λ/H_L, f_R из Table 2, f = f_R·f_ns; −dp/dL = [f·ρ_ns·v_m²/(2d) + ρ_s·g·sinθ]/(1 − Ek)."""
    state = aw_state(v_sl, v_sg, D, theta)
    result = MukherjeeBrill().calc(state)
    d = result.details
    assert result.pattern is FlowPattern.ANNULAR
    v_m = v_sl + v_sg
    assert d["h_r"] == pytest.approx(d["lambda_l"] / result.holdup, rel=1e-12)
    assert d["f_r"] == pytest.approx(annular_friction_ratio(d["h_r"]), rel=1e-12)
    assert result.dpdl_friction_pa_m == pytest.approx(
        d["f_r"] * d["f_ns"] * d["rho_ns_kg_m3"] * v_m**2 / (2.0 * D), rel=1e-12
    )
    assert result.dpdl_gravity_pa_m == pytest.approx(
        d["rho_s_kg_m3"] * 9.81 * math.sin(math.radians(theta)), rel=1e-12
    )
    base = result.dpdl_gravity_pa_m + result.dpdl_friction_pa_m
    assert result.dpdl_pa_m == pytest.approx(base / (1.0 - d["ek"]), rel=1e-12)


def test_annular_h_r_values(aw_state: Callable[..., FlowState]) -> None:
    # reference: расчёт агента: H_R = 0.5391 (45°, 0.01, 15) и 0.4836 (0°, 0.05, 20), rtol 1e-3
    assert MukherjeeBrill().calc(aw_state(0.01, 15.0, D, 45.0)).details["h_r"] == pytest.approx(
        0.5391, rel=1e-3
    )
    assert MukherjeeBrill().calc(aw_state(0.05, 20.0, D, 0.0)).details["h_r"] == pytest.approx(
        0.4836, rel=1e-3
    )


def test_annular_with_zero_holdup_is_finite(aw_state: Callable[..., FlowState]) -> None:
    """H_L underflow (спуск, малая жидкость): H_R = ∞ → f_R = 1, без деления на ноль."""
    result = MukherjeeBrill().calc(aw_state(0.001, 10.0, D, -45.0))
    assert result.pattern is FlowPattern.ANNULAR
    assert result.details["f_r"] == 1.0
    assert math.isfinite(result.dpdl_pa_m)


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), ANNULAR_POINTS)
def test_flag_annular_fr_log_interp(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    state = aw_state(v_sl, v_sg, D, theta)
    linear = MukherjeeBrill().calc(state)
    log = MukherjeeBrill(MukherjeeBrillOptions(annular_fr_log_interp=True)).calc(state)
    assert log.details["f_r"] != pytest.approx(linear.details["f_r"], rel=1e-6)
    assert log.dpdl_friction_pa_m != pytest.approx(linear.dpdl_friction_pa_m, rel=1e-6)


# --- STRATIFIED ---
STRATIFIED_POINTS = [
    (-10.0, 0.03162, 0.001),
    (-30.0, 0.05623, 0.001),
    (-45.0, 0.1, 0.1),
    (-90.0, 0.05623, 0.01778),
]


def test_equivalent_diameters_half_filled_pipe() -> None:
    # reference: MB (24)–(25) при δ = π (расчёт агента): D_EL = D_Eg = d·π/(π + 2)
    d_el, d_eg = equivalent_diameters(math.pi, 0.2)
    assert d_el == pytest.approx(0.2 * math.pi / (math.pi + 2.0), rel=1e-12)
    assert d_eg == pytest.approx(0.2 * math.pi / (math.pi + 2.0), rel=1e-12)


def test_equivalent_diameters_limits() -> None:
    """Почти пустая труба: D_EL → 0 (мало жидкости), D_Eg → d (газ занимает всё сечение)."""
    d_el, d_eg = equivalent_diameters(1e-3, 0.2)
    assert d_el < 1e-6
    assert d_eg == pytest.approx(0.2, rel=1e-3)


def expected_stratified(
    state: FlowState, *, combined: bool = False, published: bool = False
) -> dict[str, float]:
    """Независимая запись MB (10), (15), (17), (18)–(27) для сверки с моделью."""
    d, theta = state.d_m, math.radians(state.theta_deg)
    nums = dimensionless.duns_ros(
        state.v_sl_m_s, state.v_sg_m_s, state.rho_l_kg_m3, state.sigma_n_m, state.mu_l_pa_s
    )
    h_l = liquid_holdup(nums.n_lv, nums.n_gv, nums.n_l, theta, stratified=True)
    delta = stratified.delta_from_holdup(h_l)
    d_el = d * (delta - math.sin(delta)) / (delta + 2.0 * math.sin(delta / 2.0))
    d_eg = (
        d
        * (2.0 * math.pi - (delta - math.sin(delta)))
        / (2.0 * math.pi - delta + 2.0 * math.sin(delta / 2.0))
    )
    perimeter = math.pi * d
    p_g = (1.0 - delta / (2.0 * math.pi)) * perimeter
    p_l = perimeter - p_g
    v_l = state.v_sl_m_s / h_l
    v_g = state.v_sg_m_s / (1.0 - h_l)
    f_l = friction.moody(state.rho_l_kg_m3 * v_l * d_el / state.mu_l_pa_s, state.eps_m / d_el)
    f_g = friction.moody(state.rho_g_kg_m3 * v_g * d_eg / state.mu_g_pa_s, state.eps_m / d_eg)
    k = 0.5 if published else 0.125  # τ = f·ρ·v²/8 (Moody) | как напечатано f·ρ·v²/2
    tau_l = k * f_l * state.rho_l_kg_m3 * v_l**2
    tau_g = k * f_g * state.rho_g_kg_m3 * v_g**2
    area = math.pi * d**2 / 4.0
    area_g = (1.0 - h_l) * area
    if combined:  # (17): −A·dp/dL = τ_wL·P_L + τ_wg·P_g + (γ_L·A_L + γ_g·A_g)·sinθ
        fric = (tau_l * p_l + tau_g * p_g) / area
        rho_s = h_l * state.rho_l_kg_m3 + (1.0 - h_l) * state.rho_g_kg_m3
        grav = rho_s * 9.81 * math.sin(theta)
    else:  # (15) при τ_i = 0
        fric = tau_g * p_g / area_g
        grav = state.rho_g_kg_m3 * 9.81 * math.sin(theta)
    return {
        "h_l": h_l,
        "delta": delta,
        "fric": fric,
        "grav": grav,
        "tau_wg": tau_g,
        "tau_wl": tau_l,
    }


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), STRATIFIED_POINTS)
def test_stratified_gradient_by_equation_15(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    state = aw_state(v_sl, v_sg, D, theta)
    result = MukherjeeBrill().calc(state)
    expected = expected_stratified(state)
    assert result.pattern is FlowPattern.STRATIFIED
    assert result.holdup == pytest.approx(expected["h_l"], rel=1e-12)
    assert result.details["delta"] == pytest.approx(expected["delta"], rel=1e-9)
    assert result.dpdl_friction_pa_m == pytest.approx(expected["fric"], rel=1e-9)
    assert result.dpdl_gravity_pa_m == pytest.approx(expected["grav"], rel=1e-12)
    assert result.dpdl_acceleration_pa_m == 0.0  # ТЗ: ускорение = 0
    assert result.dpdl_pa_m == pytest.approx(expected["fric"] + expected["grav"], rel=1e-9)


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), STRATIFIED_POINTS)
def test_stratified_uses_gas_density_for_gravity_and_recovers_pressure(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    """(15): гравитация ρ_g·g·sinθ (жидкость в потоке без вклада), при спуске — восстановление давления."""
    state = aw_state(v_sl, v_sg, D, theta)
    result = MukherjeeBrill().calc(state)
    assert result.dpdl_gravity_pa_m == pytest.approx(
        state.rho_g_kg_m3 * 9.81 * math.sin(math.radians(theta)), rel=1e-12
    )
    assert result.dpdl_gravity_pa_m < 0.0


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), STRATIFIED_POINTS)
def test_stratified_holdup_geometry(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    result = MukherjeeBrill().calc(aw_state(v_sl, v_sg, D, theta))
    d = result.details
    assert (d["delta"] - math.sin(d["delta"])) / (2.0 * math.pi) == pytest.approx(
        result.holdup, rel=1e-9
    )
    assert d["h_l_over_d"] == pytest.approx(0.5 * (1.0 - math.cos(d["delta"] / 2.0)), rel=1e-12)


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), STRATIFIED_POINTS)
def test_flag_wall_shear_as_published_quadruples_friction(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    """ТЗ: τ_w = f·ρ·v²/2 (как напечатано) вместо f·ρ·v²/8 — трение ×4 при тех же f."""
    state = aw_state(v_sl, v_sg, D, theta)
    base = MukherjeeBrill().calc(state)
    published = MukherjeeBrill(MukherjeeBrillOptions(wall_shear_as_published=True)).calc(state)
    assert published.dpdl_friction_pa_m == pytest.approx(4.0 * base.dpdl_friction_pa_m, rel=1e-12)
    assert published.dpdl_friction_pa_m == pytest.approx(
        expected_stratified(state, published=True)["fric"], rel=1e-9
    )


@pytest.mark.parametrize(("theta", "v_sl", "v_sg"), STRATIFIED_POINTS)
def test_flag_stratified_combined_momentum_uses_equation_17(
    aw_state: Callable[..., FlowState], theta: float, v_sl: float, v_sg: float
) -> None:
    state = aw_state(v_sl, v_sg, D, theta)
    base = MukherjeeBrill().calc(state)
    combined = MukherjeeBrill(MukherjeeBrillOptions(stratified_combined_momentum=True)).calc(state)
    expected = expected_stratified(state, combined=True)
    assert combined.dpdl_friction_pa_m == pytest.approx(expected["fric"], rel=1e-9)
    assert combined.dpdl_gravity_pa_m == pytest.approx(expected["grav"], rel=1e-12)
    assert combined.dpdl_pa_m != pytest.approx(base.dpdl_pa_m, rel=1e-6)
    assert combined.dpdl_acceleration_pa_m == 0.0


def test_stratified_ignores_include_acceleration(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.05623, 0.001, D, -30.0)
    on = MukherjeeBrill().calc(state)
    off = MukherjeeBrill(EX_4_8).calc(state)
    assert on.dpdl_pa_m == off.dpdl_pa_m


def test_stratified_with_tiny_holdup_is_finite(aw_state: Callable[..., FlowState]) -> None:
    """H_L ~ 1e-15 (малая жидкость при спуске): δ ищется без потери точности на δ − sin δ."""
    result = MukherjeeBrill().calc(aw_state(0.001, 0.001, D, -10.0))
    assert result.pattern is FlowPattern.STRATIFIED
    assert 0.0 < result.holdup < 1e-6
    assert math.isfinite(result.dpdl_pa_m)


# --- downhill, не расслоённый поток ---
def test_downhill_bubble_uses_other_coefficient_set(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 0.001, D, -45.0)
    result = MukherjeeBrill().calc(state)
    assert result.pattern is FlowPattern.BUBBLE
    d = result.details
    assert result.holdup == pytest.approx(
        liquid_holdup(d["n_lv"], d["n_gv"], d["n_l"], math.radians(-45.0), stratified=False),
        rel=1e-12,
    )


def test_gravity_uses_slip_density_for_slug_in_downhill(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.46416, 10.0, D, -45.0)
    result = MukherjeeBrill(EX_4_8).calc(state)
    assert result.pattern is FlowPattern.SLUG
    rho_s = mixture.slip(
        result.holdup, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s
    ).rho_s_kg_m3
    assert result.dpdl_gravity_pa_m == pytest.approx(rho_s * 9.81 * math.sin(math.radians(-45.0)))


def test_holdup_between_zero_and_one_on_grid(aw_state: Callable[..., FlowState]) -> None:
    for theta in (-90.0, -45.0, -10.0, 0.0, 10.0, 45.0, 90.0):
        for v_sl in (0.001, 0.05, 1.0):
            for v_sg in (0.001, 0.1, 5.0, 20.0):
                result = MukherjeeBrill().calc(aw_state(v_sl, v_sg, D, theta))
                assert 0.0 <= result.holdup <= 1.0
