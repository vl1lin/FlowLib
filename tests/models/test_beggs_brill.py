"""Тесты `models/beggs_brill.py` — корреляция Beggs–Brill (ICorrelation).

Ожидаемый API (`from correlations.models.beggs_brill import ...`):
    BeggsBrillOptions(revised_map=False, payne_correction=False, rough_pipe_friction=False,
                      clip_negative_c=False, include_acceleration=True)
    BeggsBrill(options=None, flow_model=None); .name == "beggs_brill"; .options; .flow_model;
        .calc(state) -> CorrelationResult
        flow_model по умолчанию — `BeggsBrillMap(BeggsBrillMapOptions(revised_map=options.revised_map))`.
    horizontal_holdup(lam, n_fr, pattern) -> float            H_L(0) = A·λ^α/N_FR^|β|, Table 1, ≥ λ  (ур. 19–21)
    inclination_coefficient(lam, n_lv, n_fr, pattern, *, uphill) -> float   C без обрезки (ур. 22–25)
    inclination_factor(c, theta_deg) -> float                  ψ = 1 + C[sin φ − sin³φ/3], φ = 1.8·θ°  (ур. 5)
    friction_ratio_exponent(y) -> float                        S, ур. 16
details (двухфазный, не TRANSITION): lambda_l, n_fr, n_lv, rho_ns_kg_m3, mu_ns_pa_s, re_ns, f_ns,
    h_l_0, c, psi, y, s, f_tp, ek, rho_s_kg_m3; для TRANSITION вместо h_l_0/c/psi:
    transition_a, h_l_segregated, h_l_intermittent.
Источник: Beggs H.D., Brill J.P., 1973, JPT 25(5), ур. 1–25, Table 1; Payne G.A. et al., 1979, JPT 31(9);
revised-карта — монография (4.110–4.119). Таблица 4.2 монографии: a = 0.845 (напечатано 0.855) —
исправление по ТЗ, «Опечатки».
Допущения (вне ТЗ, помечены в коде TODO): при Re_ns < 2000 f_ns = 64/Re в обоих режимах трения;
Payne: множитель 0.924 при θ > 0 с нижней границей λ, 0.685 при θ < 0 без границы.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core import dimensionless, mixture
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.beggs_brill_map import BeggsBrillMap
from correlations.models.beggs_brill import (
    BeggsBrill,
    BeggsBrillOptions,
    friction_ratio_exponent,
    horizontal_holdup,
    inclination_coefficient,
    inclination_factor,
)

# reference: ТЗ, «Сквозные эталоны» BB, флаги Примера 4.7
EX_4_7 = BeggsBrillOptions(
    payne_correction=True,
    rough_pipe_friction=True,
    clip_negative_c=True,
    include_acceleration=False,
)
SEG, INT, DIST = FlowPattern.SEGREGATED, FlowPattern.INTERMITTENT, FlowPattern.DISTRIBUTED


def test_default_flags_are_original_behaviour() -> None:
    # reference: ТЗ, флаги BB: умолчание = оригинал статьи (include_acceleration = True)
    options = BeggsBrill().options
    assert options.revised_map is False
    assert options.payne_correction is False
    assert options.rough_pipe_friction is False
    assert options.clip_negative_c is False
    assert options.include_acceleration is True


# --- элементные эталоны R1 (ТЗ), rtol 1e-3 ---
def test_r1_elements_default_flags(r1_kwargs: dict[str, Any]) -> None:
    d = BeggsBrill().calc(FlowState(**r1_kwargs)).details
    assert d["n_fr"] == pytest.approx(3.808, rel=1e-3)
    assert d["n_lv"] == pytest.approx(11.86, rel=1e-3)
    assert d["re_ns"] == pytest.approx(3.152e5, rel=1e-3)
    assert d["f_ns"] == pytest.approx(0.01434, rel=1e-3)  # гладкая (13)
    assert d["h_l_0"] == pytest.approx(0.5741, rel=1e-3)
    assert d["c"] == pytest.approx(-0.0481, rel=1e-3)
    assert d["psi"] == pytest.approx(0.9856, rel=1e-3)


def test_r1_rough_pipe_friction_uses_colebrook(r1_kwargs: dict[str, Any]) -> None:
    options = BeggsBrillOptions(rough_pipe_friction=True)
    d = BeggsBrill(options).calc(FlowState(**r1_kwargs)).details
    assert d["f_ns"] == pytest.approx(0.01551, rel=1e-3)  # reference: ТЗ, f_ns Colebrook


def test_r1_default_end_to_end(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Сквозные эталоны», BB R1 по умолчанию: INTERMITTENT, H_L = 0.5659, 4804 Па/м (rtol 0.1%)
    result = BeggsBrill().calc(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.INTERMITTENT
    assert result.holdup == pytest.approx(0.5659, rel=1e-3)
    assert result.dpdl_pa_m == pytest.approx(4804, rel=1e-3)


def test_r1_default_components(r1_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R1, умолчание): гравитация 4634.8, трение 168.76, Ek = 1.13e-4
    result = BeggsBrill().calc(FlowState(**r1_kwargs))
    assert result.dpdl_gravity_pa_m == pytest.approx(4634.8, rel=1e-3)
    assert result.dpdl_friction_pa_m == pytest.approx(168.76, rel=1e-3)
    assert result.details["ek"] == pytest.approx(1.1318e-4, rel=1e-3)
    base = result.dpdl_gravity_pa_m + result.dpdl_friction_pa_m
    assert result.dpdl_pa_m == pytest.approx(base / (1.0 - result.details["ek"]), rel=1e-12)
    assert result.dpdl_acceleration_pa_m == pytest.approx(result.dpdl_pa_m - base, rel=1e-12)


def test_example_4_7_end_to_end(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Сквозные эталоны», BB R1 с флагами Примера 4.7: INTERMITTENT, H_L = 0.5305,
    # 4588 Па/м (книга 4580), rtol 1%; элементы: y / S / f_tp = 1.802 / 0.3871 / 0.02284 (rtol 1e-3)
    result = BeggsBrill(EX_4_7).calc(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.INTERMITTENT
    assert result.holdup == pytest.approx(0.5305, rel=1e-3)
    assert result.dpdl_pa_m == pytest.approx(4588, rel=1e-2)
    assert result.details["y"] == pytest.approx(1.802, rel=1e-3)
    assert result.details["s"] == pytest.approx(0.3871, rel=1e-3)
    assert result.details["f_tp"] == pytest.approx(0.02284, rel=1e-3)


def test_example_4_7_has_no_acceleration(r1_kwargs: dict[str, Any]) -> None:
    result = BeggsBrill(EX_4_7).calc(FlowState(**r1_kwargs))
    assert result.dpdl_acceleration_pa_m == 0.0
    assert result.dpdl_pa_m == pytest.approx(
        result.dpdl_gravity_pa_m + result.dpdl_friction_pa_m, rel=1e-12
    )


def test_gravity_and_friction_definitions(r1_kwargs: dict[str, Any]) -> None:
    """Гравитация = ρ_s·g·sinθ; трение = f_tp·ρ_ns·v_m²/(2d) (ур. 1)."""
    state = FlowState(**r1_kwargs)
    result = BeggsBrill().calc(state)
    d = result.details
    rho_s = mixture.slip(
        result.holdup, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s
    ).rho_s_kg_m3
    v_m = state.v_sl_m_s + state.v_sg_m_s
    assert d["rho_s_kg_m3"] == pytest.approx(rho_s, rel=1e-12)
    assert result.dpdl_gravity_pa_m == pytest.approx(rho_s * 9.81, rel=1e-12)
    assert result.dpdl_friction_pa_m == pytest.approx(
        d["f_tp"] * d["rho_ns_kg_m3"] * v_m**2 / (2.0 * state.d_m), rel=1e-12
    )


# --- Table 1: горизонтальный holdup ---
def test_horizontal_holdup_intermittent_uses_a_0_845() -> None:
    """ТЗ, «Опечатки»: Table 4.2 монографии a = 0.855 неверно, BB Table 1 — 0.845 (Пример 4.7)."""
    value = horizontal_holdup(0.50712, 3.808, INT)
    assert value == pytest.approx(0.5741, rel=1e-3)
    assert value != pytest.approx(0.5741 * 0.855 / 0.845, rel=5e-3)


def test_horizontal_holdup_segregated_and_distributed() -> None:
    # reference: BB Table 1, ур. 19 и 21 (расчёт агента)
    assert horizontal_holdup(0.1, 0.3, SEG) == pytest.approx(
        0.98 * 0.1**0.4846 / 0.3**0.0868, rel=1e-12
    )
    assert horizontal_holdup(0.3, 100.0, DIST) == pytest.approx(
        1.065 * 0.3**0.5824 / 100.0**0.0609, rel=1e-12
    )


def test_horizontal_holdup_not_below_no_slip_holdup() -> None:
    # reference: BB ур. 11: H_L(0) ≥ λ; λ = 0.5, N_FR = 1e4: 1.065·0.5^0.5824/1e4^0.0609 = 0.4064 < λ
    assert horizontal_holdup(0.5, 1.0e4, DIST) == 0.5


# --- Table 1: коэффициент наклона C ---
def test_inclination_coefficient_formulas() -> None:
    lam, n_lv, n_fr = 0.2, 3.0, 0.8
    # reference: BB Table 1, ур. 22, 23, 25 (расчёт агента)
    seg_up = (1 - lam) * math.log(0.011 * n_lv**3.539 / (lam**3.768 * n_fr**1.614))
    int_up = (1 - lam) * math.log(2.96 * lam**0.305 * n_fr**0.0978 / n_lv**0.4473)
    down = (1 - lam) * math.log(4.7 * n_lv**0.1244 / (lam**0.3692 * n_fr**0.5056))
    assert inclination_coefficient(lam, n_lv, n_fr, SEG, uphill=True) == pytest.approx(seg_up)
    assert inclination_coefficient(lam, n_lv, n_fr, INT, uphill=True) == pytest.approx(int_up)
    for pattern in (SEG, INT, DIST):
        assert inclination_coefficient(lam, n_lv, n_fr, pattern, uphill=False) == pytest.approx(
            down
        )


def test_distributed_uphill_has_zero_c() -> None:
    # reference: BB ур. 24 — для восходящего распределённого потока C = 0
    assert inclination_coefficient(0.3, 5.0, 100.0, DIST, uphill=True) == 0.0


def test_inclination_coefficient_r1_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны»: C = −0.0481 для R1 (точные λ, N_Lv, N_FR из R1)
    v_sl, v_sg = r1_kwargs["v_sl_m_s"], r1_kwargs["v_sg_m_s"]
    nums = dimensionless.duns_ros(
        v_sl, v_sg, r1_kwargs["rho_l_kg_m3"], r1_kwargs["sigma_n_m"], r1_kwargs["mu_l_pa_s"]
    )
    n_fr = dimensionless.froude_number(v_sl + v_sg, r1_kwargs["d_m"])
    c = inclination_coefficient(v_sl / (v_sl + v_sg), nums.n_lv, n_fr, INT, uphill=True)
    assert c == pytest.approx(-0.0481, rel=1e-3)


# --- ψ ---
def test_inclination_factor_is_one_for_horizontal_pipe() -> None:
    assert inclination_factor(0.7, 0.0) == 1.0


def test_inclination_factor_r1_matches_tz() -> None:
    # reference: ТЗ: ψ при θ = 90°, без clip = 0.9856 (φ = 1.8·90° = 162°)
    assert inclination_factor(-0.0481, 90.0) == pytest.approx(0.9856, rel=1e-3)


def test_inclination_factor_maximum_at_50_degrees() -> None:
    """BB ур. 5–6: ψ_max = 1 + 2C/3 при φ = 90° (θ = 50°); C = (ψ_max − 1)·1.5."""
    assert inclination_factor(0.3, 50.0) == pytest.approx(1.2, rel=1e-12)
    assert inclination_factor(0.3, -50.0) == pytest.approx(0.8, rel=1e-12)


# --- S (16) ---
def test_friction_ratio_exponent_y_equals_one() -> None:
    # reference: ТЗ, «Контрольные точки», BB (16): y = 1 → f_tp/f_ns = 1 (S = 0), rtol 1e-12
    assert friction_ratio_exponent(1.0) == pytest.approx(0.0, abs=1e-12)
    assert math.exp(friction_ratio_exponent(1.0)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("y", [1.0001, 1.1, 1.19])
def test_friction_ratio_exponent_special_interval(y: float) -> None:
    # reference: BB ур. 16: при 1 < y < 1.2 S = ln(2.2·y − 1.2)
    assert friction_ratio_exponent(y) == pytest.approx(math.log(2.2 * y - 1.2), rel=1e-12)


def test_friction_ratio_exponent_r1_matches_tz() -> None:
    assert friction_ratio_exponent(1.802) == pytest.approx(0.3871, rel=1e-3)


def test_friction_ratio_exponent_general_form() -> None:
    # reference: BB ур. 16 (расчёт агента) для y = 0.5 и y = 3
    for y in (0.5, 3.0):
        ln = math.log(y)
        expected = ln / (-0.0523 + 3.182 * ln - 0.8725 * ln**2 + 0.01853 * ln**4)
        assert friction_ratio_exponent(y) == pytest.approx(expected, rel=1e-12)


def test_friction_ratio_exponent_is_nearly_continuous_at_1_2() -> None:
    assert friction_ratio_exponent(1.2 - 1e-9) == pytest.approx(
        friction_ratio_exponent(1.2 + 1e-9), rel=5e-3
    )


def test_friction_ratio_exponent_vanishes_for_huge_y() -> None:
    # Замечание: ТЗ ожидает S → 0 точно (1e-12) при λ → 0; при y → ∞ S убывает как 1/ln³y (≈1.6e-7 при 1e300)
    assert abs(friction_ratio_exponent(1.0e300)) < 1.0e-6


# --- флаги ---
def test_flag_clip_negative_c(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    plain = BeggsBrill().calc(state).details
    clipped = BeggsBrill(BeggsBrillOptions(clip_negative_c=True)).calc(state).details
    assert plain["c"] < 0.0
    assert clipped["c"] == 0.0
    assert clipped["psi"] == 1.0
    assert clipped["h_l_0"] == pytest.approx(plain["h_l_0"], rel=1e-12)


def test_flag_payne_uphill_multiplier(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, флаг payne_correction: H_L × 0.924 при θ > 0 (не ниже λ)
    state = FlowState(**r1_kwargs)
    plain = BeggsBrill().calc(state).holdup
    payne = BeggsBrill(BeggsBrillOptions(payne_correction=True)).calc(state).holdup
    assert payne == pytest.approx(0.924 * plain, rel=1e-12)


def test_flag_payne_uphill_not_below_no_slip_holdup(aw_state: Callable[..., FlowState]) -> None:
    """λ = 0.5, N_FR = 1e4 (распределённый): H_L = λ, 0.924·λ < λ → остаётся λ."""
    v_m = math.sqrt(1.0e4 * 9.81 * 0.1524)
    state = aw_state(v_m / 2.0, v_m / 2.0, 0.1524, 30.0)
    # v_m ≈ 121 м/с при 1 бар даёт Ek ≥ 1 — ускорение выключено (иначе ConvergenceError по существу)
    options = BeggsBrillOptions(payne_correction=True, include_acceleration=False)
    result = BeggsBrill(options).calc(state)
    assert result.pattern is FlowPattern.DISTRIBUTED
    assert result.holdup == pytest.approx(0.5, rel=1e-12)


def test_flag_payne_downhill_multiplier(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, флаг payne_correction: × 0.685 при θ < 0
    state = FlowState(**{**r1_kwargs, "theta_deg": -30.0})
    plain = BeggsBrill().calc(state).holdup
    payne = BeggsBrill(BeggsBrillOptions(payne_correction=True)).calc(state).holdup
    assert payne == pytest.approx(0.685 * plain, rel=1e-12)


def test_flag_include_acceleration(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    with_acc = BeggsBrill().calc(state)
    without = BeggsBrill(BeggsBrillOptions(include_acceleration=False)).calc(state)
    assert with_acc.dpdl_acceleration_pa_m > 0.0
    assert without.dpdl_acceleration_pa_m == 0.0
    assert with_acc.dpdl_pa_m > without.dpdl_pa_m


def test_flag_rough_pipe_friction_changes_gradient(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    smooth = BeggsBrill().calc(state).dpdl_friction_pa_m
    rough = BeggsBrill(BeggsBrillOptions(rough_pipe_friction=True)).calc(state).dpdl_friction_pa_m
    assert rough > smooth


def test_laminar_no_slip_friction_is_64_over_re(aw_state: Callable[..., FlowState]) -> None:
    """Допущение (вне ТЗ): при Re_ns < 2000 f_ns = 64/Re независимо от флага шероховатости."""
    state = aw_state(0.005, 0.005, 0.01, 0.0, mu_l_pa_s=0.5)
    for rough in (False, True):
        d = BeggsBrill(BeggsBrillOptions(rough_pipe_friction=rough)).calc(state).details
        assert d["re_ns"] < 2000.0
        assert d["f_ns"] * d["re_ns"] == pytest.approx(64.0, rel=1e-12)


# --- revised_map и TRANSITION ---
def test_revised_map_flag_builds_revised_default_flow_model() -> None:
    model = BeggsBrill(BeggsBrillOptions(revised_map=True))
    assert isinstance(model.flow_model, BeggsBrillMap)
    assert model.flow_model.options.revised_map is True
    assert BeggsBrill().flow_model.options.revised_map is False  # type: ignore[attr-defined]


@pytest.mark.parametrize("theta", [0.0, 30.0, -30.0])
def test_transition_zone_interpolates_holdup(
    aw_state: Callable[..., FlowState], theta: float
) -> None:
    """Монография 4.118–4.119: H_L = A·H_seg + (1 − A)·H_int, A = (L3 − N_FR)/(L3 − L2)."""
    v = 0.19333  # λ = 0.5, N_FR = 0.1 (d = 0.1524) → TRANSITION при revised_map
    state = aw_state(v, v, 0.1524, theta)
    model = BeggsBrill(BeggsBrillOptions(revised_map=True))
    result = model.calc(state)
    assert result.pattern is FlowPattern.TRANSITION
    flow = BeggsBrillMap(model.flow_model.options).predict(state).details  # type: ignore[attr-defined]
    a = (flow["l3_froude"] - flow["n_fr"]) / (flow["l3_froude"] - flow["l2_froude"])
    assert result.details["transition_a"] == pytest.approx(a, rel=1e-12)

    def holdup(pattern: FlowPattern) -> float:
        h0 = horizontal_holdup(flow["lambda_l"], flow["n_fr"], pattern)
        c = inclination_coefficient(
            flow["lambda_l"], result.details["n_lv"], flow["n_fr"], pattern, uphill=theta > 0
        )
        h = h0 * inclination_factor(c if theta != 0.0 else 0.0, theta)
        return min(max(h, 0.0), 1.0)  # BB ур. 11: 0 ≤ H_L(θ) ≤ 1 для каждого режима до интерполяции

    expected = a * holdup(SEG) + (1.0 - a) * holdup(INT)
    assert result.holdup == pytest.approx(expected, rel=1e-12)
    assert result.details["h_l_segregated"] == pytest.approx(holdup(SEG), rel=1e-12)
    assert result.details["h_l_intermittent"] == pytest.approx(holdup(INT), rel=1e-12)


def test_original_map_has_no_transition_zone(aw_state: Callable[..., FlowState]) -> None:
    v = 0.19333
    assert BeggsBrill().calc(aw_state(v, v, 0.1524, 0.0)).pattern is FlowPattern.SEGREGATED


# --- углы и знаки ---
def test_horizontal_pipe_has_no_gravity_component(aw_state: Callable[..., FlowState]) -> None:
    result = BeggsBrill().calc(aw_state(0.5, 2.0, 0.1, 0.0))
    assert result.dpdl_gravity_pa_m == 0.0
    assert result.details["psi"] == 1.0


def test_downhill_flow_recovers_pressure(aw_state: Callable[..., FlowState]) -> None:
    result = BeggsBrill().calc(aw_state(0.5, 2.0, 0.1, -30.0))
    assert result.dpdl_gravity_pa_m < 0.0


def test_gradient_grows_with_gas_for_uphill_vertical_to_horizontal(
    aw_state: Callable[..., FlowState],
) -> None:
    vertical = BeggsBrill().calc(aw_state(0.5, 2.0, 0.1, 90.0)).dpdl_pa_m
    horizontal = BeggsBrill().calc(aw_state(0.5, 2.0, 0.1, 0.0)).dpdl_pa_m
    assert vertical > horizontal
