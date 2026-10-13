"""Тесты `models/ansari.py` — корреляция Ansari et al. 1994 (ICorrelation).

Ожидаемый API (`from correlations.models.ansari import ...`):
    AnsariOptions(developing_slug=True, vo_shoham_a_as_published=False, bridging_with_delta_min=True,
                  bubble_slug_sin_theta=False, ff_over_fsl_unity=False, eq56_as_published=False)
    Ansari(options=None, flow_model=None); .name == "ansari"; .options; .flow_model; .calc(state)
        flow_model по умолчанию — `AnsariFlowModel(AnsariFlowOptions(...))` с тремя флагами режима.
    SlugUnit (frozen dataclass): v_tb, h_gls, h_lls, v_gls, a_tilde, h_ltb, v_ltb, v_lls, v_gtb, beta, l_ls_m, l_tb_m
    solve_slug_unit(state, *, vo_shoham_a_as_published=False) -> SlugUnit            ур. (29)–(43)
    cap_length(state, unit) -> float                                                  L_c, ур. (44)–(47)
    DevelopingSlug (frozen dataclass): l_tb_star_m, beta_star, h_ltba
    developing_slug_unit(state, unit, *, eq56_as_published=False) -> DevelopingSlug | None   ур. (50)–(57), (64)
    bubble_holdup(state) -> float                                                     ур. (22): H_L пузырькового режима
Схема (ТЗ, «Ansari et al. (1994)»):
    BUBBLE: H_L из 1.53·[gσΔρ/ρ_L²]^0.25·H_L^0.5 = v_sg/(1 − H_L) − 1.2·v_m (22); ρ_TP, μ_TP по H_L (23)–(24);
        −dp/dL = ρ_TP·g·sinθ + f·ρ_TP·v_m²/(2d), f — Moody по Re = ρ_TP·v_m·d/μ_TP (v_TP = v_m, (18)); ускорение 0.
    DISPERSED_BUBBLE: то же при H_L = λ_L (no-slip).
    SLUG: ур. (29)–(43), L_LS = 30·d; L_c ≥ L_TB → развивающаяся пробка (флаг developing_slug); градиент:
        (60) или (62) + (65), μ_LS, ρ_LS (61); H_L = (1 − β)·H_LLS + β·H_LTB (или H_LTBA с β*); ускорение 0.
    ANNULAR: плёнка (96), φ_C² = Z/(1 − 2δ)^5 (99), −dp/dL = φ_C²·(dp/dL)_SC + ρ_C·g·sinθ (102);
        H_L = 4δ(1 − δ) + λ_LC·(1 − 2δ)² ; ускорение 0.
details: BUBBLE/DB: h_l, rho_tp_kg_m3, mu_tp_pa_s, re_tp, f_darcy;
    SLUG: v_tb_m_s, h_gls, h_lls, v_gls_m_s, a_tilde_m_s, h_ltb, v_ltb_m_s, v_lls_m_s, v_gtb_m_s, beta, l_tb_m, l_c_m,
        developing (0.0/1.0), rho_ls_kg_m3, mu_ls_pa_s, re_ls, f_ls; при развивающейся: l_tb_star_m, beta_star, h_ltba;
    ANNULAR: f_e, lam_lc, rho_c_kg_m3, delta, x_m_sq, y_m, z, phi_c_sq, dp_sc_pa_m.
Источник: Ansari A.M. et al., 1994, SPE Production & Facilities 9(2), 145–152, ур. (1)–(102).
Опечатки (ТЗ): (18) v_TP = v_m; (32) v_TB; (40) Ā = H_gLS·(v_TB − v_gLS) + v_m (буквальная форма — флаг
vo_shoham_a_as_published); (85) (ρ_L/ρ_g)^(1/3).
Расхождение сверх таблицы «Опечатки» (в отчёт): ур. (56) в напечатанном виде b = [v_sg − v_gLS·(2 − H_LLS)]/v_TB·L_LS
для R2 даёт комплексные корни (54); из (52)–(53) следует (1 − H_LLS) — принято как вывод из (52)–(53), корень (54) —
больший положительный (Q3 ТЗ).
Допущение Q3 ТЗ: v_NLTB в (45) — из баланса (32) с H_NLTB (46).
"""

import math
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from correlations.core import annular, bubbles, friction, mixture
from correlations.core.errors import UnsupportedPatternError
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models.ansari_fp import AnsariFlowModel, solve_annular_film
from correlations.models.ansari import (
    Ansari,
    AnsariOptions,
    bubble_holdup,
    cap_length,
    developing_slug_unit,
    solve_slug_unit,
)

G = 9.81
# reference: ТЗ, «Сквозные эталоны», Ansari R2: developing_slug = False, ff_over_fsl_unity = True
EX_4_9 = AnsariOptions(developing_slug=False, ff_over_fsl_unity=True)


class FixedFlowModel:
    name: ClassVar[str] = "fixed"

    def __init__(self, pattern: FlowPattern) -> None:
        self._pattern = pattern

    @property
    def options(self) -> BaseOptions:
        return BaseOptions()

    def predict(self, state: FlowState) -> FlowPatternResult:
        return FlowPatternResult(pattern=self._pattern, details={})


def test_default_flags() -> None:
    options = Ansari().options
    assert options.developing_slug is True
    assert options.vo_shoham_a_as_published is False
    assert options.bridging_with_delta_min is True
    assert options.bubble_slug_sin_theta is False
    assert options.ff_over_fsl_unity is False
    # reference: решение пользователя по (56): по умолчанию (1 − H_LLS), напечатанное (2 − H_LLS) — флагом
    assert options.eq56_as_published is False


def test_default_flow_model_gets_regime_flags() -> None:
    model = Ansari(AnsariOptions(bridging_with_delta_min=False, bubble_slug_sin_theta=True))
    assert isinstance(model.flow_model, AnsariFlowModel)
    assert model.flow_model.options.bridging_with_delta_min is False  # type: ignore[attr-defined]
    assert model.flow_model.options.bubble_slug_sin_theta is True  # type: ignore[attr-defined]


# --- пробковая модель: эталоны R2 (ТЗ) ---
def test_slug_unit_velocities_and_a(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», Ansari: v_TB / H_gLS / v_gLS / Ā = 3.2578 / 0.17417 / 2.9944 / 2.4269 (rtol 1e-4)
    unit = solve_slug_unit(FlowState(**r2_kwargs))
    assert unit.v_tb == pytest.approx(3.2578, rel=1e-4)
    assert unit.h_gls == pytest.approx(0.17417, rel=1e-4)
    assert unit.v_gls == pytest.approx(2.9944, rel=1e-4)
    assert unit.a_tilde == pytest.approx(2.4269, rel=1e-4)


def test_slug_unit_taylor_bubble(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ: H_LTB / v_LTB / v_LLS / v_gTB = 0.12983 / 3.1424 / 2.2516 / 3.2051 (rtol 1e-3)
    unit = solve_slug_unit(FlowState(**r2_kwargs))
    assert unit.h_ltb == pytest.approx(0.12983, rel=1e-3)
    assert unit.v_ltb == pytest.approx(3.1424, rel=1e-3)
    assert unit.v_lls == pytest.approx(2.2516, rel=1e-3)
    assert unit.v_gtb == pytest.approx(3.2051, rel=1e-3)


def test_slug_unit_lengths(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ: β / L_TB / L_c = 0.2873 / 1.843 м / 3.17 м (rtol 1e-3 / 1e-3 / 5e-2); L_LS = 30·d
    state = FlowState(**r2_kwargs)
    unit = solve_slug_unit(state)
    assert unit.beta == pytest.approx(0.2873, rel=1e-3)
    assert unit.l_ls_m == pytest.approx(30.0 * state.d_m, rel=1e-12)
    assert unit.l_tb_m == pytest.approx(1.843, rel=1e-3)
    assert cap_length(state, unit) == pytest.approx(3.17, rel=5e-2)


def test_slug_unit_vo_shoham_a_as_published(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ: H_LTB при vo_shoham_a_as_published = 0.1188 (вместо 0.1298), rtol 1e-3
    unit = solve_slug_unit(FlowState(**r2_kwargs), vo_shoham_a_as_published=True)
    assert unit.h_ltb == pytest.approx(0.1188, rel=1e-3)


def test_slug_unit_satisfies_mass_balances(r2_kwargs: dict[str, Any]) -> None:
    """(29), (30), (32), (33): баланс газа и жидкости по слагу и по пузырю Тейлора."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    h_gtb = 1.0 - u.h_ltb
    assert u.beta * u.v_gtb * h_gtb + (1.0 - u.beta) * u.v_gls * u.h_gls == pytest.approx(
        state.v_sg_m_s, rel=1e-9
    )  # (29)
    assert (1.0 - u.beta) * u.v_lls * u.h_lls - u.beta * u.v_ltb * u.h_ltb == pytest.approx(
        state.v_sl_m_s, rel=1e-9
    )  # (30)
    assert (u.v_tb - u.v_lls) * u.h_lls == pytest.approx(
        (u.v_tb + u.v_ltb) * u.h_ltb, rel=1e-9
    )  # (32)
    assert (u.v_tb - u.v_gls) * u.h_gls == pytest.approx(
        (u.v_tb - u.v_gtb) * h_gtb, rel=1e-9
    )  # (33)


def test_slug_unit_satisfies_equation_39(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    f = (
        9.916 * math.sqrt(G * state.d_m) * (1.0 - math.sqrt(1.0 - u.h_ltb)) ** 0.5 * u.h_ltb
        - u.v_tb * (1.0 - u.h_ltb)
        + u.a_tilde
    )
    assert f == pytest.approx(0.0, abs=1e-8)
    # (37): v_LTB = 9.916·[g·d·(1 − √H_gTB)]^0.5
    assert u.v_ltb == pytest.approx(
        9.916 * math.sqrt(G * state.d_m * (1.0 - math.sqrt(1.0 - u.h_ltb))), rel=1e-9
    )


def test_slug_unit_taylor_velocity_and_void_in_slug(r2_kwargs: dict[str, Any]) -> None:
    """(34): v_TB = 1.2·v_m + 0.35·√(g·d·Δρ/ρ_L); (38): H_gLS = v_sg/(0.425 + 2.65·v_m)."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    v_m = state.v_sl_m_s + state.v_sg_m_s
    drho = state.rho_l_kg_m3 - state.rho_g_kg_m3
    assert u.v_tb == pytest.approx(
        1.2 * v_m + 0.35 * math.sqrt(G * state.d_m * drho / state.rho_l_kg_m3), rel=1e-12
    )
    assert u.h_gls == pytest.approx(state.v_sg_m_s / (0.425 + 2.65 * v_m), rel=1e-12)
    assert u.h_lls == pytest.approx(1.0 - u.h_gls, rel=1e-12)


def test_slug_unit_a_equals_hgls_times_vtb_minus_vgls_plus_vm(r2_kwargs: dict[str, Any]) -> None:
    """ТЗ, «Опечатки», Ansari (40): Ā = H_gLS·(v_TB − v_gLS) + v_m; буквальная форма — флагом."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    v_m = state.v_sl_m_s + state.v_sg_m_s
    assert u.a_tilde == pytest.approx(u.h_gls * (u.v_tb - u.v_gls) + v_m, rel=1e-12)
    literal = solve_slug_unit(state, vo_shoham_a_as_published=True)
    assert literal.a_tilde != pytest.approx(u.a_tilde, rel=1e-3)


def test_slug_unit_unique_root_on_unit_interval(aw_state: Callable[..., FlowState]) -> None:
    for v_sl, v_sg in ((0.3, 0.3), (1.0, 2.0), (0.05, 1.0)):
        unit = solve_slug_unit(aw_state(v_sl, v_sg, 0.0508, 90.0))
        assert 0.0 < unit.h_ltb < 1.0
        assert 0.0 < unit.beta < 1.0


# --- развивающаяся пробка ---
def test_r2_is_developing_slug(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, главные находки: для R2 L_c = 3.17 м > L_TB = 1.84 м — пробка «развивающаяся»
    state = FlowState(**r2_kwargs)
    unit = solve_slug_unit(state)
    assert cap_length(state, unit) >= unit.l_tb_m


def test_equation_56_as_printed_has_no_real_root_for_r2(r2_kwargs: dict[str, Any]) -> None:
    """Расхождение с источником (в отчёт): напечатанное b с (2 − H_LLS) даёт для R2 дискриминант (54) < 0."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    a = 1.0 - state.v_sg_m_s / u.v_tb
    c = (u.v_tb - u.v_lls) / math.sqrt(2.0 * G) * u.h_lls
    b_printed = (state.v_sg_m_s - u.v_gls * (2.0 - u.h_lls)) / u.v_tb * u.l_ls_m
    p = (-2.0 * a * b_printed - 4.0 * c**2) / a**2
    q = b_printed**2 / a**2
    assert p**2 - 4.0 * q < 0.0


def test_developing_slug_larger_root_of_equation_54(r2_kwargs: dict[str, Any]) -> None:
    """(52)–(53): a·L − 2c·√L − b = 0, b = [v_sg − v_gLS(1 − H_LLS)]/v_TB·L_LS; (54) — после возведения в квадрат,
    поэтому берётся больший корень (меньший — посторонний)."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    dev = developing_slug_unit(state, u)
    assert dev is not None
    a = 1.0 - state.v_sg_m_s / u.v_tb  # (55)
    b = (state.v_sg_m_s - u.v_gls * (1.0 - u.h_lls)) / u.v_tb * u.l_ls_m
    c = (u.v_tb - u.v_lls) / math.sqrt(2.0 * G) * u.h_lls  # (57)
    assert a * dev.l_tb_star_m - 2.0 * c * math.sqrt(dev.l_tb_star_m) - b == pytest.approx(
        0.0, abs=1e-9
    )
    p, q = (-2.0 * a * b - 4.0 * c**2) / a**2, b**2 / a**2  # (54)
    roots = [(-p + s * math.sqrt(p**2 - 4.0 * q)) / 2.0 for s in (1.0, -1.0)]
    assert dev.l_tb_star_m == pytest.approx(max(roots), rel=1e-12)


def test_developing_slug_values(r2_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R2; (56) с (1 − H_LLS)): L_TB* = 2.322 м, β* = 0.3368, H_LTBA = 0.2462
    state = FlowState(**r2_kwargs)
    dev = developing_slug_unit(state, solve_slug_unit(state))
    assert dev is not None
    assert dev.l_tb_star_m == pytest.approx(2.322, rel=1e-3)
    assert dev.beta_star == pytest.approx(0.3368, rel=1e-3)
    assert dev.h_ltba == pytest.approx(0.2462, rel=1e-3)


def test_developing_slug_definitions(r2_kwargs: dict[str, Any]) -> None:
    """β* = L_TB*/(L_TB* + L_LS); (64): H_LTBA = 2·(v_TB − v_LLS)·H_LLS/√(2g·L_TB*)."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    dev = developing_slug_unit(state, u)
    assert dev is not None
    assert dev.beta_star == pytest.approx(dev.l_tb_star_m / (dev.l_tb_star_m + u.l_ls_m), rel=1e-12)
    assert dev.h_ltba == pytest.approx(
        2.0 * (u.v_tb - u.v_lls) * u.h_lls / math.sqrt(2.0 * G * dev.l_tb_star_m), rel=1e-12
    )


# --- SLUG: сквозные эталоны ---
def test_r2_end_to_end_developed_slug(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Сквозные эталоны», Ansari R2, developing_slug = False, ff_over_fsl_unity = True:
    # SLUG, 4912 Па/м (книга 4921), rtol 1%
    result = Ansari(EX_4_9).calc(FlowState(**r2_kwargs))
    assert result.pattern is FlowPattern.SLUG
    assert result.dpdl_pa_m == pytest.approx(4912, rel=1e-2)
    assert result.dpdl_acceleration_pa_m == 0.0
    assert result.details["developing"] == 0.0


def test_r2_developed_slug_components(r2_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R2, развитая пробка): гравитация (60) = 4777.7, трение (65) = 134.05,
    # H_L = (1 − β)·H_LLS + β·H_LTB = 0.6259
    result = Ansari(EX_4_9).calc(FlowState(**r2_kwargs))
    assert result.dpdl_gravity_pa_m == pytest.approx(4777.7, rel=1e-3)
    assert result.dpdl_friction_pa_m == pytest.approx(134.05, rel=2e-3)
    assert result.holdup == pytest.approx(0.6259, rel=1e-3)


def test_r2_developed_slug_gradient_definitions(r2_kwargs: dict[str, Any]) -> None:
    """(60) [(1 − β)·ρ_LS + β·ρ_g]·g·sinθ, ρ_LS (61); (65) f_LS·ρ_LS·v_m²/(2d)·(1 − β), Re_LS (66)."""
    state = FlowState(**r2_kwargs)
    result = Ansari(EX_4_9).calc(state)
    d = result.details
    v_m = state.v_sl_m_s + state.v_sg_m_s
    mix = mixture.slip(
        d["h_lls"], state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s
    )
    assert d["rho_ls_kg_m3"] == pytest.approx(mix.rho_s_kg_m3, rel=1e-12)
    assert d["mu_ls_pa_s"] == pytest.approx(mix.mu_s_pa_s, rel=1e-12)
    assert d["re_ls"] == pytest.approx(mix.rho_s_kg_m3 * v_m * state.d_m / mix.mu_s_pa_s, rel=1e-12)
    assert d["f_ls"] == pytest.approx(
        friction.moody(d["re_ls"], state.eps_m / state.d_m), rel=1e-12
    )
    beta = d["beta"]
    assert result.dpdl_gravity_pa_m == pytest.approx(
        ((1.0 - beta) * mix.rho_s_kg_m3 + beta * state.rho_g_kg_m3) * G, rel=1e-12
    )
    assert result.dpdl_friction_pa_m == pytest.approx(
        d["f_ls"] * mix.rho_s_kg_m3 * v_m**2 / (2.0 * state.d_m) * (1.0 - beta), rel=1e-12
    )
    assert result.holdup == pytest.approx((1.0 - beta) * d["h_lls"] + beta * d["h_ltb"], rel=1e-12)


def test_r2_default_is_developing_slug(r2_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R2, умолчание): β* = 0.3368, гравитация (62) = 5053.1, трение = 124.74, H_L = 0.6306
    result = Ansari().calc(FlowState(**r2_kwargs))
    d = result.details
    assert result.pattern is FlowPattern.SLUG
    assert d["developing"] == 1.0
    assert d["beta_star"] == pytest.approx(0.3368, rel=1e-3)
    assert result.dpdl_gravity_pa_m == pytest.approx(5053.1, rel=1e-3)
    assert result.dpdl_friction_pa_m == pytest.approx(124.74, rel=2e-3)
    assert result.holdup == pytest.approx(0.6306, rel=1e-3)
    assert result.dpdl_pa_m == pytest.approx(5177.8, rel=1e-3)


def test_developing_slug_gradient_definitions(r2_kwargs: dict[str, Any]) -> None:
    """(62): [(1 − β*)·ρ_LS + β*·ρ_TBA]·g·sinθ, ρ_TBA (63); (65) с β*; H_L с H_LTBA."""
    state = FlowState(**r2_kwargs)
    result = Ansari().calc(state)
    d = result.details
    v_m = state.v_sl_m_s + state.v_sg_m_s
    rho_tba = state.rho_l_kg_m3 * d["h_ltba"] + state.rho_g_kg_m3 * (1.0 - d["h_ltba"])
    b = d["beta_star"]
    assert result.dpdl_gravity_pa_m == pytest.approx(
        ((1.0 - b) * d["rho_ls_kg_m3"] + b * rho_tba) * G, rel=1e-12
    )
    assert result.dpdl_friction_pa_m == pytest.approx(
        d["f_ls"] * d["rho_ls_kg_m3"] * v_m**2 / (2.0 * state.d_m) * (1.0 - b), rel=1e-12
    )
    assert result.holdup == pytest.approx((1.0 - b) * d["h_lls"] + b * d["h_ltba"], rel=1e-12)


def test_flag_developing_slug_changes_result(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    on = Ansari().calc(state)
    off = Ansari(AnsariOptions(developing_slug=False)).calc(state)
    assert on.dpdl_pa_m != pytest.approx(off.dpdl_pa_m, rel=1e-3)
    assert on.holdup != pytest.approx(off.holdup, rel=1e-3)


# --- (56) как напечатано: eq56_as_published ---
def test_developing_slug_unit_default_matches_derived_56(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    assert developing_slug_unit(state, u, eq56_as_published=False) == developing_slug_unit(state, u)


def test_developing_slug_unit_as_published_has_no_root_for_r2(r2_kwargs: dict[str, Any]) -> None:
    """(2 − H_LLS): для R2 b = −3.29, c² + a·b < 0 — вещественного корня нет, развивающаяся пробка не определена."""
    state = FlowState(**r2_kwargs)
    u = solve_slug_unit(state)
    assert developing_slug_unit(state, u, eq56_as_published=True) is None


def test_flag_eq56_as_published_falls_back_to_developed_slug(r2_kwargs: dict[str, Any]) -> None:
    """True: корня (54) нет → полностью развитая пробка (L_TB, β, H_LTB); результат совпадает с developing_slug=False."""
    state = FlowState(**r2_kwargs)
    literal = Ansari(AnsariOptions(eq56_as_published=True)).calc(state)
    developed = Ansari(AnsariOptions(developing_slug=False)).calc(state)
    assert literal.details["developing"] == 0.0
    assert literal.holdup == pytest.approx(developed.holdup, rel=1e-12)
    assert literal.dpdl_pa_m == pytest.approx(developed.dpdl_pa_m, rel=1e-12)
    # reference: расчёт агента (R2, ff_over_fsl_unity по умолчанию не влияет на пробку): 5178 против 4912 Па/м
    default = Ansari().calc(state)
    assert default.details["developing"] == 1.0
    assert default.dpdl_pa_m == pytest.approx(5177.8, rel=1e-3)


def test_flag_vo_shoham_a_as_published_changes_h_ltb(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    base = Ansari(EX_4_9).calc(state)
    literal = Ansari(
        AnsariOptions(developing_slug=False, ff_over_fsl_unity=True, vo_shoham_a_as_published=True)
    ).calc(state)
    assert base.details["h_ltb"] == pytest.approx(0.12983, rel=1e-3)
    assert literal.details["h_ltb"] == pytest.approx(0.1188, rel=1e-3)
    assert literal.dpdl_pa_m != pytest.approx(base.dpdl_pa_m, rel=1e-6)


@pytest.mark.parametrize(("v_sl", "v_sg"), [(0.3, 0.3), (1.0, 2.0), (2.0, 0.5)])
def test_developing_flag_follows_cap_length_criterion(
    aw_state: Callable[..., FlowState], v_sl: float, v_sg: float
) -> None:
    """ТЗ/Ansari: L_c ≥ L_TB → развивающаяся пробка; details developing = 1 иначе 0 (при developing_slug = True)."""
    state = aw_state(v_sl, v_sg, 0.0508, 90.0)
    unit = solve_slug_unit(state)
    expected = 1.0 if cap_length(state, unit) >= unit.l_tb_m else 0.0
    assert Ansari().calc(state).details["developing"] == expected


# --- BUBBLE / DISPERSED_BUBBLE ---
def test_bubble_holdup_solves_equation_22(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 0.01, 0.1524, 90.0)
    h = bubble_holdup(state)
    v0 = (
        1.53
        * (G * state.sigma_n_m * (state.rho_l_kg_m3 - state.rho_g_kg_m3) / state.rho_l_kg_m3**2)
        ** 0.25
    )
    lhs = v0 * h**0.5
    rhs = state.v_sg_m_s / (1.0 - h) - 1.2 * (state.v_sl_m_s + state.v_sg_m_s)
    assert 0.0 < h < 1.0
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_bubble_gradient(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 0.01, 0.1524, 90.0)
    result = Ansari().calc(state)
    assert result.pattern is FlowPattern.BUBBLE
    h = result.holdup
    assert h == pytest.approx(bubble_holdup(state), rel=1e-12)
    mix = mixture.slip(h, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s)
    v_m = state.v_sl_m_s + state.v_sg_m_s
    f = friction.moody(mix.rho_s_kg_m3 * v_m * state.d_m / mix.mu_s_pa_s, 0.0)
    assert result.dpdl_gravity_pa_m == pytest.approx(mix.rho_s_kg_m3 * G, rel=1e-12)  # (26)
    assert result.dpdl_friction_pa_m == pytest.approx(
        f * mix.rho_s_kg_m3 * v_m**2 / (2.0 * state.d_m), rel=1e-12
    )  # (27), v_TP = v_m
    assert result.dpdl_acceleration_pa_m == 0.0


def test_dispersed_bubble_is_no_slip_pseudo_single_phase(
    aw_state: Callable[..., FlowState],
) -> None:
    state = aw_state(1.7783, 0.01, 0.0254, 90.0)
    result = Ansari().calc(state)
    assert result.pattern is FlowPattern.DISPERSED_BUBBLE
    lam = state.v_sl_m_s / (state.v_sl_m_s + state.v_sg_m_s)
    assert result.holdup == pytest.approx(lam, rel=1e-12)  # (16)–(19)
    mix = mixture.no_slip(
        state.v_sl_m_s,
        state.v_sg_m_s,
        state.rho_l_kg_m3,
        state.rho_g_kg_m3,
        state.mu_l_pa_s,
        state.mu_g_pa_s,
    )
    v_m = mix.v_m_m_s
    f = friction.moody(mix.rho_ns_kg_m3 * v_m * state.d_m / mix.mu_ns_pa_s, 0.0)
    assert result.dpdl_pa_m == pytest.approx(
        mix.rho_ns_kg_m3 * G + f * mix.rho_ns_kg_m3 * v_m**2 / (2.0 * state.d_m), rel=1e-12
    )


# --- ANNULAR ---
def test_annular_gradient_by_equations_96_99_102(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 17.7828, 0.0254, 90.0)
    result = Ansari().calc(state)
    d = result.details
    film = solve_annular_film(state)
    assert result.pattern is FlowPattern.ANNULAR
    assert film is not None
    assert d["delta"] == pytest.approx(film.delta, rel=1e-12)
    assert d["phi_c_sq"] == pytest.approx(film.z / (1.0 - 2.0 * film.delta) ** 5, rel=1e-12)  # (99)
    assert result.dpdl_friction_pa_m == pytest.approx(d["phi_c_sq"] * film.dp_sc_pa_m, rel=1e-12)
    assert result.dpdl_gravity_pa_m == pytest.approx(film.rho_c_kg_m3 * G, rel=1e-12)
    assert result.dpdl_acceleration_pa_m == 0.0  # (102): без ускорения
    assert result.holdup == pytest.approx(
        4.0 * film.delta * (1.0 - film.delta) + film.lam_lc * (1.0 - 2.0 * film.delta) ** 2,
        rel=1e-12,
    )


def test_annular_entrainment_and_core(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 17.7828, 0.0254, 90.0)
    d = Ansari().calc(state).details
    v_crit = annular.critical_velocity(
        state.v_sg_m_s, state.mu_g_pa_s, state.sigma_n_m, state.rho_g_kg_m3, state.rho_l_kg_m3
    )
    assert d["f_e"] == pytest.approx(annular.entrainment_wallis(v_crit), rel=1e-12)
    assert d["lam_lc"] == pytest.approx(
        annular.core_liquid_fraction(d["f_e"], state.v_sl_m_s, state.v_sg_m_s), rel=1e-12
    )


def test_flag_ff_over_fsl_unity_changes_annular_gradient(
    aw_state: Callable[..., FlowState],
) -> None:
    state = aw_state(0.001, 17.7828, 0.0254, 90.0)
    plain = Ansari().calc(state)
    unity = Ansari(AnsariOptions(ff_over_fsl_unity=True)).calc(state)
    assert plain.pattern is unity.pattern is FlowPattern.ANNULAR
    assert plain.dpdl_pa_m != pytest.approx(unity.dpdl_pa_m, rel=1e-6)


def test_flag_bridging_changes_regime_through_default_flow_model(
    aw_state: Callable[..., FlowState],
) -> None:
    state = aw_state(0.17783, 33.246, 0.0254, 90.0)
    paper = Ansari().calc(state)
    example = Ansari(AnsariOptions(bridging_with_delta_min=False)).calc(state)
    assert paper.pattern is FlowPattern.SLUG
    assert example.pattern is FlowPattern.ANNULAR


def test_flag_bubble_slug_sin_theta_changes_regime_through_default_flow_model(
    aw_state: Callable[..., FlowState],
) -> None:
    state = aw_state(0.001, 0.06224, 0.1524, 80.0)
    assert Ansari().calc(state).pattern is FlowPattern.BUBBLE
    assert Ansari(AnsariOptions(bubble_slug_sin_theta=True)).calc(state).pattern is FlowPattern.SLUG


# --- прочее ---
def test_churn_is_treated_as_slug(aw_state: Callable[..., FlowState]) -> None:
    """ТЗ: churn входит в slug; режим в результате сохраняется как выдал модуль режима."""
    state = aw_state(1.0, 1.0, 0.0508, 90.0)
    slug = Ansari(flow_model=FixedFlowModel(FlowPattern.SLUG)).calc(state)
    churn = Ansari(flow_model=FixedFlowModel(FlowPattern.CHURN)).calc(state)
    assert churn.pattern is FlowPattern.CHURN
    assert churn.holdup == pytest.approx(slug.holdup, rel=1e-12)
    assert churn.dpdl_pa_m == pytest.approx(slug.dpdl_pa_m, rel=1e-12)


@pytest.mark.parametrize(
    "pattern", [FlowPattern.STRATIFIED, FlowPattern.SEGREGATED, FlowPattern.ELONGATED_BUBBLE]
)
def test_unsupported_patterns_raise(
    pattern: FlowPattern, aw_state: Callable[..., FlowState]
) -> None:
    with pytest.raises(UnsupportedPatternError):
        Ansari(flow_model=FixedFlowModel(pattern)).calc(aw_state(1.0, 1.0, 0.0508, 90.0))


def test_gravity_uses_sin_theta_for_inclined_slug(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, 0.0508, 80.0)
    vertical = Ansari(AnsariOptions(developing_slug=False)).calc(aw_state(1.0, 1.0, 0.0508, 90.0))
    inclined = Ansari(AnsariOptions(developing_slug=False)).calc(state)
    assert inclined.dpdl_gravity_pa_m == pytest.approx(
        vertical.dpdl_gravity_pa_m * math.sin(math.radians(80.0)), rel=1e-6
    )


def test_bubble_equation_uses_harmathy_rise_velocity(aw_state: Callable[..., FlowState]) -> None:
    """v_s (21) = Harmathy v∞·H_L^0.5; v∞ совпадает с `core.bubbles.harmathy_rise_velocity`."""
    state = aw_state(0.001, 0.01, 0.1524, 90.0)
    v_inf = bubbles.harmathy_rise_velocity(state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m)
    h = bubble_holdup(state)
    rhs = state.v_sg_m_s / (1.0 - h) - 1.2 * (state.v_sl_m_s + state.v_sg_m_s)
    assert v_inf * h**0.5 == pytest.approx(rhs, rel=1e-9)
