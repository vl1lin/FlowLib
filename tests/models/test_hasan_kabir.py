"""Тесты `models/hasan_kabir.py` — корреляция Hasan–Kabir 1988 (ICorrelation).

Ожидаемый API (`from correlations.models.hasan_kabir import HasanKabir, HasanKabirOptions`):
    HasanKabirOptions(porteous_low_pressure=False, wallis_c2=False, rise_coef_150=False,
                      churn_as_slug=False, include_acceleration=True, warn_non_vertical=True)
    HasanKabir(options=None, flow_model=None); .name == "hasan_kabir"; .options; .flow_model; .calc(state)
        flow_model по умолчанию — `HasanKabirFlowModel(HasanKabirFlowOptions(options.porteous_low_pressure))`.
Схема (ТЗ, «Hasan & Kabir (1988)»; r = d_t/d, 0 для трубы):
    BUBBLE/DISPERSED_BUBBLE: f_g = v_gs/(C0·v_M + v∞), C0 = 1.2 + 0.371·r; ρ_M (21); трение f·ρ_M·v_M²/(2d),
        f — Moody по Re = ρ_L·v_M·d/μ_L; гравитация ρ_M·g·sinθ.
    SLUG: f_g = v_gs/(C1·v_M + v∞T); труба: C1 = 1.2, v∞T = C2·√(g·d·Δρ/ρ_L), C2 = 0.345;
        затрубье: C1 = 1.182 + 0.90·r, v∞T = (0.30 + 0.22·r)·√(g·(d − d_t)·Δρ/ρ_L); трение f·ρ_L·(1 − f_g)·v_M²/(2d).
    CHURN: как SLUG, но C1 = 1.15 + 0.90·r (Приложение).
    ANNULAR: v_crit, E (35)/(36), ρ_c (37b), x, X (42), f_g = (1 + X^0.8)^-0.378 (40), v_g = v_gs/f_g,
        f_c = 0.079·Re_g^-0.25·[1 + 75·(1 − f_g)] (Fanning), Re_g = ρ_g·v_gs·d/μ_g;
        −dp/dL = [ρ_c·g·sinθ + 2·f_c·ρ_c·v_g²/d]/(1 − ρ_c·v_g²/p) (33); H_L = (1 − f_g) + f_g·(1 − f_gc).
    H_L = 1 − f_g для BUBBLE/DISPERSED_BUBBLE/SLUG/CHURN.
details: v_m_m_s, v_inf_m_s, f_g, f_darcy, re_m; BUBBLE: c0, rho_m_kg_m3; SLUG/CHURN: c1, c2, v_inf_t_m_s, rho_m_kg_m3;
    ANNULAR: v_crit, e, f_gc, rho_c_kg_m3, x, x_lm, v_g_m_s, re_g, f_c_fanning, ek.
Допущения (вне ТЗ, TODO(question N)): ускорение (знаменатель 1 − Ek) — только в ANNULAR (ур. 33; в
пузырьковом/пробковом — «пренебрегается», HK); диаметр трения в затрубье — d (как в статье).
Предупреждения (ТЗ, вопрос 6, «Исключения»): θ ≠ 90° — OutOfRangeWarning (считается с sinθ); Re плёнки < 3000
(применимость Steen–Wallis) — OutOfRangeWarning.
Предупреждение о наклоне (решение пользователя): текст сообщения постоянный (без θ), чтобы стандартный фильтр
Python показывал его один раз на место вызова; угол — в атрибуте `OutOfRangeWarning.context["theta_deg"]`.
Флаг `warn_non_vertical` (по умолчанию True) отключает только это предупреждение.
Источник: Hasan A.R., Kabir C.S., 1988, SPEPE 3(2), 263–272 (+ Erratum 1989). Опечатки (ТЗ): (6) ρ_L, (23), (26b),
(30) g·(d_c − d_t), (37b) E·v_Ls.
Эталоны R2 для ANNULAR помечены в ТЗ как provisional («пересчёт», до сверки с PIPESIM).
"""

import math
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from correlations.core import annular, bubbles, friction, mixture
from correlations.core.errors import OutOfRangeWarning, UnsupportedPatternError
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel
from correlations.models.hasan_kabir import HasanKabir, HasanKabirOptions

D = 0.0508
G = 9.81


class FixedFlowModel:
    """Модуль режима-заглушка: всегда возвращает заданный режим."""

    name: ClassVar[str] = "fixed"

    def __init__(self, pattern: FlowPattern) -> None:
        self._pattern = pattern

    @property
    def options(self) -> BaseOptions:
        return BaseOptions()

    def predict(self, state: FlowState) -> FlowPatternResult:
        return FlowPatternResult(pattern=self._pattern, details={})


def hk(pattern: FlowPattern, options: HasanKabirOptions | None = None) -> HasanKabir:
    return HasanKabir(options, flow_model=FixedFlowModel(pattern))


def test_default_flags() -> None:
    options = HasanKabir().options
    assert options.porteous_low_pressure is False
    assert options.wallis_c2 is False
    assert options.rise_coef_150 is False
    assert options.churn_as_slug is False
    assert options.include_acceleration is True
    assert options.warn_non_vertical is True


def test_default_flow_model_gets_porteous_flag() -> None:
    assert isinstance(HasanKabir().flow_model, HasanKabirFlowModel)
    on = HasanKabir(HasanKabirOptions(porteous_low_pressure=True))
    assert on.flow_model.options.porteous_low_pressure is True  # type: ignore[attr-defined]


# --- R2: сквозной эталон (provisional, ТЗ) ---
def test_r2_end_to_end(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Сквозные эталоны», HK 1988 R2, по умолчанию: ANNULAR, H_L = 0.5744, 4186 Па/м
    # (пересчёт, provisional), rtol 0.1%
    result = HasanKabir().calc(FlowState(**r2_kwargs))
    assert result.pattern is FlowPattern.ANNULAR
    assert result.holdup == pytest.approx(0.5744, rel=1e-3)
    assert result.dpdl_pa_m == pytest.approx(4186, rel=1e-3)


def test_r2_annular_elements(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: E / ρ_c / X / f_g (кольцевой) = 0.5666 / 340.1 / 2.143 / 0.6740
    # (rtol 1e-3, provisional); f_gc = 0.631 (ур. 37a, подтверждено)
    d = HasanKabir().calc(FlowState(**r2_kwargs)).details
    assert d["e"] == pytest.approx(0.5666, rel=1e-3)
    assert d["rho_c_kg_m3"] == pytest.approx(340.1, rel=1e-3)
    assert d["x_lm"] == pytest.approx(2.143, rel=1e-3)
    assert d["f_g"] == pytest.approx(0.6740, rel=1e-3)
    assert d["f_gc"] == pytest.approx(0.6315, rel=1e-3)


def test_r2_annular_holdup_is_film_plus_droplets(r2_kwargs: dict[str, Any]) -> None:
    """H_L = (1 − f_g) + f_g·(1 − f_gc): плёнка + капли в ядре (ТЗ)."""
    result = HasanKabir().calc(FlowState(**r2_kwargs))
    d = result.details
    assert result.holdup == pytest.approx(
        (1.0 - d["f_g"]) + d["f_g"] * (1.0 - d["f_gc"]), rel=1e-12
    )


def test_r2_components(r2_kwargs: dict[str, Any]) -> None:
    # reference: расчёт агента (R2): гравитация ρ_c·g = 3336.4, трение 848.9, Ek = 8.795e-5
    result = HasanKabir().calc(FlowState(**r2_kwargs))
    assert result.dpdl_gravity_pa_m == pytest.approx(3336.4, rel=1e-3)
    assert result.dpdl_friction_pa_m == pytest.approx(848.9, rel=1e-3)
    assert result.details["ek"] == pytest.approx(8.795e-5, rel=1e-3)
    base = result.dpdl_gravity_pa_m + result.dpdl_friction_pa_m
    assert result.dpdl_pa_m == pytest.approx(base / (1.0 - result.details["ek"]), rel=1e-12)


def test_annular_core_density_and_x(r2_kwargs: dict[str, Any]) -> None:
    """ρ_c = f_gc·ρ_g + (1 − f_gc)·ρ_L; x = ρ_g·v_gs/(ρ_g·v_gs + E·ρ_L·v_Ls) — (37a,b), Q5 ТЗ."""
    state = FlowState(**r2_kwargs)
    d = HasanKabir().calc(state).details
    rho_c = d["f_gc"] * state.rho_g_kg_m3 + (1.0 - d["f_gc"]) * state.rho_l_kg_m3
    assert d["rho_c_kg_m3"] == pytest.approx(rho_c, rel=1e-12)
    g_gas = state.rho_g_kg_m3 * state.v_sg_m_s
    g_liq = d["e"] * state.rho_l_kg_m3 * state.v_sl_m_s
    assert d["x"] == pytest.approx(g_gas / (g_gas + g_liq), rel=1e-12)


def test_annular_friction_moody_fanning_consistency(r2_kwargs: dict[str, Any]) -> None:
    """Moody/Fanning: трение (33) = 2·f_c·ρ_c·v_g²/d с Fanning f_c = Blasius·[1 + 75(1 − f_g)]; ядро — Darcy 4·f_c."""
    state = FlowState(**r2_kwargs)
    result = HasanKabir().calc(state)
    d = result.details
    re_g = state.rho_g_kg_m3 * state.v_sg_m_s * state.d_m / state.mu_g_pa_s
    f_c = friction.fanning_blasius(re_g) * (1.0 + 75.0 * (1.0 - d["f_g"]))
    assert d["re_g"] == pytest.approx(re_g, rel=1e-12)
    assert d["f_c_fanning"] == pytest.approx(f_c, rel=1e-12)
    assert d["v_g_m_s"] == pytest.approx(state.v_sg_m_s / d["f_g"], rel=1e-12)
    assert result.dpdl_friction_pa_m == pytest.approx(
        2.0 * f_c * d["rho_c_kg_m3"] * d["v_g_m_s"] ** 2 / state.d_m, rel=1e-12
    )
    assert result.dpdl_friction_pa_m == pytest.approx(
        friction.fanning_to_darcy(f_c) * d["rho_c_kg_m3"] * d["v_g_m_s"] ** 2 / (2.0 * state.d_m),
        rel=1e-12,
    )


def test_annular_void_fraction_from_lockhart_martinelli(r2_kwargs: dict[str, Any]) -> None:
    """(40), (42): f_g = (1 + X^0.8)^-0.378; X = [(1 − x)/x]^0.9·√(ρ_g/ρ_L)·(μ_L/μ_g)^0.1."""
    state = FlowState(**r2_kwargs)
    d = HasanKabir().calc(state).details
    x_lm = (
        ((1.0 - d["x"]) / d["x"]) ** 0.9
        * math.sqrt(state.rho_g_kg_m3 / state.rho_l_kg_m3)
        * (state.mu_l_pa_s / state.mu_g_pa_s) ** 0.1
    )
    assert d["x_lm"] == pytest.approx(x_lm, rel=1e-12)
    assert d["f_g"] == pytest.approx((1.0 + x_lm**0.8) ** -0.378, rel=1e-12)


def test_annular_entrainment_uses_steen_wallis(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    d = HasanKabir().calc(state).details
    v_crit = annular.critical_velocity(
        state.v_sg_m_s,
        state.mu_g_pa_s,
        state.sigma_n_m,
        state.rho_g_kg_m3,
        state.rho_l_kg_m3,
    )
    assert d["v_crit"] == pytest.approx(v_crit, rel=1e-12)
    assert d["e"] == pytest.approx(annular.entrainment_steen_wallis(v_crit), rel=1e-12)


def test_flag_include_acceleration_only_affects_annular(r2_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r2_kwargs)
    with_acc = HasanKabir().calc(state)
    without = HasanKabir(HasanKabirOptions(include_acceleration=False)).calc(state)
    assert with_acc.dpdl_acceleration_pa_m > 0.0
    assert without.dpdl_acceleration_pa_m == 0.0
    assert without.dpdl_pa_m == pytest.approx(
        without.dpdl_gravity_pa_m + without.dpdl_friction_pa_m, rel=1e-12
    )


def test_bubble_slug_have_no_acceleration(aw_state: Callable[..., FlowState]) -> None:
    for pattern in (FlowPattern.BUBBLE, FlowPattern.SLUG, FlowPattern.CHURN):
        result = hk(pattern).calc(aw_state(0.3, 0.05, D, 90.0))
        assert result.dpdl_acceleration_pa_m == 0.0


# --- BUBBLE / DISPERSED_BUBBLE ---
def aw_props(state: FlowState) -> tuple[float, float, float, float]:
    delta_rho = state.rho_l_kg_m3 - state.rho_g_kg_m3
    return delta_rho, state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m


@pytest.mark.parametrize("d_t", [0.0, 0.03])
def test_bubble_void_fraction_and_gradient(aw_state: Callable[..., FlowState], d_t: float) -> None:
    state = aw_state(0.3, 0.05, D, 90.0, d_t_m=d_t)
    result = hk(FlowPattern.BUBBLE).calc(state)
    delta_rho, rho_l, rho_g, sigma = aw_props(state)
    r = d_t / D
    v_inf = 1.53 * (G * sigma * delta_rho / rho_l**2) ** 0.25
    v_m = 0.35
    f_g = 0.05 / ((1.2 + 0.371 * r) * v_m + v_inf)
    rho_m = rho_g * f_g + rho_l * (1.0 - f_g)
    f_darcy = friction.moody(rho_l * v_m * D / state.mu_l_pa_s, 0.0)
    assert result.holdup == pytest.approx(1.0 - f_g, rel=1e-12)
    assert result.details["c0"] == pytest.approx(1.2 + 0.371 * r, rel=1e-12)  # HK (19)
    assert result.dpdl_friction_pa_m == pytest.approx(f_darcy * rho_m * v_m**2 / (2 * D), rel=1e-12)
    assert result.dpdl_gravity_pa_m == pytest.approx(rho_m * G, rel=1e-12)


def test_dispersed_bubble_uses_bubble_formulas(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(4.0, 0.5, D, 90.0)
    result = HasanKabir().calc(state)
    assert result.pattern is FlowPattern.DISPERSED_BUBBLE
    stub = hk(FlowPattern.BUBBLE).calc(state)
    assert result.holdup == pytest.approx(stub.holdup, rel=1e-12)
    assert result.dpdl_pa_m == pytest.approx(stub.dpdl_pa_m, rel=1e-12)


def test_flag_rise_coef_150(aw_state: Callable[..., FlowState]) -> None:
    # reference: ТЗ, флаг rise_coef_150: 1.50 вместо 1.53 в v∞ для пузырькового H_L (HK ур. 17)
    state = aw_state(0.3, 0.05, D, 90.0)
    base = hk(FlowPattern.BUBBLE).calc(state)
    flagged = hk(FlowPattern.BUBBLE, HasanKabirOptions(rise_coef_150=True)).calc(state)
    assert flagged.details["v_inf_m_s"] == pytest.approx(
        base.details["v_inf_m_s"] * 1.50 / 1.53, rel=1e-12
    )
    assert flagged.holdup != pytest.approx(base.holdup, rel=1e-6)


# --- SLUG / CHURN ---
def test_slug_pipe_void_fraction_and_gradient(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, D, 90.0)
    result = hk(FlowPattern.SLUG).calc(state)
    delta_rho, rho_l, rho_g, _ = aw_props(state)
    v_m = 2.0
    v_inf_t = 0.345 * math.sqrt(G * D * delta_rho / rho_l)
    f_g = 1.0 / (1.2 * v_m + v_inf_t)
    rho_m = rho_g * f_g + rho_l * (1.0 - f_g)
    f_darcy = friction.moody(rho_l * v_m * D / state.mu_l_pa_s, 0.0)
    assert result.holdup == pytest.approx(1.0 - f_g, rel=1e-12)
    assert result.details["c1"] == 1.2
    assert result.details["v_inf_t_m_s"] == pytest.approx(v_inf_t, rel=1e-12)
    assert result.dpdl_friction_pa_m == pytest.approx(
        f_darcy * rho_l * (1.0 - f_g) * v_m**2 / (2 * D), rel=1e-12
    )  # (31)
    assert result.dpdl_gravity_pa_m == pytest.approx(rho_m * G, rel=1e-12)


def test_slug_r2_taylor_velocity(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v∞T (0.345) = 0.3949 м/с (R2)
    result = hk(FlowPattern.SLUG).calc(FlowState(**r2_kwargs))
    assert result.details["v_inf_t_m_s"] == pytest.approx(0.3949, rel=1e-3)


def test_slug_annulus_coefficients(aw_state: Callable[..., FlowState]) -> None:
    """HK (29), (30): C1 = 1.182 + 0.90·r; v∞T = (0.30 + 0.22·r)·√(g·(d_c − d_t)·Δρ/ρ_L) (Erratum 1989)."""
    d_t = 0.03
    state = aw_state(1.0, 1.0, D, 90.0, d_t_m=d_t)
    result = hk(FlowPattern.SLUG).calc(state)
    delta_rho, rho_l, _, _ = aw_props(state)
    r = d_t / D
    v_inf_t = (0.30 + 0.22 * r) * math.sqrt(G * (D - d_t) * delta_rho / rho_l)
    assert result.details["c1"] == pytest.approx(1.182 + 0.90 * r, rel=1e-12)
    assert result.details["v_inf_t_m_s"] == pytest.approx(v_inf_t, rel=1e-12)
    assert result.holdup == pytest.approx(
        1.0 - 1.0 / (result.details["c1"] * 2.0 + v_inf_t), rel=1e-12
    )


def test_churn_uses_c1_1_15(aw_state: Callable[..., FlowState]) -> None:
    # reference: ТЗ: CHURN — как SLUG, но C1 = 1.15 + 0.90·r (Приложение)
    state = aw_state(1.0, 1.0, D, 90.0)
    churn = hk(FlowPattern.CHURN).calc(state)
    slug = hk(FlowPattern.SLUG).calc(state)
    assert churn.details["c1"] == pytest.approx(1.15, rel=1e-12)
    assert churn.holdup != pytest.approx(slug.holdup, rel=1e-6)


def test_churn_annulus_c1(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, D, 90.0, d_t_m=0.03)
    churn = hk(FlowPattern.CHURN).calc(state)
    assert churn.details["c1"] == pytest.approx(1.15 + 0.90 * 0.03 / D, rel=1e-12)


def test_flag_churn_as_slug(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, D, 90.0)
    options = HasanKabirOptions(churn_as_slug=True)
    churn = hk(FlowPattern.CHURN, options).calc(state)
    slug = hk(FlowPattern.SLUG).calc(state)
    assert churn.details["c1"] == pytest.approx(1.2, rel=1e-12)
    assert churn.holdup == pytest.approx(slug.holdup, rel=1e-12)
    assert churn.dpdl_pa_m == pytest.approx(slug.dpdl_pa_m, rel=1e-12)


def test_flag_wallis_c2_for_viscous_liquid(aw_state: Callable[..., FlowState]) -> None:
    """Вязкая жидкость (μ_L = 0.5 Па·с): N_f ≈ 72 → C2 по Wallis (23), (26b) заметно ниже 0.345."""
    state = aw_state(1.0, 1.0, D, 90.0, mu_l_pa_s=0.5)
    base = hk(FlowPattern.SLUG).calc(state)
    flagged = hk(FlowPattern.SLUG, HasanKabirOptions(wallis_c2=True)).calc(state)
    delta_rho, rho_l, rho_g, sigma = aw_props(state)
    n_f = bubbles.inverse_viscosity_number(D, rho_l, rho_g, 0.5)
    n_eo = bubbles.eotvos_number(D, rho_l, rho_g, sigma)
    c2 = bubbles.wallis_taylor_coefficient(n_f, n_eo)
    assert base.details["c2"] == pytest.approx(0.345, rel=1e-12)
    assert flagged.details["c2"] == pytest.approx(c2, rel=1e-12)
    assert c2 < 0.345
    assert flagged.holdup != pytest.approx(base.holdup, rel=1e-6)


def test_wallis_c2_equals_0_345_for_water(aw_state: Callable[..., FlowState]) -> None:
    """ТЗ/HK (27): при больших N_f, N_Eo C2 = 0.345 — флаг не меняет результат для воды."""
    state = aw_state(1.0, 1.0, D, 90.0)
    base = hk(FlowPattern.SLUG).calc(state)
    flagged = hk(FlowPattern.SLUG, HasanKabirOptions(wallis_c2=True)).calc(state)
    assert flagged.holdup == pytest.approx(base.holdup, rel=1e-12)


def test_flag_porteous_changes_result_through_default_flow_model(
    aw_state: Callable[..., FlowState],
) -> None:
    state = aw_state(1.0, 1.5, D, 90.0)
    default = HasanKabir().calc(state)
    flagged = HasanKabir(HasanKabirOptions(porteous_low_pressure=True)).calc(state)
    assert default.pattern is FlowPattern.SLUG
    assert flagged.pattern is FlowPattern.CHURN
    assert flagged.holdup != pytest.approx(default.holdup, rel=1e-6)


# --- общие свойства ---
def test_gravity_uses_mixture_density_and_sin_theta(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, D, 60.0)
    with pytest.warns(OutOfRangeWarning):
        result = hk(FlowPattern.SLUG).calc(state)
    rho_m = mixture.slip(
        result.holdup, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s, state.mu_g_pa_s
    ).rho_s_kg_m3
    assert result.dpdl_gravity_pa_m == pytest.approx(
        rho_m * G * math.sin(math.radians(60.0)), rel=1e-12
    )


def test_warns_when_angle_differs_from_vertical(aw_state: Callable[..., FlowState]) -> None:
    # reference: ТЗ, вопрос 6: при θ ≠ 90° считать с sinθ и выдавать предупреждение
    for theta in (0.0, 45.0, -30.0):
        with pytest.warns(OutOfRangeWarning):
            HasanKabir().calc(aw_state(1.0, 1.0, D, theta))


def _angle_warnings(record: Any) -> list[warnings.WarningMessage]:
    return [
        w
        for w in record
        if issubclass(w.category, OutOfRangeWarning) and "вертикальн" in str(w.message)
    ]


def test_angle_warning_text_is_constant_and_angle_is_in_context(
    aw_state: Callable[..., FlowState],
) -> None:
    """Решение пользователя: текст без θ (одинаков для всех углов), угол — в `context`."""
    messages: set[str] = set()
    for theta in (30.0, 45.0, -30.0):
        with pytest.warns(OutOfRangeWarning) as record:
            HasanKabir().calc(aw_state(1.0, 1.0, D, theta))
        (angle,) = _angle_warnings(record)
        assert isinstance(angle.message, OutOfRangeWarning)
        assert angle.message.context == {"theta_deg": theta}
        messages.add(str(angle.message))
    assert len(messages) == 1


def test_angle_warning_shown_once_by_default_filter(aw_state: Callable[..., FlowState]) -> None:
    """Стандартный фильтр «default» показывает одинаковое сообщение один раз на место вызова
    (профиль скважины с меняющимся углом не засоряет вывод)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for theta in (30.0, 45.0, 60.0):
            HasanKabir().calc(aw_state(1.0, 1.0, D, theta))
    assert len(_angle_warnings(record)) == 1


@pytest.mark.parametrize("theta", [0.0, 45.0, -30.0])
def test_flag_warn_non_vertical_false_silences_angle_warning(
    aw_state: Callable[..., FlowState], theta: float
) -> None:
    state = aw_state(1.0, 1.0, D, theta)
    with pytest.warns(OutOfRangeWarning):
        enabled = HasanKabir().calc(state)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutOfRangeWarning)
        disabled = HasanKabir(HasanKabirOptions(warn_non_vertical=False)).calc(state)
    assert disabled == enabled  # флаг не влияет на результат


def test_flag_warn_non_vertical_false_keeps_other_warnings(
    aw_state: Callable[..., FlowState],
) -> None:
    """Флаг отключает только предупреждение о наклоне; Steen–Wallis остаётся."""
    with pytest.warns(OutOfRangeWarning, match="Steen"):
        HasanKabir(HasanKabirOptions(warn_non_vertical=False)).calc(aw_state(0.001, 20.0, D, 90.0))


def test_warn_non_vertical_flag_is_described() -> None:
    assert HasanKabirOptions.describe()["warn_non_vertical"].strip()


def test_no_warning_for_vertical_flow(r2_kwargs: dict[str, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HasanKabir().calc(FlowState(**r2_kwargs))


def test_no_angle_warning_for_single_phase(aw_state: Callable[..., FlowState]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HasanKabir().calc(aw_state(1.0, 0.0, D, 30.0))


def test_warns_for_thin_film_in_steen_wallis(aw_state: Callable[..., FlowState]) -> None:
    """Re плёнки d·ρ_M·v_M·(1 − x)/μ_L < 3000 — Steen–Wallis вне области применимости (HK, текст перед ур. 34)."""
    state = aw_state(0.001, 20.0, D, 90.0)
    with pytest.warns(OutOfRangeWarning):
        result = HasanKabir().calc(state)
    assert result.pattern is FlowPattern.ANNULAR


def test_no_steen_wallis_warning_for_turbulent_film(r2_kwargs: dict[str, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutOfRangeWarning)
        HasanKabir().calc(FlowState(**r2_kwargs))


@pytest.mark.parametrize(
    "pattern", [FlowPattern.STRATIFIED, FlowPattern.SEGREGATED, FlowPattern.ELONGATED_BUBBLE]
)
def test_unsupported_patterns_raise(
    pattern: FlowPattern, aw_state: Callable[..., FlowState]
) -> None:
    with pytest.raises(UnsupportedPatternError):
        hk(pattern).calc(aw_state(1.0, 1.0, D, 90.0))


def test_annular_holdup_between_zero_and_one(aw_state: Callable[..., FlowState]) -> None:
    for v_sl in (0.001, 0.05, 0.3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OutOfRangeWarning)
            result = HasanKabir().calc(aw_state(v_sl, 25.0, D, 90.0))
        assert result.pattern is FlowPattern.ANNULAR
        assert 0.0 <= result.holdup <= 1.0
