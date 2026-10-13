"""Корреляция Mukherjee–Brill для наклонных труб (полная корреляция, ICorrelation).

Схема расчёта (θ > 0 — подъём):
    1. N_Lv, N_gv, N_L (N_L с g); режим — `MukherjeeBrillFlowModel` (BUBBLE, SLUG, ANNULAR,
      STRATIFIED);
    2. H_L по ур. (10), Table 1: θ ≥ 0 — набор «uphill and horizontal»; θ < 0 и STRATIFIED —
      «downhill
       stratified»; θ < 0 и прочие — «downhill other»; степень при C3 — sin²θ;
    3. BUBBLE/SLUG: −dp/dL = [f_ns·ρ_s·v_m²/(2d) + ρ_s·g·sinθ]/(1 − Ek), Ek = ρ_s·v_m·v_sg/p, f_ns
      — Moody
       (Colebrook) по Re_ns (ур. 14);
    4. ANNULAR: H_R = λ/H_L, f_R по Table 2 (вне 0.01…10 — 1.0), f = f_R·f_ns;
       −dp/dL = [f·ρ_ns·v_m²/(2d) + ρ_s·g·sinθ]/(1 − Ek);
    5. STRATIFIED: δ из H_L = (δ − sin δ)/2π (ур. 20), D_EL, D_Eg (24)–(25), P_g = (1 − δ/2π)·πd;
       v_L = v_sl/H_L, v_g = v_sg/(1 − H_L); f_L, f_g — Moody по Re с D_E; τ_w = f·ρ·v²/8;
       −dp/dL = τ_wg·P_g/A_g + ρ_g·g·sinθ (ур. 15, τ_i = 0); ускорение = 0.

Источник: Mukherjee H., Brill J.P., 1985, J. Energy Resources Technology 107(4), 549–554, ур.
  (1)–(27),
Table 1–2; режимы — Mukherjee, Brill 1985, IJMF 11(3).
Исправления (ТЗ, «Опечатки»): N_L с g (в JERT напечатано без g); (10) — общепринятая форма скобок;
в монографии (4.135) sin³θ неверно — sin²θ.

Допущения (TODO(question N) ТЗ): в стратифицированной ветви относительная шероховатость фазы ε/D_E;
принимаются только режимы BUBBLE, SLUG, ANNULAR, STRATIFIED (иначе `UnsupportedPatternError`);
H_L ≤ 1 обеспечивается обрезкой (формула (10) может дать exp(x) > 1 при больших N_L).
"""

import bisect
import math
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import dimensionless, friction, gradient, mixture, stratified
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import UnsupportedPatternError
from correlations.core.interfaces import IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowState
from correlations.flow_models.mukherjee_brill_fp import MukherjeeBrillFlowModel
from correlations.models._common import assemble_gradient, single_phase_correlation_result

# Table 1 (C1…C6)
_COEFFICIENTS_UPHILL: Final = (-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657)
_COEFFICIENTS_DOWNHILL_STRATIFIED: Final = (
    -1.330282,
    4.808139,
    4.171584,
    56.262268,
    0.079951,
    0.504887,
)
_COEFFICIENTS_DOWNHILL_OTHER: Final = (
    -0.516644,
    0.789805,
    0.551627,
    15.519214,
    0.371771,
    0.393952,
)
_MAX_EXPONENT: Final = 700.0
# Table 2: H_R → f_R
_TABLE_2_HR: Final = (0.01, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 10.00)
_TABLE_2_FR: Final = (1.00, 0.98, 1.20, 1.25, 1.30, 1.25, 1.00, 1.00)
# Касательное напряжение: τ = f·ρ·v²/8 (Moody); «как напечатано» (γv²/2g при Moody f) — f·ρ·v²/2
_WALL_SHEAR_MOODY: Final = 1.0 / 8.0
_WALL_SHEAR_AS_PUBLISHED: Final = 1.0 / 2.0
# Режимы, обрабатываемые как BUBBLE/SLUG
_BUBBLE_SLUG: Final = (FlowPattern.BUBBLE, FlowPattern.SLUG)


class MukherjeeBrillOptions(BaseOptions):
    """Флаги корреляции Mukherjee–Brill."""

    include_acceleration: bool = Field(
        default=True,
        description=(
            "Знаменатель 1 − Ek в BUBBLE/SLUG/ANNULAR. Выключать для сверки с Примером 4.8. "
            "Источник: Mukherjee, Brill 1985, ур. (1), Δp_a."
        ),
    )
    stratified_combined_momentum: bool = Field(
        default=False,
        description=(
            "Уравнение (17) (сумма фаз, без τ_i) вместо (15) для расслоённого потока. Включать для "
            "труб большого диаметра (рекомендация авторов); граница диаметра не задана. "
            "Источник: Mukherjee, Brill 1985, ур. (15)–(17)."
        ),
    )
    wall_shear_as_published: bool = Field(
        default=False,
        description=(
            "τ_w = f·ρ·v²/2 (как напечатано: γv²/2g при Moody f) вместо f·ρ·v²/8. Только для "
            "воспроизведения кода, реализованного «буквально» (сверка с PIPESIM). "
            "Источник: Mukherjee, Brill 1985, ур. (26)–(27)."
        ),
    )
    annular_fr_log_interp: bool = Field(
        default=False,
        description=(
            "Интерполяция Table 2 по lg H_R вместо линейной по H_R. Таблица снята с графика в "
            "лог-шкале (Fig. 4); даёт более гладкую f_R. Источник: Mukherjee, Brill 1985, Table 2."
        ),
    )


def liquid_holdup(
    n_lv: float,
    n_gv: float,
    n_l: float,
    theta_rad: float,
    *,
    stratified: bool,
) -> float:
    """Истинное содержание жидкости по ур. (10), Table 1.

    H_L = exp{(C1 + C2·sinθ + C3·sin²θ + C4·N_L²)·N_gv^C5/N_Lv^C6}.
    # typo fix: в (4.135) монографии при C3 напечатано sin³θ — взято sin²θ (ТЗ, «Опечатки»).

    Параметры:
        n_lv: число скорости жидкости N_Lv (> 0).
        n_gv: число скорости газа N_gv (> 0).
        n_l: число вязкости жидкости N_L.
        theta_rad: угол от горизонтали, рад (≥ 0 — подъём и горизонталь).
        stratified: режим STRATIFIED (учитывается только при θ < 0).

    Возвращает:
        H_L, 0…1.
    """
    if theta_rad >= 0.0:
        c1, c2, c3, c4, c5, c6 = _COEFFICIENTS_UPHILL
    elif stratified:
        c1, c2, c3, c4, c5, c6 = _COEFFICIENTS_DOWNHILL_STRATIFIED
    else:
        c1, c2, c3, c4, c5, c6 = _COEFFICIENTS_DOWNHILL_OTHER
    s = math.sin(theta_rad)
    exponent = (c1 + c2 * s + c3 * s**2 + c4 * n_l**2) * n_gv**c5 / n_lv**c6
    return min(1.0, math.exp(min(exponent, _MAX_EXPONENT)))


def annular_friction_ratio(h_r: float, *, log_interpolation: bool = False) -> float:
    """Отношение коэффициентов трения f_R по H_R = λ/H_L (Table 2).

    Параметры:
        h_r: отношение holdup H_R = λ_L/H_L.
        log_interpolation: интерполировать по lg H_R (иначе линейно по H_R).

    Возвращает:
        f_R; вне диапазона 0.01…10 (и для H_R = ∞) — 1.0.
    """
    if not (_TABLE_2_HR[0] <= h_r <= _TABLE_2_HR[-1]):
        return 1.0
    index = min(bisect.bisect_right(_TABLE_2_HR, h_r), len(_TABLE_2_HR) - 1)
    x0, x1 = _TABLE_2_HR[index - 1], _TABLE_2_HR[index]
    y0, y1 = _TABLE_2_FR[index - 1], _TABLE_2_FR[index]
    if log_interpolation:
        t = (math.log10(h_r) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
    else:
        t = (h_r - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def equivalent_diameters(delta: float, d_m: float) -> tuple[float, float]:
    """Эквивалентные диаметры жидкой и газовой фаз (ур. 24–25).

    D_EL = d·(δ − sin δ)/(δ + 2·sin(δ/2)); D_Eg = d·[2π − (δ − sin δ)]/[2π − δ + 2·sin(δ/2)].

    Параметры:
        delta: угол δ границы раздела, рад (0…2π).
        d_m: диаметр трубы, м.

    Возвращает:
        Пара (D_EL, D_Eg), м. При δ = 0 (нет жидкости) — (0, d).
    """
    if delta <= 0.0:
        return 0.0, d_m
    area_term = delta - math.sin(delta)
    d_el = d_m * area_term / (delta + 2.0 * math.sin(delta / 2.0))
    d_eg = d_m * (2.0 * math.pi - area_term) / (2.0 * math.pi - delta + 2.0 * math.sin(delta / 2.0))
    return d_el, d_eg


class MukherjeeBrill:
    """Корреляция Mukherjee–Brill (ICorrelation).

    Атрибуты:
        name: "mukherjee_brill".
    """

    name: ClassVar[str] = "mukherjee_brill"

    def __init__(
        self,
        options: MukherjeeBrillOptions | None = None,
        flow_model: IFlowModel | None = None,
    ) -> None:
        """Создаёт корреляцию.

        Параметры:
            options: настройки (по умолчанию — поведение оригинала статьи).
            flow_model: модуль режима; по умолчанию `MukherjeeBrillFlowModel`.
        """
        self._options = options if options is not None else MukherjeeBrillOptions()
        self._flow_model = flow_model if flow_model is not None else MukherjeeBrillFlowModel()

    @property
    def options(self) -> MukherjeeBrillOptions:
        """Настройки корреляции."""
        return self._options

    @property
    def flow_model(self) -> IFlowModel:
        """Используемый модуль режима."""
        return self._flow_model

    def calc(self, state: FlowState) -> CorrelationResult:
        """Рассчитывает режим, holdup и градиент давления.

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `CorrelationResult`; details: `lambda_l`, `n_lv`, `n_gv`, `n_l`, `rho_ns_kg_m3`,
              `mu_ns_pa_s`,
            `re_ns`, `f_ns`, `rho_s_kg_m3`, `ek`, `h_l`, для ANNULAR — `h_r`, `f_r`, для
              STRATIFIED —
            `delta`, `h_l_over_d`, `d_el_m`, `d_eg_m`, `f_l`, `f_g`, `tau_wl_pa`, `tau_wg_pa`.

        Исключения:
            UnsupportedPatternError: модуль режима вернул режим вне BUBBLE/SLUG/ANNULAR/STRATIFIED.
        """
        single = single_phase_correlation_result(state)
        if single is not None:
            return single
        pattern = self._flow_model.predict(state).pattern
        if pattern not in (*_BUBBLE_SLUG, FlowPattern.ANNULAR, FlowPattern.STRATIFIED):
            raise UnsupportedPatternError(pattern, "mukherjee_brill")

        theta = math.radians(state.theta_deg)
        nums = dimensionless.duns_ros(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.sigma_n_m,
            state.mu_l_pa_s,
        )
        holdup = liquid_holdup(
            nums.n_lv, nums.n_gv, nums.n_l, theta, stratified=pattern is FlowPattern.STRATIFIED
        )
        mix = mixture.no_slip(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        )
        v_m = mix.v_m_m_s
        re_ns = dimensionless.reynolds_number(mix.rho_ns_kg_m3, v_m, state.d_m, mix.mu_ns_pa_s)
        f_ns = friction.moody(re_ns, state.eps_m / state.d_m)
        rho_s = mixture.slip(
            holdup,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        ).rho_s_kg_m3
        ek = gradient.kinetic_number(rho_s, v_m, state.v_sg_m_s, state.p_pa)
        details: dict[str, float] = {
            "lambda_l": mix.lam,
            "n_lv": nums.n_lv,
            "n_gv": nums.n_gv,
            "n_l": nums.n_l,
            "rho_ns_kg_m3": mix.rho_ns_kg_m3,
            "mu_ns_pa_s": mix.mu_ns_pa_s,
            "re_ns": re_ns,
            "f_ns": f_ns,
            "rho_s_kg_m3": rho_s,
            "ek": ek,
            "h_l": holdup,
        }

        if pattern is FlowPattern.STRATIFIED:
            parts = self._stratified_gradient(state, holdup, rho_s, details)
        else:
            grav = gradient.gravity_gradient(rho_s, theta)
            if pattern is FlowPattern.ANNULAR:
                h_r = mix.lam / holdup if holdup > 0.0 else math.inf
                f_r = annular_friction_ratio(
                    h_r, log_interpolation=self._options.annular_fr_log_interp
                )
                details["h_r"] = h_r
                details["f_r"] = f_r
                fric = gradient.friction_gradient(f_r * f_ns, mix.rho_ns_kg_m3, v_m, state.d_m)
            else:
                fric = gradient.friction_gradient(f_ns, rho_s, v_m, state.d_m)
            parts = assemble_gradient(
                grav, fric, ek, include_acceleration=self._options.include_acceleration
            )
        return CorrelationResult(
            pattern=pattern,
            holdup=holdup,
            dpdl_pa_m=parts.dpdl_pa_m,
            dpdl_gravity_pa_m=parts.gravity_pa_m,
            dpdl_friction_pa_m=parts.friction_pa_m,
            dpdl_acceleration_pa_m=parts.acceleration_pa_m,
            details=details,
        )

    def _stratified_gradient(
        self, state: FlowState, holdup: float, rho_s: float, details: dict[str, float]
    ) -> gradient.GradientComponents:
        """Градиент расслоённого потока: ур. (15) при τ_i = 0 или (17); ускорение = 0.

        Параметры:
            state: состояние потока.
            holdup: H_L по ур. (10) для набора «downhill stratified» (или «uphill»).
            rho_s: плотность смеси ρ_s, кг/м³.
            details: словарь для записи промежуточных величин.

        Возвращает:
            `GradientComponents` без ускорения.
        """
        d = state.d_m
        theta = math.radians(state.theta_deg)
        h_l = min(holdup, 1.0 - 1e-12)
        delta = stratified.delta_from_holdup(h_l)
        d_el, d_eg = equivalent_diameters(delta, d)
        perimeter = math.pi * d
        p_g = (1.0 - delta / (2.0 * math.pi)) * perimeter  # (23)
        p_l = perimeter - p_g  # (22)
        k = _WALL_SHEAR_AS_PUBLISHED if self._options.wall_shear_as_published else _WALL_SHEAR_MOODY
        v_g = state.v_sg_m_s / (1.0 - h_l)
        re_g = dimensionless.reynolds_number(state.rho_g_kg_m3, v_g, d_eg, state.mu_g_pa_s)
        f_g = friction.moody(re_g, state.eps_m / d_eg)
        tau_g = k * f_g * state.rho_g_kg_m3 * v_g**2  # (27)
        if h_l > 0.0:
            v_l = state.v_sl_m_s / h_l
            re_l = dimensionless.reynolds_number(state.rho_l_kg_m3, v_l, d_el, state.mu_l_pa_s)
            f_l = friction.moody(re_l, state.eps_m / d_el)
            tau_l = k * f_l * state.rho_l_kg_m3 * v_l**2  # (26)
        else:
            f_l = tau_l = 0.0
        area = math.pi * d**2 / 4.0
        if self._options.stratified_combined_momentum:  # (17)
            fric = (tau_l * p_l + tau_g * p_g) / area
            grav = rho_s * GRAVITY_M_S2 * math.sin(theta)
        else:  # (15)
            fric = tau_g * p_g / ((1.0 - h_l) * area)
            grav = state.rho_g_kg_m3 * GRAVITY_M_S2 * math.sin(theta)
        details.update(
            {
                "delta": delta,
                "h_l_over_d": stratified.level_from_delta(delta),
                "d_el_m": d_el,
                "d_eg_m": d_eg,
                "f_l": f_l,
                "f_g": f_g,
                "tau_wl_pa": tau_l,
                "tau_wg_pa": tau_g,
            }
        )
        return assemble_gradient(grav, fric, 0.0, include_acceleration=False)
