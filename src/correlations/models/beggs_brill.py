"""Корреляция Beggs–Brill для наклонных труб (полная корреляция, ICorrelation).

Схема расчёта:
    1. λ_L = v_sl/v_m, N_FR = v_m²/(g·d), N_Lv, ρ_ns, μ_ns, Re_ns (ур. 14);
    2. режим — `BeggsBrillMap` (SEGREGATED / INTERMITTENT / DISTRIBUTED, при revised_map — и
      TRANSITION);
    3. H_L(0) = A·λ^α/N_FR^|β| (Table 1, ур. 19–21), H_L(0) ≥ λ;
    4. C: подъём — по режиму (ур. 22–24, распределённый C = 0), спуск — одна формула (ур. 25);
       θ = 0 → ψ = 1;
    5. ψ = 1 + C·[sin φ − sin³φ/3], φ = 1.8·θ° (ур. 5); H_L(θ) = H_L(0)·ψ, 0 ≤ H_L ≤ 1 (ур. 11);
       в зоне TRANSITION: H_L = A·H_seg + (1 − A)·H_int, A = (L3 − N_FR)/(L3 − L2) (монография
         4.118–4.119);
    6. y = λ/H_L²; S по ур. 16 (при 1 < y < 1.2 — ln(2.2y − 1.2)); f_tp = f_ns·e^S;
    7. f_ns — гладкая формула (13) или Colebrook (флаг rough_pipe_friction);
    8. −dp/dL = [ρ_s·g·sinθ + f_tp·ρ_ns·v_m²/(2d)]/(1 − Ek), Ek = ρ_s·v_m·v_sg/p (ур. 1).

Источники:
    * Beggs H.D., Brill J.P., 1973, JPT 25(5), 607–617, ур. 1–25, Table 1;
    * Payne G.A., Palmer C.L., Brill J.P., 1979, JPT 31(9), 1198–1208 — поправка на holdup;
    * Brill J.P., Mukherjee H., «Multiphase Flow in Wells», 1999, ур. 4.110–4.119 — revised-карта.
Исправление (ТЗ, «Опечатки»): Table 4.2 монографии — a = 0.845 (напечатано 0.855).

Допущения (TODO(question N) ТЗ): при Re_ns < 2000 f_ns = 64/Re в обоих режимах трения; поправка
  Payne:
множитель 0.924 при θ > 0 с нижней границей λ, 0.685 при θ < 0 без границы.
"""

import math
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import dimensionless, friction, gradient, mixture
from correlations.core.errors import UnsupportedPatternError
from correlations.core.interfaces import IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowState
from correlations.flow_models.beggs_brill_map import BeggsBrillMap, BeggsBrillMapOptions
from correlations.models._common import assemble_gradient, single_phase_correlation_result

# Table 1, горизонтальный holdup: H_L(0) = A·λ^α/N_FR^|β|, (A, α, |β|)
_HORIZONTAL_HOLDUP: Final = {
    FlowPattern.SEGREGATED: (0.98, 0.4846, 0.0868),  # ур. 19
    FlowPattern.INTERMITTENT: (0.845, 0.5351, 0.0173),  # ур. 20; typo fix: монография 0.855
    FlowPattern.DISTRIBUTED: (1.065, 0.5824, 0.0609),  # ур. 21
}
# Table 1, C для подъёма: ln[e·λ^f·N_Lv^g·N_FR^h]
_C_UPHILL_SEGREGATED: Final = (0.011, -3.768, 3.539, -1.614)  # ур. 22
_C_UPHILL_INTERMITTENT: Final = (2.96, 0.305, -0.4473, 0.0978)  # ур. 23
_C_DOWNHILL: Final = (4.7, -0.3692, 0.1244, -0.5056)  # ур. 25
# ψ: φ = 1.8·θ (θ в градусах), ур. 5
_PSI_ANGLE_FACTOR: Final = 1.8
# S (ур. 16): ln y/(−0.0523 + 3.182·ln y − 0.8725·ln²y + 0.01853·ln⁴y); при 1 < y < 1.2 — ln(2.2y
# − 1.2)
_S_COEFFICIENTS: Final = (-0.0523, 3.182, -0.8725, 0.01853)
_S_SPECIAL_RANGE: Final = (1.0, 1.2)
_S_SPECIAL_A: Final = 2.2
_S_SPECIAL_B: Final = 1.2
# Поправка Payne et al. 1979
_PAYNE_UPHILL: Final = 0.924
_PAYNE_DOWNHILL: Final = 0.685


class BeggsBrillOptions(BaseOptions):
    """Флаги корреляции Beggs–Brill."""

    revised_map: bool = Field(
        default=False,
        description=(
            "Карта L1–L4 с переходной зоной (монография 4.110–4.119) вместо оригинальной карты "
            "(ур. 9–10); H_L в зоне — интерполяция между segregated и intermittent. Включать для "
            "сверки с PIPESIM «BB Revised»; убирает скачок H_L на границе segregated/intermittent."
        ),
    )
    payne_correction: bool = Field(
        default=False,
        description=(
            "H_L × 0.924 при θ > 0 (не ниже λ), × 0.685 при θ < 0. Включать для скважин и "
            "трубопроводов: оригинал завышает H_L и dp (Пример 4.7). Источник: Payne et al. 1979."
        ),
    )
    rough_pipe_friction: bool = Field(
        default=False,
        description=(
            "f_ns по Colebrook с шероховатостью ε вместо гладкой формулы (13). Включать для "
            "реальных (шероховатых) труб (Пример 4.7). Источник: Beggs, Brill 1973, ур. 13."
        ),
    )
    clip_negative_c: bool = Field(
        default=False,
        description=(
            "C < 0 → C = 0 (ψ ≥ 1 при подъёме). Включать для совместимости с монографией и "
            "PIPESIM (Пример 4.7). Источник: Beggs, Brill 1973, ур. 22–25; монография."
        ),
    )
    include_acceleration: bool = Field(
        default=True,
        description=(
            "Знаменатель 1 − Ek в ур. 1. Выключать для сверки с примерами монографии и при "
            "Ek → 1 (высокоскоростной газ). Источник: Beggs, Brill 1973, ур. 1."
        ),
    )


def horizontal_holdup(lam: float, n_fr: float, pattern: FlowPattern) -> float:
    """Holdup горизонтального потока H_L(0) = A·λ^α/N_FR^|β| (Table 1, ур. 19–21), не ниже λ.

    Параметры:
        lam: доля жидкости без проскальзывания λ_L.
        n_fr: число Фруда смеси.
        pattern: горизонтальный режим (SEGREGATED, INTERMITTENT или DISTRIBUTED).

    Возвращает:
        H_L(0), безразм. (ограничение H_L(0) ≥ λ, ур. 11).

    Исключения:
        UnsupportedPatternError: режим не относится к карте Beggs–Brill.
    """
    if pattern not in _HORIZONTAL_HOLDUP:
        raise UnsupportedPatternError(pattern, "beggs_brill")
    a, alpha, beta = _HORIZONTAL_HOLDUP[pattern]
    return max(a * lam**alpha / n_fr**beta, lam)


def inclination_coefficient(
    lam: float, n_lv: float, n_fr: float, pattern: FlowPattern, *, uphill: bool
) -> float:
    """Коэффициент наклона C = (1 − λ)·ln[e·λ^f·N_Lv^g·N_FR^h] (Table 1, ур. 22–25), без обрезки.

    Подъём: сегрегированный — ур. 22, перемежающийся — ур. 23, распределённый — C = 0 (ур. 24).
    Спуск: одна формула для всех режимов (ур. 25).

    Параметры:
        lam: доля жидкости без проскальзывания λ_L.
        n_lv: число скорости жидкости N_Lv.
        n_fr: число Фруда смеси.
        pattern: горизонтальный режим.
        uphill: True — подъём (θ > 0), False — спуск (θ < 0).

    Возвращает:
        C, безразм.

    Исключения:
        UnsupportedPatternError: режим не относится к карте Beggs–Brill.
    """
    if pattern not in _HORIZONTAL_HOLDUP:
        raise UnsupportedPatternError(pattern, "beggs_brill")
    if not uphill:
        coefficients = _C_DOWNHILL
    elif pattern is FlowPattern.SEGREGATED:
        coefficients = _C_UPHILL_SEGREGATED
    elif pattern is FlowPattern.INTERMITTENT:
        coefficients = _C_UPHILL_INTERMITTENT
    else:
        return 0.0
    e, f, g, h = coefficients
    return (1.0 - lam) * math.log(e * lam**f * n_lv**g * n_fr**h)


def inclination_factor(c: float, theta_deg: float) -> float:
    """Поправка на наклон ψ = 1 + C·[sin φ − sin³φ/3], φ = 1.8·θ (θ в градусах), ур. 5.

    Параметры:
        c: коэффициент наклона C.
        theta_deg: угол от горизонтали, град (> 0 — подъём).

    Возвращает:
        ψ, безразм. (при θ = 0 равен 1).
    """
    phi = math.radians(_PSI_ANGLE_FACTOR * theta_deg)
    sin_phi = math.sin(phi)
    return 1.0 + c * (sin_phi - sin_phi**3 / 3.0)


def friction_ratio_exponent(y: float) -> float:
    """Показатель S отношения f_tp/f_ns = e^S (ур. 16), y = λ/H_L².

    При 1 < y < 1.2 знаменатель обращается в нуль — S = ln(2.2·y − 1.2). При y = 1 S = 0.

    Параметры:
        y: λ_L/H_L².

    Возвращает:
        S, безразм. (для нефинитного y — 0).
    """
    if not math.isfinite(y) or y <= 0.0:
        return 0.0
    if _S_SPECIAL_RANGE[0] < y < _S_SPECIAL_RANGE[1]:
        return math.log(_S_SPECIAL_A * y - _S_SPECIAL_B)
    ln_y = math.log(y)
    a, b, c, d = _S_COEFFICIENTS
    return ln_y / (a + b * ln_y + c * ln_y**2 + d * ln_y**4)


def _theta_holdup(
    lam: float,
    n_lv: float,
    n_fr: float,
    pattern: FlowPattern,
    theta_deg: float,
    clip_negative_c: bool,
) -> tuple[float, float, float, float]:
    """H_L(0), C, ψ и H_L(θ) (с ограничением 0…1) для одного горизонтального режима."""
    h0 = horizontal_holdup(lam, n_fr, pattern)
    if theta_deg == 0.0:
        c = 0.0
    else:
        c = inclination_coefficient(lam, n_lv, n_fr, pattern, uphill=theta_deg > 0.0)
        if clip_negative_c:
            c = max(c, 0.0)
    psi = inclination_factor(c, theta_deg)
    return h0, c, psi, min(max(h0 * psi, 0.0), 1.0)


class BeggsBrill:
    """Корреляция Beggs–Brill (ICorrelation).

    Атрибуты:
        name: "beggs_brill".
    """

    name: ClassVar[str] = "beggs_brill"

    def __init__(
        self,
        options: BeggsBrillOptions | None = None,
        flow_model: IFlowModel | None = None,
    ) -> None:
        """Создаёт корреляцию.

        Параметры:
            options: настройки (по умолчанию — поведение оригинала статьи).
            flow_model: модуль режима; по умолчанию `BeggsBrillMap` с тем же `revised_map`.
        """
        self._options = options if options is not None else BeggsBrillOptions()
        if flow_model is None:
            map_options = BeggsBrillMapOptions(revised_map=self._options.revised_map)
            flow_model = BeggsBrillMap(map_options)
        self._flow_model = flow_model

    @property
    def options(self) -> BeggsBrillOptions:
        """Настройки корреляции."""
        return self._options

    @property
    def flow_model(self) -> IFlowModel:
        """Используемый модуль режима."""
        return self._flow_model

    def _friction_factor(self, state: FlowState, re_ns: float) -> float:
        """f_ns (Darcy): Colebrook при `rough_pipe_friction`, иначе (13); при Re < 2000 — 64/Re."""
        if re_ns < friction.RE_LAMINAR_MAX:
            return friction.laminar_darcy(re_ns)
        if self._options.rough_pipe_friction:
            return friction.colebrook(re_ns, state.eps_m / state.d_m)
        return friction.bb_smooth(re_ns)

    def calc(self, state: FlowState) -> CorrelationResult:
        """Рассчитывает режим, holdup и градиент давления.

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `CorrelationResult`; details: `lambda_l`, `n_fr`, `n_lv`, `rho_ns_kg_m3`, `mu_ns_pa_s`,
            `re_ns`, `f_ns`, `y`, `s`, `f_tp`, `ek`, `rho_s_kg_m3`, а также `h_l_0`, `c`, `psi`
            (кроме зоны TRANSITION) или `transition_a`, `h_l_segregated`, `h_l_intermittent`.

        Исключения:
            UnsupportedPatternError: модуль режима вернул режим вне карты Beggs–Brill.
        """
        single = single_phase_correlation_result(state)
        if single is not None:
            return single
        flow = self._flow_model.predict(state)
        pattern = flow.pattern
        nums = dimensionless.duns_ros(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.sigma_n_m,
            state.mu_l_pa_s,
        )
        mix = mixture.no_slip(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        )
        lam, v_m = mix.lam, mix.v_m_m_s
        n_fr = dimensionless.froude_number(v_m, state.d_m)
        details: dict[str, float] = {
            "lambda_l": lam,
            "n_fr": n_fr,
            "n_lv": nums.n_lv,
            "rho_ns_kg_m3": mix.rho_ns_kg_m3,
            "mu_ns_pa_s": mix.mu_ns_pa_s,
        }
        clip = self._options.clip_negative_c
        if pattern is FlowPattern.TRANSITION:
            holdup = self._transition_holdup(state, flow.details, lam, nums.n_lv, n_fr, details)
        else:
            h0, c, psi, holdup = _theta_holdup(lam, nums.n_lv, n_fr, pattern, state.theta_deg, clip)
            details.update({"h_l_0": h0, "c": c, "psi": psi})
        holdup = self._payne(holdup, lam, state.theta_deg)

        y = lam / holdup**2 if holdup > 0.0 else math.inf
        s = friction_ratio_exponent(y)
        re_ns = dimensionless.reynolds_number(mix.rho_ns_kg_m3, v_m, state.d_m, mix.mu_ns_pa_s)
        f_ns = self._friction_factor(state, re_ns)
        f_tp = f_ns * math.exp(s)
        rho_s = mixture.slip(
            holdup,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        ).rho_s_kg_m3
        grav = gradient.gravity_gradient(rho_s, math.radians(state.theta_deg))
        fric = f_tp * mix.rho_ns_kg_m3 * v_m**2 / (2.0 * state.d_m)
        ek = gradient.kinetic_number(rho_s, v_m, state.v_sg_m_s, state.p_pa)
        parts = assemble_gradient(
            grav, fric, ek, include_acceleration=self._options.include_acceleration
        )
        details.update(
            {
                "re_ns": re_ns,
                "f_ns": f_ns,
                "y": y,
                "s": s,
                "f_tp": f_tp,
                "ek": ek,
                "rho_s_kg_m3": rho_s,
            }
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

    def _transition_holdup(
        self,
        state: FlowState,
        flow_details: dict[str, float],
        lam: float,
        n_lv: float,
        n_fr: float,
        details: dict[str, float],
    ) -> float:
        """H_L в переходной зоне: A·H_seg + (1 − A)·H_int, A = (L3 − N_FR)/(L3 − L2) "
        "(4.118–4.119)."""
        try:
            l2 = flow_details["l2_froude"]
            l3 = flow_details["l3_froude"]
        except KeyError as error:
            raise UnsupportedPatternError(
                FlowPattern.TRANSITION, "beggs_brill (нет границ l2_froude, l3_froude)"
            ) from error
        a = (l3 - n_fr) / (l3 - l2)
        clip = self._options.clip_negative_c
        theta = state.theta_deg
        h_seg = _theta_holdup(lam, n_lv, n_fr, FlowPattern.SEGREGATED, theta, clip)[3]
        h_int = _theta_holdup(lam, n_lv, n_fr, FlowPattern.INTERMITTENT, theta, clip)[3]
        details.update({"transition_a": a, "h_l_segregated": h_seg, "h_l_intermittent": h_int})
        return min(max(a * h_seg + (1.0 - a) * h_int, 0.0), 1.0)

    def _payne(self, holdup: float, lam: float, theta_deg: float) -> float:
        """Поправка Payne: × 0.924 при θ > 0 (не ниже λ), × 0.685 при θ < 0."""
        if not self._options.payne_correction or theta_deg == 0.0:
            return holdup
        if theta_deg > 0.0:
            corrected = max(_PAYNE_UPHILL * holdup, lam)
        else:
            corrected = _PAYNE_DOWNHILL * holdup
        return min(max(corrected, 0.0), 1.0)
