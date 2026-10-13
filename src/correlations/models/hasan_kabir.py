"""Корреляция Hasan–Kabir для вертикального подъёма в трубе и затрубье (ICorrelation).

Схема расчёта (r = d_t/d, 0 для трубы):
    BUBBLE / DISPERSED_BUBBLE: f_g = v_gs/(C0·v_M + v∞), C0 = 1.2 + 0.371·r (ур. 16–19);
        ρ_M = ρ_g·f_g + ρ_L·(1 − f_g) (21); трение f·ρ_M·v_M²/(2d), f — Moody по Re = ρ_L·v_M·d/μ_L;
    SLUG: f_g = v_gs/(C1·v_M + v∞T) (22); труба: C1 = 1.2, v∞T = C2·√(g·d·Δρ/ρ_L), C2 = 0.345 (27,
      28);
        затрубье: C1 = 1.182 + 0.90·r (29), v∞T = (0.30 + 0.22·r)·√(g·(d − d_t)·Δρ/ρ_L) (30);
        трение f·ρ_L·(1 − f_g)·v_M²/(2d) (31);
    CHURN: как SLUG, но C1 = 1.15 + 0.90·r (Приложение);
    ANNULAR: v_crit (34), E (35)/(36), ρ_c (37b), x = ρ_g·v_gs/(ρ_g·v_gs + E·ρ_L·v_Ls), X (42),
        f_g = (1 + X^0.8)^-0.378 (40), v_g = v_gs/f_g, f_c = 0.079·Re_g^-0.25·[1 + 75·(1 − f_g)]
          (Fanning, 38–39),
        −dp/dL = [ρ_c·g·sinθ + 2·f_c·ρ_c·v_g²/d]/(1 − ρ_c·v_g²/p) (33);
        H_L = (1 − f_g) + f_g·(1 − f_gc), f_gc = v_gs/(v_gs + E·v_Ls) (37a).
    H_L = 1 − f_g для BUBBLE / DISPERSED_BUBBLE / SLUG / CHURN.

Источник: Hasan A.R., Kabir C.S., 1988, SPE Production Engineering 3(2), 263–272 (+ Erratum 1989),
  ур. (1)–(42).
Исправления опечаток (ТЗ): (6) ρ_L, (23) exp((3.37 − N_Eo)/m), (26b) N_f^-0.35, (30) g·(d_c −
  d_t), (37b) E·v_Ls.

Допущения (TODO(question N) ТЗ): ускорение (знаменатель 1 − Ek) — только в ANNULAR (ур. 33); в
  пузырьковом, пробковом и
пульсирующем режимах оно пренебрегается, как в статье. Диаметр трения в затрубье — d (в статье
  эквивалентный d_c − d_t
используется только в v∞T). При θ ≠ 90° расчёт с sinθ и OutOfRangeWarning (модель вертикального
  подъёма).
"""

import math
import warnings
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import annular, bubbles, dimensionless, friction, gradient, mixture
from correlations.core.errors import OutOfRangeWarning, UnsupportedPatternError
from correlations.core.interfaces import IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowState
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel, HasanKabirFlowOptions
from correlations.models._common import assemble_gradient, single_phase_correlation_result

# Коэффициенты (ур. 3, 17, 19, 27, 29, 30, Приложение)
_RISE_COEFFICIENT_150: Final = 1.50
_C0_BASE: Final = 1.2
_C0_ANNULUS_SLOPE: Final = 0.371
_C1_SLUG_PIPE: Final = 1.2
_C1_SLUG_ANNULUS: Final = (1.182, 0.90)
_C1_CHURN: Final = (1.15, 0.90)
_C2_PIPE: Final = bubbles.TAYLOR_COEFFICIENT_HK
_C2_ANNULUS: Final = (0.30, 0.22)
# Показатели ур. (40), (42)
_VOID_X_EXPONENT: Final = 0.8
_VOID_OUTER_EXPONENT: Final = -0.378
_LM_QUALITY_EXPONENT: Final = 0.9
_LM_VISCOSITY_EXPONENT: Final = 0.1
# Re плёнки, ниже которого Steen–Wallis не применима (HK, перед ур. 34)
_FILM_RE_MIN: Final = 3000.0
_VERTICAL_DEG: Final = 90.0
_NON_VERTICAL_MESSAGE: Final = (
    "модель Hasan–Kabir — вертикальный подъём; угол отличен от 90° (гравитация считается с sinθ)"
)
_BUBBLE_PATTERNS: Final = (FlowPattern.BUBBLE, FlowPattern.DISPERSED_BUBBLE)
_SLUG_PATTERNS: Final = (FlowPattern.SLUG, FlowPattern.CHURN)


class HasanKabirOptions(BaseOptions):
    """Флаги корреляции Hasan–Kabir."""

    porteous_low_pressure: bool = Field(
        default=False,
        description=(
            "При p < 1013 кПа граница slug/churn по Porteous (7): v_M = 0.103·√(g·d·Δρ/ρ_g). "
            "Включать для низконапорных скважин и лабораторных данных (так рекомендует текст "
            "статьи). Источник: Hasan, Kabir 1988, ур. 7."
        ),
    )
    wallis_c2: bool = Field(
        default=False,
        description=(
            "C2 скорости пузыря Тейлора по Wallis (23)–(26) вместо 0.345. Включать для вязкой "
            "нефти или малого диаметра: N_f < 300 или N_Eo < 100. Источник: Hasan, Kabir 1988, "
            "ур. 23–26 (Wallis 1969)."
        ),
    )
    rise_coef_150: bool = Field(
        default=False,
        description=(
            "1.50 вместо 1.53 в v∞ для пузырькового H_L (ур. 17). Включать для буквального "
            "воспроизведения ур. 17. Источник: Hasan, Kabir 1988, ур. 3 и 17."
        ),
    )
    churn_as_slug: bool = Field(
        default=False,
        description=(
            "CHURN считается с C1 пробкового режима. Рекомендация Govier–Aziz из текста; "
            "сглаживает скачок slug/churn. Источник: Hasan, Kabir 1988, раздел «Churn Flow»."
        ),
    )
    include_acceleration: bool = Field(
        default=True,
        description=(
            "Знаменатель 1 − ρ_c·v_g²/p в ур. (33) (кольцевой режим). Выключать для оценки вклада "
            "ускорения. Источник: Hasan, Kabir 1988, ур. 33."
        ),
    )
    warn_non_vertical: bool = Field(
        default=True,
        description=(
            "OutOfRangeWarning при θ ≠ 90° (модель создана для вертикального подъёма; гравитация "
            "считается с sinθ). False отключает только это предупреждение, результат не меняется; "
            "отключать при расчёте профиля наклонно-направленной скважины. Источник: Hasan, Kabir "
            "1988 (вертикальный подъём); вопрос 6 ТЗ."
        ),
    )


def gas_void_fraction(v_gs_m_s: float, v_m_m_s: float, c: float, v_rise_m_s: float) -> float:
    """Газосодержание дрейфовой модели f_g = v_gs/(C·v_M + v_rise) (HK 1988, ур. 16b, 22).

    Параметры:
        v_gs_m_s: приведённая скорость газа, м/с.
        v_m_m_s: скорость смеси, м/с.
        c: параметр профиля (C0 пузырькового, C1 пробкового режима).
        v_rise_m_s: скорость всплытия (v∞ или v∞T), м/с.

    Возвращает:
        f_g, безразм.
    """
    return v_gs_m_s / (c * v_m_m_s + v_rise_m_s)


class HasanKabir:
    """Корреляция Hasan–Kabir (ICorrelation).

    Атрибуты:
        name: "hasan_kabir".
    """

    name: ClassVar[str] = "hasan_kabir"

    def __init__(
        self,
        options: HasanKabirOptions | None = None,
        flow_model: IFlowModel | None = None,
    ) -> None:
        """Создаёт корреляцию.

        Параметры:
            options: настройки (по умолчанию — поведение оригинала статьи).
            flow_model: модуль режима; по умолчанию `HasanKabirFlowModel` с тем же
              `porteous_low_pressure`.
        """
        self._options = options if options is not None else HasanKabirOptions()
        if flow_model is None:
            flow_options = HasanKabirFlowOptions(
                porteous_low_pressure=self._options.porteous_low_pressure
            )
            flow_model = HasanKabirFlowModel(flow_options)
        self._flow_model = flow_model

    @property
    def options(self) -> HasanKabirOptions:
        """Настройки корреляции."""
        return self._options

    @property
    def flow_model(self) -> IFlowModel:
        """Используемый модуль режима."""
        return self._flow_model

    def calc(self, state: FlowState) -> CorrelationResult:
        """Рассчитывает режим, holdup и градиент давления.

        Параметры:
            state: состояние потока (СИ); модель вертикального подъёма (θ = 90°).

        Возвращает:
            `CorrelationResult`; details зависят от режима (см. docstring модуля).

        Исключения:
            UnsupportedPatternError: модуль режима вернул режим вне BUBBLE, DISPERSED_BUBBLE, SLUG,
                CHURN, ANNULAR.
        """
        single = single_phase_correlation_result(state)
        if single is not None:
            return single
        pattern = self._flow_model.predict(state).pattern
        if pattern is FlowPattern.ANNULAR:
            computed = self._annular(state)
        elif pattern in _BUBBLE_PATTERNS:
            computed = self._bubble(state)
        elif pattern in _SLUG_PATTERNS:
            computed = self._slug(state, churn=pattern is FlowPattern.CHURN)
        else:
            raise UnsupportedPatternError(pattern, "hasan_kabir")
        if self._options.warn_non_vertical and state.theta_deg != _VERTICAL_DEG:
            # Текст постоянный (без θ): фильтр "default" показывает его один раз на место вызова;
            # угол — в context предупреждения.
            warnings.warn(
                OutOfRangeWarning(_NON_VERTICAL_MESSAGE, {"theta_deg": state.theta_deg}),
                stacklevel=2,
            )
        holdup, grav, fric, ek, details = computed
        parts = assemble_gradient(
            grav,
            fric,
            ek,
            include_acceleration=self._options.include_acceleration
            and pattern is FlowPattern.ANNULAR,
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

    def _mixture_friction_factor(self, state: FlowState, v_m: float) -> tuple[float, float]:
        """Re = ρ_L·v_M·d/μ_L и коэффициент Moody f (HK, ур. 14, 31)."""
        re = dimensionless.reynolds_number(state.rho_l_kg_m3, v_m, state.d_m, state.mu_l_pa_s)
        return re, friction.moody(re, state.eps_m / state.d_m)

    def _bubble(self, state: FlowState) -> tuple[float, float, float, float, dict[str, float]]:
        """BUBBLE / DISPERSED_BUBBLE: (H_L, гравитация, трение, Ek, details)."""
        v_m = state.v_sl_m_s + state.v_sg_m_s
        coefficient = _RISE_COEFFICIENT_150 if self._options.rise_coef_150 else None
        if coefficient is None:
            v_inf = bubbles.harmathy_rise_velocity(
                state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m
            )
        else:
            v_inf = bubbles.harmathy_rise_velocity(
                state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m, coef=coefficient
            )
        c0 = _C0_BASE + _C0_ANNULUS_SLOPE * state.d_t_m / state.d_m
        f_g = gas_void_fraction(state.v_sg_m_s, v_m, c0, v_inf)
        holdup = 1.0 - f_g
        rho_m = self._slip_density(state, holdup)
        re, f_darcy = self._mixture_friction_factor(state, v_m)
        fric = gradient.friction_gradient(f_darcy, rho_m, v_m, state.d_m)
        grav = gradient.gravity_gradient(rho_m, math.radians(state.theta_deg))
        details = {
            "v_m_m_s": v_m,
            "v_inf_m_s": v_inf,
            "c0": c0,
            "f_g": f_g,
            "rho_m_kg_m3": rho_m,
            "re_m": re,
            "f_darcy": f_darcy,
        }
        return holdup, grav, fric, 0.0, details

    def _slug(
        self, state: FlowState, *, churn: bool
    ) -> tuple[float, float, float, float, dict[str, float]]:
        """SLUG / CHURN: (H_L, гравитация, трение, Ek, details)."""
        v_m = state.v_sl_m_s + state.v_sg_m_s
        rho_l, rho_g = state.rho_l_kg_m3, state.rho_g_kg_m3
        r = state.d_t_m / state.d_m
        annulus = state.d_t_m > 0.0
        if churn and not self._options.churn_as_slug:
            c1 = _C1_CHURN[0] + _C1_CHURN[1] * r
        elif annulus:
            c1 = _C1_SLUG_ANNULUS[0] + _C1_SLUG_ANNULUS[1] * r
        else:
            c1 = _C1_SLUG_PIPE
        if annulus:
            c2 = _C2_ANNULUS[0] + _C2_ANNULUS[1] * r
            v_inf_t = bubbles.taylor_rise_velocity(state.d_m - state.d_t_m, rho_l, rho_g, coef=c2)
        else:
            c2 = self._pipe_taylor_coefficient(state)
            v_inf_t = bubbles.taylor_rise_velocity(state.d_m, rho_l, rho_g, coef=c2)
        f_g = gas_void_fraction(state.v_sg_m_s, v_m, c1, v_inf_t)
        holdup = 1.0 - f_g
        rho_m = self._slip_density(state, holdup)
        re, f_darcy = self._mixture_friction_factor(state, v_m)
        fric = gradient.friction_gradient(f_darcy, rho_l * holdup, v_m, state.d_m)
        grav = gradient.gravity_gradient(rho_m, math.radians(state.theta_deg))
        details = {
            "v_m_m_s": v_m,
            "v_inf_m_s": bubbles.harmathy_rise_velocity(rho_l, rho_g, state.sigma_n_m),
            "c1": c1,
            "c2": c2,
            "v_inf_t_m_s": v_inf_t,
            "f_g": f_g,
            "rho_m_kg_m3": rho_m,
            "re_m": re,
            "f_darcy": f_darcy,
        }
        return holdup, grav, fric, 0.0, details

    def _pipe_taylor_coefficient(self, state: FlowState) -> float:
        """C2 трубы: 0.345 (ур. 27) или по Wallis (ур. 23, 26) при `wallis_c2`."""
        if not self._options.wallis_c2:
            return _C2_PIPE
        n_f = bubbles.inverse_viscosity_number(
            state.d_m, state.rho_l_kg_m3, state.rho_g_kg_m3, state.mu_l_pa_s
        )
        n_eo = bubbles.eotvos_number(
            state.d_m, state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m
        )
        return bubbles.wallis_taylor_coefficient(n_f, n_eo)

    @staticmethod
    def _slip_density(state: FlowState, holdup: float) -> float:
        """ρ_M = ρ_g·f_g + ρ_L·(1 − f_g) (ур. 21)."""
        return mixture.slip(
            holdup,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        ).rho_s_kg_m3

    def _annular(self, state: FlowState) -> tuple[float, float, float, float, dict[str, float]]:
        """ANNULAR: (H_L, гравитация, трение, Ek, details) по ур. (33)–(42)."""
        v_sl, v_sg, d = state.v_sl_m_s, state.v_sg_m_s, state.d_m
        rho_l, rho_g = state.rho_l_kg_m3, state.rho_g_kg_m3
        v_crit = annular.critical_velocity(v_sg, state.mu_g_pa_s, state.sigma_n_m, rho_g, rho_l)
        entrainment = annular.entrainment_steen_wallis(v_crit)  # (35), (36)
        film_re = dimensionless.reynolds_number(rho_l, v_sl, d, state.mu_l_pa_s)
        if film_re < _FILM_RE_MIN:
            warnings.warn(
                f"Re плёнки = {film_re:.0f} < {_FILM_RE_MIN:.0f}: корреляция Steen–Wallis вне "
                "области применимости (HK 1988, перед ур. 34)",
                OutOfRangeWarning,
                stacklevel=3,
            )
        f_gc = 1.0 - annular.core_liquid_fraction(entrainment, v_sl, v_sg)  # (37a)
        rho_c = mixture.slip(
            1.0 - f_gc, rho_l, rho_g, state.mu_l_pa_s, state.mu_g_pa_s
        ).rho_s_kg_m3  # (37b); typo fix: E·v_Ls
        gas_flux = rho_g * v_sg
        x = gas_flux / (gas_flux + entrainment * rho_l * v_sl)
        x_lm = (
            ((1.0 - x) / x) ** _LM_QUALITY_EXPONENT
            * math.sqrt(rho_g / rho_l)
            * (state.mu_l_pa_s / state.mu_g_pa_s) ** _LM_VISCOSITY_EXPONENT
        )  # (42)
        f_g = (1.0 + x_lm**_VOID_X_EXPONENT) ** _VOID_OUTER_EXPONENT  # (40)
        v_g = v_sg / f_g
        re_g = dimensionless.reynolds_number(rho_g, v_sg, d, state.mu_g_pa_s)
        f_c = friction.fanning_blasius(re_g) * annular.wallis_z_from_holdup(1.0 - f_g)  # (39)
        fric = gradient.friction_gradient(friction.fanning_to_darcy(f_c), rho_c, v_g, d)
        grav = gradient.gravity_gradient(rho_c, math.radians(state.theta_deg))
        ek = gradient.kinetic_number(rho_c, v_g, v_g, state.p_pa)
        holdup = (1.0 - f_g) + f_g * (1.0 - f_gc)
        details = {
            "v_m_m_s": v_sl + v_sg,
            "v_inf_m_s": bubbles.harmathy_rise_velocity(rho_l, rho_g, state.sigma_n_m),
            "v_crit": v_crit,
            "e": entrainment,
            "f_gc": f_gc,
            "rho_c_kg_m3": rho_c,
            "x": x,
            "x_lm": x_lm,
            "f_g": f_g,
            "v_g_m_s": v_g,
            "re_g": re_g,
            "f_c_fanning": f_c,
            "ek": ek,
        }
        return holdup, grav, fric, ek, details
