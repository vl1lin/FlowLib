"""Модуль режимов Hasan–Kabir для вертикального подъёма (IFlowModel).

Схема расчёта (r = d_t/d, 0 для трубы; C0 = 1.2 + 0.371·r):
    (а) v_M > v_M,DB (ур. 6) и f_g,bubbly = v_gs/(C0·v_M + v∞) < 0.52 → DISPERSED_BUBBLE;
    (б) v_gs < α·(C0·v_Ls + v∞)/(1 − α·C0), α = 0.25 (ур. 1–2) → BUBBLE;
    (в) v_gs > 3.1·[σ·g·Δρ/ρ_g²]^0.25 (ур. 13) → ANNULAR;
    (г) иначе граница slug/churn:
        ρ_L·v_Ls² > 3300 lbm/(ft·s²) → CHURN и OutOfRangeWarning;
        porteous_low_pressure и p < 1013 кПа: v_M ≥ 0.103·√(g·d·Δρ/ρ_g) (ур. 7) → CHURN, иначе SLUG;
        иначе ρ_g·v_gs² выше границы Hewitt–Roberts (ур. 8, 9) → CHURN, иначе SLUG.

Источник: Hasan A.R., Kabir C.S., 1988, SPE Production Engineering 3(2), 263–272 (+ Erratum 1989),
ур. (1)–(13). Опечатка (ТЗ): в (6) ρ_M → ρ_L (Erratum 1989; реализовано в `core.transitions`).

Допущения (TODO(question 4) ТЗ): порядок проверок (а)–(г) как указан; при ρ_L·v_Ls² > 3300
lbm/(ft·s²) вне DB и ANNULAR режим CHURN с предупреждением.
"""

import math
import warnings
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import bubbles, transitions
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import OutOfRangeWarning
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# 1 lbm/(ft·s²) = 1.488 кг/(м·с²) (HK, ур. 8–9 даны в промысловых единицах)
LBM_FT_S2_IN_SI: Final = 1.488
# Hewitt–Roberts: (8) ρ_g·v_gs² = 0.00673·(ρ_L·v_Ls²)^1.7 при ρ_L·v_Ls² < 50;
# (9) ρ_g·v_gs² = 17.1·lg(ρ_L·v_Ls²) − 23.2 при 50 < ρ_L·v_Ls² < 3300 (lbm/(ft·s²))
_HR_LOW_COEFFICIENT: Final = 0.00673
_HR_LOW_EXPONENT: Final = 1.7
_HR_SPLIT_LBM: Final = 50.0
_HR_HIGH_COEFFICIENT: Final = 17.1
_HR_HIGH_OFFSET: Final = 23.2
_HR_MAX_LBM: Final = 3300.0
# Porteous (7): v_M = 0.103·√(g·d·Δρ/ρ_g); применяется при p < 10 атм
_PORTEOUS_COEFFICIENT: Final = 0.103
_PORTEOUS_MAX_PRESSURE_PA: Final = 1.013e6
# HK: C0 = 1.2 + 0.371·(d_t/d_c) (ур. 19)
_C0_BASE: Final = 1.2
_C0_ANNULUS_SLOPE: Final = 0.371
# Предельное газосодержание дисперсно-пузырькового режима (Taitel 1980)
_DISPERSED_MAX_VOID: Final = 0.52


class HasanKabirFlowOptions(BaseOptions):
    """Флаги модуля режима Hasan–Kabir."""

    porteous_low_pressure: bool = Field(
        default=False,
        description=(
            "При p < 1013 кПа граница slug/churn по Porteous (7): v_M = 0.103·√(g·d·Δρ/ρ_g) вместо "
            "карты Hewitt–Roberts. Включать для низконапорных скважин и лабораторных данных (так "
            "рекомендует текст статьи). Источник: Hasan, Kabir 1988, ур. 7."
        ),
    )


def hewitt_roberts_boundary_lbm(rho_l_vls_sq_lbm: float) -> float:
    """Граница slug/churn по Hewitt–Roberts: значение ρ_g·v_gs² (HK 1988, ур. 8, 9).

    Параметры:
        rho_l_vls_sq_lbm: ρ_L·v_Ls², lbm/(ft·s²).

    Возвращает:
        ρ_g·v_gs² на границе, lbm/(ft·s²); при ρ_L·v_Ls² > 3300 — +∞ (выше — только DB или
          кольцевой).
    """
    if rho_l_vls_sq_lbm < _HR_SPLIT_LBM:
        return _HR_LOW_COEFFICIENT * rho_l_vls_sq_lbm**_HR_LOW_EXPONENT
    if rho_l_vls_sq_lbm <= _HR_MAX_LBM:
        return _HR_HIGH_COEFFICIENT * math.log10(rho_l_vls_sq_lbm) - _HR_HIGH_OFFSET
    return math.inf


def porteous_velocity(d_m: float, rho_l_kg_m3: float, rho_g_kg_m3: float) -> float:
    """Скорость смеси перехода slug → churn по Porteous v_M = 0.103·√(g·d·Δρ/ρ_g) (HK, ур. 7).

    Параметры:
        d_m: диаметр, м.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.

    Возвращает:
        v_M, м/с.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return _PORTEOUS_COEFFICIENT * math.sqrt(GRAVITY_M_S2 * d_m * delta_rho / rho_g_kg_m3)


class HasanKabirFlowModel:
    """Модуль режима Hasan–Kabir (IFlowModel).

    Атрибуты:
        name: "hasan_kabir_fp".
    """

    name: ClassVar[str] = "hasan_kabir_fp"

    def __init__(self, options: HasanKabirFlowOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (по умолчанию — карта Hewitt–Roberts).
        """
        self._options = options if options is not None else HasanKabirFlowOptions()

    @property
    def options(self) -> HasanKabirFlowOptions:
        """Настройки модели."""
        return self._options

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по порядку проверок (а)–(г).

        Параметры:
            state: состояние потока (СИ); угол не используется (модель вертикального подъёма).

        Возвращает:
            `FlowPatternResult`: режим и details (`v_inf_m_s`, `v_gs_bubble_slug_m_s`,
            `v_m_dispersed_m_s`, `v_gs_annular_m_s`, `f_g_bubbly`, `hr_rho_l_vls_sq_lbm`,
            `hr_rho_g_vgs_sq_lbm`, `hr_boundary_lbm`, `v_m_porteous_m_s`).
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        rho_l, rho_g = state.rho_l_kg_m3, state.rho_g_kg_m3
        v_sl, v_sg, d = state.v_sl_m_s, state.v_sg_m_s, state.d_m
        v_m = v_sl + v_sg
        c0 = _C0_BASE + _C0_ANNULUS_SLOPE * state.d_t_m / d
        v_inf = bubbles.harmathy_rise_velocity(rho_l, rho_g, state.sigma_n_m)
        v_gs_bs = transitions.bubble_slug_drift_flux(v_sl, v_inf, c0=c0)
        v_m_db = transitions.dispersed_bubble_mixture_velocity_taitel1980(
            d, rho_l, rho_g, state.sigma_n_m, state.mu_l_pa_s
        )
        v_gs_annular = transitions.annular_turner_velocity(rho_l, rho_g, state.sigma_n_m)
        f_g_bubbly = v_sg / (c0 * v_m + v_inf)
        x_l = rho_l * v_sl**2 / LBM_FT_S2_IN_SI
        x_g = rho_g * v_sg**2 / LBM_FT_S2_IN_SI
        boundary = hewitt_roberts_boundary_lbm(x_l)
        v_m_porteous = porteous_velocity(d, rho_l, rho_g)
        details = {
            "v_inf_m_s": v_inf,
            "v_gs_bubble_slug_m_s": v_gs_bs,
            "v_m_dispersed_m_s": v_m_db,
            "v_gs_annular_m_s": v_gs_annular,
            "f_g_bubbly": f_g_bubbly,
            "hr_rho_l_vls_sq_lbm": x_l,
            "hr_rho_g_vgs_sq_lbm": x_g,
            "hr_boundary_lbm": boundary,
            "v_m_porteous_m_s": v_m_porteous,
        }
        if v_m > v_m_db and f_g_bubbly < _DISPERSED_MAX_VOID:
            pattern = FlowPattern.DISPERSED_BUBBLE
        elif v_sg < v_gs_bs:
            pattern = FlowPattern.BUBBLE
        elif v_sg > v_gs_annular:
            pattern = FlowPattern.ANNULAR
        elif x_l > _HR_MAX_LBM:
            warnings.warn(
                f"ρ_L·v_Ls² = {x_l:.0f} lbm/(ft·s²) > {_HR_MAX_LBM:.0f}: вне DB и "
                "кольцевого режима; принят CHURN (HK 1988, карта Hewitt–Roberts)",
                OutOfRangeWarning,
                stacklevel=2,
            )
            pattern = FlowPattern.CHURN
        elif self._options.porteous_low_pressure and state.p_pa < _PORTEOUS_MAX_PRESSURE_PA:
            pattern = FlowPattern.CHURN if v_m >= v_m_porteous else FlowPattern.SLUG
        else:
            pattern = FlowPattern.CHURN if x_g > boundary else FlowPattern.SLUG
        return FlowPatternResult(pattern=pattern, details=details)
