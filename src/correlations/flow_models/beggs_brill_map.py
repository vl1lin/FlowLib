"""Карта режимов Beggs–Brill для горизонтального потока (модуль режима, IFlowModel).

Схема расчёта:
    1. λ_L = v_sl/v_m, N_FR = v_m²/(g·d);
    2. оригинал (1973): X = ln λ; L1 (9), L2 (10): N_FR < L1 → SEGREGATED;
       N_FR > L1 и > L2 → DISTRIBUTED; L1 < N_FR < L2 → INTERMITTENT;
    3. revised_map: границы L1…L4 (монография, 4.110–4.113) с переходной зоной TRANSITION.

Режим не зависит от угла: карта построена для горизонтали, а режим определяет, по каким
коэффициентам считается holdup коррелятором.

Источники:
    * Beggs H.D., Brill J.P., 1973, JPT 25(5), ур. 9–10 — оригинальная карта;
    * Brill J.P., Mukherjee H., «Multiphase Flow in Wells», 1999, ур. 4.110–4.119 — revised-карта
      (единственное допустимое использование монографии как источника формул, ТЗ).
"""

import math
from typing import ClassVar, Final

from pydantic import Field

from correlations.core.constants import GRAVITY_M_S2
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# BB 1973, ур. 9: L1 = exp(−4.62 − 3.757·X − 0.481·X² − 0.0207·X³)
_L1_COEFFICIENTS: Final = (-4.62, -3.757, -0.481, -0.0207)
# BB 1973, ур. 10: L2 = exp(1.061 − 4.602·X − 1.609·X² − 0.179·X³ + 0.635e-3·X⁵)
_L2_COEFFICIENTS: Final = (1.061, -4.602, -1.609, -0.179)
_L2_X5_COEFFICIENT: Final = 0.635e-3
# Монография 4.110–4.113: L1 = 316·λ^0.302, L2 = 0.000925·λ^-2.468, L3 = 0.10·λ^-1.452, L4 =
# 0.5·λ^-6.738
_REV_L1: Final = (316.0, 0.302)
_REV_L2: Final = (0.000925, -2.468)
_REV_L3: Final = (0.10, -1.452)
_REV_L4: Final = (0.5, -6.738)
# Границы по λ_L в определении режимов revised-карты
_LAMBDA_SEGREGATED_SPLIT: Final = 0.01
_LAMBDA_DISTRIBUTED_SPLIT: Final = 0.4
# Верхняя граница показателя экспоненты: выше — результат считается +∞ (полиномы L1, L2 при λ → 0)
_MAX_EXPONENT: Final = 700.0


def _safe_exp(value: float) -> float:
    """exp(value) без OverflowError: при value > 700 возвращает +∞."""
    return math.exp(min(value, _MAX_EXPONENT))


class BeggsBrillMapOptions(BaseOptions):
    """Флаги карты режимов Beggs–Brill."""

    revised_map: bool = Field(
        default=False,
        description=(
            "Карта L1–L4 с переходной зоной TRANSITION (монография, 4.110–4.119) вместо "
            "оригинальной карты 1973 (ур. 9–10). Включать для сверки с PIPESIM «BB Revised»: "
            "убирает скачок H_L на границе segregated/intermittent. Источник: Brill, Mukherjee "
            "1999."
        ),
    )


def froude_boundaries_original(lam: float) -> tuple[float, float]:
    """Границы L1 и L2 оригинальной карты (BB 1973, ур. 9–10).

    Параметры:
        lam: доля жидкости без проскальзывания λ_L (0 < λ_L < 1; при λ_L = 1 X = 0).

    Возвращает:
        Пара (L1, L2) по числу Фруда N_FR, безразм.
    """
    x = math.log(lam)
    a, b, c, d = _L1_COEFFICIENTS
    l1 = _safe_exp(a + b * x + c * x**2 + d * x**3)
    a2, b2, c2, d2 = _L2_COEFFICIENTS
    l2 = _safe_exp(a2 + b2 * x + c2 * x**2 + d2 * x**3 + _L2_X5_COEFFICIENT * x**5)
    return l1, l2


def froude_boundaries_revised(lam: float) -> tuple[float, float, float, float]:
    """Границы L1…L4 revised-карты (монография, 4.110–4.113).

    Параметры:
        lam: доля жидкости без проскальзывания λ_L (> 0).

    Возвращает:
        Четвёрка (L1, L2, L3, L4) по числу Фруда N_FR, безразм.
    """
    ln_lam = math.log(lam)
    return (
        _REV_L1[0] * _safe_exp(_REV_L1[1] * ln_lam),
        _REV_L2[0] * _safe_exp(_REV_L2[1] * ln_lam),
        _REV_L3[0] * _safe_exp(_REV_L3[1] * ln_lam),
        _REV_L4[0] * _safe_exp(_REV_L4[1] * ln_lam),
    )


def _original_pattern(n_fr: float, l1: float, l2: float) -> FlowPattern:
    """Режим по оригинальной карте: BB 1973, правила 1–3 к ур. 9–10."""
    if n_fr < l1:
        return FlowPattern.SEGREGATED
    if n_fr > l2:
        return FlowPattern.DISTRIBUTED
    return FlowPattern.INTERMITTENT


def _revised_pattern(
    n_fr: float, lam: float, bounds: tuple[float, float, float, float]
) -> FlowPattern:
    """Режим по revised-карте (монография, неравенства после 4.113)."""
    l1, l2, l3, l4 = bounds
    if lam < _LAMBDA_SEGREGATED_SPLIT:
        return FlowPattern.SEGREGATED if n_fr < l1 else FlowPattern.DISTRIBUTED
    if n_fr < l2:
        return FlowPattern.SEGREGATED
    if n_fr < l3:
        return FlowPattern.TRANSITION
    upper = l1 if lam < _LAMBDA_DISTRIBUTED_SPLIT else l4
    return FlowPattern.INTERMITTENT if n_fr <= upper else FlowPattern.DISTRIBUTED


class BeggsBrillMap:
    """Модуль режима Beggs–Brill (IFlowModel).

    Атрибуты:
        name: "beggs_brill_map".
    """

    name: ClassVar[str] = "beggs_brill_map"

    def __init__(self, options: BeggsBrillMapOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (по умолчанию — оригинальная карта 1973).
        """
        self._options = options if options is not None else BeggsBrillMapOptions()

    @property
    def options(self) -> BeggsBrillMapOptions:
        """Настройки модели."""
        return self._options

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по карте Beggs–Brill.

        Параметры:
            state: состояние потока (СИ); угол не используется.

        Возвращает:
            `FlowPatternResult`: режим и details (`lambda_l`, `n_fr`, границы Фруда:
            оригинал — `l1_froude`, `l2_froude`; revised — `l1_froude`…`l4_froude`).
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        v_m = state.v_sl_m_s + state.v_sg_m_s
        lam = state.v_sl_m_s / v_m
        n_fr = v_m**2 / (GRAVITY_M_S2 * state.d_m)
        details: dict[str, float] = {"lambda_l": lam, "n_fr": n_fr}
        if self._options.revised_map:
            bounds = froude_boundaries_revised(lam)
            for index, value in enumerate(bounds, start=1):
                details[f"l{index}_froude"] = value
            pattern = _revised_pattern(n_fr, lam, bounds)
        else:
            l1, l2 = froude_boundaries_original(lam)
            details["l1_froude"] = l1
            details["l2_froude"] = l2
            pattern = _original_pattern(n_fr, l1, l2)
        return FlowPatternResult(pattern=pattern, details=details)
