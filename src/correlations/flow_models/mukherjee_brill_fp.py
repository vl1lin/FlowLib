"""Карта режимов Mukherjee–Brill для наклонных труб (модуль режима, IFlowModel).

Схема расчёта (блок-схема рис. 16 статьи; θ > 0 — подъём):
    1. N_Lv, N_gv, N_L (числа Данса–Роса, N_L с g);
    2. N_gv > N_gvSM [2] → ANNULAR (граница не зависит от угла);
    3. θ > 0: N_Lv > N_LvBS [1] → BUBBLE, иначе SLUG;
    4. θ ≤ 0 и |θ| > 30°: N_gv > N_gvBS [3] → (N_Lv > N_LvST [4] → SLUG, иначе STRATIFIED), иначе
      BUBBLE;
    5. θ ≤ 0 и |θ| ≤ 30°: N_Lv > N_LvST [4] → (N_gv > N_gvBS [3] → SLUG, иначе BUBBLE), иначе
      STRATIFIED.

Флагов нет. Опечатки оригинала: в [3] напечатано «sin 0» — взято sinθ (ТЗ, «Опечатки»);
член −0.017·N_gv в [4] линейный в обоих оригиналах — не исправляется.

Источник: Mukherjee H., Brill J.P., 1985, Int. J. Multiphase Flow 11(3), 299–315, ур. [1]–[4],
  рис. 16.
"""

import math
from typing import ClassVar, Final

from correlations.core import dimensionless
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# [2]: lg N_gvSM = 1.401 − 2.694·N_L + 0.521·N_Lv^0.329
_SM_A: Final = 1.401
_SM_B: Final = -2.694
_SM_C: Final = 0.521
_SM_POWER: Final = 0.329
# [1]: lg N_LvBS = lg N_gv + 0.940 + 0.074·sinθ − 0.855·sin²θ + 3.695·N_L
_BS_UP: Final = (0.940, 0.074, -0.855, 3.695)
# [3]: lg N_gvBS = 0.431 + 1.132·sinθ − 3.003·N_L − 1.138·lg(N_Lv)·sinθ − 0.429·lg²(N_Lv)·sinθ
_BS_DOWN: Final = (0.431, 1.132, -3.003, -1.138, -0.429)
# [4]: lg N_LvST = 0.321 − 0.017·N_gv − 4.267·sinθ − 2.972·N_L − 0.033·lg²(N_gv) − 3.925·sin²θ
_ST: Final = (0.321, -0.017, -4.267, -2.972, -0.033, -3.925)
# Граница ветвей спуска по |θ|, градусы
_STEEP_DOWNFLOW_DEG: Final = 30.0


class MukherjeeBrillFlowOptions(BaseOptions):
    """Флаги карты режимов Mukherjee–Brill: флагов нет (ТЗ)."""


def n_gv_slug_annular(n_l: float, n_lv: float) -> float:
    """Граница пробковый → кольцевой N_gvSM, [2] (не зависит от θ).

    Параметры:
        n_l: число вязкости жидкости N_L.
        n_lv: число скорости жидкости N_Lv.

    Возвращает:
        N_gvSM, безразм.
    """
    return 10.0 ** (_SM_A + _SM_B * n_l + _SM_C * n_lv**_SM_POWER)


def n_lv_bubble_slug(n_gv: float, n_l: float, theta_rad: float) -> float:
    """Граница пузырьковый → пробковый при подъёме N_LvBS, [1].

    Параметры:
        n_gv: число скорости газа N_gv (> 0).
        n_l: число вязкости жидкости N_L.
        theta_rad: угол от горизонтали, рад (> 0 — подъём).

    Возвращает:
        N_LvBS, безразм.
    """
    s = math.sin(theta_rad)
    a, b, c, d = _BS_UP
    return 10.0 ** (math.log10(n_gv) + a + b * s + c * s**2 + d * n_l)


def n_gv_bubble_slug(n_lv: float, n_l: float, theta_rad: float) -> float:
    """Граница пузырьковый → пробковый при спуске и в горизонтали N_gvBS, [3].

    # typo fix: в IJMF [3] напечатано «sin 0» в члене с 1.138 — взято sinθ (JERT (8); ТЗ,
    # «Опечатки»).

    Параметры:
        n_lv: число скорости жидкости N_Lv (> 0).
        n_l: число вязкости жидкости N_L.
        theta_rad: угол от горизонтали, рад (≤ 0 — спуск или горизонталь).

    Возвращает:
        N_gvBS, безразм.
    """
    s = math.sin(theta_rad)
    lg = math.log10(n_lv)
    a, b, c, d, e = _BS_DOWN
    return 10.0 ** (a + b * s + c * n_l + d * lg * s + e * lg**2 * s)


def n_lv_stratified(n_gv: float, n_l: float, theta_rad: float) -> float:
    """Граница расслоённого потока при спуске и в горизонтали N_LvST, [4].

    Член −0.017·N_gv — линейный (так в обоих оригиналах, не исправляется).

    Параметры:
        n_gv: число скорости газа N_gv (> 0).
        n_l: число вязкости жидкости N_L.
        theta_rad: угол от горизонтали, рад (≤ 0).

    Возвращает:
        N_LvST, безразм.
    """
    s = math.sin(theta_rad)
    a, b, c, d, e, f = _ST
    return 10.0 ** (a + b * n_gv + c * s + d * n_l + e * math.log10(n_gv) ** 2 + f * s**2)


class MukherjeeBrillFlowModel:
    """Модуль режима Mukherjee–Brill (IFlowModel).

    Атрибуты:
        name: "mukherjee_brill_fp".
    """

    name: ClassVar[str] = "mukherjee_brill_fp"

    def __init__(self, options: MukherjeeBrillFlowOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (флагов нет).
        """
        self._options = options if options is not None else MukherjeeBrillFlowOptions()

    @property
    def options(self) -> MukherjeeBrillFlowOptions:
        """Настройки модели."""
        return self._options

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по блок-схеме рис. 16.

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `FlowPatternResult`: режим и details (`n_lv`, `n_gv`, `n_l`, `n_gv_sm` и
            границы использованной ветви: `n_lv_bs` | `n_gv_bs`, `n_lv_st`).
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        nums = dimensionless.duns_ros(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.sigma_n_m,
            state.mu_l_pa_s,
        )
        n_lv, n_gv, n_l = nums.n_lv, nums.n_gv, nums.n_l
        n_gv_sm = n_gv_slug_annular(n_l, n_lv)
        details: dict[str, float] = {"n_lv": n_lv, "n_gv": n_gv, "n_l": n_l, "n_gv_sm": n_gv_sm}
        if n_gv > n_gv_sm:
            return FlowPatternResult(pattern=FlowPattern.ANNULAR, details=details)
        theta = math.radians(state.theta_deg)
        if state.theta_deg > 0.0:
            n_lv_bs = n_lv_bubble_slug(n_gv, n_l, theta)
            details["n_lv_bs"] = n_lv_bs
            pattern = FlowPattern.BUBBLE if n_lv > n_lv_bs else FlowPattern.SLUG
            return FlowPatternResult(pattern=pattern, details=details)
        n_gv_bs = n_gv_bubble_slug(n_lv, n_l, theta)
        n_lv_st = n_lv_stratified(n_gv, n_l, theta)
        details["n_gv_bs"] = n_gv_bs
        details["n_lv_st"] = n_lv_st
        if abs(state.theta_deg) > _STEEP_DOWNFLOW_DEG:
            if n_gv > n_gv_bs:
                pattern = FlowPattern.SLUG if n_lv > n_lv_st else FlowPattern.STRATIFIED
            else:
                pattern = FlowPattern.BUBBLE
        elif n_lv > n_lv_st:
            pattern = FlowPattern.SLUG if n_gv > n_gv_bs else FlowPattern.BUBBLE
        else:
            pattern = FlowPattern.STRATIFIED
        return FlowPatternResult(pattern=pattern, details=details)
