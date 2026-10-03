"""
Механистическая модель Беггса-Брилла для определения режима потока в
горизонтальных и наклонных трубах (упрощённая классификация,
без расчёта удержания и градиента давления)

Если нужна повышенная точность для расчета горизонтальной трубы: см.
Если нужна повышенная точность для расчета наклонной трубы: см.
"""

from correlations.base import (
    ClassificationResult,
    FlowPatternCode,
    IFlowModel,
    check_velocities,
    single_phase_pattern,
)
from correlations.constants import GRAVITY

_FP_SEG = 0
_FP_INT = 1
_FP_DIST = 2
_FP_TRANS = 3

_FLOW_PATTERN_MAP = {
    _FP_SEG: FlowPatternCode.STRATIFIED,
    _FP_TRANS: FlowPatternCode.SLUG,
    _FP_INT: FlowPatternCode.SLUG,
    _FP_DIST: FlowPatternCode.DISPERSED_BUBBLE,
}


class BeggsBrillModel(IFlowModel):
    """
    Модель Беггса-Брилла для расчета режима потока (угол от −90° до 90°).
    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    """

    name = "Beggs-Brill"
    angle_range = (-90.0, 90.0)

    def classify(self, vsl: float, vsg: float) -> ClassificationResult:
        """
        Определяет режим течения по значениям приведённых скоростей.
        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        """
        check_velocities(vsl, vsg)
        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return ClassificationResult(single_phase)

        diameter = self.pipe.diameter
        vm = vsl + vsg
        lam_l = vsl / vm
        n_fr = vm**2 / (GRAVITY * diameter)

        fp = self._flow_pattern(lam_l, n_fr)
        return ClassificationResult(_FLOW_PATTERN_MAP[fp])

    def _flow_pattern(self, lam_l: float, n_fr: float) -> int:
        """
        Определение базового режима потока по карте Beggs-Brill.
        :param lam_l: безотрывное удержание жидкости (vsl / (vsl+vsg))
        :param n_fr: число Фруда смеси
        """
        if n_fr >= 316.0 * lam_l**0.302 or n_fr >= 0.5 * lam_l**-6.738:
            return _FP_DIST
        if n_fr <= 0.000925 * lam_l**-2.468:
            return _FP_SEG
        if n_fr <= 0.1 * lam_l**-1.452:
            return _FP_TRANS
        return _FP_INT
