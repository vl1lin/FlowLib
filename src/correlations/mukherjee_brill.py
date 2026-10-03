"""
Эмпирическая модель режимов течения Mukherjee–Brill (1985) для наклонных труб.

Источники: Mukherjee H., Brill J.P. Empirical equations to predict flow patterns
in two-phase inclined flow // Int. J. Multiphase Flow. 1985. 11(3):299–315;
Брилл Дж. П., Мукерджи Х. «Многофазный поток в скважинах», §4.2.1 «Метод
Мукерджи и Брилла», ур. 4.128–4.134, рис. 4.19, пример 4.8. Номера уравнений
в комментариях и docstring — по этой книге.

Диапазон применимости: модель получена на трубе диаметром 38 мм (системы
воздух–керосин и воздух–смазочное масло) для углов от −90° до +90°. Расчёт на
других диаметрах и флюидах — экстраполяция. Реализовано только определение
режима; удержание жидкости и градиент давления не рассчитываются.

Возвращаемые режимы: BUBBLE (пузырьковый), SLUG (пробковый), ANNULAR
(кольцевой/эмульсионный), STRATIFIED (расслоённый), а также SINGLE_GAS и
SINGLE_LIQUID для однофазных случаев. Модель не различает гладкое и волновое
расслоённое течение и не выделяет дисперсно-пузырьковый режим.

Ограничения и неоднозначности источника:

* Реализована схема выбора режима по рис. 4.19. В крутом нисходящем потоке
  (θ < −30°) пузырькового режима нет: при N_gv ≤ N_gv,BS течение пробковое.
* Текст книги («при низких дебитах переход из пузырькового в расслоённый режим
  происходит, когда угол ниже −30°») со схемой расходится; реализована схема.
* При θ < −30° на срезе по v_SL возможна последовательность
  SLUG → STRATIFIED → SLUG по мере роста v_Sg. Это свойство модели.
* При θ ≤ 0 граница «пузырьковый/пробковый» задаётся по газу: N_gv против
  N_gv,BS (ур. 4.131). Уравнение 4.128 (N_Lv,BS) определено только для
  восходящего потока и в ветвях θ ≤ 0 не используется.
* θ = 0 относится к ветви «горизонтальный и нисходящий» (ур. 4.131, 4.133).
* Граница крутого нисходящего потока строгая: θ = −30° относится к пологой ветви.
* Эмпирические границы могут давать немонотонную картину (например, при θ = 0
  в узкой области малых v_Sg чередуются BUBBLE и STRATIFIED). Это свойство
  регрессий, а не ошибка.
* В книге в примере 4.8 N_gv,SM вычисляется «по уравнению (4.109)», фактически
  по (4.130).
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Final

from correlations.base import (
    ClassificationResult,
    FlowPatternCode,
    IFlowModel,
    check_velocities,
    single_phase_pattern,
)
from correlations.constants import GRAVITY

STEEP_DOWNFLOW_ANGLE_DEG: Final = 30.0  # Граница крутого нисходящего потока, градусы
_NUMBER_EXPONENT: Final = 0.25  # Показатель в безразмерных группах Duns & Ros

# Слаг → кольцевой, ур. 4.130
_SLUG_ANNULAR_BASE: Final = 1.401
_SLUG_ANNULAR_VISCOSITY: Final = -2.694
_SLUG_ANNULAR_LIQUID: Final = 0.521
_SLUG_ANNULAR_LIQUID_EXPONENT: Final = 0.329

# Пузырьковый → пробковый, восходящий поток, ур. 4.128–4.129
_UPFLOW_BS_BASE: Final = 0.940
_UPFLOW_BS_SIN: Final = 0.074
_UPFLOW_BS_SIN_SQUARED: Final = -0.855
_UPFLOW_BS_VISCOSITY: Final = 3.695

# Пузырьковый → пробковый, горизонтальный и нисходящий поток, ур. 4.131–4.132
_DOWNFLOW_BS_BASE: Final = 0.431
_DOWNFLOW_BS_VISCOSITY: Final = -3.003
_DOWNFLOW_BS_LOG_SIN: Final = -1.138
_DOWNFLOW_BS_LOG_SQUARED_SIN: Final = -0.429
_DOWNFLOW_BS_SIN: Final = 1.132

# Расслоённый режим, горизонтальный и нисходящий поток, ур. 4.133–4.134
_STRATIFIED_BASE: Final = 0.321
_STRATIFIED_GAS: Final = -0.017
_STRATIFIED_SIN: Final = -4.267
_STRATIFIED_VISCOSITY: Final = -2.972
_STRATIFIED_LOG_GAS_SQUARED: Final = -0.033
_STRATIFIED_SIN_SQUARED: Final = -3.925


class MukherjeeBrillReason(Enum):
    """Ветвь схемы выбора режима (рис. 4.19), по которой получен результат."""

    SINGLE_PHASE = auto()
    ANNULAR = auto()
    BUBBLE_UPFLOW = auto()
    SLUG_UPFLOW = auto()
    SLUG_STEEP_DOWNFLOW_LOW_GAS = auto()  # θ < −30°, N_gv ≤ N_gv,BS
    SLUG_STEEP_DOWNFLOW = auto()
    STRATIFIED_STEEP_DOWNFLOW = auto()
    BUBBLE_MILD_DOWNFLOW = auto()
    SLUG_MILD_DOWNFLOW = auto()
    STRATIFIED_MILD_DOWNFLOW = auto()


@dataclass(frozen=True, slots=True)
class MukherjeeBrillResult(ClassificationResult):
    """
    Результат классификации.

    Attributes
    ----------
    pattern: FlowPatternCode
        Режим течения (унаследовано от ClassificationResult)
    reason: MukherjeeBrillReason
        Ветвь схемы, по которой получен режим
    liquid_velocity_number: float
        N_Lv, показатель скорости жидкости
    gas_velocity_number: float
        N_gv, показатель скорости газа
    liquid_viscosity_number: float
        N_L, показатель вязкости жидкости
    slug_annular_boundary: float
        N_gv,SM, граница пробковый → кольцевой (ур. 4.130)
    bubble_slug_boundary: float
        N_Lv,BS при θ > 0 (ур. 4.128–4.129) или N_gv,BS при θ ≤ 0 (ур. 4.131–4.132)
    stratified_boundary: float | None
        N_Lv,ST при θ ≤ 0 (ур. 4.133–4.134), при θ > 0 None

    Для однофазных случаев границы не определены: slug_annular_boundary и
    bubble_slug_boundary равны NaN, stratified_boundary равен None.
    """

    reason: MukherjeeBrillReason
    liquid_velocity_number: float
    gas_velocity_number: float
    liquid_viscosity_number: float
    slug_annular_boundary: float
    bubble_slug_boundary: float
    stratified_boundary: float | None


def velocity_number(
    *, velocity: float, liquid_density: float, surface_tension: float
) -> float:
    """
    Показатель скорости фазы (Duns & Ros): N_v = v·(ρ_L/(g·σ))^0.25.

    :param velocity: приведённая скорость фазы v_SL или v_Sg, м/с
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    :return: N_Lv (для v_SL) или N_gv (для v_Sg), безразмерный
    """
    return velocity * (liquid_density / (GRAVITY * surface_tension)) ** _NUMBER_EXPONENT


def liquid_viscosity_number(
    *, liquid_viscosity: float, liquid_density: float, surface_tension: float
) -> float:
    """
    Показатель вязкости жидкости (Duns & Ros): N_L = μ_L·(g/(ρ_L·σ³))^0.25.

    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    :return: N_L, безразмерный
    """
    return (
        liquid_viscosity
        * (GRAVITY / (liquid_density * surface_tension**3)) ** _NUMBER_EXPONENT
    )


def slug_annular_boundary(
    *, viscosity_number: float, liquid_velocity_number: float
) -> float:
    """
    Граница пробковый → кольцевой, не зависит от угла (ур. 4.130):
    N_gv,SM = 10^(1.401 − 2.694·N_L + 0.521·N_Lv^0.329).

    :param viscosity_number: N_L
    :param liquid_velocity_number: N_Lv
    :return: N_gv,SM
    """
    exponent = (
        _SLUG_ANNULAR_BASE
        + _SLUG_ANNULAR_VISCOSITY * viscosity_number
        + _SLUG_ANNULAR_LIQUID * liquid_velocity_number**_SLUG_ANNULAR_LIQUID_EXPONENT
    )
    return 10.0**exponent


def upflow_bubble_slug_boundary(
    *, gas_velocity_number: float, viscosity_number: float, sin_angle: float
) -> float:
    """
    Граница пузырьковый → пробковый для восходящего потока θ > 0
    (ур. 4.128–4.129): x = lg N_gv + 0.940 + 0.074·s − 0.855·s² + 3.695·N_L,
    N_Lv,BS = 10^x.

    :param gas_velocity_number: N_gv, > 0
    :param viscosity_number: N_L
    :param sin_angle: s = sin θ
    :return: N_Lv,BS
    """
    exponent = (
        math.log10(gas_velocity_number)
        + _UPFLOW_BS_BASE
        + _UPFLOW_BS_SIN * sin_angle
        + _UPFLOW_BS_SIN_SQUARED * sin_angle**2
        + _UPFLOW_BS_VISCOSITY * viscosity_number
    )
    return 10.0**exponent


def downflow_bubble_slug_boundary(
    *, liquid_velocity_number: float, viscosity_number: float, sin_angle: float
) -> float:
    """
    Граница пузырьковый → пробковый по газу для θ ≤ 0 (ур. 4.131–4.132):
    y = 0.431 − 3.003·N_L − 1.138·(lg N_Lv)·s − 0.429·(lg N_Lv)²·s + 1.132·s,
    N_gv,BS = 10^y.

    :param liquid_velocity_number: N_Lv, > 0
    :param viscosity_number: N_L
    :param sin_angle: s = sin θ
    :return: N_gv,BS
    """
    log_liquid = math.log10(liquid_velocity_number)
    exponent = (
        _DOWNFLOW_BS_BASE
        + _DOWNFLOW_BS_VISCOSITY * viscosity_number
        + _DOWNFLOW_BS_LOG_SIN * log_liquid * sin_angle
        + _DOWNFLOW_BS_LOG_SQUARED_SIN * log_liquid**2 * sin_angle
        + _DOWNFLOW_BS_SIN * sin_angle
    )
    return 10.0**exponent


def stratified_boundary(
    *, gas_velocity_number: float, viscosity_number: float, sin_angle: float
) -> float:
    """
    Граница расслоённого режима по жидкости для θ ≤ 0 (ур. 4.133–4.134):
    z = 0.321 − 0.017·N_gv − 4.267·s − 2.972·N_L − 0.033·(lg N_gv)² − 3.925·s²,
    N_Lv,ST = 10^z.

    :param gas_velocity_number: N_gv, > 0
    :param viscosity_number: N_L
    :param sin_angle: s = sin θ
    :return: N_Lv,ST
    """
    exponent = (
        _STRATIFIED_BASE
        + _STRATIFIED_GAS * gas_velocity_number
        + _STRATIFIED_SIN * sin_angle
        + _STRATIFIED_VISCOSITY * viscosity_number
        + _STRATIFIED_LOG_GAS_SQUARED * math.log10(gas_velocity_number) ** 2
        + _STRATIFIED_SIN_SQUARED * sin_angle**2
    )
    return 10.0**exponent


@dataclass(frozen=True, slots=True)
class _Invariants:
    """Величины, не зависящие от (v_SL, v_Sg)."""

    sin_angle: float
    is_upflow: bool
    is_steep_downflow: bool
    viscosity_number: float


class MukherjeeBrillModel(IFlowModel):
    """
    Модель режимов Mukherjee–Brill для углов от −90° до +90°.

    Порядок выбора режима (рис. 4.19):

    1. N_gv > N_gv,SM → ANNULAR.
    2. θ > 0: N_Lv > N_Lv,BS → BUBBLE, иначе SLUG.
    3. θ < −30° (крутой нисходящий): при N_gv > N_gv,BS — SLUG, если
       N_Lv > N_Lv,ST, иначе STRATIFIED; при N_gv ≤ N_gv,BS — SLUG.
    4. −30° ≤ θ ≤ 0: при N_Lv > N_Lv,ST — SLUG, если N_gv > N_gv,BS, иначе
       BUBBLE; при N_Lv ≤ N_Lv,ST — STRATIFIED.

    Диапазон применимости, ограничения и неоднозначности источника — в
    docstring модуля. Настроек нет.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    """

    name = "Mukherjee-Brill"
    angle_range = (-90.0, 90.0)

    @cached_property
    def _invariants(self) -> _Invariants:
        """Величины, зависящие только от трубы и флюида (PipeParams/FluidParams неизменяемы)."""
        angle = self.pipe.angle
        return _Invariants(
            sin_angle=math.sin(math.radians(angle)),
            is_upflow=angle > 0.0,
            is_steep_downflow=angle < -STEEP_DOWNFLOW_ANGLE_DEG,
            viscosity_number=liquid_viscosity_number(
                liquid_viscosity=self.fluid.viscosity_liquid,
                liquid_density=self.fluid.density_liquid,
                surface_tension=self.fluid.surface_tension,
            ),
        )

    def classify(self, vsl: float, vsg: float) -> MukherjeeBrillResult:
        """
        Определяет режим течения и ветвь схемы (для диагностики).

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная или не конечная
        """
        check_velocities(vsl, vsg)
        invariants = self._invariants
        viscosity_number = invariants.viscosity_number
        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return MukherjeeBrillResult(
                pattern=single_phase,
                reason=MukherjeeBrillReason.SINGLE_PHASE,
                liquid_velocity_number=self._velocity_number(vsl),
                gas_velocity_number=self._velocity_number(vsg),
                liquid_viscosity_number=viscosity_number,
                slug_annular_boundary=float("nan"),
                bubble_slug_boundary=float("nan"),
                stratified_boundary=None,
            )

        liquid_number = self._velocity_number(vsl)
        gas_number = self._velocity_number(vsg)
        sin_angle = invariants.sin_angle
        annular_boundary = slug_annular_boundary(
            viscosity_number=viscosity_number, liquid_velocity_number=liquid_number
        )

        def result(
            pattern: FlowPatternCode,
            reason: MukherjeeBrillReason,
            bubble_slug: float,
            stratified: float | None,
        ) -> MukherjeeBrillResult:
            return MukherjeeBrillResult(
                pattern=pattern,
                reason=reason,
                liquid_velocity_number=liquid_number,
                gas_velocity_number=gas_number,
                liquid_viscosity_number=viscosity_number,
                slug_annular_boundary=annular_boundary,
                bubble_slug_boundary=bubble_slug,
                stratified_boundary=stratified,
            )

        if invariants.is_upflow:
            bubble_slug = upflow_bubble_slug_boundary(
                gas_velocity_number=gas_number,
                viscosity_number=viscosity_number,
                sin_angle=sin_angle,
            )
            if gas_number > annular_boundary:
                return result(
                    FlowPatternCode.ANNULAR,
                    MukherjeeBrillReason.ANNULAR,
                    bubble_slug,
                    None,
                )
            if liquid_number > bubble_slug:
                return result(
                    FlowPatternCode.BUBBLE,
                    MukherjeeBrillReason.BUBBLE_UPFLOW,
                    bubble_slug,
                    None,
                )
            return result(
                FlowPatternCode.SLUG,
                MukherjeeBrillReason.SLUG_UPFLOW,
                bubble_slug,
                None,
            )

        bubble_slug = downflow_bubble_slug_boundary(
            liquid_velocity_number=liquid_number,
            viscosity_number=viscosity_number,
            sin_angle=sin_angle,
        )
        stratified = stratified_boundary(
            gas_velocity_number=gas_number,
            viscosity_number=viscosity_number,
            sin_angle=sin_angle,
        )
        if gas_number > annular_boundary:
            return result(
                FlowPatternCode.ANNULAR,
                MukherjeeBrillReason.ANNULAR,
                bubble_slug,
                stratified,
            )

        if invariants.is_steep_downflow:
            if gas_number <= bubble_slug:
                return result(
                    FlowPatternCode.SLUG,
                    MukherjeeBrillReason.SLUG_STEEP_DOWNFLOW_LOW_GAS,
                    bubble_slug,
                    stratified,
                )
            if liquid_number > stratified:
                return result(
                    FlowPatternCode.SLUG,
                    MukherjeeBrillReason.SLUG_STEEP_DOWNFLOW,
                    bubble_slug,
                    stratified,
                )
            return result(
                FlowPatternCode.STRATIFIED,
                MukherjeeBrillReason.STRATIFIED_STEEP_DOWNFLOW,
                bubble_slug,
                stratified,
            )

        if liquid_number <= stratified:
            return result(
                FlowPatternCode.STRATIFIED,
                MukherjeeBrillReason.STRATIFIED_MILD_DOWNFLOW,
                bubble_slug,
                stratified,
            )
        if gas_number > bubble_slug:
            return result(
                FlowPatternCode.SLUG,
                MukherjeeBrillReason.SLUG_MILD_DOWNFLOW,
                bubble_slug,
                stratified,
            )
        return result(
            FlowPatternCode.BUBBLE,
            MukherjeeBrillReason.BUBBLE_MILD_DOWNFLOW,
            bubble_slug,
            stratified,
        )

    def _velocity_number(self, velocity: float) -> float:
        """N_Lv или N_gv для приведённой скорости `velocity` (м/с)."""
        return velocity_number(
            velocity=velocity,
            liquid_density=self.fluid.density_liquid,
            surface_tension=self.fluid.surface_tension,
        )
