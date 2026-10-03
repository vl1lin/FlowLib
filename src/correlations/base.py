"""
Базовый интерфейс для всех моделей корреляции режима течения.

Здесь — только контракт модели (что общего у Ansari, Barnea, Beggs-Brill и
остальных: угол, скорости, код режима), не физика конкретной корреляции. Формулы,
общие для нескольких моделей (трение, критерии перехода и т.п.), выносятся в
`physics/`, а не сюда.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from correlations.constants import SINGLE_PHASE_VELOCITY_EPS
from correlations.domain.params import FluidParams, PipeParams


class FlowPatternCode(IntEnum):
    """Единый словарь режимов для всех корреляций"""

    SINGLE_LIQUID = 100
    SINGLE_GAS = 101
    BUBBLE = 102
    SLUG = 103
    DISPERSED_BUBBLE = 104
    ANNULAR = 105
    STRATIFIED = 106  # Расслоённый гладкий
    STRATIFIED_WAVY = 107  # Расслоённый волновой
    CHURN = 108  # Эмульсионный (churn)
    UNKNOWN = 199


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """
    Базовый результат классификации режима течения.

    Модели с диагностическими величинами (уровень, толщина плёнки, граничные
    скорости и т.п.) наследуют свой `*Result` от этого класса, добавляя свои поля.
    """

    pattern: FlowPatternCode


def check_velocities(vsl: float, vsg: float) -> None:
    """
    Общая валидация приведённых скоростей: конечны и неотрицательны.

    :param vsl: приведённая скорость жидкости v_SL, м/с
    :param vsg: приведённая скорость газа v_Sg, м/с
    :raises ValueError: если скорость отрицательная, NaN или бесконечная
    """
    if not (math.isfinite(vsl) and math.isfinite(vsg) and vsl >= 0.0 and vsg >= 0.0):
        raise ValueError(f"Скорости должны быть конечными и ≥ 0: vsl={vsl}, vsg={vsg}")


def single_phase_pattern(vsl: float, vsg: float) -> FlowPatternCode | None:
    """
    Код однофазного режима, если одна из скоростей ниже порога отсутствия фазы,
    иначе None (двухфазный случай — классификация остаётся за моделью).

    :param vsl: приведённая скорость жидкости v_SL, м/с
    :param vsg: приведённая скорость газа v_Sg, м/с
    """
    if vsl < SINGLE_PHASE_VELOCITY_EPS:
        return FlowPatternCode.SINGLE_GAS
    if vsg < SINGLE_PHASE_VELOCITY_EPS:
        return FlowPatternCode.SINGLE_LIQUID
    return None


class IFlowModel(ABC):
    """
    Базовый класс для всех моделей корреляции.

    Наследники задают `name` и `angle_range` как атрибуты класса и реализуют
    `classify` — саму физику модели для точки (v_SL, v_Sg), используя при этом
    `check_velocities`/`single_phase_pattern` из этого модуля вместо повторной
    реализации той же проверки.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    :raises ValueError: если угол трубы вне `angle_range` модели
    """

    name: ClassVar[str]
    angle_range: ClassVar[tuple[float, float]]

    def __init__(self, pipe: PipeParams, fluid: FluidParams) -> None:
        self.pipe = pipe
        self.fluid = fluid
        self.validate_angle()

    def validate_angle(self) -> None:
        """
        Валидировать угол течения.
        :raises ValueError: если угол выходит за допустимый диапазон
        """
        angle_min, angle_max = self.angle_range
        if not angle_min <= self.pipe.angle <= angle_max:
            raise ValueError(
                f"Модель '{self.name}' работает только для углов "
                f"{angle_min}°-{angle_max}°.\nТекущий угол: {self.pipe.angle}°"
            )

    @abstractmethod
    def classify(self, vsl: float, vsg: float) -> ClassificationResult:
        """
        Определяет режим течения (и диагностику перехода, если модель её даёт)
        по значениям приведённых скоростей.

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная, NaN или бесконечная
        """
        ...

    def get_pattern_code(self, vsl: float, vsg: float) -> FlowPatternCode:
        """
        Код режима течения по значениям приведённых скоростей.
        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        """
        return self.classify(vsl, vsg).pattern
