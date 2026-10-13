"""Интерфейсы библиотеки (структурная типизация, `typing.Protocol`).

`IFlowModel` — модуль режима потока: по состоянию потока `FlowState` определяет режим
(`FlowPatternResult`). Любая модель обязана иметь имя `name`, настройки `options`
(`BaseOptions` из флагов `bool`) и метод `predict`.

`ICorrelation` — полная корреляция: режим (через внедряемый `IFlowModel`), holdup и градиент
давления (`CorrelationResult`).

Конструктор любой модели режима: `Model(options: ModelOptions | None = None)`;
конструктор любой корреляции: `Correlation(options=None, flow_model: IFlowModel | None = None)`.
Источник требований: ТЗ проекта, раздел «Интерфейсы».
"""

from typing import ClassVar, Protocol, runtime_checkable

from correlations.core.options import BaseOptions
from correlations.core.state import CorrelationResult, FlowPatternResult, FlowState


@runtime_checkable
class IFlowModel(Protocol):
    """Модуль режима потока.

    Атрибуты:
        name: уникальное имя модели (например, "barnea").
        options: неизменяемые настройки модели (флаги `bool`).
    """

    name: ClassVar[str]

    @property
    def options(self) -> BaseOptions:
        """Настройки модели."""
        ...

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим потока.

        Параметры:
            state: локальное состояние потока (СИ).

        Возвращает:
            `FlowPatternResult`: режим и промежуточные величины (`details`).
        """
        ...


@runtime_checkable
class ICorrelation(Protocol):
    """Полная корреляция: режим + holdup + градиент давления.

    Атрибуты:
        name: уникальное имя корреляции (например, "beggs_brill").
        options: неизменяемые настройки (флаги `bool`).
        flow_model: модуль режима, внедряемый через конструктор.
    """

    name: ClassVar[str]

    @property
    def options(self) -> BaseOptions:
        """Настройки корреляции."""
        ...

    @property
    def flow_model(self) -> IFlowModel:
        """Используемый модуль режима потока."""
        ...

    def calc(self, state: FlowState) -> CorrelationResult:
        """Рассчитывает режим, holdup и градиент давления.

        Параметры:
            state: локальное состояние потока (СИ).

        Возвращает:
            `CorrelationResult`: режим, H_L, −dp/dL и его составляющие, `details`.
        """
        ...
