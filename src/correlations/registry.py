"""Реестр корреляций и моделей режима потока.

Позволяет создавать модели по имени и флагам:

    get_correlation("beggs_brill", payne_correction=True)
    get_flow_model("barnea", some_flag=True)

Флаги проверяются `<Name>Options.model_validate(flags)`: неизвестный флаг или значение не `bool`
дают `pydantic.ValidationError`; неизвестное имя модели — `ValueError`.
Источник требований: ТЗ проекта, раздел «Реестр и публичный API».
"""

from collections.abc import Callable
from typing import Any, TypeVar

from correlations.core.interfaces import ICorrelation, IFlowModel
from correlations.core.options import BaseOptions
from correlations.flow_models.ansari_fp import AnsariFlowModel, AnsariFlowOptions
from correlations.flow_models.barnea import BarneaModel, BarneaOptions
from correlations.flow_models.beggs_brill_map import BeggsBrillMap, BeggsBrillMapOptions
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel, HasanKabirFlowOptions
from correlations.flow_models.mukherjee_brill_fp import (
    MukherjeeBrillFlowModel,
    MukherjeeBrillFlowOptions,
)
from correlations.flow_models.taitel_dukler import TaitelDuklerModel, TaitelDuklerOptions
from correlations.models.ansari import Ansari, AnsariOptions
from correlations.models.beggs_brill import BeggsBrill, BeggsBrillOptions
from correlations.models.hasan_kabir import HasanKabir, HasanKabirOptions
from correlations.models.mukherjee_brill import MukherjeeBrill, MukherjeeBrillOptions

_T = TypeVar("_T")
_CorrelationFactory = Callable[[Any, IFlowModel | None], ICorrelation]
_FlowModelFactory = Callable[[Any], IFlowModel]

_CORRELATIONS: dict[str, tuple[type[BaseOptions], _CorrelationFactory]] = {
    "ansari": (AnsariOptions, lambda o, f: Ansari(o, f)),
    "beggs_brill": (BeggsBrillOptions, lambda o, f: BeggsBrill(o, f)),
    "hasan_kabir": (HasanKabirOptions, lambda o, f: HasanKabir(o, f)),
    "mukherjee_brill": (MukherjeeBrillOptions, lambda o, f: MukherjeeBrill(o, f)),
}

_FLOW_MODELS: dict[str, tuple[type[BaseOptions], _FlowModelFactory]] = {
    "ansari_fp": (AnsariFlowOptions, lambda o: AnsariFlowModel(o)),
    "barnea": (BarneaOptions, lambda o: BarneaModel(o)),
    "beggs_brill_map": (BeggsBrillMapOptions, lambda o: BeggsBrillMap(o)),
    "hasan_kabir_fp": (HasanKabirFlowOptions, lambda o: HasanKabirFlowModel(o)),
    "mukherjee_brill_fp": (MukherjeeBrillFlowOptions, lambda o: MukherjeeBrillFlowModel(o)),
    "taitel_dukler": (TaitelDuklerOptions, lambda o: TaitelDuklerModel(o)),
}


def _lookup(registry: dict[str, _T], name: str, kind: str) -> _T:
    """Возвращает запись реестра или выбрасывает `ValueError` со списком имён.

    Параметры:
        registry: словарь «имя -> запись».
        name: запрошенное имя.
        kind: название вида моделей для сообщения об ошибке.

    Исключения:
        ValueError: имя не зарегистрировано.
    """
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Неизвестное имя {kind} {name!r}; доступны: {available}") from None


def list_correlations() -> tuple[str, ...]:
    """Возвращает отсортированные имена корреляций."""
    return tuple(sorted(_CORRELATIONS))


def list_flow_models() -> tuple[str, ...]:
    """Возвращает отсортированные имена моделей режима потока."""
    return tuple(sorted(_FLOW_MODELS))


def get_flow_model(name: str, **flags: Any) -> IFlowModel:
    """Создаёт модель режима потока по имени.

    Параметры:
        name: имя из `list_flow_models()`.
        **flags: флаги настроек модели (остальные — по умолчанию).

    Возвращает:
        Новый экземпляр `IFlowModel`.

    Исключения:
        ValueError: неизвестное имя.
        pydantic.ValidationError: неизвестный флаг или значение не `bool`.
    """
    options_cls, factory = _lookup(_FLOW_MODELS, name, "модели режима")
    return factory(options_cls.model_validate(flags))


def get_correlation(
    name: str, *, flow_model: IFlowModel | str | None = None, **flags: Any
) -> ICorrelation:
    """Создаёт корреляцию по имени.

    Параметры:
        name: имя из `list_correlations()`.
        flow_model: экземпляр `IFlowModel`, имя модели режима (создаётся с настройками по
            умолчанию) или None (модель режима по умолчанию корреляции).
        **flags: флаги настроек корреляции (остальные — по умолчанию).

    Возвращает:
        Новый экземпляр `ICorrelation`.

    Исключения:
        ValueError: неизвестное имя корреляции или модели режима.
        pydantic.ValidationError: неизвестный флаг или значение не `bool`.
    """
    options_cls, factory = _lookup(_CORRELATIONS, name, "корреляции")
    options = options_cls.model_validate(flags)
    resolved = get_flow_model(flow_model) if isinstance(flow_model, str) else flow_model
    return factory(options, resolved)


def describe_correlation(name: str) -> dict[str, str]:
    """Возвращает описания флагов корреляции (`Options.describe()`).

    Параметры:
        name: имя из `list_correlations()`.

    Возвращает:
        Словарь «флаг -> описание».

    Исключения:
        ValueError: неизвестное имя.
    """
    return _lookup(_CORRELATIONS, name, "корреляции")[0].describe()


def describe_flow_model(name: str) -> dict[str, str]:
    """Возвращает описания флагов модели режима (`Options.describe()`).

    Параметры:
        name: имя из `list_flow_models()`.

    Возвращает:
        Словарь «флаг -> описание».

    Исключения:
        ValueError: неизвестное имя.
    """
    return _lookup(_FLOW_MODELS, name, "модели режима")[0].describe()
