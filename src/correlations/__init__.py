"""
Correlations — корреляции для определения режима течения газожидкостной смеси
в трубах.

Публичный API:

* модели (`*Model`) — по одной на каждую корреляцию;
* их настройки (`*Settings`), результаты классификации (`*Result`) и причины
  перехода (`*Reason`) — там, где модель их предоставляет;
* базовые типы (`IFlowModel`, `FlowPatternCode`, `ClassificationResult`);
* входные параметры (`PipeParams`, `FluidParams`, `SystemParams`).

Формулы, общие для нескольких моделей (трение, расслоённое течение), — в
`correlations.physics`; для собственных моделей их можно импортировать оттуда
напрямую.
"""

from correlations.ansari import (
    AnsariModel,
    AnsariResult,
    AnsariSettings,
    FilmState,
    TransitionReason,
)
from correlations.ansari_vba import AnsariVBAModel
from correlations.barnea import BarneaModel, BarneaReason, BarneaResult, BarneaSettings
from correlations.base import ClassificationResult, FlowPatternCode, IFlowModel
from correlations.beggs_brill import BeggsBrillModel
from correlations.domain.params import FluidParams, PipeParams, SystemParams
from correlations.hasan_kabir import (
    AnnularRejection,
    HasanKabirModel,
    HasanKabirReason,
    HasanKabirResult,
    HasanKabirSettings,
)
from correlations.mukherjee_brill import (
    MukherjeeBrillModel,
    MukherjeeBrillReason,
    MukherjeeBrillResult,
)
from correlations.taitel_dukler import (
    TaitelDuklerModel,
    TaitelDuklerReason,
    TaitelDuklerResult,
    TaitelDuklerSettings,
)

__all__ = [
    "AnnularRejection",
    "AnsariModel",
    "AnsariResult",
    "AnsariSettings",
    "AnsariVBAModel",
    "BarneaModel",
    "BarneaReason",
    "BarneaResult",
    "BarneaSettings",
    "BeggsBrillModel",
    "ClassificationResult",
    "FilmState",
    "FlowPatternCode",
    "FluidParams",
    "HasanKabirModel",
    "HasanKabirReason",
    "HasanKabirResult",
    "HasanKabirSettings",
    "IFlowModel",
    "MukherjeeBrillModel",
    "MukherjeeBrillReason",
    "MukherjeeBrillResult",
    "PipeParams",
    "SystemParams",
    "TaitelDuklerModel",
    "TaitelDuklerReason",
    "TaitelDuklerResult",
    "TaitelDuklerSettings",
    "TransitionReason",
]
