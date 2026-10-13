"""Correlations — корреляции градиента давления газожидкостного потока в трубах и скважинах.

Публичный API:

* реестр: `get_correlation`, `get_flow_model`, `list_*`, `describe_*`;
* типы: `FlowState`, `FlowPattern`, `FlowPatternResult`, `CorrelationResult`, `BaseOptions`,
  интерфейсы `IFlowModel`, `ICorrelation`;
* ошибки: `ConvergenceError`, `OutOfRangeWarning`, `UnsupportedPatternError`;
* корреляции (`BeggsBrill`, `MukherjeeBrill`, `HasanKabir`, `Ansari`) и модели режима
  с их `*Options`.

Все величины — в СИ.
"""

from correlations.core.errors import ConvergenceError, OutOfRangeWarning, UnsupportedPatternError
from correlations.core.interfaces import ICorrelation, IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowPatternResult, FlowState
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
from correlations.registry import (
    describe_correlation,
    describe_flow_model,
    get_correlation,
    get_flow_model,
    list_correlations,
    list_flow_models,
)

__all__ = [
    "Ansari",
    "AnsariFlowModel",
    "AnsariFlowOptions",
    "AnsariOptions",
    "BarneaModel",
    "BarneaOptions",
    "BaseOptions",
    "BeggsBrill",
    "BeggsBrillMap",
    "BeggsBrillMapOptions",
    "BeggsBrillOptions",
    "ConvergenceError",
    "CorrelationResult",
    "FlowPattern",
    "FlowPatternResult",
    "FlowState",
    "HasanKabir",
    "HasanKabirFlowModel",
    "HasanKabirFlowOptions",
    "HasanKabirOptions",
    "ICorrelation",
    "IFlowModel",
    "MukherjeeBrill",
    "MukherjeeBrillFlowModel",
    "MukherjeeBrillFlowOptions",
    "MukherjeeBrillOptions",
    "OutOfRangeWarning",
    "TaitelDuklerModel",
    "TaitelDuklerOptions",
    "UnsupportedPatternError",
    "describe_correlation",
    "describe_flow_model",
    "get_correlation",
    "get_flow_model",
    "list_correlations",
    "list_flow_models",
]
