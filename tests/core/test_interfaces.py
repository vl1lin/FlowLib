"""Тесты `core/interfaces.py` и общего поведения всех модулей режима (IFlowModel).

Ожидаемый API: `from correlations.core.interfaces import IFlowModel` — `Protocol`, `runtime_checkable`:
    name: ClassVar[str]
    options -> BaseOptions (property)
    predict(state: FlowState) -> FlowPatternResult
Конструктор любой модели режима: `Model(options: ModelOptions | None = None)`.
"""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from correlations.core.interfaces import IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models.ansari_fp import AnsariFlowModel
from correlations.flow_models.barnea import BarneaModel
from correlations.flow_models.beggs_brill_map import BeggsBrillMap
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel
from correlations.flow_models.mukherjee_brill_fp import MukherjeeBrillFlowModel
from correlations.flow_models.taitel_dukler import TaitelDuklerModel

MODEL_CLASSES = [
    BeggsBrillMap,
    MukherjeeBrillFlowModel,
    TaitelDuklerModel,
    BarneaModel,
    HasanKabirFlowModel,
    AnsariFlowModel,
]
FLAGGED_MODEL_CLASSES = [
    BeggsBrillMap,
    TaitelDuklerModel,
    BarneaModel,
    HasanKabirFlowModel,
    AnsariFlowModel,
]
EXPECTED_NAMES = {
    BeggsBrillMap: "beggs_brill_map",
    MukherjeeBrillFlowModel: "mukherjee_brill_fp",
    TaitelDuklerModel: "taitel_dukler",
    BarneaModel: "barnea",
    HasanKabirFlowModel: "hasan_kabir_fp",
    AnsariFlowModel: "ansari_fp",
}


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_is_flow_model(model_cls: type) -> None:
    assert isinstance(model_cls(), IFlowModel)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_name(model_cls: type) -> None:
    assert model_cls.name == EXPECTED_NAMES[model_cls]


def test_names_are_unique() -> None:
    assert len({cls.name for cls in MODEL_CLASSES}) == len(MODEL_CLASSES)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_options_are_base_options_and_frozen(model_cls: type) -> None:
    options = model_cls().options
    assert isinstance(options, BaseOptions)
    with pytest.raises(ValidationError):
        options.some_flag = True  # type: ignore[attr-defined]  # frozen/extra=forbid


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_every_flag_has_description_and_is_bool(model_cls: type) -> None:
    options_cls = type(model_cls().options)
    for name, description in options_cls.describe().items():
        assert description.strip(), f"{options_cls.__name__}.{name}: пустое описание"
        assert options_cls.model_fields[name].annotation is bool


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_unknown_flag_rejected(model_cls: type) -> None:
    options_cls = type(model_cls().options)
    with pytest.raises(ValidationError):
        options_cls.model_validate({"no_such_flag": True})


@pytest.mark.parametrize("model_cls", FLAGGED_MODEL_CLASSES)
def test_string_flag_rejected(model_cls: type) -> None:
    options_cls = type(model_cls().options)
    name = next(iter(options_cls.describe()))
    with pytest.raises(ValidationError):
        options_cls.model_validate({name: "True"})


def test_mukherjee_brill_flow_model_has_no_flags() -> None:
    # reference: ТЗ, «Mukherjee & Brill flow patterns»: «Флагов нет»
    assert type(MukherjeeBrillFlowModel().options).describe() == {}


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_predict_returns_flow_pattern_result(
    model_cls: type, aw_state: Callable[..., FlowState]
) -> None:
    result = model_cls().predict(aw_state(0.1, 1.0, 0.0508, 0.0))
    assert isinstance(result, FlowPatternResult)
    assert isinstance(result.pattern, FlowPattern)
    assert all(isinstance(v, float) for v in result.details.values())


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_single_phase_limits(model_cls: type, aw_state: Callable[..., FlowState]) -> None:
    """ТЗ: v_sg = 0 → SINGLE_PHASE_LIQUID, v_sl = 0 → SINGLE_PHASE_GAS (модель не падает)."""
    model = model_cls()
    assert model.predict(aw_state(1.0, 0.0, 0.0508, 0.0)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert model.predict(aw_state(0.0, 1.0, 0.0508, 0.0)).pattern is FlowPattern.SINGLE_PHASE_GAS


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_predict_is_deterministic(model_cls: type, aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.3, 2.0, 0.0508, 5.0)
    model = model_cls()
    assert model.predict(state) == model.predict(state)


def test_not_flow_model_is_rejected() -> None:
    class NotAModel:
        name = "x"

    assert not isinstance(NotAModel(), IFlowModel)
