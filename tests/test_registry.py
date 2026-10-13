"""Тесты реестра `correlations.registry` (этап 6, ТЗ «Реестр и публичный API»).

Ожидаемый API (модуль `correlations.registry`):
    list_correlations() -> tuple[str, ...]      — имена корреляций, отсортированы;
    list_flow_models() -> tuple[str, ...]       — имена моделей режима, отсортированы;
    get_correlation(name, *, flow_model=None, **flags) -> ICorrelation
        flags валидируются `<Name>Options.model_validate(flags)`; `flow_model` — экземпляр
        `IFlowModel`, либо имя из `list_flow_models()` (создаётся с настройками по умолчанию),
        либо None (модель режима по умолчанию корреляции);
    get_flow_model(name, **flags) -> IFlowModel;
    describe_correlation(name) -> dict[str, str] и describe_flow_model(name) -> dict[str, str]
        — «флаг -> описание» (`Options.describe()`).
Неизвестное имя -> ValueError (в сообщении перечислены допустимые имена);
неизвестный флаг / не-bool значение -> pydantic.ValidationError.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from correlations.core.interfaces import ICorrelation, IFlowModel
from correlations.core.state import CorrelationResult, FlowPatternResult, FlowState
from correlations.flow_models.ansari_fp import AnsariFlowModel
from correlations.flow_models.barnea import BarneaModel
from correlations.flow_models.beggs_brill_map import BeggsBrillMap
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel
from correlations.flow_models.mukherjee_brill_fp import MukherjeeBrillFlowModel
from correlations.flow_models.taitel_dukler import TaitelDuklerModel
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

CORRELATION_CLASSES = {
    "ansari": Ansari,
    "beggs_brill": BeggsBrill,
    "hasan_kabir": HasanKabir,
    "mukherjee_brill": MukherjeeBrill,
}
FLOW_MODEL_CLASSES = {
    "ansari_fp": AnsariFlowModel,
    "barnea": BarneaModel,
    "beggs_brill_map": BeggsBrillMap,
    "hasan_kabir_fp": HasanKabirFlowModel,
    "mukherjee_brill_fp": MukherjeeBrillFlowModel,
    "taitel_dukler": TaitelDuklerModel,
}


# --- перечни ---
def test_list_correlations() -> None:
    assert list_correlations() == tuple(sorted(CORRELATION_CLASSES))


def test_list_flow_models() -> None:
    assert list_flow_models() == tuple(sorted(FLOW_MODEL_CLASSES))


def test_registered_names_match_class_names() -> None:
    for name, cls in CORRELATION_CLASSES.items():
        assert cls.name == name
    for name, flow_cls in FLOW_MODEL_CLASSES.items():
        assert flow_cls.name == name


# --- get_correlation ---
@pytest.mark.parametrize("name", sorted(CORRELATION_CLASSES))
def test_get_correlation_default(name: str) -> None:
    model = get_correlation(name)
    assert type(model) is CORRELATION_CLASSES[name]
    assert isinstance(model, ICorrelation)
    assert model.options == type(model.options)()


def test_get_correlation_flags_are_applied() -> None:
    model = get_correlation("beggs_brill", payne_correction=True, revised_map=True)
    assert isinstance(model, BeggsBrill)
    assert model.options == BeggsBrillOptions(payne_correction=True, revised_map=True)
    # default-флаги сохраняются
    assert model.options.include_acceleration is True


@pytest.mark.parametrize(
    ("name", "options_cls"),
    [
        ("ansari", AnsariOptions),
        ("beggs_brill", BeggsBrillOptions),
        ("hasan_kabir", HasanKabirOptions),
        ("mukherjee_brill", MukherjeeBrillOptions),
    ],
)
def test_every_flag_can_be_set_through_registry(name: str, options_cls: type) -> None:
    for flag in options_cls.describe():
        default = getattr(options_cls(), flag)
        flags: dict[str, Any] = {flag: not default}
        model = get_correlation(name, **flags)
        assert getattr(model.options, flag) is (not default)


@pytest.mark.parametrize("name", sorted(CORRELATION_CLASSES))
def test_get_correlation_unknown_flag(name: str) -> None:
    with pytest.raises(ValidationError):
        get_correlation(name, no_such_flag=True)


@pytest.mark.parametrize("name", sorted(CORRELATION_CLASSES))
def test_get_correlation_string_flag_rejected(name: str) -> None:
    first = next(iter(describe_correlation(name)))
    with pytest.raises(ValidationError):
        get_correlation(name, **{first: "True"})


def test_get_correlation_unknown_name() -> None:
    with pytest.raises(ValueError, match="beggs_brill") as info:
        get_correlation("no_such_correlation")
    assert "no_such_correlation" in str(info.value)


def test_get_correlation_name_is_case_sensitive() -> None:
    with pytest.raises(ValueError):
        get_correlation("Beggs_Brill")


def test_get_correlation_returns_new_instance_each_call() -> None:
    assert get_correlation("ansari") is not get_correlation("ansari")


# --- внедрение модели режима ---
def test_get_correlation_with_flow_model_instance() -> None:
    flow = get_flow_model("barnea")
    model = get_correlation("hasan_kabir", flow_model=flow)
    assert model.flow_model is flow


def test_get_correlation_with_flow_model_name() -> None:
    model = get_correlation("beggs_brill", flow_model="beggs_brill_map")
    assert isinstance(model.flow_model, BeggsBrillMap)


def test_get_correlation_flow_model_unknown_name() -> None:
    with pytest.raises(ValueError, match="no_such_flow_model"):
        get_correlation("ansari", flow_model="no_such_flow_model")


def test_get_correlation_flow_model_does_not_swallow_flags() -> None:
    flow = get_flow_model("beggs_brill_map")
    model = get_correlation("beggs_brill", flow_model=flow, payne_correction=True)
    assert model.flow_model is flow
    assert isinstance(model.options, BeggsBrillOptions)
    assert model.options.payne_correction is True


# --- get_flow_model ---
@pytest.mark.parametrize("name", sorted(FLOW_MODEL_CLASSES))
def test_get_flow_model_default(name: str) -> None:
    model = get_flow_model(name)
    assert type(model) is FLOW_MODEL_CLASSES[name]
    assert isinstance(model, IFlowModel)
    assert model.options == type(model.options)()


@pytest.mark.parametrize("name", sorted(FLOW_MODEL_CLASSES))
def test_get_flow_model_unknown_flag(name: str) -> None:
    with pytest.raises(ValidationError):
        get_flow_model(name, no_such_flag=True)


def test_get_flow_model_flags_are_applied() -> None:
    model = get_flow_model("beggs_brill_map", revised_map=True)
    assert isinstance(model, BeggsBrillMap)
    assert model.options.revised_map is True  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", sorted(FLOW_MODEL_CLASSES))
def test_every_flow_model_flag_can_be_set(name: str) -> None:
    options_cls = type(get_flow_model(name).options)
    for flag in options_cls.describe():
        default = getattr(options_cls(), flag)
        model = get_flow_model(name, **{flag: not default})
        assert getattr(model.options, flag) is (not default)


def test_get_flow_model_unknown_name() -> None:
    with pytest.raises(ValueError, match="no_such_flow_model") as info:
        get_flow_model("no_such_flow_model")
    assert "barnea" in str(info.value)


def test_get_flow_model_rejects_correlation_name() -> None:
    with pytest.raises(ValueError):
        get_flow_model("beggs_brill")


# --- describe ---
@pytest.mark.parametrize("name", sorted(CORRELATION_CLASSES))
def test_describe_correlation_matches_options(name: str) -> None:
    description = describe_correlation(name)
    assert description == type(get_correlation(name).options).describe()
    assert description
    assert all(text.strip() for text in description.values())


@pytest.mark.parametrize("name", sorted(FLOW_MODEL_CLASSES))
def test_describe_flow_model_matches_options(name: str) -> None:
    description = describe_flow_model(name)
    assert description == type(get_flow_model(name).options).describe()
    assert all(text.strip() for text in description.values())


def test_describe_unknown_name() -> None:
    with pytest.raises(ValueError):
        describe_correlation("no_such_correlation")
    with pytest.raises(ValueError):
        describe_flow_model("no_such_flow_model")


# --- сквозной расчёт через реестр ---
@pytest.mark.parametrize("name", sorted(CORRELATION_CLASSES))
def test_registry_correlation_calculates(name: str, aw_state: Callable[..., FlowState]) -> None:
    result = get_correlation(name).calc(aw_state(0.1, 1.0, 0.0508, 20.0))
    assert isinstance(result, CorrelationResult)
    assert 0.0 <= result.holdup <= 1.0


@pytest.mark.parametrize("name", sorted(FLOW_MODEL_CLASSES))
def test_registry_flow_model_predicts(name: str, aw_state: Callable[..., FlowState]) -> None:
    result = get_flow_model(name).predict(aw_state(0.1, 1.0, 0.0508, 90.0))
    assert isinstance(result, FlowPatternResult)


def test_registry_matches_direct_construction(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.3, 2.0, 0.0508, 5.0)
    flags: dict[str, Any] = {"payne_correction": True, "rough_pipe_friction": True}
    via_registry = get_correlation("beggs_brill", **flags).calc(state)
    direct = BeggsBrill(BeggsBrillOptions(**flags)).calc(state)
    assert via_registry == direct
