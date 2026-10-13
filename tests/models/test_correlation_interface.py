"""Общие тесты `ICorrelation` для всех полных корреляций (ТЗ, «Общие тесты свойств»).

Ожидаемый API:
    `from correlations.core.interfaces import ICorrelation` — Protocol, runtime_checkable:
        name: ClassVar[str]; options -> BaseOptions; flow_model -> IFlowModel; calc(state) -> CorrelationResult.
    `from correlations.core.errors import UnsupportedPatternError` — ValueError, атрибут `pattern`;
        выбрасывается, если внедрённая модель режима вернула режим, который корреляция не обрабатывает.
Конструктор любой корреляции: `Model(options=None, flow_model=None)`.
Однофазные пределы (ТЗ): v_sg = 0 → SINGLE_PHASE_LIQUID, H_L = 1; v_sl = 0 → SINGLE_PHASE_GAS, H_L = 0;
градиент — однофазный (Colebrook, без ускорения).
"""

import math
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from correlations.core import friction
from correlations.core.errors import UnsupportedPatternError
from correlations.core.interfaces import ICorrelation, IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowPatternResult, FlowState
from correlations.flow_models.ansari_fp import AnsariFlowModel
from correlations.flow_models.beggs_brill_map import BeggsBrillMap
from correlations.flow_models.hasan_kabir_fp import HasanKabirFlowModel
from correlations.flow_models.mukherjee_brill_fp import MukherjeeBrillFlowModel
from correlations.models.ansari import Ansari, AnsariOptions
from correlations.models.beggs_brill import BeggsBrill, BeggsBrillOptions
from correlations.models.hasan_kabir import HasanKabir, HasanKabirOptions
from correlations.models.mukherjee_brill import MukherjeeBrill, MukherjeeBrillOptions

CORRELATIONS = [BeggsBrill, MukherjeeBrill, HasanKabir, Ansari]
EXPECTED_NAMES = {
    BeggsBrill: "beggs_brill",
    MukherjeeBrill: "mukherjee_brill",
    HasanKabir: "hasan_kabir",
    Ansari: "ansari",
}
# Параметры для тестов предельного перехода: ускорение выключено (однофазный градиент без Ek);
# BB — шероховатая формула (Colebrook), как в однофазном пределе ТЗ.
LIMIT_OPTIONS: dict[type, Any] = {
    BeggsBrill: BeggsBrillOptions(rough_pipe_friction=True, include_acceleration=False),
    MukherjeeBrill: MukherjeeBrillOptions(include_acceleration=False),
    HasanKabir: HasanKabirOptions(include_acceleration=False),
    Ansari: AnsariOptions(),
}


class FixedFlowModel:
    """Модуль режима-заглушка: всегда возвращает заданный режим (проверка инъекции flow_model)."""

    name: ClassVar[str] = "fixed"

    def __init__(self, pattern: FlowPattern, details: dict[str, float] | None = None) -> None:
        self._pattern = pattern
        self._details = details or {}
        self.calls = 0

    @property
    def options(self) -> BaseOptions:
        return BaseOptions()

    def predict(self, state: FlowState) -> FlowPatternResult:
        self.calls += 1
        return FlowPatternResult(pattern=self._pattern, details=dict(self._details))


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_is_correlation(model_cls: type) -> None:
    assert isinstance(model_cls(), ICorrelation)


def test_flow_model_is_not_a_correlation() -> None:
    assert not isinstance(BeggsBrillMap(), ICorrelation)
    assert not isinstance(BeggsBrill(), IFlowModel)


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_name(model_cls: type) -> None:
    assert model_cls.name == EXPECTED_NAMES[model_cls]


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_options_are_frozen_and_described(model_cls: type) -> None:
    model = model_cls()
    assert isinstance(model.options, BaseOptions)
    with pytest.raises(ValidationError):
        model.options.some_flag = True  # type: ignore[attr-defined]
    for name, description in type(model.options).describe().items():
        assert description.strip(), f"{type(model.options).__name__}.{name}: пустое описание"
        assert type(model.options).model_fields[name].annotation is bool


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_unknown_and_string_flags_rejected(model_cls: type) -> None:
    options_cls = type(model_cls().options)
    with pytest.raises(ValidationError):
        options_cls.model_validate({"no_such_flag": True})
    first = next(iter(options_cls.describe()))
    with pytest.raises(ValidationError):
        options_cls.model_validate({first: "True"})


def test_default_flow_models() -> None:
    assert isinstance(BeggsBrill().flow_model, BeggsBrillMap)
    assert isinstance(MukherjeeBrill().flow_model, MukherjeeBrillFlowModel)
    assert isinstance(HasanKabir().flow_model, HasanKabirFlowModel)
    assert isinstance(Ansari().flow_model, AnsariFlowModel)
    assert isinstance(BeggsBrill().flow_model, IFlowModel)


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_calc_returns_correlation_result(
    model_cls: type, aw_state: Callable[..., FlowState]
) -> None:
    result = model_cls().calc(aw_state(0.1, 1.0, 0.0508, 20.0))
    assert isinstance(result, CorrelationResult)
    assert isinstance(result.pattern, FlowPattern)
    assert 0.0 <= result.holdup <= 1.0
    assert all(isinstance(v, float) for v in result.details.values())


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_calc_is_deterministic(model_cls: type, aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.3, 2.0, 0.0508, 5.0)
    model = model_cls()
    assert model.calc(state) == model.calc(state)


# --- инъекция модуля режима ---
def test_beggs_brill_uses_injected_flow_model(aw_state: Callable[..., FlowState]) -> None:
    stub = FixedFlowModel(FlowPattern.DISTRIBUTED)
    model = BeggsBrill(flow_model=stub)
    result = model.calc(aw_state(0.3, 2.0, 0.0508, 5.0))
    assert model.flow_model is stub
    assert stub.calls == 1
    assert result.pattern is FlowPattern.DISTRIBUTED


def test_mukherjee_brill_uses_injected_flow_model(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(0.001, 10.0, 0.1, 90.0)  # по умолчанию — ANNULAR
    assert MukherjeeBrill().calc(state).pattern is FlowPattern.ANNULAR
    stub = FixedFlowModel(FlowPattern.SLUG)
    model = MukherjeeBrill(flow_model=stub)
    result = model.calc(state)
    assert model.flow_model is stub
    assert stub.calls == 1
    assert result.pattern is FlowPattern.SLUG


@pytest.mark.parametrize(
    ("model_cls", "pattern"),
    [
        (BeggsBrill, FlowPattern.CHURN),
        (BeggsBrill, FlowPattern.ANNULAR),
        (MukherjeeBrill, FlowPattern.TRANSITION),
        (MukherjeeBrill, FlowPattern.CHURN),
        (HasanKabir, FlowPattern.STRATIFIED),
        (HasanKabir, FlowPattern.INTERMITTENT),
        (Ansari, FlowPattern.STRATIFIED),
        (Ansari, FlowPattern.TRANSITION),
    ],
)
def test_unsupported_pattern_from_injected_flow_model(
    model_cls: type, pattern: FlowPattern, aw_state: Callable[..., FlowState]
) -> None:
    model = model_cls(flow_model=FixedFlowModel(pattern))
    with pytest.raises(UnsupportedPatternError) as info:
        model.calc(aw_state(0.3, 2.0, 0.0508, 5.0))
    assert info.value.pattern is pattern
    assert isinstance(info.value, ValueError)


# --- однофазные пределы ---
@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_single_phase_liquid_exact(
    model_cls: type, aw_state: Callable[..., FlowState], aw_fluid_kwargs: dict[str, Any]
) -> None:
    state = aw_state(1.5, 0.0, 0.0508, 30.0, eps_m=1.0e-5)
    result = model_cls().calc(state)
    re = aw_fluid_kwargs["rho_l_kg_m3"] * 1.5 * 0.0508 / aw_fluid_kwargs["mu_l_pa_s"]
    fric = (
        friction.moody(re, 1.0e-5 / 0.0508) * aw_fluid_kwargs["rho_l_kg_m3"] * 1.5**2 / (2 * 0.0508)
    )
    assert result.pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert result.holdup == 1.0
    assert result.dpdl_friction_pa_m == pytest.approx(fric, rel=1e-12)
    assert result.dpdl_gravity_pa_m == pytest.approx(
        aw_fluid_kwargs["rho_l_kg_m3"] * 9.81 * math.sin(math.radians(30.0)), rel=1e-12
    )
    assert result.dpdl_acceleration_pa_m == 0.0


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_single_phase_gas_exact(
    model_cls: type, aw_state: Callable[..., FlowState], aw_fluid_kwargs: dict[str, Any]
) -> None:
    state = aw_state(0.0, 8.0, 0.0508, -20.0)
    result = model_cls().calc(state)
    re = aw_fluid_kwargs["rho_g_kg_m3"] * 8.0 * 0.0508 / aw_fluid_kwargs["mu_g_pa_s"]
    fric = friction.moody(re, 0.0) * aw_fluid_kwargs["rho_g_kg_m3"] * 8.0**2 / (2 * 0.0508)
    assert result.pattern is FlowPattern.SINGLE_PHASE_GAS
    assert result.holdup == 0.0
    assert result.dpdl_friction_pa_m == pytest.approx(fric, rel=1e-12)
    assert result.dpdl_gravity_pa_m == pytest.approx(
        aw_fluid_kwargs["rho_g_kg_m3"] * 9.81 * math.sin(math.radians(-20.0)), rel=1e-12
    )
    assert result.dpdl_acceleration_pa_m == 0.0


@pytest.mark.parametrize("model_cls", CORRELATIONS)
@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0])
def test_vanishing_gas_gives_single_phase_liquid_gradient(
    model_cls: type, theta: float, aw_state: Callable[..., FlowState]
) -> None:
    """ТЗ: v_sg → 0 даёт однофазный градиент жидкости (Colebrook), rtol 1e-6."""
    near = model_cls(LIMIT_OPTIONS[model_cls]).calc(aw_state(1.5, 1.0e-30, 0.0508, theta))
    exact = model_cls(LIMIT_OPTIONS[model_cls]).calc(aw_state(1.5, 0.0, 0.0508, theta))
    assert near.dpdl_pa_m == pytest.approx(exact.dpdl_pa_m, rel=1e-6)
    assert near.holdup == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0])
def test_mukherjee_brill_vanishing_liquid_gives_single_phase_gas_gradient(
    theta: float, aw_state: Callable[..., FlowState]
) -> None:
    """ТЗ: v_sl → 0 даёт однофазный градиент газа (Colebrook), rtol 1e-6 (для MB выполняется).

    Для BB это свойство не выполняется по построению корреляции: S (ур. 16) при λ → 0 не стремится
    к 0 (≈ 0.19–0.37 при λ = 1e-4…1e-31, расчёт агента), f_tp на 20–45% выше f_ns. Расхождение с
    ТЗ («λ → 0 → S → 0») вынесено в отчёт; тест для BB не пишется и не ослабляется.
    """
    options = LIMIT_OPTIONS[MukherjeeBrill]
    near = MukherjeeBrill(options).calc(aw_state(1.0e-30, 8.0, 0.0508, theta))
    exact = MukherjeeBrill(options).calc(aw_state(0.0, 8.0, 0.0508, theta))
    assert near.dpdl_pa_m == pytest.approx(exact.dpdl_pa_m, rel=1e-6)
    assert near.holdup == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("model_cls", CORRELATIONS)
def test_holdup_in_unit_interval_on_grid(
    model_cls: type, aw_state: Callable[..., FlowState]
) -> None:
    model = model_cls()
    for theta in (-90.0, -30.0, 0.0, 30.0, 90.0):
        for v_sl in (0.001, 0.1, 2.0):
            for v_sg in (0.001, 0.5, 15.0):
                result = model.calc(aw_state(v_sl, v_sg, 0.0508, theta))
                assert 0.0 <= result.holdup <= 1.0
                parts = (
                    result.dpdl_gravity_pa_m
                    + result.dpdl_friction_pa_m
                    + result.dpdl_acceleration_pa_m
                )
                assert result.dpdl_pa_m == pytest.approx(parts, rel=1e-9)
