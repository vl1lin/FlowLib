"""Тесты `core/state.py::FlowPatternResult` и `CorrelationResult` (ТЗ, раздел «Выход»).

Ожидаемый API: `from correlations.core.state import CorrelationResult, FlowPatternResult`;
`from correlations.core.patterns import FlowPattern`.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowPatternResult


def corr_kwargs(**changes: Any) -> dict[str, Any]:
    """Согласованный набор полей `CorrelationResult`: 4500 = 3000 + 1400 + 100."""
    base: dict[str, Any] = {
        "pattern": FlowPattern.SLUG,
        "holdup": 0.55,
        "dpdl_pa_m": 4500.0,
        "dpdl_gravity_pa_m": 3000.0,
        "dpdl_friction_pa_m": 1400.0,
        "dpdl_acceleration_pa_m": 100.0,
        "details": {"x_m": 1.0},
    }
    return {**base, **changes}


class TestFlowPatternResult:
    def test_valid(self) -> None:
        result = FlowPatternResult(pattern=FlowPattern.BUBBLE, details={"h_l_over_d": 0.3})
        assert result.pattern is FlowPattern.BUBBLE
        assert result.details == {"h_l_over_d": 0.3}

    def test_details_must_be_floats(self) -> None:
        with pytest.raises(ValidationError):
            FlowPatternResult.model_validate(
                {"pattern": FlowPattern.BUBBLE, "details": {"x": "not a number"}}
            )

    def test_frozen(self) -> None:
        result = FlowPatternResult(pattern=FlowPattern.BUBBLE, details={})
        with pytest.raises(ValidationError):
            result.pattern = FlowPattern.SLUG  # type: ignore[misc]  # проверка frozen


class TestCorrelationResult:
    def test_valid(self) -> None:
        result = CorrelationResult(**corr_kwargs())
        assert result.pattern is FlowPattern.SLUG
        assert result.holdup == 0.55
        assert result.dpdl_pa_m == 4500.0

    @pytest.mark.parametrize("holdup", [0.0, 1.0])
    def test_holdup_bounds_inclusive(self, holdup: float) -> None:
        assert CorrelationResult(**corr_kwargs(holdup=holdup)).holdup == holdup

    @pytest.mark.parametrize("holdup", [-0.001, 1.001])
    def test_holdup_out_of_range_rejected(self, holdup: float) -> None:
        with pytest.raises(ValidationError):
            CorrelationResult(**corr_kwargs(holdup=holdup))

    def test_components_sum_enforced(self) -> None:
        with pytest.raises(ValidationError):
            CorrelationResult(**corr_kwargs(dpdl_pa_m=4600.0))

    def test_components_sum_within_rtol_1e10_accepted(self) -> None:
        # reference: ТЗ «Выход» — сумма компонент проверяется с rtol 1e-10
        result = CorrelationResult(**corr_kwargs(dpdl_pa_m=4500.0 * (1 + 1e-11)))
        assert result.dpdl_pa_m == pytest.approx(4500.0, rel=1e-10)

    def test_components_sum_beyond_rtol_1e10_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorrelationResult(**corr_kwargs(dpdl_pa_m=4500.0 * (1 + 1e-8)))

    def test_zero_acceleration_allowed(self) -> None:
        result = CorrelationResult(
            **corr_kwargs(dpdl_pa_m=4400.0, dpdl_acceleration_pa_m=0.0),
        )
        assert result.dpdl_acceleration_pa_m == 0.0

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CorrelationResult(**corr_kwargs(unknown=1.0))

    def test_frozen(self) -> None:
        result = CorrelationResult(**corr_kwargs())
        with pytest.raises(ValidationError):
            result.holdup = 0.1  # type: ignore[misc]  # проверка frozen
