"""Тесты `core/errors.py` (ТЗ, «Общие правила / Исключения»)."""

import warnings

import pytest

from correlations.core.errors import ConvergenceError, OutOfRangeWarning


def test_convergence_error_carries_params() -> None:
    err = ConvergenceError("не сошлось", params={"x0": 0.15, "iterations": 50.0})
    assert err.params == {"x0": 0.15, "iterations": 50.0}
    text = str(err)
    assert "не сошлось" in text
    assert "x0" in text  # параметры видны в сообщении для диагностики


def test_convergence_error_is_exception() -> None:
    with pytest.raises(ConvergenceError):
        raise ConvergenceError("fail", params={})


def test_out_of_range_warning_is_warning() -> None:
    assert issubclass(OutOfRangeWarning, Warning)


def test_out_of_range_warning_can_be_emitted_and_caught() -> None:
    with pytest.warns(OutOfRangeWarning, match="Re плёнки"):
        warnings.warn("Re плёнки < 3000", OutOfRangeWarning, stacklevel=1)


def test_out_of_range_warning_context_defaults_to_empty() -> None:
    warning = OutOfRangeWarning("текст")
    assert str(warning) == "текст"
    assert warning.context == {}


def test_out_of_range_warning_carries_context() -> None:
    """Решение пользователя: угол и другие величины — атрибутом, текст сообщения не меняется."""
    warning = OutOfRangeWarning("текст", {"theta_deg": 45.0})
    assert str(warning) == "текст"
    assert warning.context == {"theta_deg": 45.0}


def test_out_of_range_warning_context_is_copied() -> None:
    source = {"theta_deg": 45.0}
    warning = OutOfRangeWarning("текст", source)
    source["theta_deg"] = 0.0
    assert warning.context == {"theta_deg": 45.0}


def test_out_of_range_warning_with_context_can_be_emitted() -> None:
    with pytest.warns(OutOfRangeWarning) as record:
        warnings.warn(OutOfRangeWarning("x", {"a": 1.0}), stacklevel=1)
    message = record[0].message
    assert isinstance(message, OutOfRangeWarning)
    assert message.context == {"a": 1.0}


def test_unsupported_pattern_error_carries_pattern() -> None:
    from correlations.core.errors import UnsupportedPatternError
    from correlations.core.patterns import FlowPattern

    err = UnsupportedPatternError(FlowPattern.CHURN, "beggs_brill")
    assert isinstance(err, ValueError)
    assert err.pattern is FlowPattern.CHURN
    assert "churn" in str(err)
    assert "beggs_brill" in str(err)
