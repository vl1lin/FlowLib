"""Тесты `core/state.py::FlowState` — валидация входа (pydantic), ТЗ «Вход FlowState».

Ожидаемый API: `from correlations.core.state import FlowState`.
Любое нарушение условий -> `pydantic.ValidationError`.
"""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from correlations.core.state import FlowState


def make(base: dict[str, Any], **changes: Any) -> FlowState:
    """Строит `FlowState` из набора `base` с заменой полей `changes`."""
    return FlowState(**{**base, **changes})


def test_r1_is_valid_and_defaults(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    assert state.v_sl_m_s == 1.21
    assert state.theta_deg == 90.0
    assert state.d_t_m == 0.0  # труба, без НКТ — умолчание ТЗ


def test_r2_is_valid(r2_kwargs: dict[str, Any]) -> None:
    assert FlowState(**r2_kwargs).rho_l_kg_m3 == 761.7


def test_eps_defaults_to_zero(r1_kwargs: dict[str, Any]) -> None:
    kwargs = {k: v for k, v in r1_kwargs.items() if k != "eps_m"}
    assert FlowState(**kwargs).eps_m == 0.0


@pytest.mark.parametrize("field", ["v_sl_m_s", "v_sg_m_s"])
def test_negative_velocity_rejected(r1_kwargs: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, **{field: -0.1})


@pytest.mark.parametrize(
    ("v_sl", "v_sg"), [(0.0, 1.0), (1.0, 0.0)], ids=["single_gas", "single_liquid"]
)
def test_single_phase_velocities_allowed(
    r1_kwargs: dict[str, Any], v_sl: float, v_sg: float
) -> None:
    state = make(r1_kwargs, v_sl_m_s=v_sl, v_sg_m_s=v_sg)
    assert (state.v_sl_m_s, state.v_sg_m_s) == (v_sl, v_sg)


def test_both_velocities_zero_rejected(r1_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, v_sl_m_s=0.0, v_sg_m_s=0.0)


@pytest.mark.parametrize(
    "field",
    ["rho_l_kg_m3", "rho_g_kg_m3", "mu_l_pa_s", "mu_g_pa_s", "sigma_n_m", "d_m", "p_pa"],
)
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_strictly_positive_fields(r1_kwargs: dict[str, Any], field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, **{field: bad})


@pytest.mark.parametrize("field", ["d_t_m", "eps_m"])
def test_negative_d_t_and_eps_rejected(r1_kwargs: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, **{field: -1e-6})


@pytest.mark.parametrize("rho_g", [762.64, 800.0], ids=["equal", "gas_heavier"])
def test_rho_l_must_exceed_rho_g(r1_kwargs: dict[str, Any], rho_g: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, rho_g_kg_m3=rho_g)


def test_annulus_d_t_less_than_d_allowed(r1_kwargs: dict[str, Any]) -> None:
    assert make(r1_kwargs, d_t_m=0.06).d_t_m == 0.06


@pytest.mark.parametrize("d_t", [0.1524, 0.2], ids=["equal", "larger"])
def test_d_t_must_be_less_than_d(r1_kwargs: dict[str, Any], d_t: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, d_t_m=d_t)


@pytest.mark.parametrize("eps", [0.1524, 0.5], ids=["equal", "larger"])
def test_eps_must_be_less_than_d(r1_kwargs: dict[str, Any], eps: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, eps_m=eps)


@pytest.mark.parametrize("theta", [-90.0, 0.0, 90.0])
def test_theta_bounds_inclusive(r1_kwargs: dict[str, Any], theta: float) -> None:
    assert make(r1_kwargs, theta_deg=theta).theta_deg == theta


@pytest.mark.parametrize("theta", [-90.001, 90.001, 180.0])
def test_theta_out_of_range_rejected(r1_kwargs: dict[str, Any], theta: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, theta_deg=theta)


# Добавлено агентом (не из ТЗ напрямую): NaN/inf во входе — заведомо некорректные данные.
@pytest.mark.parametrize("field", ["v_sl_m_s", "rho_l_kg_m3", "d_m", "theta_deg", "p_pa"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_rejected(r1_kwargs: dict[str, Any], field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, **{field: bad})


def test_strict_rejects_string_numbers(r1_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, d_m="0.1524")


def test_extra_field_forbidden(r1_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make(r1_kwargs, unknown_field=1.0)


@pytest.mark.parametrize("missing", ["v_sl_m_s", "rho_l_kg_m3", "theta_deg", "p_pa"])
def test_required_fields(r1_kwargs: dict[str, Any], missing: str) -> None:
    kwargs = {k: v for k, v in r1_kwargs.items() if k != missing}
    with pytest.raises(ValidationError):
        FlowState(**kwargs)


def test_frozen(r1_kwargs: dict[str, Any]) -> None:
    state = FlowState(**r1_kwargs)
    with pytest.raises(ValidationError):
        state.d_m = 0.2  # type: ignore[misc]  # проверяем именно запрет присваивания
