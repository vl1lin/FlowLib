"""Тесты `core/gradient.py` — сборка градиента давления (ТЗ «Дополнительно общие»).

Ожидаемый API (`from correlations.core import gradient`):
    friction_gradient(f_darcy, rho, v, d_m) -> float      f_D·ρ·v²/(2d), Па/м (положительный)
    gravity_gradient(rho, theta_rad) -> float             ρ·g·sinθ, Па/м
    kinetic_number(rho, v_m, v_g, p_pa) -> float          Ek = ρ·v_m·v_g/p   (BB ур. 1; HK (33): ρ_c·v_g²/p)
    with_kinetic_term(gravity, friction, ek) -> GradientComponents
        dpdl = (gravity + friction)/(1 − Ek); acceleration = dpdl − (gravity + friction)
        поля: dpdl_pa_m, gravity_pa_m, friction_pa_m, acceleration_pa_m
        Ek ≥ 1 -> ConvergenceError (особенность знаменателя)
"""

import math
from typing import Any

import pytest

from correlations.core import friction, gradient
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import ConvergenceError


def test_friction_gradient_definition() -> None:
    # reference: Darcy–Weisbach, dp/dL = f·ρ·v²/(2d) (расчёт агента): 0.02·1000·4/(2·0.1) = 400
    assert gradient.friction_gradient(0.02, 1000.0, 2.0, 0.1) == pytest.approx(400.0, rel=1e-12)


def test_friction_gradient_fanning_consistency() -> None:
    """Moody/Fanning: Darcy f_D·ρv²/(2d) ≡ Fanning 2·f_F·ρv²/d при f_D = 4·f_F."""
    f_fanning = friction.fanning_taitel_dukler(1.0e5, laminar=False)
    rho, v, d = 997.0, 1.7, 0.0508
    darcy_form = gradient.friction_gradient(friction.fanning_to_darcy(f_fanning), rho, v, d)
    assert darcy_form == pytest.approx(2.0 * f_fanning * rho * v**2 / d, rel=1e-12)


def test_gravity_gradient() -> None:
    assert gradient.gravity_gradient(500.0, math.pi / 2) == pytest.approx(
        500.0 * GRAVITY_M_S2, rel=1e-12
    )
    assert gradient.gravity_gradient(500.0, 0.0) == 0.0
    assert gradient.gravity_gradient(500.0, -math.pi / 2) == pytest.approx(
        -500.0 * GRAVITY_M_S2, rel=1e-12
    )


def test_kinetic_number_r1(r1_kwargs: dict[str, Any]) -> None:
    # reference: определение Ek = ρ_s·v_m·v_sg/p (BB ур. 1); ρ_s = 448.52 (ТЗ, Пример 4.7)
    v_m = r1_kwargs["v_sl_m_s"] + r1_kwargs["v_sg_m_s"]
    ek = gradient.kinetic_number(448.52, v_m, r1_kwargs["v_sg_m_s"], r1_kwargs["p_pa"])
    assert ek == pytest.approx(448.52 * v_m * r1_kwargs["v_sg_m_s"] / r1_kwargs["p_pa"], rel=1e-12)
    assert 0.0 < ek < 0.001  # при 117 бар ускорение мало


def test_with_kinetic_term_values() -> None:
    # (3000 + 1500)/(1 − 0.1) = 5000; ускорение = 500
    result = gradient.with_kinetic_term(3000.0, 1500.0, 0.1)
    assert result.dpdl_pa_m == pytest.approx(5000.0, rel=1e-12)
    assert result.acceleration_pa_m == pytest.approx(500.0, rel=1e-12)
    assert result.gravity_pa_m == 3000.0
    assert result.friction_pa_m == 1500.0


def test_with_kinetic_term_zero_ek_is_pure_sum() -> None:
    result = gradient.with_kinetic_term(3000.0, 1500.0, 0.0)
    assert result.dpdl_pa_m == 4500.0
    assert result.acceleration_pa_m == 0.0


def test_with_kinetic_term_components_sum_to_total() -> None:
    result = gradient.with_kinetic_term(2345.6, 789.1, 0.37)
    total = result.gravity_pa_m + result.friction_pa_m + result.acceleration_pa_m
    assert result.dpdl_pa_m == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("ek", [1.0, 1.5])
def test_with_kinetic_term_singular_denominator_raises(ek: float) -> None:
    with pytest.raises(ConvergenceError) as info:
        gradient.with_kinetic_term(3000.0, 1500.0, ek)
    assert info.value.params["ek"] == ek
