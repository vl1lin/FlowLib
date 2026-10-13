"""Тесты `core/dimensionless.py` — безразмерные числа (пересечение №1 ТЗ).

Ожидаемый API (`from correlations.core import dimensionless`):
    froude_number(v_m_m_s, d_m) -> float                    N_FR = v_m²/(g·d)
    duns_ros(v_sl, v_sg, rho_l, sigma, mu_l) -> DunsRosNumbers   (n_lv, n_gv, n_l)
    diameter_number(d_m, rho_l, sigma) -> float             N_d = d·√(ρ_L·g/σ)
    reynolds_number(rho, v_m_s, d_m, mu) -> float           Re = ρ·v·d/μ
`from correlations.core.constants import GRAVITY_M_S2`  (= 9.81 м/с²).
"""

from typing import Any

import pytest

from correlations.core import dimensionless
from correlations.core.constants import GRAVITY_M_S2


def test_gravity_constant() -> None:
    # reference: ТЗ/BB — g = 9.81 м/с²
    assert GRAVITY_M_S2 == 9.81


def test_froude_number_r1(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», BB: N_FR = 3.808 (rtol 1e-3)
    v_m = r1_kwargs["v_sl_m_s"] + r1_kwargs["v_sg_m_s"]
    assert dimensionless.froude_number(v_m, r1_kwargs["d_m"]) == pytest.approx(3.808, rel=1e-3)


def test_duns_ros_r1(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны»: BB N_Lv = 11.86; MB N_gv = 11.53, N_L = 0.01176
    nums = dimensionless.duns_ros(
        v_sl_m_s=r1_kwargs["v_sl_m_s"],
        v_sg_m_s=r1_kwargs["v_sg_m_s"],
        rho_l_kg_m3=r1_kwargs["rho_l_kg_m3"],
        sigma_n_m=r1_kwargs["sigma_n_m"],
        mu_l_pa_s=r1_kwargs["mu_l_pa_s"],
    )
    assert nums.n_lv == pytest.approx(11.86, rel=1e-3)
    assert nums.n_gv == pytest.approx(11.53, rel=1e-3)
    assert nums.n_l == pytest.approx(0.01176, rel=1e-3)


def test_duns_ros_n_l_includes_gravity(r1_kwargs: dict[str, Any]) -> None:
    """ТЗ, п. 1 «Пересечений»: N_L с g (в JERT опечатка без g).

    Для R1 значение без g было бы ≈ 0.0066, а не 0.01176.
    """
    nums = dimensionless.duns_ros(
        v_sl_m_s=1.0,
        v_sg_m_s=1.0,
        rho_l_kg_m3=r1_kwargs["rho_l_kg_m3"],
        sigma_n_m=r1_kwargs["sigma_n_m"],
        mu_l_pa_s=r1_kwargs["mu_l_pa_s"],
    )
    without_g = (
        r1_kwargs["mu_l_pa_s"]
        * (1.0 / (r1_kwargs["rho_l_kg_m3"] * r1_kwargs["sigma_n_m"] ** 3)) ** 0.25
    )
    assert nums.n_l != pytest.approx(without_g, rel=0.1)
    assert nums.n_l / without_g == pytest.approx(GRAVITY_M_S2**0.25, rel=1e-12)


def test_duns_ros_velocity_numbers_are_linear_in_velocity() -> None:
    a = dimensionless.duns_ros(1.0, 2.0, 1000.0, 0.07, 1.0e-3)
    b = dimensionless.duns_ros(3.0, 6.0, 1000.0, 0.07, 1.0e-3)
    assert b.n_lv == pytest.approx(3.0 * a.n_lv, rel=1e-12)
    assert b.n_gv == pytest.approx(3.0 * a.n_gv, rel=1e-12)
    assert b.n_l == pytest.approx(a.n_l, rel=1e-12)


def test_diameter_number_r1(r1_kwargs: dict[str, Any]) -> None:
    # reference: определение N_d = d·√(ρ_L g/σ) (расчёт агента, R1): 143.74
    n_d = dimensionless.diameter_number(
        r1_kwargs["d_m"], r1_kwargs["rho_l_kg_m3"], r1_kwargs["sigma_n_m"]
    )
    assert n_d == pytest.approx(143.74, rel=1e-3)


def test_reynolds_number_definition() -> None:
    # reference: определение Re = ρvd/μ (расчёт агента): вода, 1 м/с, 0.1 м
    assert dimensionless.reynolds_number(1000.0, 1.0, 0.1, 1.0e-3) == pytest.approx(1.0e5)
