"""Тесты `core/mixture.py` — свойства смеси (пересечение №2 ТЗ).

Ожидаемый API (`from correlations.core import mixture`):
    no_slip(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g) -> NoSlipMixture
        поля: lam, rho_ns_kg_m3, mu_ns_pa_s, v_m_m_s     (λ_L = v_sl/v_m; линейное смешение)
    slip(holdup, rho_l, rho_g, mu_l, mu_g) -> SlipMixture
        поля: rho_s_kg_m3, mu_s_pa_s                     (смешение по истинному H_L)
"""

from typing import Any

import pytest

from correlations.core import dimensionless, mixture


def no_slip_r1(r1: dict[str, Any]) -> mixture.NoSlipMixture:
    return mixture.no_slip(
        v_sl_m_s=r1["v_sl_m_s"],
        v_sg_m_s=r1["v_sg_m_s"],
        rho_l_kg_m3=r1["rho_l_kg_m3"],
        rho_g_kg_m3=r1["rho_g_kg_m3"],
        mu_l_pa_s=r1["mu_l_pa_s"],
        mu_g_pa_s=r1["mu_g_pa_s"],
    )


def test_no_slip_fraction_r1(r1_kwargs: dict[str, Any]) -> None:
    # reference: определение λ_L = v_sl/(v_sl+v_sg) = 1.21/2.386 (расчёт агента) = 0.50712
    mix = no_slip_r1(r1_kwargs)
    assert mix.lam == pytest.approx(0.50712, rel=1e-4)
    assert mix.v_m_m_s == pytest.approx(2.386, rel=1e-12)


def test_reynolds_ns_r1_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», BB: Re_ns = 3.152e5 (rtol 1e-3)
    mix = no_slip_r1(r1_kwargs)
    re_ns = dimensionless.reynolds_number(
        mix.rho_ns_kg_m3, mix.v_m_m_s, r1_kwargs["d_m"], mix.mu_ns_pa_s
    )
    assert re_ns == pytest.approx(3.152e5, rel=1e-3)


def test_no_slip_density_and_viscosity_are_linear_mixes(r1_kwargs: dict[str, Any]) -> None:
    mix = no_slip_r1(r1_kwargs)
    lam = mix.lam
    assert mix.rho_ns_kg_m3 == pytest.approx(
        lam * r1_kwargs["rho_l_kg_m3"] + (1 - lam) * r1_kwargs["rho_g_kg_m3"], rel=1e-12
    )
    assert mix.mu_ns_pa_s == pytest.approx(
        lam * r1_kwargs["mu_l_pa_s"] + (1 - lam) * r1_kwargs["mu_g_pa_s"], rel=1e-12
    )


def test_no_slip_single_phase_limits(r1_kwargs: dict[str, Any]) -> None:
    liquid = mixture.no_slip(1.0, 0.0, 1000.0, 1.2, 1.0e-3, 1.8e-5)
    assert liquid.lam == 1.0
    assert liquid.rho_ns_kg_m3 == 1000.0
    assert liquid.mu_ns_pa_s == 1.0e-3
    gas = mixture.no_slip(0.0, 1.0, 1000.0, 1.2, 1.0e-3, 1.8e-5)
    assert gas.lam == 0.0
    assert gas.rho_ns_kg_m3 == 1.2
    assert gas.mu_ns_pa_s == 1.8e-5


def test_slip_density_matches_tz_example_4_7(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, таблица «Опечатки», Пример 4.7: ρ_s = 762.64·0.53 + … → итог 448.52
    mix = mixture.slip(
        holdup=0.53,
        rho_l_kg_m3=r1_kwargs["rho_l_kg_m3"],
        rho_g_kg_m3=r1_kwargs["rho_g_kg_m3"],
        mu_l_pa_s=r1_kwargs["mu_l_pa_s"],
        mu_g_pa_s=r1_kwargs["mu_g_pa_s"],
    )
    assert mix.rho_s_kg_m3 == pytest.approx(448.52, rel=1e-3)


@pytest.mark.parametrize(("holdup", "rho", "mu"), [(1.0, 1000.0, 1.0e-3), (0.0, 1.2, 1.8e-5)])
def test_slip_limits(holdup: float, rho: float, mu: float) -> None:
    mix = mixture.slip(holdup, 1000.0, 1.2, 1.0e-3, 1.8e-5)
    assert mix.rho_s_kg_m3 == rho
    assert mix.mu_s_pa_s == mu


def test_slip_is_linear_in_holdup() -> None:
    mix = mixture.slip(0.5, 1000.0, 1.0, 2.0e-3, 2.0e-5)
    assert mix.rho_s_kg_m3 == pytest.approx(500.5, rel=1e-12)
    assert mix.mu_s_pa_s == pytest.approx(1.01e-3, rel=1e-12)
