"""Тесты `core/stratified.py` — расслоённый поток (пересечение №14 + критерии KH и Jeffreys).

Соглашение знака: θ > 0 — подъём (как во всей библиотеке). У Taitel–Dukler α = −θ; параметр
Y здесь определяется для θ: Y = Δρ·g·sinθ/(dp/dx)_G (Barnea [12b]); в уравнении (7) TD член
−4Y_TD превращается в +4Y.

Ожидаемый API (`from correlations.core import stratified`):
    geometry(h_tilde) -> StratifiedGeometry    поля: h, a_l, a_g, s_l, s_g, s_i,
                                               u_l, u_g (безразм. скорости Ã/Ã_L, Ã/Ã_G),
                                               d_l, d_g (безразм. гидравл. диаметры, TD (6))
    delta_from_holdup(h_l) -> float            δ из H_L = (δ − sin δ)/(2π)      MB (20)
    level_from_delta(delta) -> float           h_L/d = 0.5·(1 − cos(δ/2))       MB (18)
    x_squared(grad_l, grad_g) -> float         X² = (dp/dx)_LS/(dp/dx)_GS       TD (8)
    y_parameter(rho_l, rho_g, theta_rad, grad_g) -> float                       TD (9), Barnea [12b]
    equilibrium_residual(h, x_sq, y, *, n=0.2, m=0.2) -> float                  TD (7)
    x_squared_for_level(h, y, *, n=0.2, m=0.2) -> float                         решение (7) относительно X²
    equilibrium_levels(x_sq, y, *, n=0.2, m=0.2) -> tuple[float, ...]           все корни, по возрастанию
    equilibrium_level(x_sq, y, *, n=0.2, m=0.2) -> float | None                 наименьший корень (Q9 ТЗ)
    modified_froude(v_sg, rho_l, rho_g, d_m, theta_rad) -> float                F, TD / Barnea [10]
    kelvin_helmholtz_parameter(froude, h) -> float                              левая часть [9]/(25); < 1 — устойчиво
    jeffreys_wavy(k, h, *, sheltering=0.01) -> bool                             TD (30), Barnea [20]
"""

import math
from typing import Any

import pytest

from correlations.core import friction, gradient, stratified
from correlations.core.constants import GRAVITY_M_S2


# --- геометрия -------------------------------------------------------------------------
def test_geometry_half_filled_pipe() -> None:
    geo = stratified.geometry(0.5)
    # reference: TD (10)–(14) при h̃ = 0.5 (расчёт агента)
    assert geo.a_l == pytest.approx(math.pi / 8, rel=1e-12)
    assert geo.a_g == pytest.approx(math.pi / 8, rel=1e-12)
    assert geo.s_l == pytest.approx(math.pi / 2, rel=1e-12)
    assert geo.s_g == pytest.approx(math.pi / 2, rel=1e-12)
    assert geo.s_i == pytest.approx(1.0, rel=1e-12)
    assert geo.u_l == pytest.approx(2.0, rel=1e-12)  # Ã/Ã_L = (π/4)/(π/8)
    assert geo.u_g == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("h", [0.05, 0.25, 0.5, 0.75, 0.95])
def test_geometry_invariants(h: float) -> None:
    geo = stratified.geometry(h)
    assert geo.a_l + geo.a_g == pytest.approx(math.pi / 4, rel=1e-12)
    assert geo.s_l + geo.s_g == pytest.approx(math.pi, rel=1e-12)
    assert geo.d_l == pytest.approx(4.0 * geo.a_l / geo.s_l, rel=1e-12)
    assert geo.d_g == pytest.approx(4.0 * geo.a_g / (geo.s_g + geo.s_i), rel=1e-12)


def test_geometry_liquid_area_increases_with_level() -> None:
    areas = [stratified.geometry(h).a_l for h in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert areas == sorted(areas)


def test_liquid_area_derivative_equals_interface_width() -> None:
    """dÃ_L/dh̃ = S̃_i (используется в критерии Кельвина–Гельмгольца)."""
    h, step = 0.4, 1e-6
    derivative = (stratified.geometry(h + step).a_l - stratified.geometry(h - step).a_l) / (
        2 * step
    )
    assert derivative == pytest.approx(stratified.geometry(h).s_i, rel=1e-6)


@pytest.mark.parametrize("holdup", [0.05, 0.3, 0.5, 0.8])
def test_delta_from_holdup_roundtrip(holdup: float) -> None:
    # reference: MB (20): H_L = (δ − sin δ)/(2π)
    delta = stratified.delta_from_holdup(holdup)
    assert (delta - math.sin(delta)) / (2.0 * math.pi) == pytest.approx(holdup, rel=1e-10)


def test_delta_and_level_for_half_filled_pipe() -> None:
    delta = stratified.delta_from_holdup(0.5)
    assert delta == pytest.approx(math.pi, rel=1e-10)
    # reference: MB (18): h_L/d = 0.5·(1 − cos(δ/2)) = 0.5 при δ = π
    assert stratified.level_from_delta(delta) == pytest.approx(0.5, rel=1e-10)


def test_level_from_holdup_matches_tz_geometry() -> None:
    """Согласованность двух записей геометрии: площадь жидкости Ã_L(h̃) = H_L·π/4."""
    holdup = 0.3
    h = stratified.level_from_delta(stratified.delta_from_holdup(holdup))
    assert stratified.geometry(h).a_l == pytest.approx(holdup * math.pi / 4.0, rel=1e-9)


# --- X, Y, равновесный уровень ------------------------------------------------------------
def test_x_squared_for_level_half_filled_matches_tz() -> None:
    # reference: ТЗ, «Контрольные точки», TD: X при h̃_L = 0.5, Y = 0, турб.–турб.: 1.6 (пересчёт 1.584), rtol 2%
    x_sq = stratified.x_squared_for_level(0.5, 0.0)
    assert math.sqrt(x_sq) == pytest.approx(1.6, rel=2e-2)
    assert math.sqrt(x_sq) == pytest.approx(1.584, rel=1e-3)


@pytest.mark.parametrize("y", [0.0, 5.0, -5.0, -50.0])
@pytest.mark.parametrize("h", [0.1, 0.3, 0.5, 0.7])
def test_equilibrium_level_roundtrip(h: float, y: float) -> None:
    """Заданный уровень — один из корней (7). При Y > 0 корней может быть несколько (Q9 ТЗ),
    поэтому проверяется вхождение h в список всех корней, а не равенство наименьшему."""
    x_sq = stratified.x_squared_for_level(h, y)
    levels = stratified.equilibrium_levels(x_sq, y)
    assert any(level == pytest.approx(h, rel=1e-8) for level in levels)
    assert stratified.equilibrium_residual(h, x_sq, y) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("y", [0.0, -5.0, -50.0])
@pytest.mark.parametrize("h", [0.1, 0.3, 0.5, 0.7])
def test_equilibrium_level_roundtrip_unique_root(h: float, y: float) -> None:
    """При Y ≤ 0 корень единственный: `equilibrium_level` возвращает заданный уровень."""
    x_sq = stratified.x_squared_for_level(h, y)
    assert stratified.equilibrium_level(x_sq, y) == pytest.approx(h, rel=1e-8)


def test_equilibrium_level_grows_with_liquid_to_gas_gradient_ratio() -> None:
    levels = [stratified.equilibrium_level(x_sq, 0.0) for x_sq in (0.01, 0.1, 1.0, 10.0)]
    found = [level for level in levels if level is not None]
    assert len(found) == len(levels)
    assert found == sorted(found)


def test_equilibrium_level_upflow_raises_level_downflow_lowers_it() -> None:
    """Подъём (Y > 0, жидкость тормозится гравитацией) — уровень выше; спуск (Y < 0) — ниже."""
    horizontal = stratified.equilibrium_level(1.0, 0.0)
    up = stratified.equilibrium_level(1.0, 5.0)
    down = stratified.equilibrium_level(1.0, -5.0)
    assert horizontal is not None and up is not None and down is not None
    assert down < horizontal < up


def test_equilibrium_levels_are_sorted_and_first_is_smallest() -> None:
    levels = stratified.equilibrium_levels(1.0, 0.0)
    assert list(levels) == sorted(levels)
    assert stratified.equilibrium_level(1.0, 0.0) == levels[0]


def test_equilibrium_exponents_laminar_gas_changes_solution() -> None:
    """TD: возможны режимы турб./лам. (n = 0.2, m = 1). Решение близко, но не совпадает."""
    turb = stratified.equilibrium_level(1.0, 0.0, n=0.2, m=0.2)
    lam_gas = stratified.equilibrium_level(1.0, 0.0, n=0.2, m=1.0)
    assert turb is not None and lam_gas is not None
    assert lam_gas != turb
    assert lam_gas == pytest.approx(turb, abs=0.1)  # «remarkably close» (TD, текст к рис. 2)


def test_y_parameter_sign_follows_theta() -> None:
    up = stratified.y_parameter(1000.0, 1.0, math.radians(10.0), 50.0)
    down = stratified.y_parameter(1000.0, 1.0, math.radians(-10.0), 50.0)
    horizontal = stratified.y_parameter(1000.0, 1.0, 0.0, 50.0)
    assert up > 0.0 > down
    assert horizontal == 0.0
    # reference: TD (9), Barnea [12b] (расчёт агента): Δρ·g·sinθ/grad_g
    assert up == pytest.approx(999.0 * GRAVITY_M_S2 * math.sin(math.radians(10.0)) / 50.0)


def test_x_squared_is_ratio_of_gradients() -> None:
    assert stratified.x_squared(30.0, 120.0) == pytest.approx(0.25, rel=1e-12)


def test_td_x_and_y_same_for_darcy_and_fanning_definitions(aw_fluid_kwargs: dict[str, Any]) -> None:
    """Moody/Fanning для X², Y и равновесного уровня.

    TD (8): X² через Fanning: (4/D)·f_F·ρu²/2 для каждой фазы. Ядро — через Darcy
    (`gradient.friction_gradient(4·f_F, …)`). X², Y и найденный уровень должны совпасть.
    """
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    mu_l, mu_g, d = aw_fluid_kwargs["mu_l_pa_s"], aw_fluid_kwargs["mu_g_pa_s"], 0.0508
    v_sl, v_sg, theta = 0.05, 8.0, math.radians(1.0)
    re_l = rho_l * v_sl * d / mu_l
    re_g = rho_g * v_sg * d / mu_g
    f_l = friction.fanning_taitel_dukler(re_l, laminar=False)
    f_g = friction.fanning_taitel_dukler(re_g, laminar=False)

    fanning_grad_l = 4.0 / d * f_l * rho_l * v_sl**2 / 2.0  # TD (8)
    fanning_grad_g = 4.0 / d * f_g * rho_g * v_sg**2 / 2.0
    darcy_grad_l = gradient.friction_gradient(friction.fanning_to_darcy(f_l), rho_l, v_sl, d)
    darcy_grad_g = gradient.friction_gradient(friction.fanning_to_darcy(f_g), rho_g, v_sg, d)
    assert darcy_grad_l == pytest.approx(fanning_grad_l, rel=1e-12)
    assert darcy_grad_g == pytest.approx(fanning_grad_g, rel=1e-12)

    x_darcy = stratified.x_squared(darcy_grad_l, darcy_grad_g)
    x_fanning = stratified.x_squared(fanning_grad_l, fanning_grad_g)
    y_darcy = stratified.y_parameter(rho_l, rho_g, theta, darcy_grad_g)
    y_fanning = stratified.y_parameter(rho_l, rho_g, theta, fanning_grad_g)
    assert x_darcy == pytest.approx(x_fanning, rel=1e-12)
    assert y_darcy == pytest.approx(y_fanning, rel=1e-12)
    level_darcy = stratified.equilibrium_level(x_darcy, y_darcy)
    level_fanning = stratified.equilibrium_level(x_fanning, y_fanning)
    assert level_darcy is not None and level_fanning is not None
    assert level_darcy == pytest.approx(0.519, rel=1e-2)  # расчёт агента: 0.5189
    assert level_darcy == pytest.approx(level_fanning, rel=1e-9)


def test_missing_factor_four_changes_the_gradient_and_y(aw_fluid_kwargs: dict[str, Any]) -> None:
    """Контроль чувствительности: Fanning вместо Darcy (потерян ×4) искажает градиент и Y в 4 раза."""
    rho_l, rho_g = aw_fluid_kwargs["rho_l_kg_m3"], aw_fluid_kwargs["rho_g_kg_m3"]
    d, v_sg = 0.0508, 8.0
    f_g = friction.fanning_taitel_dukler(
        rho_g * v_sg * d / aw_fluid_kwargs["mu_g_pa_s"], laminar=False
    )
    right = gradient.friction_gradient(friction.fanning_to_darcy(f_g), rho_g, v_sg, d)
    wrong = gradient.friction_gradient(f_g, rho_g, v_sg, d)
    assert right / wrong == pytest.approx(4.0, rel=1e-12)
    theta = math.radians(1.0)
    assert stratified.y_parameter(rho_l, rho_g, theta, right) != pytest.approx(
        stratified.y_parameter(rho_l, rho_g, theta, wrong), rel=0.5
    )


# --- критерии KH и Jeffreys ---------------------------------------------------------------
def test_modified_froude_horizontal() -> None:
    # reference: TD (F), Barnea [10] (расчёт агента): √(ρ_G/Δρ)·U_GS/√(D·g·cosθ); AW, d = 0.0508, 10 м/с
    f = stratified.modified_froude(10.0, 997.0, 1.18, 0.0508, 0.0)
    assert f == pytest.approx(0.48762, rel=1e-4)


def test_modified_froude_cos_theta_dependence() -> None:
    f0 = stratified.modified_froude(10.0, 997.0, 1.18, 0.0508, 0.0)
    f60 = stratified.modified_froude(10.0, 997.0, 1.18, 0.0508, math.radians(60.0))
    assert f60 / f0 == pytest.approx(1.0 / math.sqrt(0.5), rel=1e-12)
    f_neg = stratified.modified_froude(10.0, 997.0, 1.18, 0.0508, math.radians(-60.0))
    assert f_neg == pytest.approx(f60, rel=1e-12)  # cos чётна


def test_kelvin_helmholtz_parameter_half_filled() -> None:
    # reference: TD (25) с ũ_G² (ТЗ, «Опечатки») при h̃ = 0.5: F²·[ũ_G²·S̃_i/(C2²·Ã_G)],
    # C2 = 1 − h̃ = 0.5, ũ_G = 2, S̃_i = 1, Ã_G = π/8 -> F²·128/π
    assert stratified.kelvin_helmholtz_parameter(1.0, 0.5) == pytest.approx(
        128.0 / math.pi, rel=1e-12
    )


def test_kelvin_helmholtz_parameter_is_quadratic_in_froude() -> None:
    a = stratified.kelvin_helmholtz_parameter(0.3, 0.4)
    b = stratified.kelvin_helmholtz_parameter(0.6, 0.4)
    assert b / a == pytest.approx(4.0, rel=1e-12)


def test_kelvin_helmholtz_stable_below_one_unstable_above() -> None:
    """Критерий TD (25): < 1 — расслоённый поток устойчив. Граница по F при h̃ = 0.5: √(π/128)."""
    f_crit = math.sqrt(math.pi / 128.0)
    assert stratified.kelvin_helmholtz_parameter(0.9 * f_crit, 0.5) < 1.0
    assert stratified.kelvin_helmholtz_parameter(1.1 * f_crit, 0.5) > 1.0


def test_jeffreys_wavy_threshold_half_filled() -> None:
    # reference: Barnea [20], TD (30): K ≥ 2/(ũ_G·√ũ_L·√s); h̃ = 0.5, s = 0.01 -> 2/(2·√2·0.1) = 7.0711
    threshold = 2.0 / (2.0 * math.sqrt(2.0) * 0.1)
    assert stratified.jeffreys_wavy(threshold * 1.001, 0.5)
    assert not stratified.jeffreys_wavy(threshold * 0.999, 0.5)


def test_jeffreys_sheltering_coefficient_is_parameter() -> None:
    k = 7.5
    assert stratified.jeffreys_wavy(k, 0.5, sheltering=0.01)
    assert not stratified.jeffreys_wavy(k, 0.5, sheltering=0.005)  # порог растёт как 1/√s


@pytest.mark.parametrize("holdup", [1e-12, 1e-9, 1e-6])
def test_delta_from_holdup_small_holdup_keeps_precision(holdup: float) -> None:
    """При H_L ≪ 1 разность δ − sin δ теряет точность; решение должно удовлетворять δ³/6 ≈ 2π·H_L."""
    delta = stratified.delta_from_holdup(holdup)
    assert delta**3 / 6.0 == pytest.approx(2.0 * math.pi * holdup, rel=1e-4)


def test_delta_from_holdup_zero_and_one() -> None:
    assert stratified.delta_from_holdup(0.0) == 0.0
    assert stratified.delta_from_holdup(1.0) == pytest.approx(2.0 * math.pi, rel=1e-10)
