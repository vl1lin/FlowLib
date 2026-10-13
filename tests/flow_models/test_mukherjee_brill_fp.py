"""Тесты `flow_models/mukherjee_brill_fp.py` — карта режимов Mukherjee–Brill (IFlowModel).

Ожидаемый API (`from correlations.flow_models.mukherjee_brill_fp import ...`):
    MukherjeeBrillFlowOptions()     — флагов нет (ТЗ)
    MukherjeeBrillFlowModel(options=None); .name == "mukherjee_brill_fp"
    n_gv_slug_annular(n_l, n_lv) -> float                   N_gvSM  [2]
    n_lv_bubble_slug(n_gv, n_l, theta_rad) -> float         N_LvBS  [1]   (подъём)
    n_gv_bubble_slug(n_lv, n_l, theta_rad) -> float         N_gvBS  [3]   (спуск и горизонталь)
    n_lv_stratified(n_gv, n_l, theta_rad) -> float          N_LvST  [4]   (спуск и горизонталь)
details: n_lv, n_gv, n_l, n_gv_sm и (по ветви) n_lv_bs / n_gv_bs / n_lv_st.
Источник: Mukherjee H., Brill J.P., 1985, IJMF 11(3), ур. [1]–[4] и рис. 16 (блок-схема);
в [3] sin θ (опечатка «sin 0» исправлена, ТЗ «Опечатки»); в [4] член −0.017·N_gv линейный
(так в обоих оригиналах — не исправлять).
"""

import math
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core import dimensionless
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.mukherjee_brill_fp import (
    MukherjeeBrillFlowModel,
    MukherjeeBrillFlowOptions,
    n_gv_bubble_slug,
    n_gv_slug_annular,
    n_lv_bubble_slug,
    n_lv_stratified,
)

D = 0.1


def r1_numbers(r1: dict[str, Any]) -> dimensionless.DunsRosNumbers:
    return dimensionless.duns_ros(
        r1["v_sl_m_s"], r1["v_sg_m_s"], r1["rho_l_kg_m3"], r1["sigma_n_m"], r1["mu_l_pa_s"]
    )


def test_name_and_no_flags() -> None:
    model = MukherjeeBrillFlowModel()
    assert model.name == "mukherjee_brill_fp"
    assert MukherjeeBrillFlowOptions.describe() == {}  # флагов нет
    assert model.options == MukherjeeBrillFlowOptions()


def test_n_gv_sm_r1_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», MB: N_gvSM = 350.7 (rtol 1e-3)
    nums = r1_numbers(r1_kwargs)
    assert n_gv_slug_annular(nums.n_l, nums.n_lv) == pytest.approx(350.7, rel=1e-3)


def test_n_lv_bs_r1_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», MB: N_LvBS = 18.38 (rtol 1e-3), θ = 90°
    nums = r1_numbers(r1_kwargs)
    assert n_lv_bubble_slug(nums.n_gv, nums.n_l, math.pi / 2) == pytest.approx(18.38, rel=1e-3)


def test_r1_pattern_is_slug_with_details(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, сквозной эталон MB: режим R1 — SLUG
    result = MukherjeeBrillFlowModel().predict(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.SLUG
    assert result.details["n_gv"] == pytest.approx(11.53, rel=1e-3)
    assert result.details["n_lv"] == pytest.approx(11.86, rel=1e-3)
    assert result.details["n_l"] == pytest.approx(0.01176, rel=1e-3)
    assert result.details["n_gv_sm"] == pytest.approx(350.7, rel=1e-3)
    assert result.details["n_lv_bs"] == pytest.approx(18.38, rel=1e-3)


@pytest.mark.parametrize(("n_lv_a", "n_lv_b"), [(1.0, 100.0)])
def test_n_gv_bs_horizontal_independent_of_n_lv(n_lv_a: float, n_lv_b: float) -> None:
    # reference: ТЗ, «Контрольные точки», MB [3]: θ = 0 -> N_gvBS не зависит от N_Lv (rtol 1e-12)
    a = n_gv_bubble_slug(n_lv_a, 0.002, 0.0)
    b = n_gv_bubble_slug(n_lv_b, 0.002, 0.0)
    assert a == pytest.approx(b, rel=1e-12)


def test_n_gv_sm_independent_of_theta(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Контрольные точки», MB [2]: N_gvSM не зависит от θ (θ = −90…90, rtol 1e-12)
    values = [
        MukherjeeBrillFlowModel()
        .predict(FlowState(**{**r1_kwargs, "theta_deg": theta}))
        .details["n_gv_sm"]
        for theta in (-90.0, -45.0, 0.0, 45.0, 90.0)
    ]
    assert values == pytest.approx([values[0]] * 5, rel=1e-12)


def test_n_gv_bs_has_sin_theta_in_log_term() -> None:
    """ТЗ, «Опечатки», MB [3]: член −1.138·lg(N_Lv)·sinθ (не sin 0). θ = −45°, N_Lv = 10, N_L = 0.002."""
    theta, n_lv, n_l = math.radians(-45.0), 10.0, 0.002
    s = math.sin(theta)
    expected = 10 ** (
        0.431
        + 1.132 * s
        - 3.003 * n_l
        - 1.138 * math.log10(n_lv) * s
        - 0.429 * math.log10(n_lv) ** 2 * s
    )
    assert n_gv_bubble_slug(n_lv, n_l, theta) == pytest.approx(expected, rel=1e-12)


def test_n_lv_st_has_linear_n_gv_term() -> None:
    """MB [4]: член −0.017·N_gv — линейный (так в обоих оригиналах, ТЗ: не исправлять)."""
    theta, n_gv, n_l = math.radians(-10.0), 5.0, 0.002
    s = math.sin(theta)
    expected = 10 ** (
        0.321
        - 0.017 * n_gv
        - 4.267 * s
        - 2.972 * n_l
        - 0.033 * math.log10(n_gv) ** 2
        - 3.925 * s**2
    )
    assert n_lv_stratified(n_gv, n_l, theta) == pytest.approx(expected, rel=1e-12)


def test_n_lv_bs_formula() -> None:
    # reference: MB [1] (расчёт агента): lg N_LvBS = lg N_gv + 0.940 + 0.074·sinθ − 0.855·sin²θ + 3.695·N_L
    theta, n_gv, n_l = math.radians(30.0), 2.0, 0.003
    s = math.sin(theta)
    expected = 10 ** (math.log10(n_gv) + 0.940 + 0.074 * s - 0.855 * s**2 + 3.695 * n_l)
    assert n_lv_bubble_slug(n_gv, n_l, theta) == pytest.approx(expected, rel=1e-12)


# --- ветви блок-схемы (рис. 16): (θ°, v_sl, v_sg, режим) ---
# reference: расчёт агента по [1]–[4] и рис. 16, воздух–вода; точки устойчивы к ±10% скоростей
BRANCH_POINTS = [
    # подъём (ANG > 0)
    (90.0, 0.001, 10.0, FlowPattern.ANNULAR),
    (90.0, 0.002154, 0.001, FlowPattern.BUBBLE),
    (90.0, 0.001, 0.001, FlowPattern.SLUG),
    (45.0, 0.00464, 0.001, FlowPattern.BUBBLE),
    (10.0, 0.02154, 0.001, FlowPattern.BUBBLE),
    (10.0, 0.001, 0.001, FlowPattern.SLUG),
    # спуск, |ANG| > 30°
    (-45.0, 0.001, 0.001, FlowPattern.BUBBLE),
    (-45.0, 0.001, 0.04642, FlowPattern.STRATIFIED),
    (-45.0, 0.46416, 10.0, FlowPattern.SLUG),
    (-90.0, 0.001, 10.0, FlowPattern.ANNULAR),
    (-90.0, 0.001, 0.02154, FlowPattern.STRATIFIED),
    (-90.0, 0.21544, 10.0, FlowPattern.SLUG),
    # |ANG| ≤ 30°, спуск и горизонталь
    (-10.0, 2.15443, 0.001, FlowPattern.BUBBLE),
    (-10.0, 0.001, 0.001, FlowPattern.STRATIFIED),
    (-10.0, 0.21544, 10.0, FlowPattern.SLUG),
    (0.0, 0.46416, 0.001, FlowPattern.BUBBLE),
    (0.0, 0.001, 0.001, FlowPattern.STRATIFIED),
    (0.0, 0.21544, 2.15443, FlowPattern.SLUG),
    (0.0, 0.001, 10.0, FlowPattern.ANNULAR),
]


@pytest.mark.parametrize(("theta", "v_sl", "v_sg", "expected"), BRANCH_POINTS)
def test_flow_chart_branches(
    aw_state: Callable[..., FlowState],
    theta: float,
    v_sl: float,
    v_sg: float,
    expected: FlowPattern,
) -> None:
    result = MukherjeeBrillFlowModel().predict(aw_state(v_sl, v_sg, D, theta))
    assert result.pattern is expected


def test_horizontal_flow_uses_downhill_branch(aw_state: Callable[..., FlowState]) -> None:
    """θ = 0 относится к ветке «downhill & horizontal»: details содержат N_gvBS и N_LvST, но не N_LvBS."""
    result = MukherjeeBrillFlowModel().predict(aw_state(0.46416, 0.001, D, 0.0))
    assert "n_gv_bs" in result.details
    assert "n_lv_st" in result.details
    assert "n_lv_bs" not in result.details


def test_uphill_branch_details(aw_state: Callable[..., FlowState]) -> None:
    result = MukherjeeBrillFlowModel().predict(aw_state(0.001, 0.001, D, 45.0))
    assert "n_lv_bs" in result.details
    assert "n_gv_bs" not in result.details


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    model = MukherjeeBrillFlowModel()
    assert model.predict(aw_state(1.0, 0.0, D, 30.0)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert model.predict(aw_state(0.0, 1.0, D, 30.0)).pattern is FlowPattern.SINGLE_PHASE_GAS
