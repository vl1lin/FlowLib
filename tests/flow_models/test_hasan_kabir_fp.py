"""Тесты `flow_models/hasan_kabir_fp.py` — модуль режима Hasan–Kabir 1988 (IFlowModel).

Ожидаемый API (`from correlations.flow_models.hasan_kabir_fp import ...`):
    HasanKabirFlowOptions(porteous_low_pressure=False)
    HasanKabirFlowModel(options=None); .name == "hasan_kabir_fp"; .options; .predict(state)
    hewitt_roberts_boundary_lbm(rho_l_vls_sq_lbm) -> float     ρ_g·v_gs² на границе slug/churn, lbm/(ft·s²)  (ур. 8, 9)
    porteous_velocity(d_m, rho_l, rho_g) -> float               v_M = 0.103·√(g·d·Δρ/ρ_g)                        (ур. 7)
    LBM_FT_S2_IN_SI = 1.488                                     1 lbm/(ft·s²) = 1.488 кг/(м·с²)
Порядок проверок (ТЗ, «Hasan & Kabir (1988)»; r = d_t/d, C0 = 1.2 + 0.371·r):
    (а) v_M > v_M,DB (6) и f_g,bubbly < 0.52 → DISPERSED_BUBBLE;
    (б) v_gs < α·(C0·v_Ls + v∞)/(1 − α·C0) (α = 0.25; ур. 2) → BUBBLE;
    (в) v_gs > 3.1·[σ·g·Δρ/ρ_g²]^0.25 (13) → ANNULAR;
    (г) иначе: ρ_L·v_Ls² > 3300 lbm/(ft·s²) → CHURN + OutOfRangeWarning; при porteous_low_pressure и
        p < 1013 кПа v_M ≥ Porteous (7) → CHURN, иначе SLUG; иначе ρ_g·v_gs² выше границы Hewitt–Roberts → CHURN, иначе SLUG.
details: v_inf_m_s, v_gs_bubble_slug_m_s, v_m_dispersed_m_s, v_gs_annular_m_s, f_g_bubbly,
    hr_rho_l_vls_sq_lbm, hr_rho_g_vgs_sq_lbm, hr_boundary_lbm, v_m_porteous_m_s.
Источник: Hasan A.R., Kabir C.S., 1988, SPEPE 3(2), 263–272 (+ Erratum, SPEPE 4(2), 1989), ур. (1)–(13).
Допущения (ТЗ, вопрос 4): порядок проверок как указан; при ρ_L·v_Ls² > 3300 вне DB и ANNULAR — CHURN + предупреждение.
"""

import itertools
import math
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from correlations.core.errors import OutOfRangeWarning
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.hasan_kabir_fp import (
    LBM_FT_S2_IN_SI,
    HasanKabirFlowModel,
    HasanKabirFlowOptions,
    hewitt_roberts_boundary_lbm,
    porteous_velocity,
)

D = 0.0508


def test_name_and_default_flags() -> None:
    model = HasanKabirFlowModel()
    assert model.name == "hasan_kabir_fp"
    assert model.options.porteous_low_pressure is False


def test_unit_conversion_constant() -> None:
    # reference: ТЗ, HK: ρv²_SI / 1.488 → lbm/(ft·s²)
    assert LBM_FT_S2_IN_SI == 1.488


# --- R2: эталоны ТЗ ---
def test_r2_pattern_is_annular(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, главные находки: для точки Примера 3.2 по оригиналу HK режим — кольцевой (v_gs = 1.173 > 0.871)
    result = HasanKabirFlowModel().predict(FlowState(**r2_kwargs))
    assert result.pattern is FlowPattern.ANNULAR


def test_r2_transition_values(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: v∞ / v_gs,BS / v_M,DB / v_gs,A = 0.1510 / 0.5721 / 4.405 / 0.8706 (rtol 1e-3)
    d = HasanKabirFlowModel().predict(FlowState(**r2_kwargs)).details
    assert d["v_inf_m_s"] == pytest.approx(0.1510, rel=1e-3)
    assert d["v_gs_bubble_slug_m_s"] == pytest.approx(0.5721, rel=1e-3)
    assert d["v_m_dispersed_m_s"] == pytest.approx(4.405, rel=1e-3)
    assert d["v_gs_annular_m_s"] == pytest.approx(0.8706, rel=1e-3)


def test_r2_hewitt_roberts_values(r2_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», HK: ρ_L·v_Ls² / ρ_g·v_gs² / граница (lbm/(ft·s²)) = 747.0 / 87.0 / 25.93
    d = HasanKabirFlowModel().predict(FlowState(**r2_kwargs)).details
    assert d["hr_rho_l_vls_sq_lbm"] == pytest.approx(747.0, rel=1e-3)
    assert d["hr_rho_g_vgs_sq_lbm"] == pytest.approx(87.0, rel=1e-3)
    assert d["hr_boundary_lbm"] == pytest.approx(25.93, rel=1e-3)


def test_hewitt_roberts_boundary_equations() -> None:
    # reference: HK (8): 0.00673·x^1.7 при x < 50; (9): 17.1·lg x − 23.2 при 50 < x < 3300
    assert hewitt_roberts_boundary_lbm(10.0) == pytest.approx(0.00673 * 10.0**1.7, rel=1e-12)
    assert hewitt_roberts_boundary_lbm(747.0) == pytest.approx(25.93, rel=1e-3)
    assert hewitt_roberts_boundary_lbm(100.0) == pytest.approx(17.1 * 2.0 - 23.2, rel=1e-12)


def test_porteous_velocity() -> None:
    # reference: HK (7), расчёт агента: AW, d = 0.0508: 0.103·√(9.81·0.0508·995.82/1.18) = 2.112 м/с
    assert porteous_velocity(D, 997.0, 1.18) == pytest.approx(2.112, rel=1e-3)


# --- ветви порядка (а)–(г): (v_sl, v_sg) для AW, d = 0.0508, θ = 90°, p = 0.1 МПа ---
# reference: расчёт агента по HK (2), (6), (7), (8), (9), (13); точки устойчивы к ±10% скоростей
BRANCH_POINTS = [
    (0.3, 0.05, FlowPattern.BUBBLE),  # (б)
    (4.0, 0.5, FlowPattern.DISPERSED_BUBBLE),  # (а): v_M > v_M,DB, f_g,bubbly < 0.52
    (0.01, 20.0, FlowPattern.ANNULAR),  # (в)
    (1.0, 1.0, FlowPattern.SLUG),  # (г): ρ_g·v_gs² ниже границы Hewitt–Roberts
    (0.05, 5.0, FlowPattern.CHURN),  # (г): выше границы Hewitt–Roberts
    (0.1, 4.0, FlowPattern.CHURN),  # v_M > v_M,DB, но f_g,bubbly ≥ 0.52 → не DISPERSED_BUBBLE
]


@pytest.mark.parametrize(("v_sl", "v_sg", "expected"), BRANCH_POINTS)
def test_regime_branches(
    aw_state: Callable[..., FlowState], v_sl: float, v_sg: float, expected: FlowPattern
) -> None:
    assert HasanKabirFlowModel().predict(aw_state(v_sl, v_sg, D, 90.0)).pattern is expected


def test_dispersed_bubble_requires_gas_fraction_below_0_52(
    aw_state: Callable[..., FlowState],
) -> None:
    result = HasanKabirFlowModel().predict(aw_state(0.1, 4.0, D, 90.0))
    assert result.details["f_g_bubbly"] >= 0.52
    assert result.pattern is not FlowPattern.DISPERSED_BUBBLE
    dispersed = HasanKabirFlowModel().predict(aw_state(4.0, 0.5, D, 90.0))
    assert dispersed.details["f_g_bubbly"] < 0.52


def test_dispersed_bubble_has_priority_over_bubble(aw_state: Callable[..., FlowState]) -> None:
    """Порядок (а) → (б): при v_M > v_M,DB DISPERSED_BUBBLE даже если v_gs ниже границы пузырькового."""
    result = HasanKabirFlowModel().predict(aw_state(4.0, 0.5, D, 90.0))
    assert result.details["v_gs_bubble_slug_m_s"] > 0.5  # (б) тоже выполнено
    assert result.pattern is FlowPattern.DISPERSED_BUBBLE


def test_annular_independent_of_liquid_velocity(aw_state: Callable[..., FlowState]) -> None:
    """HK: переход в кольцевой не зависит от расхода жидкости (ТЗ/HK после ур. 13) — при вне DB и пузырькового."""
    for v_sl in (0.01, 0.05, 0.2):
        assert (
            HasanKabirFlowModel().predict(aw_state(v_sl, 20.0, D, 90.0)).pattern
            is FlowPattern.ANNULAR
        )


def test_churn_above_3300_warns_and_is_churn(aw_state: Callable[..., FlowState]) -> None:
    """ρ_L·v_Ls² > 3300 lbm/(ft·s²) вне DB и кольцевого → CHURN + OutOfRangeWarning (ТЗ, вопрос 4)."""
    state = aw_state(2.5, 1.5, 0.1524, 90.0)
    with pytest.warns(OutOfRangeWarning):
        result = HasanKabirFlowModel().predict(state)
    assert result.pattern is FlowPattern.CHURN
    assert result.details["hr_rho_l_vls_sq_lbm"] > 3300.0


def test_no_warning_below_3300(aw_state: Callable[..., FlowState]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HasanKabirFlowModel().predict(aw_state(1.0, 1.0, D, 90.0))


# --- флаг porteous_low_pressure ---
def test_flag_porteous_low_pressure_changes_regime(aw_state: Callable[..., FlowState]) -> None:
    """(1.0, 1.5): v_M = 2.5 ≥ Porteous 2.11 м/с → CHURN при p = 0.1 МПа < 1013 кПа; по умолчанию — SLUG."""
    state = aw_state(1.0, 1.5, D, 90.0)
    assert HasanKabirFlowModel().predict(state).pattern is FlowPattern.SLUG
    on = HasanKabirFlowModel(HasanKabirFlowOptions(porteous_low_pressure=True))
    assert on.predict(state).pattern is FlowPattern.CHURN


def test_porteous_applies_only_below_1013_kpa(aw_state: Callable[..., FlowState]) -> None:
    """HK: выше 10 атм (1013 кПа) Porteous не применяется — используется карта Hewitt–Roberts."""
    state = aw_state(1.0, 1.5, D, 90.0, p_pa=2.0e6)
    on = HasanKabirFlowModel(HasanKabirFlowOptions(porteous_low_pressure=True))
    assert on.predict(state).pattern is FlowPattern.SLUG


def test_porteous_below_threshold_velocity_is_slug(aw_state: Callable[..., FlowState]) -> None:
    state = aw_state(1.0, 1.0, D, 90.0)  # v_M = 2.0 < 2.11
    on = HasanKabirFlowModel(HasanKabirFlowOptions(porteous_low_pressure=True))
    assert on.predict(state).pattern is FlowPattern.SLUG


# --- кольцевое пространство ---
def test_annulus_changes_bubble_slug_boundary(aw_state: Callable[..., FlowState]) -> None:
    """C0 = 1.2 + 0.371·r: граница пузырьковый/пробковый зависит от r = d_t/d."""
    pipe = HasanKabirFlowModel().predict(aw_state(0.3, 0.05, 0.1, 90.0))
    annulus = HasanKabirFlowModel().predict(aw_state(0.3, 0.05, 0.1, 90.0, d_t_m=0.06))
    r = 0.6
    c0 = 1.2 + 0.371 * r
    v_inf = pipe.details["v_inf_m_s"]
    assert annulus.details["v_gs_bubble_slug_m_s"] == pytest.approx(
        0.25 * (c0 * 0.3 + v_inf) / (1.0 - 0.25 * c0), rel=1e-12
    )
    assert annulus.details["v_gs_bubble_slug_m_s"] > pipe.details["v_gs_bubble_slug_m_s"]


def test_pipe_boundary_matches_hk_equation_2(aw_state: Callable[..., FlowState]) -> None:
    # reference: HK ур. 2: v_gs = 0.429·v_Ls + 0.357·v∞ (α = 0.25, C0 = 1.2)
    d = HasanKabirFlowModel().predict(aw_state(2.0, 0.05, D, 90.0)).details
    assert d["v_gs_bubble_slug_m_s"] == pytest.approx(
        0.429 * 2.0 + 0.357 * d["v_inf_m_s"], rel=2e-3
    )


def test_regime_on_grid_is_always_defined(aw_state: Callable[..., FlowState]) -> None:
    model = HasanKabirFlowModel()
    seen = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OutOfRangeWarning)
        for v_sl, v_sg in itertools.product((0.001, 0.05, 0.5, 2.0, 4.0), (0.01, 0.5, 3.0, 25.0)):
            seen.add(model.predict(aw_state(v_sl, v_sg, D, 90.0)).pattern)
    assert {
        FlowPattern.BUBBLE,
        FlowPattern.DISPERSED_BUBBLE,
        FlowPattern.SLUG,
        FlowPattern.CHURN,
        FlowPattern.ANNULAR,
    } <= seen


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    model = HasanKabirFlowModel()
    assert model.predict(aw_state(1.0, 0.0, D, 90.0)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert model.predict(aw_state(0.0, 1.0, D, 90.0)).pattern is FlowPattern.SINGLE_PHASE_GAS


def test_math_sanity_of_details(aw_state: Callable[..., FlowState]) -> None:
    d = HasanKabirFlowModel().predict(aw_state(1.0, 1.0, D, 90.0)).details
    assert all(math.isfinite(v) for v in d.values())
