"""Тесты `flow_models/beggs_brill_map.py` — карта режимов Beggs–Brill (IFlowModel).

Ожидаемый API (`from correlations.flow_models.beggs_brill_map import ...`):
    BeggsBrillMapOptions(revised_map: bool = False)
    BeggsBrillMap(options: BeggsBrillMapOptions | None = None)
        .name == "beggs_brill_map"; .options; .predict(state) -> FlowPatternResult
    froude_boundaries_original(lam) -> tuple[float, float]            (L1, L2) — BB 1973, ур. 9–10
    froude_boundaries_revised(lam) -> tuple[float, float, float, float]  (L1…L4) — монография 4.110–4.113
Режимы: SEGREGATED, INTERMITTENT, DISTRIBUTED (+ TRANSITION при revised_map).
details: lambda_l, n_fr; оригинал: l1_froude, l2_froude; revised: l1_froude…l4_froude.
Источник revised_map: Brill J.P., Mukherjee H., «Multiphase Flow in Wells» (монография), ур. 4.110–4.119
(ТЗ: единственное допустимое использование монографии как источника формул).
"""

from collections.abc import Callable
from typing import Any

import pytest

from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowState
from correlations.flow_models.beggs_brill_map import (
    BeggsBrillMap,
    BeggsBrillMapOptions,
    froude_boundaries_original,
    froude_boundaries_revised,
)

D = 0.1524


def test_name_and_default_options() -> None:
    model = BeggsBrillMap()
    assert model.name == "beggs_brill_map"
    assert model.options == BeggsBrillMapOptions()
    assert model.options.revised_map is False


def test_r1_original_map_matches_tz(r1_kwargs: dict[str, Any]) -> None:
    # reference: ТЗ, «Элементные эталоны», BB: L1 / L2 (оригинал) = 0.1018 / 33.11 (rtol 1e-3);
    # N_FR = 3.808; режим для R1 — INTERMITTENT (ТЗ, сквозной эталон BB)
    result = BeggsBrillMap().predict(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.INTERMITTENT
    assert result.details["l1_froude"] == pytest.approx(0.1018, rel=1e-3)
    assert result.details["l2_froude"] == pytest.approx(33.11, rel=1e-3)
    assert result.details["n_fr"] == pytest.approx(3.808, rel=1e-3)
    assert result.details["lambda_l"] == pytest.approx(0.50712, rel=1e-4)


def test_original_boundaries_function() -> None:
    l1, l2 = froude_boundaries_original(0.50712)
    assert l1 == pytest.approx(0.1018, rel=1e-3)
    assert l2 == pytest.approx(33.11, rel=1e-3)


def test_revised_boundaries_function() -> None:
    # reference: монография 4.110–4.113 (расчёт агента, λ_L = 0.50712): L1…L4
    l1, l2, l3, l4 = froude_boundaries_revised(0.50712)
    assert l1 == pytest.approx(257.4, rel=2e-3)
    assert l2 == pytest.approx(0.004942, rel=2e-3)
    assert l3 == pytest.approx(0.2680, rel=2e-3)
    assert l4 == pytest.approx(48.52, rel=2e-3)


def test_r1_revised_map_is_intermittent_and_has_four_boundaries(r1_kwargs: dict[str, Any]) -> None:
    model = BeggsBrillMap(BeggsBrillMapOptions(revised_map=True))
    result = model.predict(FlowState(**r1_kwargs))
    assert result.pattern is FlowPattern.INTERMITTENT
    assert {"l1_froude", "l2_froude", "l3_froude", "l4_froude"} <= set(result.details)


# --- ветви карты: (v_sl, v_sg, режим); d = 0.1524, воздух–вода. Точки устойчивы к ±10% скоростей. ---
# reference: расчёт агента по ур. 9–10 (оригинал) и 4.110–4.113 (revised)
ORIGINAL_POINTS = [
    (1.0e-4, 1.0e-4, FlowPattern.SEGREGATED),
    (1.0e-4, 1.778e-4, FlowPattern.SEGREGATED),
    (1.0e-4, 1.0e-2, FlowPattern.SEGREGATED),
    (0.316228, 0.562341, FlowPattern.INTERMITTENT),
    (0.177828, 1.0e-4, FlowPattern.INTERMITTENT),
    (3.16228, 1.0e-4, FlowPattern.DISTRIBUTED),
    (0.316228, 17.7828, FlowPattern.DISTRIBUTED),
    (3.16e-4, 56.2341, FlowPattern.DISTRIBUTED),
]
REVISED_POINTS = [
    (1.0e-4, 1.0e-4, FlowPattern.SEGREGATED),  # λ ≥ 0.4, N_FR < L2
    (1.0e-4, 1.778e-4, FlowPattern.SEGREGATED),  # 0.01 ≤ λ < 0.4
    (1.0e-4, 1.0e-2, FlowPattern.SEGREGATED),  # λ < 0.01, N_FR < L1
    (0.0562341, 0.1, FlowPattern.TRANSITION),  # λ < 0.4
    (0.0562341, 1.0e-4, FlowPattern.TRANSITION),  # λ ≥ 0.4
    (0.177828, 5.62341, FlowPattern.INTERMITTENT),  # 0.01 ≤ λ < 0.4, L3 < N_FR ≤ L1
    (0.562341, 1.0e-4, FlowPattern.INTERMITTENT),  # λ ≥ 0.4, L3 < N_FR ≤ L4
    (1.0, 1.0e-4, FlowPattern.DISTRIBUTED),  # λ ≥ 0.4, N_FR > L4
    (0.316228, 17.7828, FlowPattern.DISTRIBUTED),  # λ < 0.4, N_FR ≥ L1
    (1.0e-4, 5.62341, FlowPattern.DISTRIBUTED),  # λ < 0.01, N_FR ≥ L1
]


@pytest.mark.parametrize(("v_sl", "v_sg", "expected"), ORIGINAL_POINTS)
def test_original_map_branches(
    aw_state: Callable[..., FlowState], v_sl: float, v_sg: float, expected: FlowPattern
) -> None:
    assert BeggsBrillMap().predict(aw_state(v_sl, v_sg, D)).pattern is expected


@pytest.mark.parametrize(("v_sl", "v_sg", "expected"), REVISED_POINTS)
def test_revised_map_branches(
    aw_state: Callable[..., FlowState], v_sl: float, v_sg: float, expected: FlowPattern
) -> None:
    model = BeggsBrillMap(BeggsBrillMapOptions(revised_map=True))
    assert model.predict(aw_state(v_sl, v_sg, D)).pattern is expected


def test_original_map_never_returns_transition(aw_state: Callable[..., FlowState]) -> None:
    result = BeggsBrillMap().predict(aw_state(0.0562341, 0.1, D))
    assert result.pattern is not FlowPattern.TRANSITION


def test_revised_map_flag_changes_result(aw_state: Callable[..., FlowState]) -> None:
    """ТЗ: каждый флаг меняет результат хотя бы в одной точке. λ = 0.5, N_FR = 0.1."""
    v = 0.19333  # v_sl = v_sg, v_m = 0.3867 -> N_FR = v_m²/(g·d) = 0.1
    state = aw_state(v, v, D)
    original = BeggsBrillMap().predict(state)
    revised = BeggsBrillMap(BeggsBrillMapOptions(revised_map=True)).predict(state)
    assert original.pattern is FlowPattern.SEGREGATED
    assert revised.pattern is FlowPattern.TRANSITION


def test_froude_number_independent_of_angle(aw_state: Callable[..., FlowState]) -> None:
    """Карта построена для горизонтали: режим не зависит от угла (ТЗ, «Beggs & Brill map»)."""
    a = BeggsBrillMap().predict(aw_state(0.316228, 0.562341, D, theta_deg=0.0))
    b = BeggsBrillMap().predict(aw_state(0.316228, 0.562341, D, theta_deg=-60.0))
    assert a == b


def test_single_phase_limits(aw_state: Callable[..., FlowState]) -> None:
    """При v_sg = 0 (λ = 1) и v_sl = 0 (λ = 0) модель возвращает однофазный предел, а не падает на ln λ."""
    assert BeggsBrillMap().predict(aw_state(1.0, 0.0, D)).pattern is FlowPattern.SINGLE_PHASE_LIQUID
    assert BeggsBrillMap().predict(aw_state(0.0, 1.0, D)).pattern is FlowPattern.SINGLE_PHASE_GAS


@pytest.mark.parametrize("v_sl", [1.0e-9, 1.0e-12, 1.0e-30])
def test_extremely_small_lambda_does_not_overflow(
    aw_state: Callable[..., FlowState], v_sl: float
) -> None:
    """Полином L2 (ур. 10) даёт exp(>700) при λ ≲ 1e-8: модель не должна падать с OverflowError."""
    result = BeggsBrillMap().predict(aw_state(v_sl, 8.0, 0.0508))
    assert result.pattern is FlowPattern.SEGREGATED
    revised = BeggsBrillMap(BeggsBrillMapOptions(revised_map=True)).predict(
        aw_state(v_sl, 8.0, 0.0508)
    )
    # revised: λ < 0.01 и N_FR = 128 ≥ L1 = 316·λ^0.302 (≈ 0.08 и меньше) → DISTRIBUTED
    assert revised.pattern is FlowPattern.DISTRIBUTED
