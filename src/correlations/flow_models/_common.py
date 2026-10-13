"""Общие помощники модулей режима потока."""

from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState


def single_phase_result(state: FlowState) -> FlowPatternResult | None:
    """Однофазный предел: v_sg = 0 → жидкость, v_sl = 0 → газ (ТЗ, «Общие правила»).

    Параметры:
        state: состояние потока.

    Возвращает:
        `FlowPatternResult` с однофазным режимом и пустыми `details`, либо None для двухфазного
          потока.
    """
    if state.v_sg_m_s == 0.0:
        return FlowPatternResult(pattern=FlowPattern.SINGLE_PHASE_LIQUID, details={})
    if state.v_sl_m_s == 0.0:
        return FlowPatternResult(pattern=FlowPattern.SINGLE_PHASE_GAS, details={})
    return None
