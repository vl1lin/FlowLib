"""Общие помощники полных корреляций."""

import math

from correlations.core import dimensionless, friction, gradient
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowState


def single_phase_correlation_result(state: FlowState) -> CorrelationResult | None:
    """Однофазный предел: градиент Colebrook (Moody), без ускорения (ТЗ, «Общие правила»).

    v_sg = 0 → SINGLE_PHASE_LIQUID, H_L = 1; v_sl = 0 → SINGLE_PHASE_GAS, H_L = 0.

    Параметры:
        state: состояние потока.

    Возвращает:
        `CorrelationResult` однофазного потока (details: `re`, `f_darcy`) либо None для двухфазного.
    """
    if state.v_sg_m_s == 0.0:
        pattern, holdup = FlowPattern.SINGLE_PHASE_LIQUID, 1.0
        rho, mu, velocity = state.rho_l_kg_m3, state.mu_l_pa_s, state.v_sl_m_s
    elif state.v_sl_m_s == 0.0:
        pattern, holdup = FlowPattern.SINGLE_PHASE_GAS, 0.0
        rho, mu, velocity = state.rho_g_kg_m3, state.mu_g_pa_s, state.v_sg_m_s
    else:
        return None
    re = dimensionless.reynolds_number(rho, velocity, state.d_m, mu)
    f_darcy = friction.moody(re, state.eps_m / state.d_m)
    fric = gradient.friction_gradient(f_darcy, rho, velocity, state.d_m)
    grav = gradient.gravity_gradient(rho, math.radians(state.theta_deg))
    return CorrelationResult(
        pattern=pattern,
        holdup=holdup,
        dpdl_pa_m=grav + fric,
        dpdl_gravity_pa_m=grav,
        dpdl_friction_pa_m=fric,
        dpdl_acceleration_pa_m=0.0,
        details={"re": re, "f_darcy": f_darcy},
    )


def assemble_gradient(
    gravity_pa_m: float, friction_pa_m: float, ek: float, *, include_acceleration: bool
) -> gradient.GradientComponents:
    """Собирает градиент: с ускорением (1 − Ek) или простой суммой.

    Параметры:
        gravity_pa_m: гравитационная составляющая, Па/м.
        friction_pa_m: составляющая трения, Па/м.
        ek: кинетическое число.
        include_acceleration: учитывать ускорение (знаменатель 1 − Ek).

    Возвращает:
        `GradientComponents`.
    """
    if include_acceleration:
        return gradient.with_kinetic_term(gravity_pa_m, friction_pa_m, ek)
    return gradient.GradientComponents(
        dpdl_pa_m=gravity_pa_m + friction_pa_m,
        gravity_pa_m=gravity_pa_m,
        friction_pa_m=friction_pa_m,
        acceleration_pa_m=0.0,
    )
