"""Модель режимов Taitel–Dukler для горизонтальных и близких к ним труб (IFlowModel).

Схема расчёта (θ > 0 — подъём; у TD α = −θ, α > 0 — спуск; знак учтён в `stratified.y_parameter`):
    1. X² = (dp/dx)_LS/(dp/dx)_GS (8), Y (9) — градиенты однофазных потоков, Fanning TD → Darcy
      (×4);
    2. равновесный уровень h̃_L — наименьший корень уравнения (7);
    3. (25): F²·[ũ_G²·S̃_i/(C2²·Ã_G)] < 1 (C2 = 1 − h̃) → расслоённый поток:
       (30) выполнено → STRATIFIED_WAVY, иначе STRATIFIED_SMOOTH (s = 0.01);
    4. иначе h̃ < 0.5 → ANNULAR; иначе (36) T² ≥ 8·Ã_G/(S̃_i·ũ_L²·(ũ_L·D̃_L)^-n) → DISPERSED_BUBBLE,
       иначе INTERMITTENT.

Источник: Taitel Y., Dukler A.E., 1976, AIChE J. 22(1), 47–55, ур. (6)–(16), (25), (30), (36), (37).
В (25) напечатано ũ_G — взято ũ_G² (ТЗ, «Опечатки»; реализовано в
  `stratified.kelvin_helmholtz_parameter`).

Допущения (TODO(question N) ТЗ): `laminar_switch` — ламинарность фазы определяется по Re с
  фактической
скоростью и гидравлическим диаметром (6) при критическом Re = 2000 (TD порог не задаёт);
  перебираются
комбинации «турб.–турб., турб.–лам., лам.–турб., лам.–лам.» и берётся первая согласованная, иначе
турб.–турб.
"""

import math
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import friction, gradient, stratified
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import ConvergenceError
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# TD (5): турбулентные n = m = 0.2; ламинарные n = m = 1
_TURBULENT_EXPONENT: Final = 0.2
_LAMINAR_EXPONENT: Final = 1.0
# Уровень, разделяющий кольцевой (h̃ < 0.5) и перемежающийся/дисперсный режимы, TD, раздел «I–AD»
_ANNULAR_LEVEL: Final = 0.5
# TD (36): T² ≥ 8·Ã_G/(S̃_i·ũ_L²·(ũ_L·D̃_L)^-n)
_DISPERSION_FACTOR: Final = 8.0
# Комбинации (жидкость ламинарна, газ ламинарен) в порядке перебора
_REGIME_COMBINATIONS: Final = ((False, False), (False, True), (True, False), (True, True))


class TaitelDuklerOptions(BaseOptions):
    """Флаги модели Taitel–Dukler."""

    laminar_switch: bool = Field(
        default=True,
        description=(
            "Выбор коэффициентов C, n трения по фактическому Re каждой фазы (скорость и "
            "гидравлический диаметр (6)) вместо всегда турбулентных. Выключать (False) для "
            "сверки с картой Fig. 4 (турбулентный расчёт). Источник: Taitel, Dukler 1976, (5)."
        ),
    )
    use_moody_friction: bool = Field(
        default=False,
        description=(
            "Градиенты однофазных потоков по Colebrook с шероховатостью ε вместо гладких "
            "0.046·Re^-0.2 / 16/Re. Включать для шероховатых труб (предложение авторов в конце "
            "статьи). Источник: Taitel, Dukler 1976."
        ),
    )


class _Solution:
    """Решение уравнения (7) для выбранной комбинации режимов трения."""

    def __init__(self, level: float, grad_l: float, grad_g: float, n: float, m: float) -> None:
        self.level = level
        self.grad_l = grad_l
        self.grad_g = grad_g
        self.n = n
        self.m = m


class TaitelDuklerModel:
    """Модуль режима Taitel–Dukler (IFlowModel).

    Атрибуты:
        name: "taitel_dukler".
    """

    name: ClassVar[str] = "taitel_dukler"

    def __init__(self, options: TaitelDuklerOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (по умолчанию `laminar_switch=True`, `use_moody_friction=False`).
        """
        self._options = options if options is not None else TaitelDuklerOptions()

    @property
    def options(self) -> TaitelDuklerOptions:
        """Настройки модели."""
        return self._options

    def _gradients(
        self, state: FlowState, re_sl: float, re_sg: float, laminar_l: bool, laminar_g: bool
    ) -> tuple[float, float]:
        """Градиенты трения однофазных потоков (dp/dx)_LS и (dp/dx)_GS, Па/м (через Darcy)."""
        if self._options.use_moody_friction:
            f_l = friction.moody(re_sl, state.eps_m / state.d_m)
            f_g = friction.moody(re_sg, state.eps_m / state.d_m)
        else:
            f_l = friction.fanning_to_darcy(
                friction.fanning_taitel_dukler(re_sl, laminar=laminar_l)
            )
            f_g = friction.fanning_to_darcy(
                friction.fanning_taitel_dukler(re_sg, laminar=laminar_g)
            )
        grad_l = gradient.friction_gradient(f_l, state.rho_l_kg_m3, state.v_sl_m_s, state.d_m)
        grad_g = gradient.friction_gradient(f_g, state.rho_g_kg_m3, state.v_sg_m_s, state.d_m)
        return grad_l, grad_g

    def _solve(
        self, state: FlowState, re_sl: float, re_sg: float, laminar_l: bool, laminar_g: bool
    ) -> _Solution | None:
        """Решает (7) для заданной комбинации режимов трения; None — корня нет."""
        grad_l, grad_g = self._gradients(state, re_sl, re_sg, laminar_l, laminar_g)
        n = _LAMINAR_EXPONENT if laminar_l else _TURBULENT_EXPONENT
        m = _LAMINAR_EXPONENT if laminar_g else _TURBULENT_EXPONENT
        x_sq = stratified.x_squared(grad_l, grad_g)
        y = stratified.y_parameter(
            state.rho_l_kg_m3, state.rho_g_kg_m3, math.radians(state.theta_deg), grad_g
        )
        level = stratified.equilibrium_level(x_sq, y, n=n, m=m)
        if level is None:
            return None
        return _Solution(level, grad_l, grad_g, n, m)

    def _equilibrium(self, state: FlowState, re_sl: float, re_sg: float) -> _Solution:
        """Равновесное решение с учётом `laminar_switch` (см. допущения в docstring модуля)."""
        if self._options.laminar_switch:
            for laminar_l, laminar_g in _REGIME_COMBINATIONS:
                candidate = self._solve(state, re_sl, re_sg, laminar_l, laminar_g)
                if candidate is None:
                    continue
                geo = stratified.geometry(candidate.level)
                re_l = state.rho_l_kg_m3 * state.v_sl_m_s * geo.u_l * geo.d_l * state.d_m
                re_l /= state.mu_l_pa_s
                re_g = state.rho_g_kg_m3 * state.v_sg_m_s * geo.u_g * geo.d_g * state.d_m
                re_g /= state.mu_g_pa_s
                if (re_l < friction.RE_LAMINAR_MAX) == laminar_l and (
                    re_g < friction.RE_LAMINAR_MAX
                ) == laminar_g:
                    return candidate
        solution = self._solve(state, re_sl, re_sg, False, False)
        if solution is None:
            raise ConvergenceError(
                "равновесный уровень расслоённого потока не найден",
                {"v_sl_m_s": state.v_sl_m_s, "v_sg_m_s": state.v_sg_m_s, "d_m": state.d_m},
            )
        return solution

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по Taitel–Dukler.

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `FlowPatternResult`: режим и details (`h_l_over_d`, `holdup_stratified`, `x`, `y`, `f`,
            `k`, `t`, `n_liquid`, `m_gas`).

        Исключения:
            ConvergenceError: равновесный уровень не найден.
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        re_sl = state.rho_l_kg_m3 * state.v_sl_m_s * state.d_m / state.mu_l_pa_s
        re_sg = state.rho_g_kg_m3 * state.v_sg_m_s * state.d_m / state.mu_g_pa_s
        theta = math.radians(state.theta_deg)
        solution = self._equilibrium(state, re_sl, re_sg)
        level = solution.level
        geo = stratified.geometry(level)
        froude = stratified.modified_froude(
            state.v_sg_m_s, state.rho_l_kg_m3, state.rho_g_kg_m3, state.d_m, theta
        )
        k = froude * math.sqrt(re_sl)
        delta_rho = state.rho_l_kg_m3 - state.rho_g_kg_m3
        t_sq = solution.grad_l / (delta_rho * GRAVITY_M_S2 * math.cos(theta))
        x_sq = stratified.x_squared(solution.grad_l, solution.grad_g)
        details: dict[str, float] = {
            "h_l_over_d": level,
            "holdup_stratified": geo.a_l / (math.pi / 4.0),
            "x": math.sqrt(x_sq),
            "y": stratified.y_parameter(
                state.rho_l_kg_m3, state.rho_g_kg_m3, theta, solution.grad_g
            ),
            "f": froude,
            "k": k,
            "t": math.sqrt(t_sq),
            "n_liquid": solution.n,
            "m_gas": solution.m,
        }
        if stratified.kelvin_helmholtz_parameter(froude, level) < 1.0:
            wavy = stratified.jeffreys_wavy(k, level)
            pattern = FlowPattern.STRATIFIED_WAVY if wavy else FlowPattern.STRATIFIED_SMOOTH
        elif level < _ANNULAR_LEVEL:
            pattern = FlowPattern.ANNULAR
        else:
            threshold = (
                _DISPERSION_FACTOR
                * geo.a_g
                / (geo.s_i * geo.u_l**2 * (geo.u_l * geo.d_l) ** -solution.n)
            )
            dispersed = t_sq >= threshold
            pattern = FlowPattern.DISPERSED_BUBBLE if dispersed else FlowPattern.INTERMITTENT
        return FlowPatternResult(pattern=pattern, details=details)
