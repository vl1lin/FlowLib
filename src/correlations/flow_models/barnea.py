"""Унифицированная модель режимов Barnea для всего диапазона наклонов (модуль режима, IFlowModel).

Схема расчёта (Fig. 6 статьи; θ = β > 0 — подъём):
    1. дисперсно-пузырьковый: d_max < d_C ([5]–[7]) и α_ns < 0.52 → DISPERSED_BUBBLE;
    2. при cosβ ≠ 0: равновесный уровень h̃_L; если [9] < 1 (устойчиво) и [14] не выполнено →
       расслоённый: [20] или [23] → STRATIFIED_WAVY, иначе STRATIFIED_SMOOTH;
       при |β| = 90° расслоённый режим невозможен;
    3. уравнение плёнки [15] (наименьший корень α_L); [16] (неустойчивость) или [17] (перемыкание)
      → шаг 4,
       иначе ANNULAR;
    4. пузырьковый режим возможен, если D > D_crit [1] и |β| ≥ β_crit [2]: v_sg ниже [4] → BUBBLE,
       иначе шаг 6; невозможен → шаг 5;
    5. α_s по [18]; α_s = 0 и β = 0 → ELONGATED_BUBBLE;
    6. α_s ≤ 0.52 → SLUG, иначе CHURN.

Источник: Barnea D., 1987, Int. J. Multiphase Flow 13(1), 1–12, ур. [1]–[23], Fig. 6; Barnea 1986,
  IJMF 12(5).

Исправления опечаток (ТЗ, «Опечатки»): [5] — показатели 3/5 и −2/5 (напечатано 3/2, −2/3);
[15] — (1 − α_L)^(5/2) (напечатано 3/2); Fig. 6, ветка L — «Condition [14]» (напечатано [4]);
[19] — g·cosβ (напечатано ρ·cosβ).

Допущения (TODO(question N) ТЗ): [2] — d = d_CD, γ = 1.3, C_L = 0.8 (Q7); при нескольких корнях
  берётся
наименьший (Q9); если корня [15] нет — переход к шагу 4.

f_LS/f_L в [14] (флаг `friction_ratio_as_published`, по умолчанию True): по статье (D̃_L·Ũ_L)^-n,
n = 0.2 (турбулентное течение жидкости), n = 1 (ламинарное). Для гладкой трубы с f ∝ Re^-n
определение f_LS/f_L даёт (D̃_L·Ũ_L)^+n, т.е. запись статьи обратна определению; по решению
пользователя (текст статьи проверен по оригиналу) расчёт ведётся по статье. False — отношение
f(Re_LS)/f(Re_L) по выбранной модели трения (учёт шероховатости). Допущение: n выбирается по Re
слоя жидкости (порог 2000; в статье не указано), см. `published_friction_ratio`.
details["f_ls_over_f_l"] — использованное отношение (если [9] < 1).
"""

import math
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import annular, bubbles, friction, gradient, mixture, stratified, transitions
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import ConvergenceError
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# Предел плотной упаковки пузырьков: [8], максимум α_ns в дисперсно-пузырьковом режиме и в пробке
_MAX_PACKING_ALPHA: Final = 0.52
# Barnea [1]: D_crit = 19·[Δρ·σ/(ρ_L²·g)]^0.5
_MIN_DIAMETER_COEFFICIENT: Final = 19.0
# Barnea [2]: cosβ/sin²β = (3/4)·cos45°·U0²/g·(C_L·γ²/d); C_L = 0.8, γ = 1.3 (Q7), d = d_CD
_MIGRATION_CL: Final = 0.8
_MIGRATION_GAMMA: Final = 1.3
_MIGRATION_ANGLE_FACTOR: Final = 0.75 * math.cos(math.radians(45.0))
# Barnea [18]: α_s = 0.058·[…− 0.725]²; 0.725 — свободный член [5]
_ALPHA_S_COEFFICIENT: Final = 0.058
_DISPERSION_BASE: Final = 0.725
_DISPERSION_GAS_COEFFICIENT: Final = 4.15
# Barnea [14]: Z ≥ 2·(Ã_L/Ã)²·(1 − h̃)·f_LS/f_L
_STRATIFIED_ANNULAR_FACTOR: Final = 2.0
# Barnea [23]: W = U_LS/√(gD) ≥ 1.5·√h̃·Ã_L/Ã
_WAVE_FROUDE: Final = 1.5
_AREA_TILDE: Final = math.pi / 4.0
# |cosβ| меньше — вертикальная труба (стратифицированного режима нет)
_COS_ZERO_TOLERANCE: Final = 1e-12


class BarneaOptions(BaseOptions):
    """Флаги модели Barnea."""

    fm_noslip_reynolds: bool = Field(
        default=False,
        description=(
            "f_M по Re no-slip (ρ_ns·v_m·D/μ_ns) вместо Re = ρ_L·v_m·D/μ_L. Включать при высоком "
            "газосодержании и для согласования с Ansari. Источник: Barnea 1987, [5]–[7] (вопрос 8 "
            "ТЗ)."
        ),
    )
    elongated_bubble_horizontal_only: bool = Field(
        default=True,
        description=(
            "ELONGATED_BUBBLE выделяется только при β = 0 (как в статье). False — выделять при "
            "любом угле, где α_s = 0. Источник: Barnea 1987, раздел «elongated bubble–slug»."
        ),
    )
    use_moody_friction: bool = Field(
        default=True,
        description=(
            "Коэффициенты трения для X, Y, Z, f_M — Colebrook с шероховатостью ε. False — гладкие "
            "Fanning-формулы Taitel–Dukler (для сверки с Fig. 1–2, гладкие лабораторные трубы). "
            "Источник: Barnea 1987, [12]; Taitel, Dukler 1976."
        ),
    )
    split_churn: bool = Field(
        default=True,
        description=(
            "Выделять CHURN при α_s > 0.52. False — отдавать как SLUG (для моделей, где churn не "
            "считается отдельно). Источник: Barnea 1987, раздел «slug–churn transition»."
        ),
    )
    friction_ratio_as_published: bool = Field(
        default=True,
        description=(
            "f_LS/f_L в [14] по статье: (D̃_L·Ũ_L)^-n, n = 0.2 (турбулентное течение жидкости), "
            "n = 1 (ламинарное); режим выбирается по Re слоя жидкости Re_L < 2000. False — "
            "отношение коэффициентов трения f(Re_LS)/f(Re_L) по выбранной модели трения. "
            "Источник: Barnea 1987, [14]."
        ),
    )


def published_friction_ratio(d_tilde_l: float, u_tilde_l: float, re_l: float) -> float:
    """Отношение f_LS/f_L в критерии [14] в записи статьи: (D̃_L·Ũ_L)^-n.

    Показатель n: 0.2 для турбулентного течения жидкости, 1 для ламинарного (Barnea 1987, текст
    к [14]; значения подтверждены по статье).

    Допущение (TODO(question): в статье не сказано, по какому Re выбирать n): режим течения
    определяется по числу Рейнольдса слоя жидкости Re_L = ρ_L·U_L·D_L/μ_L; течение ламинарное при
    Re_L < `friction.RE_LAMINAR_MAX` (2000), иначе турбулентное. Одно и то же n применяется
    к обоим коэффициентам трения (f_LS и f_L), как в выводе формулы для гладкой трубы. Варианты
    выбора по Re_LS или по газовому слою могут давать другой результат вблизи порога 2000.

    Параметры:
        d_tilde_l: безразмерный гидравлический диаметр слоя жидкости D̃_L = D_L/D (-).
        u_tilde_l: безразмерная скорость слоя жидкости Ũ_L = U_L/U_LS (-).
        re_l: число Рейнольдса слоя жидкости Re_L (-).

    Возвращает:
        f_LS/f_L (-).
    """
    n = 1.0 if re_l < friction.RE_LAMINAR_MAX else 0.2
    return (d_tilde_l * u_tilde_l) ** -n


class BarneaModel:
    """Модуль режима Barnea (IFlowModel).

    Атрибуты:
        name: "barnea".
    """

    name: ClassVar[str] = "barnea"

    def __init__(self, options: BarneaOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (по умолчанию — поведение оригинала статьи).
        """
        self._options = options if options is not None else BarneaOptions()

    @property
    def options(self) -> BarneaOptions:
        """Настройки модели."""
        return self._options

    def _f_darcy(self, re: float, rel_roughness: float) -> float:
        """Коэффициент трения Darcy: Colebrook (Moody) или гладкие Fanning-формулы TD ×4."""
        if self._options.use_moody_friction:
            return friction.moody(re, rel_roughness)
        laminar = re < friction.RE_LAMINAR_MAX
        return friction.fanning_to_darcy(friction.fanning_taitel_dukler(re, laminar=laminar))

    def _mixture_reynolds(self, state: FlowState, v_m: float) -> float:
        """Re смеси для f_M: по свойствам жидкости или no-slip (флаг `fm_noslip_reynolds`)."""
        if not self._options.fm_noslip_reynolds:
            return state.rho_l_kg_m3 * v_m * state.d_m / state.mu_l_pa_s
        mix = mixture.no_slip(
            state.v_sl_m_s,
            state.v_sg_m_s,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        )
        return mix.rho_ns_kg_m3 * v_m * state.d_m / mix.mu_ns_pa_s

    def _bubbly_possible(self, state: FlowState, theta: float, u0: float) -> bool:
        """Пузырьковый режим возможен: D > D_crit [1] и |β| ≥ β_crit [2]."""
        d_min = transitions.min_bubble_diameter(
            state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m, coef=_MIN_DIAMETER_COEFFICIENT
        )
        if state.d_m <= d_min:
            return False
        sin_theta = math.sin(theta)
        if abs(sin_theta) < _COS_ZERO_TOLERANCE:
            return False
        delta_rho = state.rho_l_kg_m3 - state.rho_g_kg_m3
        d_bubble = 2.0 * math.sqrt(0.4 * state.sigma_n_m / (delta_rho * GRAVITY_M_S2))
        rhs = (
            _MIGRATION_ANGLE_FACTOR
            * u0**2
            / GRAVITY_M_S2
            * (_MIGRATION_CL * _MIGRATION_GAMMA**2 / d_bubble)
        )
        return math.cos(theta) / sin_theta**2 <= rhs

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по Fig. 6 Barnea 1987.

        Параметры:
            state: состояние потока (СИ); θ > 0 — подъём.

        Возвращает:
            `FlowPatternResult`: режим и details (`d_max_m`, `d_critical_m`, `alpha_ns`, `alpha_s`,
            `h_l_over_d`, `f_ls_over_f_l` (при [9] < 1), `alpha_l_film`, `film_unstable`,
            `film_blocked` — последние три при расчёте).

        Исключения:
            ConvergenceError: равновесный уровень расслоённого потока не найден.
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        rho_l, rho_g = state.rho_l_kg_m3, state.rho_g_kg_m3
        v_sl, v_sg, d = state.v_sl_m_s, state.v_sg_m_s, state.d_m
        rel_roughness = state.eps_m / d
        theta = math.radians(state.theta_deg)
        v_m = v_sl + v_sg

        f_m = self._f_darcy(self._mixture_reynolds(state, v_m), rel_roughness)
        boundary = transitions.dispersed_bubble_barnea(
            v_sl, v_sg, rho_l, rho_g, state.sigma_n_m, d, f_m, theta_rad=theta
        )
        bracket = boundary.ratio * (
            _DISPERSION_BASE + _DISPERSION_GAS_COEFFICIENT * math.sqrt(boundary.alpha_ns)
        )
        alpha_s = _ALPHA_S_COEFFICIENT * max(bracket - _DISPERSION_BASE, 0.0) ** 2  # [18]
        details: dict[str, float] = {
            "d_max_m": boundary.d_max_m,
            "d_critical_m": boundary.d_critical_m,
            "alpha_ns": boundary.alpha_ns,
            "alpha_s": alpha_s,
        }

        # Шаг 1: дисперсно-пузырьковый
        if boundary.ratio >= 1.0 and boundary.alpha_ns < _MAX_PACKING_ALPHA:
            return FlowPatternResult(pattern=FlowPattern.DISPERSED_BUBBLE, details=details)

        # Градиенты однофазных потоков, X², Y [12a,b]
        re_sl = rho_l * v_sl * d / state.mu_l_pa_s
        re_sg = rho_g * v_sg * d / state.mu_g_pa_s
        f_sl = self._f_darcy(re_sl, rel_roughness)
        grad_l = gradient.friction_gradient(f_sl, rho_l, v_sl, d)
        grad_g = gradient.friction_gradient(self._f_darcy(re_sg, rel_roughness), rho_g, v_sg, d)
        x_sq = stratified.x_squared(grad_l, grad_g)
        y = stratified.y_parameter(rho_l, rho_g, theta, grad_g)

        # Шаг 2: расслоённый режим
        if abs(math.cos(theta)) > _COS_ZERO_TOLERANCE:
            level = stratified.equilibrium_level(x_sq, y)
            if level is None:
                raise ConvergenceError(
                    "равновесный уровень расслоённого потока не найден",
                    {"v_sl_m_s": v_sl, "v_sg_m_s": v_sg, "d_m": d, "theta_deg": state.theta_deg},
                )
            details["h_l_over_d"] = level
            geo = stratified.geometry(level)
            froude = stratified.modified_froude(v_sg, rho_l, rho_g, d, theta)
            if stratified.kelvin_helmholtz_parameter(froude, level) < 1.0:  # [9]
                u_l = v_sl * geo.u_l
                re_l = rho_l * u_l * geo.d_l * d / state.mu_l_pa_s
                if self._options.friction_ratio_as_published:
                    f_ratio = published_friction_ratio(geo.d_l, geo.u_l, re_l)
                else:
                    f_ratio = f_sl / self._f_darcy(re_l, rel_roughness)  # f_LS/f_L
                details["f_ls_over_f_l"] = f_ratio
                z = grad_l / (rho_l * GRAVITY_M_S2 * math.cos(theta))
                limit = _STRATIFIED_ANNULAR_FACTOR * (1.0 / geo.u_l) ** 2 * (1.0 - level) * f_ratio
                if z < limit:  # [14] не выполнено
                    k = froude * math.sqrt(re_sl)
                    wave_froude = v_sl / math.sqrt(GRAVITY_M_S2 * d)
                    gravity_waves = wave_froude >= _WAVE_FROUDE * math.sqrt(level) * (
                        geo.a_l / _AREA_TILDE
                    )  # [23]
                    wavy = stratified.jeffreys_wavy(k, level) or gravity_waves  # [20] или [23]
                    pattern = FlowPattern.STRATIFIED_WAVY if wavy else FlowPattern.STRATIFIED_SMOOTH
                    return FlowPatternResult(pattern=pattern, details=details)

        # Шаг 3: кольцевой режим
        alpha_l = annular.solve_film_holdup(x_sq, y, annular.wallis_z_from_holdup)
        if alpha_l is not None:
            unstable = y >= annular.film_instability_y(alpha_l, x_sq)  # [16]
            blocked = annular.film_blocked_barnea(alpha_l)  # [17]
            details["alpha_l_film"] = alpha_l
            details["film_unstable"] = float(unstable)
            details["film_blocked"] = float(blocked)
            if not unstable and not blocked:
                return FlowPatternResult(pattern=FlowPattern.ANNULAR, details=details)

        # Шаги 4–6: пузырьковый, вытянутые пузыри, пробковый, пульсирующий
        u0 = bubbles.harmathy_rise_velocity(rho_l, rho_g, state.sigma_n_m)
        if self._bubbly_possible(state, theta, u0):
            limit_v_sg = transitions.bubble_slug_slip(v_sl, u0, sin_theta=math.sin(theta))  # [4]
            if v_sg < limit_v_sg:
                return FlowPatternResult(pattern=FlowPattern.BUBBLE, details=details)
        elif alpha_s == 0.0 and (
            state.theta_deg == 0.0 or not self._options.elongated_bubble_horizontal_only
        ):
            return FlowPatternResult(pattern=FlowPattern.ELONGATED_BUBBLE, details=details)
        if alpha_s <= _MAX_PACKING_ALPHA or not self._options.split_churn:
            return FlowPatternResult(pattern=FlowPattern.SLUG, details=details)
        return FlowPatternResult(pattern=FlowPattern.CHURN, details=details)
