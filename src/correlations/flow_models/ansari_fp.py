"""Модуль режимов Ansari для вертикального и наклонного подъёма и решение плёнки (IFlowModel).

Схема расчёта (порядок по Примеру 4.9, в статье не задан — вопрос 2 ТЗ):
    (а) DISPERSED_BUBBLE: левая часть (4) ≥ 0.725 + 4.15·(v_sg/v_m)^0.5 (f — Moody по Re_ns) и v_sg
      < 3.17·v_sl (5);
    (б) ANNULAR: v_sg > 3.1·[g·σ·Δρ/ρ_g²]^0.25 (6) и не выполнено H_LF + λ_LC·A_C/A > 0.12 (12),
      где δ — δ_min
        из (8) (флаг bridging_with_delta_min, наименьший корень, вопрос 1 ТЗ) либо δ из (96);
    (в) BUBBLE: d > d_min = 19.01·[Δρ·σ/(ρ_L²·g)]^0.5 и v_sg < v_sl/3 + 0.25·v_s (2); при
      bubble_slug_sin_theta —
        с множителем sinθ у v_s (как Barnea [4]);
    (г) иначе SLUG (churn входит в slug).

Плёнка (кольцевой режим): F_E (71)–(72), λ_LC (70), ρ_C (69), μ_C (90), v_SC (89), (dp/dL)_SC
  (87)–(88), (dp/dL)_SL
(79)–(80), f_F/f_SL по Re плёнки (74)–(76) (или 1 при ff_over_fsl_unity), X_M² (9), Y_M (10), Z
  (84)/(85), δ — корень (96).

Источник: Ansari A.M., Sylvester N.D., Sarica C., Shoham O., Brill J.P., 1994, SPE Production &
  Facilities 9(2),
145–152, ур. (2)–(15), (69)–(72), (74)–(85), (94)–(102). Опечатка (ТЗ): в (85) (ρ_L/ρ_g)^(1/3) —
  реализовано
в `core.annular.whalley_hewitt_z`.

Допущения (вне ТЗ, TODO(question 1) ТЗ): критерий «δ < δ_min» отдельно не применяется (статья
  проверяет только
(12), δ_min лишь определяет H_LF); при отсутствии корня (8) в (12) берётся δ из (96); при
  отсутствии корня (96)
плёнка не определена (`solve_annular_film` → None) и режим не кольцевой. Корни (96) и (8) ищутся на
  сетке по ln δ
(δ от 1e-9 до 0.5): при больших v_sg плёнка очень тонкая; полюс правой части (8) при 4δ(1 − δ) =
  2/3 отбрасывается.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import (
    annular,
    bubbles,
    dimensionless,
    friction,
    gradient,
    mixture,
    solvers,
    transitions,
)
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import FlowPatternResult, FlowState
from correlations.flow_models._common import single_phase_result

# (5): v_sg < 3.17·v_sl ⇔ α_ns < 0.76 (Scott, Kouba)
_PACKING_ALPHA: Final = 0.76
# d_min: коэффициент 19.01 (Taitel 1980; Ansari)
_D_MIN_COEFFICIENT: Final = 19.01
# (12): H_LF + λ_LC·A_C/A > 0.12 — перемыкание ядра
_BLOCKAGE_LIMIT: Final = 0.12
# (84)/(85): граница выбора Z по уносу
_Z_ENTRAINMENT_SPLIT: Final = 0.9
# Верхний предел ε/d диаграммы Moody (для тонкой плёнки ε/d_HF формально ≫ 1)
_MAX_RELATIVE_ROUGHNESS: Final = 0.05
# Сетка поиска δ (по ln δ) и порог отбрасывания полюса (8)
_DELTA_MIN_SEARCH: Final = 1e-9
_DELTA_MAX_SEARCH: Final = 0.4999
_DELTA_GRID: Final = 1500
_POLE_RESIDUAL_TOLERANCE: Final = 1e-6


class AnsariFlowOptions(BaseOptions):
    """Флаги модуля режима Ansari."""

    bridging_with_delta_min: bool = Field(
        default=True,
        description=(
            "В критерии (12) берётся δ_min из (8) (текст статьи); False — δ из (96) (Пример 4.9). "
            "False физически ближе к Barnea. "
            "Источник: Ansari et al. 1994, ур. 8, 11, 12 (вопрос 1 ТЗ)."
        ),
    )
    bubble_slug_sin_theta: bool = Field(
        default=False,
        description=(
            "0.25·v_s·sinθ в (2), как у Barnea [4]. Включать для наклонных скважин. "
            "Источник: Ansari et al. 1994, ур. 2; Barnea 1987, [4]."
        ),
    )
    ff_over_fsl_unity: bool = Field(
        default=False,
        description=(
            "f_F/f_SL = 1 без итерации по Re плёнки (74)–(76). Быстрый расчёт; Пример 4.9. "
            "Источник: Ansari et al. 1994, ур. 74–78."
        ),
    )


@dataclass(frozen=True, slots=True)
class AnnularFilm:
    """Решение плёнки кольцевого режима Ansari.

    Атрибуты:
        f_e: доля унесённой жидкости F_E (71).
        lam_lc: доля жидкости в ядре λ_LC (70).
        rho_c_kg_m3: плотность ядра ρ_C (69), кг/м³.
        mu_c_pa_s: вязкость ядра μ_C (90), Па·с.
        v_sc_m_s: приведённая скорость ядра v_SC (89), м/с.
        f_sc: коэффициент трения Moody по Re_SC (88).
        dp_sc_pa_m: (dp/dL)_SC (87), Па/м.
        f_sl: коэффициент трения Moody по Re_SL (80).
        dp_sl_pa_m: (dp/dL)_SL (79), Па/м.
        y_m: параметр Y_M (10).
        f_f_over_f_sl: отношение f_F/f_SL (1 при ff_over_fsl_unity).
        x_m_sq: параметр X_M² (9).
        z: фактор межфазного трения Z (84)/(85).
        delta: безразмерная толщина плёнки δ/d — корень (96).
        delta_min: наименьший корень (8) или None, если корня нет.
        h_lf: доля сечения плёнки H_LF = 4δ(1 − δ) (11) при δ из (96).
    """

    f_e: float
    lam_lc: float
    rho_c_kg_m3: float
    mu_c_pa_s: float
    v_sc_m_s: float
    f_sc: float
    dp_sc_pa_m: float
    f_sl: float
    dp_sl_pa_m: float
    y_m: float
    f_f_over_f_sl: float
    x_m_sq: float
    z: float
    delta: float
    delta_min: float | None
    h_lf: float


def solve_annular_film(state: FlowState, *, ff_over_fsl_unity: bool = False) -> AnnularFilm | None:
    """Решает уравнение плёнки Ansari (96) относительно δ (наименьший корень).

    Параметры:
        state: состояние потока (СИ, v_sl > 0, v_sg > 0).
        ff_over_fsl_unity: принять f_F/f_SL = 1 (без итерации по Re плёнки).

    Возвращает:
        `AnnularFilm` или None, если корня (96) на (1e-9, 0.5) нет.
    """
    rho_l, rho_g = state.rho_l_kg_m3, state.rho_g_kg_m3
    v_sl, v_sg, d = state.v_sl_m_s, state.v_sg_m_s, state.d_m
    v_crit = annular.critical_velocity(v_sg, state.mu_g_pa_s, state.sigma_n_m, rho_g, rho_l)
    f_e = annular.entrainment_wallis(v_crit)  # (71)–(72)
    lam_lc = annular.core_liquid_fraction(f_e, v_sl, v_sg)  # (70)
    core = mixture.slip(lam_lc, rho_l, rho_g, state.mu_l_pa_s, state.mu_g_pa_s)  # (69), (90)
    v_sc = f_e * v_sl + v_sg  # (89)
    re_sc = dimensionless.reynolds_number(core.rho_s_kg_m3, v_sc, d, core.mu_s_pa_s)  # (88)
    f_sc = friction.moody(re_sc, state.eps_m / d)
    dp_sc = gradient.friction_gradient(f_sc, core.rho_s_kg_m3, v_sc, d)  # (87)
    re_sl = dimensionless.reynolds_number(rho_l, v_sl, d, state.mu_l_pa_s)  # (80)
    f_sl = friction.moody(re_sl, state.eps_m / d)
    dp_sl = gradient.friction_gradient(f_sl, rho_l, v_sl, d)  # (79)
    sin_theta = math.sin(math.radians(state.theta_deg))
    y_m = GRAVITY_M_S2 * sin_theta * (rho_l - core.rho_s_kg_m3) / dp_sc  # (10)

    def f_ratio(delta: float) -> float:
        """f_F/f_SL по Re плёнки (74), d_HF (76), v_F (75)."""
        if ff_over_fsl_unity:
            return 1.0
        alpha = annular.film_holdup_from_delta(delta)
        v_f = v_sl * (1.0 - f_e) / alpha
        d_hf = alpha * d
        re_f = dimensionless.reynolds_number(rho_l, v_f, d_hf, state.mu_l_pa_s)
        rel_roughness = min(state.eps_m / d_hf, _MAX_RELATIVE_ROUGHNESS)
        return friction.moody(re_f, rel_roughness) / f_sl

    def x_m_sq(delta: float) -> float:
        return (1.0 - f_e) ** 2 * f_ratio(delta) * dp_sl / dp_sc  # (9), B = (1 − F_E)²·f_F/f_SL

    def z_factor(delta: float) -> float:
        if f_e > _Z_ENTRAINMENT_SPLIT:
            return annular.wallis_z(delta)  # (84)
        return annular.whalley_hewitt_z(delta, rho_l, rho_g)  # (85)

    def residual_96(log_delta: float) -> float:
        delta = math.exp(log_delta)
        alpha = annular.film_holdup_from_delta(delta)
        return annular.film_equation_residual(alpha, x_m_sq(delta), y_m, z_factor(delta))

    roots = solvers.bracketed_roots(
        residual_96,
        math.log(_DELTA_MIN_SEARCH),
        math.log(_DELTA_MAX_SEARCH),
        n_grid=_DELTA_GRID,
    )
    if not roots:
        return None
    delta = math.exp(roots[0])
    x_sq = x_m_sq(delta)
    return AnnularFilm(
        f_e=f_e,
        lam_lc=lam_lc,
        rho_c_kg_m3=core.rho_s_kg_m3,
        mu_c_pa_s=core.mu_s_pa_s,
        v_sc_m_s=v_sc,
        f_sc=f_sc,
        dp_sc_pa_m=dp_sc,
        f_sl=f_sl,
        dp_sl_pa_m=dp_sl,
        y_m=y_m,
        f_f_over_f_sl=f_ratio(delta),
        x_m_sq=x_sq,
        z=z_factor(delta),
        delta=delta,
        delta_min=_delta_min(y_m, x_sq),
        h_lf=annular.film_holdup_from_delta(delta),
    )


def _delta_min(y_m: float, x_m_sq: float) -> float | None:
    """Наименьший корень (8): Y_M = (2 − 1.5·H_LF)/(H_LF³·(1 − 1.5·H_LF))·X_M², H_LF = 4δ(1 − δ).

    Полюс при H_LF = 2/3 (знаменатель → 0) даёт ложную смену знака и отбрасывается.

    Параметры:
        y_m: параметр Y_M.
        x_m_sq: параметр X_M².

    Возвращает:
        δ_min или None, если корня нет.
    """

    def residual_8(log_delta: float) -> float:
        alpha = annular.film_holdup_from_delta(math.exp(log_delta))
        return y_m - annular.film_instability_y(alpha, x_m_sq)

    roots = solvers.bracketed_roots(
        residual_8,
        math.log(_DELTA_MIN_SEARCH),
        math.log(_DELTA_MAX_SEARCH),
        n_grid=_DELTA_GRID,
    )
    scale = 1.0 + abs(y_m)
    for root in roots:
        if abs(residual_8(root)) <= _POLE_RESIDUAL_TOLERANCE * scale:
            return math.exp(root)
    return None


def film_blockage(film: AnnularFilm, *, use_delta_min: bool) -> float:
    """Критерий перемыкания Ansari (12): H_LF + λ_LC·A_C/A, A_C/A = (1 − 2δ)².

    Параметры:
        film: решение плёнки.
        use_delta_min: взять δ_min из (8) (если корень есть) вместо δ из (96).

    Возвращает:
        Значение критерия; > 0.12 — ядро перемыкается (не кольцевой режим).
    """
    delta = film.delta_min if use_delta_min and film.delta_min is not None else film.delta
    h_lf = annular.film_holdup_from_delta(delta)
    return annular.film_blockage_ansari(h_lf, film.lam_lc, delta)


class AnsariFlowModel:
    """Модуль режима Ansari (IFlowModel).

    Атрибуты:
        name: "ansari_fp".
    """

    name: ClassVar[str] = "ansari_fp"

    def __init__(self, options: AnsariFlowOptions | None = None) -> None:
        """Создаёт модель.

        Параметры:
            options: настройки (по умолчанию — поведение статьи).
        """
        self._options = options if options is not None else AnsariFlowOptions()

    @property
    def options(self) -> AnsariFlowOptions:
        """Настройки модели."""
        return self._options

    def predict(self, state: FlowState) -> FlowPatternResult:
        """Определяет режим по порядку (а)–(г).

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `FlowPatternResult`: режим и details (`db_left`, `db_right`, `v_sg_annular_m_s`,
            `d_min_m`, `v_sg_bubble_slug_m_s`; при расчёте плёнки — `film_delta`, `film_delta_min`,
            `film_blockage`).
        """
        single = single_phase_result(state)
        if single is not None:
            return single
        rho_l, rho_g, sigma = state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m
        v_sl, v_sg, d = state.v_sl_m_s, state.v_sg_m_s, state.d_m
        theta = math.radians(state.theta_deg)
        mix = mixture.no_slip(v_sl, v_sg, rho_l, rho_g, state.mu_l_pa_s, state.mu_g_pa_s)
        re_ns = dimensionless.reynolds_number(mix.rho_ns_kg_m3, mix.v_m_m_s, d, mix.mu_ns_pa_s)
        f_ns = friction.moody(re_ns, state.eps_m / d)
        boundary = transitions.dispersed_bubble_barnea(
            v_sl, v_sg, rho_l, rho_g, sigma, d, f_ns, theta_rad=math.pi / 2
        )
        v_s = bubbles.harmathy_rise_velocity(rho_l, rho_g, sigma)
        sin_theta = math.sin(theta) if self._options.bubble_slug_sin_theta else 1.0
        v_sg_bubble = transitions.bubble_slug_slip(v_sl, v_s, sin_theta=sin_theta)
        v_sg_annular = transitions.annular_turner_velocity(rho_l, rho_g, sigma)
        d_min = transitions.min_bubble_diameter(rho_l, rho_g, sigma, coef=_D_MIN_COEFFICIENT)
        details = {
            "db_left": boundary.ratio * boundary.dispersion_factor,
            "db_right": boundary.dispersion_factor,
            "v_sg_annular_m_s": v_sg_annular,
            "d_min_m": d_min,
            "v_sg_bubble_slug_m_s": v_sg_bubble,
        }
        if boundary.ratio >= 1.0 and not transitions.exceeds_max_packing(
            v_sl, v_sg, _PACKING_ALPHA
        ):
            return FlowPatternResult(pattern=FlowPattern.DISPERSED_BUBBLE, details=details)
        if v_sg > v_sg_annular:
            film = solve_annular_film(state, ff_over_fsl_unity=self._options.ff_over_fsl_unity)
            if film is not None:
                blockage = film_blockage(film, use_delta_min=self._options.bridging_with_delta_min)
                details["film_delta"] = film.delta
                details["film_delta_min"] = (
                    film.delta_min if film.delta_min is not None else math.nan
                )
                details["film_blockage"] = blockage
                if blockage <= _BLOCKAGE_LIMIT:
                    return FlowPatternResult(pattern=FlowPattern.ANNULAR, details=details)
        if d > d_min and v_sg < v_sg_bubble:
            return FlowPatternResult(pattern=FlowPattern.BUBBLE, details=details)
        return FlowPatternResult(pattern=FlowPattern.SLUG, details=details)
