"""Общие критерии переходов между режимами (пересечения №6–№10 ТЗ).

Схема: каждая функция реализует один критерий; различия между авторами (α, C0,
коэффициент диаметра, sinθ, вид коэффициента трения) передаются аргументами.

Источники:
    * Taitel Y., Barnea D., Dukler A.E., 1980, AIChE J. 26(3) — переходы в вертикальных трубах;
    * Barnea D., 1986, IJMF 12(5) и 1987, IJMF 13(1) — [1], [4]–[8];
    * Hasan A.R., Kabir C.S., 1988 (+ Erratum 1989) — ур. 1–2, 6, 13;
    * Ansari A.M. et al., 1994, SPE PF 9(2) — (2), (4)–(6);
    * Turner R.G. et al., 1969, JPT — критерий выноса капель.
"""

import math
from dataclasses import dataclass
from typing import Final

from correlations.core.constants import GRAVITY_M_S2

# Критическое газосодержание перехода пузырьковый -> пробковый (Taitel 1980; HK; Ansari)
BUBBLE_SLUG_ALPHA: Final = 0.25

# Barnea [5]: d_max = [0.725 + 4.15·(v_sg/v_m)^0.5]·(σ/ρ_L)^(3/5)·(2·f_M·v_m³/D)^(−2/5)
_DISPERSION_BASE: Final = 0.725
_DISPERSION_GAS_COEFFICIENT: Final = 4.15
# typo fix: напечатано (σ/ρ_L)^(3/2) и (…)^(−2/3) — взято 3/5 и −2/5 (ТЗ, «Опечатки», Barnea [5])
_DISPERSION_SIGMA_EXPONENT: Final = 3.0 / 5.0
_DISPERSION_POWER_EXPONENT: Final = -2.0 / 5.0
# Barnea [6]: d_CD = 2·[0.4·σ/(Δρ·g)]^0.5
_DEFORMATION_FACTOR: Final = 2.0
_DEFORMATION_COEFFICIENT: Final = 0.4
# Barnea [7]: d_CB = (3/8)·ρ_L/Δρ·f_M·v_m²/(g·cosβ)
_MIGRATION_COEFFICIENT: Final = 3.0 / 8.0
_COS_ZERO_TOLERANCE: Final = 1e-12  # |cosβ| меньше — миграция к стенке невозможна (d_CB = ∞)
_FANNING_PER_DARCY: Final = 0.25  # f_Fanning = f_Darcy/4

# HK 1988, ур. 6: v_M^1.12 = 4.68·d^0.48·[gΔρ/σ]^0.5·(σ/ρ_L)^0.6·(ρ_L/μ_L)^0.08
_TAITEL1980_COEFFICIENT: Final = 4.68
_TAITEL1980_D_EXPONENT: Final = 0.48
_TAITEL1980_SIGMA_EXPONENT: Final = 0.6
_TAITEL1980_VISCOSITY_EXPONENT: Final = 0.08
_TAITEL1980_VELOCITY_EXPONENT: Final = 1.12

# Критерий выноса капель (Turner 1969; HK ур. 13; Ansari (6))
_TURNER_COEFFICIENT: Final = 3.1


@dataclass(frozen=True, slots=True)
class DispersedBubbleBoundary:
    """Результат проверки перехода в дисперсно-пузырьковый режим (Barnea [5]–[7]).

    Атрибуты:
        d_max_m: максимальный устойчивый диаметр пузырька при турбулентном дроблении, м ([5]).
        d_deformation_m: критический диаметр деформации пузырька d_CD, м ([6]).
        d_migration_m: критический диаметр миграции к стенке d_CB, м ([7]); ∞ при cosβ = 0.
        d_critical_m: d_C = min(d_CD, d_CB), м.
        alpha_ns: газосодержание без проскальзывания v_sg/v_m.
    """

    d_max_m: float
    d_deformation_m: float
    d_migration_m: float
    d_critical_m: float
    alpha_ns: float

    @property
    def ratio(self) -> float:
        """Отношение d_C/d_max; значение ≥ 1 означает дисперсно-пузырьковый режим."""
        return self.d_critical_m / self.d_max_m

    @property
    def dispersion_factor(self) -> float:
        """Правая часть Ansari (4): 0.725 + 4.15·(v_sg/v_m)^0.5 (множитель в d_max, Barnea [5])."""
        return _DISPERSION_BASE + _DISPERSION_GAS_COEFFICIENT * math.sqrt(self.alpha_ns)


def bubble_slug_drift_flux(
    v_sl_m_s: float,
    v_rise_m_s: float,
    *,
    c0: float,
    alpha: float = BUBBLE_SLUG_ALPHA,
) -> float:
    """Граница пузырьковый -> пробковый по дрейфовой модели v_g = C0·v_m + v∞.

    v_sg = α·(C0·v_sl + v∞)/(1 − α·C0) (HK 1988, ур. 1–2; при α = 0.25, C0 = 1.2 —
    0.429·v_sl + 0.357·v∞).

    Параметры:
        v_sl_m_s: приведённая скорость жидкости, м/с.
        v_rise_m_s: скорость всплытия пузырька v∞, м/с.
        c0: коэффициент профиля скорости (HK: 1.2 + 0.371·d_t/d_c).
        alpha: критическое газосодержание.

    Возвращает:
        Приведённая скорость газа на границе, м/с.
    """
    return alpha * (c0 * v_sl_m_s + v_rise_m_s) / (1.0 - alpha * c0)


def bubble_slug_slip(
    v_sl_m_s: float,
    v_rise_m_s: float,
    *,
    alpha: float = BUBBLE_SLUG_ALPHA,
    sin_theta: float = 1.0,
) -> float:
    """Граница пузырьковый -> пробковый при скольжении относительно жидкости.

    v_sg = α/(1 − α)·v_sl + α·v_rise·sinθ (Barnea 1987, [4]; Taitel 1980; Ansari 1994, (2):
    0.333·v_sl + 0.25·v_s). Ansari принимает sinθ = 1.

    Параметры:
        v_sl_m_s: приведённая скорость жидкости, м/с.
        v_rise_m_s: скорость всплытия пузырька, м/с.
        alpha: критическое газосодержание.
        sin_theta: sin угла наклона к горизонту (флаг `bubble_slug_sin_theta` Ansari).

    Возвращает:
        Приведённая скорость газа на границе, м/с.
    """
    return alpha / (1.0 - alpha) * v_sl_m_s + alpha * v_rise_m_s * sin_theta


def min_bubble_diameter(
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    sigma_n_m: float,
    *,
    coef: float,
) -> float:
    """Минимальный диаметр трубы для пузырькового режима d_min = coef·[Δρ·σ/(ρ_L²·g)]^0.5.

    Barnea 1987, [1] (coef = 19); Ansari 1994 (coef = 19.01, Taitel 1980).

    Параметры:
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.
        coef: коэффициент (19.0 или 19.01).

    Возвращает:
        d_min, м.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return coef * math.sqrt(delta_rho * sigma_n_m / (rho_l_kg_m3**2 * GRAVITY_M_S2))


def dispersed_bubble_barnea(
    v_sl_m_s: float,
    v_sg_m_s: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    sigma_n_m: float,
    d_m: float,
    f_darcy_mixture: float,
    *,
    theta_rad: float,
) -> DispersedBubbleBoundary:
    """Переход в дисперсно-пузырьковый режим по Barnea 1986/1987, [5]–[7].

    Режим дисперсно-пузырьковый, если d_max < d_C (`ratio` ≥ 1) и α_ns ≤ 0.52. Та же граница
    записывается у Ansari 1994 (4) через коэффициент Moody f (при d_C = d_CD).

    Коэффициент трения f_M в [5], [7] — Fanning; здесь принимается Darcy (Moody) и
    переводится как f_F = f_D/4.

    Параметры:
        v_sl_m_s, v_sg_m_s: приведённые скорости жидкости и газа, м/с.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.
        d_m: диаметр трубы, м.
        f_darcy_mixture: коэффициент трения Darcy по скорости смеси, безразм.
        theta_rad: угол наклона от горизонтали, рад (> 0 — вверх).

    Возвращает:
        `DispersedBubbleBoundary` (d_max, d_CD, d_CB, d_C, α_ns).
    """
    v_m = v_sl_m_s + v_sg_m_s
    alpha_ns = v_sg_m_s / v_m
    f_fanning = f_darcy_mixture * _FANNING_PER_DARCY
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    d_max = (
        (_DISPERSION_BASE + _DISPERSION_GAS_COEFFICIENT * math.sqrt(alpha_ns))
        * (sigma_n_m / rho_l_kg_m3) ** _DISPERSION_SIGMA_EXPONENT
        * (2.0 * f_fanning / d_m * v_m**3) ** _DISPERSION_POWER_EXPONENT
    )
    d_deformation = _DEFORMATION_FACTOR * math.sqrt(
        _DEFORMATION_COEFFICIENT * sigma_n_m / (delta_rho * GRAVITY_M_S2)
    )
    cos_theta = math.cos(theta_rad)
    if abs(cos_theta) < _COS_ZERO_TOLERANCE:
        d_migration = math.inf
    else:
        d_migration = (
            _MIGRATION_COEFFICIENT
            * rho_l_kg_m3
            / delta_rho
            * f_fanning
            * v_m**2
            / (GRAVITY_M_S2 * cos_theta)
        )
    return DispersedBubbleBoundary(
        d_max_m=d_max,
        d_deformation_m=d_deformation,
        d_migration_m=d_migration,
        d_critical_m=min(d_deformation, d_migration),
        alpha_ns=alpha_ns,
    )


def dispersed_bubble_mixture_velocity_taitel1980(
    d_m: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    sigma_n_m: float,
    mu_l_pa_s: float,
) -> float:
    """Скорость смеси перехода в дисперсно-пузырьковый режим (вертикаль, Taitel 1980).

    v_M^1.12 = 4.68·d^0.48·[g·Δρ/σ]^0.5·(σ/ρ_L)^0.6·(ρ_L/μ_L)^0.08 (HK 1988, ур. 6).
    # typo fix: напечатано (ρ_M/μ_L)^0.08 — взято ρ_L (Erratum 1989; ТЗ, «Опечатки», HK (6)).

    Параметры:
        d_m: диаметр трубы, м.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.
        mu_l_pa_s: вязкость жидкости, Па·с.

    Возвращает:
        Скорость смеси v_M на границе, м/с.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    rhs = (
        _TAITEL1980_COEFFICIENT
        * d_m**_TAITEL1980_D_EXPONENT
        * math.sqrt(GRAVITY_M_S2 * delta_rho / sigma_n_m)
        * (sigma_n_m / rho_l_kg_m3) ** _TAITEL1980_SIGMA_EXPONENT
        * (rho_l_kg_m3 / mu_l_pa_s) ** _TAITEL1980_VISCOSITY_EXPONENT
    )
    return rhs ** (1.0 / _TAITEL1980_VELOCITY_EXPONENT)


def max_packing_velocity_ratio(alpha: float) -> float:
    """Отношение v_sl/v_sg на границе плотной упаковки пузырьков: (1 − α)/α.

    Barnea 1987, [8] (α = 0.52); Ansari 1994, (5) (α = 0.76: v_sg = 3.17·v_sl).

    Параметры:
        alpha: предельное газосодержание.

    Возвращает:
        v_sl/v_sg на границе, безразм.
    """
    return (1.0 - alpha) / alpha


def exceeds_max_packing(v_sl_m_s: float, v_sg_m_s: float, alpha: float) -> bool:
    """Проверяет, что газосодержание без проскальзывания превышает предел упаковки.

    Параметры:
        v_sl_m_s, v_sg_m_s: приведённые скорости жидкости и газа, м/с.
        alpha: предельное газосодержание.

    Возвращает:
        True, если v_sg/(v_sl + v_sg) > α.
    """
    return v_sg_m_s / (v_sl_m_s + v_sg_m_s) > alpha


def annular_turner_velocity(rho_l_kg_m3: float, rho_g_kg_m3: float, sigma_n_m: float) -> float:
    """Скорость газа перехода в кольцевой режим v_sg = 3.1·[σ·g·Δρ/ρ_g²]^0.25.

    Turner 1969; HK 1988, ур. 13; Ansari 1994, (6).

    Параметры:
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.

    Возвращает:
        Приведённая скорость газа на границе, м/с.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return _TURNER_COEFFICIENT * (sigma_n_m * GRAVITY_M_S2 * delta_rho / rho_g_kg_m3**2) ** 0.25
