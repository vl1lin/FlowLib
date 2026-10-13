"""Скорости всплытия пузырьков (пересечения №4 и №5 ТЗ).

Схема: скорость всплытия малого пузырька v∞ (Harmathy) и пузыря Тейлора v∞T
(коэффициент C2 задаётся автором: 0.345 HK, 0.35 Ansari, формула Wallis для HK).

Источники:
    * Harmathy T.Z., 1960, AIChE J. 6(2) — v∞ (HK 1988 ур. 3; Barnea 1987 [3]; Ansari 1994 (3));
    * Hasan A.R., Kabir C.S., 1988, SPEPE 3(2) (+ Erratum 1989) — ур. 5, 23–27;
    * Wallis G.B., 1969, «One-dimensional two-phase flow» — формула C2.
"""

import math
from typing import Final

from correlations.core.constants import GRAVITY_M_S2

TAYLOR_COEFFICIENT_HK: Final = 0.345  # HK 1988, ур. 27
TAYLOR_COEFFICIENT_ANSARI: Final = 0.35  # Ansari 1994, (34)

_HARMATHY_COEFFICIENT: Final = 1.53  # HK ур. 3, Barnea [3], Ansari (3)
# HK ур. 23–26: параметры формулы Wallis для C2
_WALLIS_C2_MAX: Final = 0.345
_WALLIS_EOTVOS_SHIFT: Final = 3.37
_WALLIS_NF_HIGH: Final = 250.0  # N_f > 250: m = 10 (26a)
_WALLIS_NF_LOW: Final = 18.0  # N_f < 18: m = 25 (26c)
_WALLIS_M_HIGH: Final = 10.0
_WALLIS_M_LOW: Final = 25.0
_WALLIS_M_COEFFICIENT: Final = 69.0  # (26b): m = 69·N_f^-0.35
_WALLIS_M_EXPONENT: Final = -0.35  # typo fix: напечатано +0.35 (ТЗ, «Опечатки», HK (26b))


def harmathy_rise_velocity(
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    sigma_n_m: float,
    *,
    coef: float = _HARMATHY_COEFFICIENT,
) -> float:
    """Скорость всплытия малого пузырька v∞ = coef·[g·σ·Δρ/ρ_L²]^0.25.

    Harmathy 1960; HK 1988, ур. 3; Barnea 1987, [3]; Ansari 1994, (3).

    Параметры:
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.
        coef: коэффициент (1.53; HK ур. 17 использует 1.50).

    Возвращает:
        v∞, м/с.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return coef * (GRAVITY_M_S2 * sigma_n_m * delta_rho / rho_l_kg_m3**2) ** 0.25


def taylor_rise_velocity(
    d_m: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    *,
    coef: float,
) -> float:
    """Скорость всплытия пузыря Тейлора v∞T = coef·√(g·d·Δρ/ρ_L).

    HK 1988, ур. 5 и 28; Ansari 1994, (34). Для затрубья (HK ур. 30) передать
    d = d_c − d_t и коэффициент 0.30 + 0.22·(d_t/d_c).

    Параметры:
        d_m: диаметр (эквивалентный диаметр затрубья), м.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        coef: коэффициент C2 (`TAYLOR_COEFFICIENT_HK`, `TAYLOR_COEFFICIENT_ANSARI`
            или `wallis_taylor_coefficient`).

    Возвращает:
        v∞T, м/с.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return coef * math.sqrt(GRAVITY_M_S2 * d_m * delta_rho / rho_l_kg_m3)


def inverse_viscosity_number(
    d_m: float, rho_l_kg_m3: float, rho_g_kg_m3: float, mu_l_pa_s: float
) -> float:
    """Обратное число вязкости N_f = [d³·g·Δρ·ρ_L]^0.5/μ_L (HK 1988, ур. 24).

    Параметры:
        d_m: диаметр трубы, м.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        mu_l_pa_s: вязкость жидкости, Па·с.

    Возвращает:
        N_f, безразм.
    """
    delta_rho = rho_l_kg_m3 - rho_g_kg_m3
    return math.sqrt(d_m**3 * GRAVITY_M_S2 * delta_rho * rho_l_kg_m3) / mu_l_pa_s


def eotvos_number(d_m: float, rho_l_kg_m3: float, rho_g_kg_m3: float, sigma_n_m: float) -> float:
    """Число Этвёша N_Eo = g·d²·Δρ/σ (HK 1988, ур. 25).

    Параметры:
        d_m: диаметр трубы, м.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.

    Возвращает:
        N_Eo, безразм.
    """
    return GRAVITY_M_S2 * d_m**2 * (rho_l_kg_m3 - rho_g_kg_m3) / sigma_n_m


def wallis_taylor_coefficient(n_f: float, n_eo: float) -> float:
    """Коэффициент C2 скорости пузыря Тейлора по Wallis (HK 1988, ур. 23, 26a–c).

    C2 = 0.345·[1 − exp(−0.01·N_f/0.345)]·[1 − exp((3.37 − N_Eo)/m)];
    m = 10 (N_f > 250), 69·N_f^-0.35 (18 < N_f < 250), 25 (N_f < 18).

    Исправления напечатанных форм (ТЗ, «Опечатки»):
        # typo fix: напечатано exp(3.37·N_Eo/m) — взято exp((3.37 − N_Eo)/m) (Wallis 1969);
        # typo fix: напечатано m = 69·N_f^0.35 — взято 69·N_f^-0.35 (сшивка с (26a), (26c)).

    Параметры:
        n_f: обратное число вязкости, безразм.
        n_eo: число Этвёша, безразм.

    Возвращает:
        C2, безразм.
    """
    if n_f > _WALLIS_NF_HIGH:
        m = _WALLIS_M_HIGH
    elif n_f > _WALLIS_NF_LOW:
        m = _WALLIS_M_COEFFICIENT * n_f**_WALLIS_M_EXPONENT
    else:
        m = _WALLIS_M_LOW
    viscous = 1.0 - math.exp(-0.01 * n_f / _WALLIS_C2_MAX)
    surface = 1.0 - math.exp((_WALLIS_EOTVOS_SHIFT - n_eo) / m)
    return _WALLIS_C2_MAX * viscous * surface
