"""Кольцевой режим: унос, фактор межфазного трения, устойчивость и перемыкание плёнки.

Схема (пересечения №11–№13 ТЗ):
    1. v_crit → доля унесённой жидкости (Wallis для Ansari, Steen–Wallis для HK);
    2. ядро: λ_LC и плотность ядра (смешение через `mixture.slip`);
    3. фактор Z межфазного трения (Wallis / Whalley–Hewitt);
    4. уравнение плёнки (Barnea [15] ≡ Ansari (96)) → наименьший корень α_L;
    5. критерии: неустойчивость плёнки (Barnea [16] ≡ Ansari (8)) и перемыкание
       ядра (Barnea [17]; Ansari (12)).

Источники:
    * Barnea D., 1986, IJMF 12(5), 1987, IJMF 13(1) — [15]–[17];
    * Ansari A.M. et al., 1994, SPE PF 9(2) — (7)–(12), (70)–(72), (84)–(85), (96);
    * Hasan A.R., Kabir C.S., 1988 (+ Erratum 1989) — (34)–(39);
    * Wallis G.B., 1969; Steen D.A., Wallis G.B., 1964; Whalley P.B., Hewitt G.F., 1978.
"""

import math
from collections.abc import Callable
from typing import Final

from correlations.core import solvers

# Ansari (71): F_E = 1 − exp[−0.125·(v_crit − 1.5)]
_WALLIS_ENTRAINMENT_RATE: Final = 0.125
_WALLIS_ENTRAINMENT_THRESHOLD: Final = 1.5
# v_crit = 10⁴·v_sg·μ_g/σ·(ρ_g/ρ_L)^0.5 (Ansari (72); HK (34))
_V_CRIT_SCALE: Final = 1.0e4
# HK (35)/(36): граница ветвей и коэффициенты
_STEEN_WALLIS_SPLIT: Final = 4.0
_STEEN_WALLIS_LOW_COEFFICIENT: Final = 0.0055
_STEEN_WALLIS_LOW_EXPONENT: Final = 2.86
_STEEN_WALLIS_HIGH_COEFFICIENT: Final = 0.857
_STEEN_WALLIS_HIGH_OFFSET: Final = 0.20
# Z: Wallis 1 + 300·δ/d (Ansari (84), HK (38)); 1 + 75·α_L (Barnea [15], HK (39))
_WALLIS_Z_DELTA_COEFFICIENT: Final = 300.0
_WALLIS_Z_HOLDUP_COEFFICIENT: Final = 75.0
# Whalley–Hewitt: 1 + 24·(ρ_L/ρ_g)^(1/3)·δ/d (Ansari (85); typo fix: напечатано (ρ_g/ρ_g))
_WHALLEY_HEWITT_COEFFICIENT: Final = 24.0
# Barnea [15]/Ansari (96): степень (1 − α)
_FILM_POWER: Final = 2.5
# Barnea [17]: α_L/R_sm ≥ 0.5; R_sm ≈ 0.48
_BARNEA_BLOCKAGE_RATIO: Final = 0.5
BARNEA_R_SM: Final = 0.48
# Диапазон поиска корня уравнения плёнки и частота сетки
_FILM_LO: Final = 1e-6
_FILM_HI: Final = 1.0 - 1e-6
_FILM_GRID: Final = 2000


def critical_velocity(
    v_sg_m_s: float,
    mu_g_pa_s: float,
    sigma_n_m: float,
    rho_g_kg_m3: float,
    rho_l_kg_m3: float,
) -> float:
    """Безразмерная критическая скорость v_crit = 10⁴·v_sg·μ_g/σ·(ρ_g/ρ_L)^0.5.

    Ansari 1994, (72); HK 1988, (34) — величина (v_gs)_c × 10⁴.

    Параметры:
        v_sg_m_s: приведённая скорость газа, м/с.
        mu_g_pa_s: вязкость газа, Па·с.
        sigma_n_m: поверхностное натяжение, Н/м.
        rho_g_kg_m3, rho_l_kg_m3: плотности газа и жидкости, кг/м³.

    Возвращает:
        v_crit, безразм.
    """
    return _V_CRIT_SCALE * v_sg_m_s * mu_g_pa_s / sigma_n_m * math.sqrt(rho_g_kg_m3 / rho_l_kg_m3)


def entrainment_wallis(v_crit: float) -> float:
    """Доля унесённой жидкости F_E = 1 − exp[−0.125·(v_crit − 1.5)] (Wallis; Ansari (71)).

    Параметры:
        v_crit: безразмерная критическая скорость (см. `critical_velocity`).

    Возвращает:
        F_E, 0…1 (при v_crit < 1.5 формула даёт отрицательное значение — берётся 0).
    """
    f_e = 1.0 - math.exp(-_WALLIS_ENTRAINMENT_RATE * (v_crit - _WALLIS_ENTRAINMENT_THRESHOLD))
    return max(0.0, f_e)


def entrainment_steen_wallis(v_crit: float) -> float:
    """Доля унесённой жидкости E по Steen–Wallis (HK 1988, ур. 35–36).

    E = 0.0055·v_crit^2.86 при v_crit < 4; E = 0.857·lg(v_crit) − 0.20 при v_crit > 4.
    Применима при турбулентной плёнке (Re плёнки > 3000).

    Параметры:
        v_crit: безразмерная критическая скорость (см. `critical_velocity`).

    Возвращает:
        E, 0…1 (значения > 1 обрезаются до 1 — HK, текст после ур. 36).
    """
    if v_crit < _STEEN_WALLIS_SPLIT:
        e = _STEEN_WALLIS_LOW_COEFFICIENT * v_crit**_STEEN_WALLIS_LOW_EXPONENT
    else:
        e = _STEEN_WALLIS_HIGH_COEFFICIENT * math.log10(v_crit) - _STEEN_WALLIS_HIGH_OFFSET
    return min(1.0, max(0.0, e))


def core_liquid_fraction(f_e: float, v_sl_m_s: float, v_sg_m_s: float) -> float:
    """Доля жидкости в ядре без проскальзывания λ_LC = F_E·v_sl/(v_sg + F_E·v_sl).

    Ansari 1994, (70); HK 1988, (37a) (f_gc = 1 − λ_LC).

    Параметры:
        f_e: доля унесённой жидкости, 0…1.
        v_sl_m_s, v_sg_m_s: приведённые скорости жидкости и газа, м/с.

    Возвращает:
        λ_LC, безразм.
    """
    return f_e * v_sl_m_s / (v_sg_m_s + f_e * v_sl_m_s)


def wallis_z(delta_over_d: float) -> float:
    """Фактор межфазного трения Z = 1 + 300·δ/d (Wallis; Ansari (84); HK (38)).

    Параметры:
        delta_over_d: безразмерная толщина плёнки δ/d.

    Возвращает:
        Z, безразм.
    """
    return 1.0 + _WALLIS_Z_DELTA_COEFFICIENT * delta_over_d


def wallis_z_from_holdup(alpha_l: float) -> float:
    """Фактор межфазного трения Z = 1 + 75·α_L (Barnea [15]; HK (39)).

    Параметры:
        alpha_l: доля сечения, занятая плёнкой (4δ(1 − δ) ≈ 4δ).

    Возвращает:
        Z, безразм.
    """
    return 1.0 + _WALLIS_Z_HOLDUP_COEFFICIENT * alpha_l


def whalley_hewitt_z(delta_over_d: float, rho_l_kg_m3: float, rho_g_kg_m3: float) -> float:
    """Фактор межфазного трения Z = 1 + 24·(ρ_L/ρ_g)^(1/3)·δ/d (Whalley–Hewitt; Ansari (85)).

    # typo fix: напечатано (ρ_g/ρ_g)^(1/3) — взято (ρ_L/ρ_g) (ТЗ, «Опечатки», Ansari (85)).

    Параметры:
        delta_over_d: безразмерная толщина плёнки δ/d.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.

    Возвращает:
        Z, безразм.
    """
    return 1.0 + _WHALLEY_HEWITT_COEFFICIENT * (rho_l_kg_m3 / rho_g_kg_m3) ** (1.0 / 3.0) * (
        delta_over_d
    )


def film_holdup_from_delta(delta: float) -> float:
    """Доля сечения, занятая плёнкой, α = 4·δ·(1 − δ) (Ansari (11)).

    Параметры:
        delta: безразмерная толщина плёнки δ/d (0…0.5).

    Возвращает:
        α, 0…1.
    """
    return 4.0 * delta * (1.0 - delta)


def delta_from_film_holdup(alpha_l: float) -> float:
    """Безразмерная толщина плёнки δ/d по доле сечения: δ = (1 − √(1 − α))/2 (корень < 0.5).

    Параметры:
        alpha_l: доля сечения, занятая плёнкой, 0…1.

    Возвращает:
        δ/d, безразм.
    """
    return 0.5 * (1.0 - math.sqrt(1.0 - alpha_l))


def film_equation_residual(alpha_l: float, x_sq: float, y: float, z: float) -> float:
    """Невязка уравнения плёнки Y − Z/(α(1−α)^2.5) + X²/α³ (Barnea [15]; Ansari (96)).

    # typo fix: у Barnea напечатано (1 − α_L)^(3/2) — взято 5/2 (ТЗ, «Опечатки», Barnea [15]).

    Параметры:
        alpha_l: доля сечения, занятая плёнкой.
        x_sq: параметр Локкарта–Мартинелли X² (модифицированный для Ansari).
        y: параметр Y.
        z: фактор межфазного трения Z.

    Возвращает:
        Невязка; на решении равна 0.
    """
    return y - z / (alpha_l * (1.0 - alpha_l) ** _FILM_POWER) + x_sq / alpha_l**3


def solve_film_holdup(
    x_sq: float,
    y: float,
    z_of_alpha: Callable[[float], float],
) -> float | None:
    """Решает уравнение плёнки относительно α_L; при нескольких корнях — наименьший (Q9 ТЗ).

    Параметры:
        x_sq: параметр X².
        y: параметр Y.
        z_of_alpha: фактор Z как функция α_L (`wallis_z_from_holdup` или композиция с δ(α)).

    Возвращает:
        α_L на (0, 1) или None, если корня нет.
    """

    def residual(alpha: float) -> float:
        return film_equation_residual(alpha, x_sq, y, z_of_alpha(alpha))

    return solvers.smallest_root(residual, _FILM_LO, _FILM_HI, n_grid=_FILM_GRID)


def film_instability_y(alpha_l: float, x_sq: float) -> float:
    """Граница устойчивости плёнки Y = (2 − 1.5·α)/(α³·(1 − 1.5·α))·X².

    Barnea 1987, [16]; Ansari 1994, (8). Плёнка неустойчива (режим перемежающийся),
    если Y ≥ значения функции. Имеет особенность при α = 2/3.

    Параметры:
        alpha_l: доля сечения, занятая плёнкой.
        x_sq: параметр X².

    Возвращает:
        Граничное значение Y.
    """
    return (2.0 - 1.5 * alpha_l) / (alpha_l**3 * (1.0 - 1.5 * alpha_l)) * x_sq


def film_blockage_ansari(film_holdup: float, lam_lc: float, delta: float) -> float:
    """Критерий перемыкания Ansari (12): H_LF + λ_LC·A_C/A, где A_C/A = (1 − 2δ)².

    Параметры:
        film_holdup: доля сечения плёнки H_LF = 4δ(1 − δ).
        lam_lc: доля жидкости в ядре λ_LC.
        delta: безразмерная толщина плёнки δ/d.

    Возвращает:
        Значение критерия; > 0.12 — ядро перемыкается (не кольцевой режим).
    """
    return film_holdup + lam_lc * (1.0 - 2.0 * delta) ** 2


def film_blocked_barnea(alpha_l: float, r_sm: float = BARNEA_R_SM) -> bool:
    """Критерий перемыкания Barnea [17]: α_L/R_sm ≥ 0.5 (при R_sm = 0.48 — α_L ≥ 0.24).

    Параметры:
        alpha_l: доля сечения, занятая плёнкой.
        r_sm: минимальное содержание жидкости в пробке, 0.48.

    Возвращает:
        True, если ядро перемыкается плёнкой.
    """
    return alpha_l / r_sm >= _BARNEA_BLOCKAGE_RATIO
