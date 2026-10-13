"""Расслоённый поток: геометрия, равновесный уровень, критерии KH и Jeffreys.

Схема (пересечение №14 ТЗ):
    1. геометрия по безразмерному уровню h̃ = h_L/D (TD (10)–(16)) или по углу δ (MB (18), (20));
    2. параметры X² и Y из градиентов однофазных потоков (TD (8), (9));
    3. уравнение равновесного уровня TD (7) → корни h̃ (наименьший — Q9 ТЗ);
    4. критерий Кельвина–Гельмгольца (TD (25) ≡ Barnea [9]) и критерий ветровых волн
       Джеффриса (TD (30) ≡ Barnea [20]).

Соглашение знака: θ > 0 — подъём. У Taitel–Dukler α > 0 — нисходящий поток (α = −θ);
здесь Y считается для θ (Barnea [12b]), поэтому в уравнении (7) член −4Y_TD → +4Y.
Градиенты однофазных потоков — через Darcy (`gradient.friction_gradient`); Fanning TD
переводится в Darcy множителем 4 (`friction.fanning_to_darcy`).

Источники:
    * Taitel Y., Dukler A.E., 1976, AIChE J. 22(1) — (6)–(16), (25), (30);
    * Barnea D., 1987, IJMF 13(1) — [9]–[12], [20];
    * Mukherjee H., Brill J.P., 1985, IJMF 11(3) — (18), (20).
"""

import math
from dataclasses import dataclass
from typing import Final

from correlations.core import solvers
from correlations.core.constants import GRAVITY_M_S2

# Площадь сечения в долях D²: Ã = π/4
_AREA_TILDE: Final = math.pi / 4.0
# TD: турбулентные показатели трения n = m = 0.2 (C = 0.046); ламинарные n = m = 1 (C = 16)
DEFAULT_EXPONENT: Final = 0.2
# TD (30), Barnea [20]: коэффициент экранирования s
SHELTERING_COEFFICIENT: Final = 0.01
# Диапазон и частота сетки поиска уровня
_LEVEL_LO: Final = 1e-6
_LEVEL_HI: Final = 1.0 - 1e-6
_LEVEL_GRID: Final = 2000
# δ, ниже которого δ − sin δ считается рядом Тейлора
_SERIES_LIMIT: Final = 0.5
_SERIES_TERMS: Final = 12
# Множитель члена Y в (7)
_Y_FACTOR: Final = 4.0


@dataclass(frozen=True, slots=True)
class StratifiedGeometry:
    """Безразмерная геометрия расслоённого потока (длины — в D, площади — в D²).

    Атрибуты:
        h: уровень жидкости h̃ = h_L/D.
        a_l, a_g: площади жидкости и газа Ã_L, Ã_G (TD (10), (11)).
        s_l, s_g, s_i: периметры S̃_L, S̃_G и ширина границы раздела S̃_i (TD (12)–(14)).
        u_l, u_g: безразмерные скорости ũ_L = Ã/Ã_L, ũ_G = Ã/Ã_G (TD (15), (16)).
        d_l, d_g: безразмерные гидравлические диаметры D̃_L = 4Ã_L/S̃_L,
            D̃_G = 4Ã_G/(S̃_G + S̃_i) (TD (6)).
    """

    h: float
    a_l: float
    a_g: float
    s_l: float
    s_g: float
    s_i: float
    u_l: float
    u_g: float
    d_l: float
    d_g: float


def geometry(h_tilde: float) -> StratifiedGeometry:
    """Геометрия расслоённого потока по уровню h̃ (TD (6), (10)–(16)).

    Параметры:
        h_tilde: безразмерный уровень жидкости h_L/D, 0 < h̃ < 1.

    Возвращает:
        `StratifiedGeometry`.
    """
    t = 2.0 * h_tilde - 1.0
    angle = math.acos(max(-1.0, min(1.0, t)))
    root = math.sqrt(max(0.0, 1.0 - t * t))
    a_l = 0.25 * (math.pi - angle + t * root)
    a_g = 0.25 * (angle - t * root)
    s_l = math.pi - angle
    s_g = angle
    s_i = root
    return StratifiedGeometry(
        h=h_tilde,
        a_l=a_l,
        a_g=a_g,
        s_l=s_l,
        s_g=s_g,
        s_i=s_i,
        u_l=_AREA_TILDE / a_l,
        u_g=_AREA_TILDE / a_g,
        d_l=4.0 * a_l / s_l,
        d_g=4.0 * a_g / (s_g + s_i),
    )


def _delta_minus_sin(delta: float) -> float:
    """δ − sin δ без потери точности при малых δ (ряд Тейлора δ³/3! − δ⁵/5! + …).

    Параметры:
        delta: угол, рад.

    Возвращает:
        δ − sin δ.
    """
    if delta >= _SERIES_LIMIT:
        return delta - math.sin(delta)
    term = delta**3 / 6.0
    total = 0.0
    k = 3
    for _ in range(_SERIES_TERMS):
        total += term
        term *= -(delta * delta) / ((k + 1) * (k + 2))
        k += 2
    return total


def delta_from_holdup(h_l: float) -> float:
    """Угол δ расслоённого потока из H_L = (δ − sin δ)/(2π) (MB 1985, (20)).

    Параметры:
        h_l: истинное содержание жидкости H_L, 0…1.

    Возвращает:
        δ, рад (0…2π).

    Исключения:
        ConvergenceError: решатель не сошёлся.
    """

    if h_l <= 0.0:
        return 0.0
    target = 2.0 * math.pi * h_l

    def residual(delta: float) -> float:
        return _delta_minus_sin(delta) - target

    def derivative(delta: float) -> float:
        return 1.0 - math.cos(delta)

    guess = min((6.0 * target) ** (1.0 / 3.0), math.pi)  # δ³/6 ≈ 2π·H_L при малых δ
    return solvers.newton_bracketed(
        residual, derivative, guess, (0.0, 2.0 * math.pi), rtol=1e-13, atol=0.0
    )


def level_from_delta(delta: float) -> float:
    """Безразмерный уровень h_L/d = 0.5·(1 − cos(δ/2)) (MB 1985, (18)).

    Параметры:
        delta: угол δ, рад.

    Возвращает:
        h_L/d, 0…1.
    """
    return 0.5 * (1.0 - math.cos(0.5 * delta))


def x_squared(grad_l_pa_m: float, grad_g_pa_m: float) -> float:
    """Параметр Локкарта–Мартинелли X² = (dp/dx)_LS/(dp/dx)_GS (TD (8)).

    Параметры:
        grad_l_pa_m: градиент трения жидкости, текущей одной в трубе, Па/м.
        grad_g_pa_m: то же для газа, Па/м.

    Возвращает:
        X², безразм.
    """
    return grad_l_pa_m / grad_g_pa_m


def y_parameter(
    rho_l_kg_m3: float, rho_g_kg_m3: float, theta_rad: float, grad_g_pa_m: float
) -> float:
    """Параметр Y = (ρ_L − ρ_g)·g·sinθ/(dp/dx)_GS (TD (9); Barnea [12b]).

    Знак — для θ > 0 вверх (у TD sinα с α = −θ).

    Параметры:
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        theta_rad: угол от горизонтали, рад (> 0 — вверх).
        grad_g_pa_m: градиент трения газа, текущего одного в трубе, Па/м.

    Возвращает:
        Y, безразм.
    """
    return (rho_l_kg_m3 - rho_g_kg_m3) * GRAVITY_M_S2 * math.sin(theta_rad) / grad_g_pa_m


def _liquid_and_gas_terms(h_tilde: float, n: float, m: float) -> tuple[float, float]:
    """Члены уравнения (7): L(h̃) при X² и G(h̃).

    L = (ũ_L·D̃_L)^-n·ũ_L²·S̃_L/Ã_L; G = (ũ_G·D̃_G)^-m·ũ_G²·(S̃_G/Ã_G + S̃_i/Ã_L + S̃_i/Ã_G).

    Параметры:
        h_tilde: безразмерный уровень.
        n, m: показатели трения жидкости и газа.

    Возвращает:
        Пара (L, G).
    """
    geo = geometry(h_tilde)
    liquid = (geo.u_l * geo.d_l) ** -n * geo.u_l**2 * geo.s_l / geo.a_l
    gas = (
        (geo.u_g * geo.d_g) ** -m
        * geo.u_g**2
        * (geo.s_g / geo.a_g + geo.s_i / geo.a_l + geo.s_i / geo.a_g)
    )
    return liquid, gas


def equilibrium_residual(
    h_tilde: float,
    x_sq: float,
    y: float,
    *,
    n: float = DEFAULT_EXPONENT,
    m: float = DEFAULT_EXPONENT,
) -> float:
    """Невязка уравнения равновесного уровня TD (7): X²·L(h̃) − G(h̃) + 4Y.

    Параметры:
        h_tilde: безразмерный уровень.
        x_sq: параметр X².
        y: параметр Y (θ > 0 вверх).
        n, m: показатели трения жидкости и газа (0.2 — турбулентный, 1 — ламинарный).

    Возвращает:
        Невязка; на равновесном уровне равна 0.
    """
    liquid, gas = _liquid_and_gas_terms(h_tilde, n, m)
    return x_sq * liquid - gas + _Y_FACTOR * y


def x_squared_for_level(
    h_tilde: float,
    y: float,
    *,
    n: float = DEFAULT_EXPONENT,
    m: float = DEFAULT_EXPONENT,
) -> float:
    """X², при котором равновесный уровень равен h̃ (решение (7) относительно X²).

    Параметры:
        h_tilde: безразмерный уровень.
        y: параметр Y (θ > 0 вверх).
        n, m: показатели трения жидкости и газа.

    Возвращает:
        X² = (G(h̃) − 4Y)/L(h̃); при больших Y > 0 может быть отрицательным (нет физического
          решения).
    """
    liquid, gas = _liquid_and_gas_terms(h_tilde, n, m)
    return (gas - _Y_FACTOR * y) / liquid


def equilibrium_levels(
    x_sq: float,
    y: float,
    *,
    n: float = DEFAULT_EXPONENT,
    m: float = DEFAULT_EXPONENT,
) -> tuple[float, ...]:
    """Все корни уравнения равновесного уровня TD (7) на (0, 1) по возрастанию.

    Параметры:
        x_sq: параметр X².
        y: параметр Y (θ > 0 вверх).
        n, m: показатели трения жидкости и газа.

    Возвращает:
        Кортеж уровней h̃ (пустой, если расслоённого равновесия нет).
    """

    def residual(h_tilde: float) -> float:
        return equilibrium_residual(h_tilde, x_sq, y, n=n, m=m)

    return tuple(solvers.bracketed_roots(residual, _LEVEL_LO, _LEVEL_HI, n_grid=_LEVEL_GRID))


def equilibrium_level(
    x_sq: float,
    y: float,
    *,
    n: float = DEFAULT_EXPONENT,
    m: float = DEFAULT_EXPONENT,
) -> float | None:
    """Равновесный уровень h̃ расслоённого потока; при нескольких корнях — наименьший (Q9 ТЗ).

    Параметры:
        x_sq: параметр X².
        y: параметр Y (θ > 0 вверх).
        n, m: показатели трения жидкости и газа.

    Возвращает:
        h̃ на (0, 1) или None, если равновесия нет.
    """

    def residual(h_tilde: float) -> float:
        return equilibrium_residual(h_tilde, x_sq, y, n=n, m=m)

    return solvers.smallest_root(residual, _LEVEL_LO, _LEVEL_HI, n_grid=_LEVEL_GRID)


def modified_froude(
    v_sg_m_s: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    d_m: float,
    theta_rad: float,
) -> float:
    """Модифицированное число Фруда F = √(ρ_g/Δρ)·v_sg/√(D·g·cosθ) (TD; Barnea [10]).

    Параметры:
        v_sg_m_s: приведённая скорость газа, м/с.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        d_m: диаметр трубы, м.
        theta_rad: угол от горизонтали, рад.

    Возвращает:
        F, безразм.
    """
    ratio = math.sqrt(rho_g_kg_m3 / (rho_l_kg_m3 - rho_g_kg_m3))
    return ratio * v_sg_m_s / math.sqrt(d_m * GRAVITY_M_S2 * math.cos(theta_rad))


def kelvin_helmholtz_parameter(froude: float, h_tilde: float) -> float:
    """Левая часть критерия Кельвина–Гельмгольца F²·ũ_G²·(dÃ_L/dh̃)/(C2²·Ã_G), C2 = 1 − h̃.

    TD 1976, (25) и Barnea 1987, [9]; dÃ_L/dh̃ = S̃_i.
    # typo fix: в TD (25) напечатано ũ_G — взято ũ_G² (ТЗ, «Опечатки»; TD (23), Barnea [9]).
    Значение < 1 — расслоённый поток устойчив.

    Параметры:
        froude: модифицированное число Фруда F.
        h_tilde: безразмерный уровень.

    Возвращает:
        Значение левой части, безразм.
    """
    geo = geometry(h_tilde)
    c2 = 1.0 - h_tilde
    return froude**2 * geo.u_g**2 * geo.s_i / (c2**2 * geo.a_g)


def jeffreys_wavy(k: float, h_tilde: float, *, sheltering: float = SHELTERING_COEFFICIENT) -> bool:
    """Критерий ветровых волн Джеффриса K ≥ 2/(ũ_G·√ũ_L·√s) (TD (30); Barnea [20]).

    Параметры:
        k: число K = F·√Re_LS (Barnea [21]).
        h_tilde: безразмерный уровень.
        sheltering: коэффициент экранирования s (0.01).

    Возвращает:
        True — расслоённый волновой режим (STRATIFIED_WAVY), иначе гладкий.
    """
    geo = geometry(h_tilde)
    threshold = 2.0 / (geo.u_g * math.sqrt(geo.u_l) * math.sqrt(sheltering))
    return k >= threshold
