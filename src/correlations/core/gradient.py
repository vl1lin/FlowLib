"""Сборка градиента давления: гравитация, трение, ускорение.

Схема: −dp/dL = (гравитация + трение)/(1 − Ek); составляющая ускорения =
−dp/dL − (гравитация + трение). Здесь Ek — безразмерное кинетическое число
(BB 1973 ур. 1: Ek = ρ_s·v_m·v_sg/p; HK 1988 ур. 33: ρ_c·v_g²/p).

Коэффициент трения во всех функциях — Darcy (Moody); Fanning переводить через
`correlations.core.friction.fanning_to_darcy`.
"""

import math
from dataclasses import dataclass

from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import ConvergenceError


@dataclass(frozen=True, slots=True)
class GradientComponents:
    """Составляющие градиента давления, Па/м (положительный — давление падает по потоку).

    Атрибуты:
        dpdl_pa_m: полный градиент −dp/dL.
        gravity_pa_m: гравитационная составляющая.
        friction_pa_m: составляющая трения.
        acceleration_pa_m: составляющая ускорения (dpdl − gravity − friction).
    """

    dpdl_pa_m: float
    gravity_pa_m: float
    friction_pa_m: float
    acceleration_pa_m: float


def friction_gradient(f_darcy: float, rho_kg_m3: float, v_m_s: float, d_m: float) -> float:
    """Градиент трения dp/dL = f_D·ρ·v²/(2·d) (Darcy–Weisbach).

    Параметры:
        f_darcy: коэффициент трения Darcy, безразм.
        rho_kg_m3: плотность, кг/м³.
        v_m_s: скорость, м/с.
        d_m: диаметр, м.

    Возвращает:
        Градиент давления на трение, Па/м (≥ 0).
    """
    return f_darcy * rho_kg_m3 * v_m_s**2 / (2.0 * d_m)


def gravity_gradient(rho_kg_m3: float, theta_rad: float) -> float:
    """Гравитационный градиент ρ·g·sinθ.

    Параметры:
        rho_kg_m3: плотность, кг/м³.
        theta_rad: угол от горизонтали, рад (> 0 — вверх).

    Возвращает:
        Градиент давления на преодоление веса, Па/м.
    """
    return rho_kg_m3 * GRAVITY_M_S2 * math.sin(theta_rad)


def kinetic_number(rho_kg_m3: float, v_m_s: float, v_g_m_s: float, p_pa: float) -> float:
    """Кинетическое число Ek = ρ·v_m·v_g/p.

    Для BB (ур. 1) и MB: ρ = ρ_s, v_g = v_sg; для HK кольцевого (ур. 33): ρ = ρ_c,
    v_m = v_g = v_g (газ ядра).

    Параметры:
        rho_kg_m3: плотность, кг/м³.
        v_m_s: скорость смеси (или ядра), м/с.
        v_g_m_s: скорость газа, м/с.
        p_pa: давление, Па.

    Возвращает:
        Ek, безразм.
    """
    return rho_kg_m3 * v_m_s * v_g_m_s / p_pa


def with_kinetic_term(gravity_pa_m: float, friction_pa_m: float, ek: float) -> GradientComponents:
    """Добавляет ускорение: −dp/dL = (гравитация + трение)/(1 − Ek).

    Параметры:
        gravity_pa_m: гравитационная составляющая, Па/м.
        friction_pa_m: составляющая трения, Па/м.
        ek: кинетическое число (< 1).

    Возвращает:
        `GradientComponents`; ускорение = полный градиент − (гравитация + трение).

    Исключения:
        ConvergenceError: Ek ≥ 1 (особенность знаменателя).
    """
    if ek >= 1.0:
        raise ConvergenceError(
            "кинетическое число Ek ≥ 1: знаменатель 1 − Ek не положителен",
            {"ek": ek, "gravity_pa_m": gravity_pa_m, "friction_pa_m": friction_pa_m},
        )
    base = gravity_pa_m + friction_pa_m
    total = base / (1.0 - ek)
    return GradientComponents(
        dpdl_pa_m=total,
        gravity_pa_m=gravity_pa_m,
        friction_pa_m=friction_pa_m,
        acceleration_pa_m=total - base,
    )
