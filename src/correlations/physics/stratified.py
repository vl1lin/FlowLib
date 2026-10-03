"""
Расслоённое течение в круглой трубе: геометрия, равновесный уровень жидкости и
критерии устойчивости (Taitel & Dukler, 1976).

Используется моделями Taitel–Dukler и Barnea. Источники: Taitel Y., Dukler A.E.
A model for predicting flow regime transitions in horizontal and near horizontal
gas-liquid flow // AIChE J. 1976. 22(1):47–55; Bratland O. Pipe Flow 2:
Multi-phase Flow Assurance, §11.2 (ур. 11.2.11–11.2.22).

Уравнения импульса фаз при расслоённом течении (x — вдоль трубы, θ — угол от
горизонтали, положительный для восходящего потока):

    жидкость: −A_L·dp/dx − τ_WL·S_L + τ_i·S_i − ρ_L·A_L·g·sin θ = 0
    газ:      −A_G·dp/dx − τ_WG·S_G − τ_i·S_i − ρ_G·A_G·g·sin θ = 0

Делим каждое уравнение на площадь своей фазы и вычитаем одно из другого
(исключается dp/dx). Получается уравнение равновесного уровня R(h̃) = 0:

    R = τ_WL·S_L/A_L − τ_WG·S_G/A_G − τ_i·S_i·(1/A_L + 1/A_G)
        + (ρ_L − ρ_G)·g·sin θ

Знак гравитационного члена — плюс при θ > 0 (восходящий поток): при ошибке знака
восходящий поток ведёт себя как нисходящий. Трение — Фаннинга (τ = f·ρ·u²/2),
граница раздела гладкая: f_i = f_G.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from scipy.optimize import brentq

from correlations.constants import GRAVITY
from correlations.physics.friction import (
    fanning_friction_taitel_dukler,
    fanning_friction_taitel_dukler_array,
)

_LEVEL_MIN: Final = 1e-4  # Нижняя граница поиска безразмерного уровня h̃
_LEVEL_MAX: Final = 1.0 - 1e-4  # Верхняя граница поиска h̃
_LEVEL_GRID_NODES: Final = 401  # Число узлов равномерной сетки по h̃ (≥ 400)
_LEVEL_GRID: Final = np.linspace(_LEVEL_MIN, _LEVEL_MAX, _LEVEL_GRID_NODES)
_LEVEL_XTOL: Final = 1e-10  # Допуск по h̃ при уточнении корня

_QUARTER: Final = 0.25  # Множитель ¼ в формулах площадей (ур. 11.2.20)
_HYDRAULIC_DIAMETER_FACTOR: Final = 4.0  # D = 4·A/S
_KH_EXPONENT: Final = 0.5  # Показатель степени в критерии Кельвина–Гельмгольца
_JEFFREYS_FACTOR: Final = 4.0  # Множитель в критерии Джеффриса (ур. 11.2.13)


@dataclass(frozen=True, slots=True)
class StratifiedGeometry:
    """
    Безразмерная геометрия расслоённого течения (площади отнесены к d²,
    периметры — к d).

    Attributes
    ----------
    level: float
        h̃ = h_L/d, безразмерный уровень жидкости
    liquid_area: float
        Ã_L, площадь сечения жидкости
    gas_area: float
        Ã_G, площадь сечения газа
    liquid_perimeter: float
        S̃_L, периметр, смоченный жидкостью
    gas_perimeter: float
        S̃_G, периметр, смоченный газом
    interface_width: float
        S̃_i, ширина границы раздела
    """

    level: float
    liquid_area: float
    gas_area: float
    liquid_perimeter: float
    gas_perimeter: float
    interface_width: float


@dataclass(frozen=True, slots=True)
class EquilibriumLevel:
    """
    Результат поиска равновесного уровня.

    Attributes
    ----------
    level: float | None
        Наименьший корень h̃; None, если расслоённое течение невозможно
    all_roots: tuple[float, ...]
        Все найденные корни (для диагностики), по возрастанию
    """

    level: float | None
    all_roots: tuple[float, ...]


def stratified_geometry(level: float) -> StratifiedGeometry:
    """
    Геометрия расслоённого течения (Taitel & Dukler, 1976), x = 2h̃ − 1:
    Ã_L = ¼·[π − arccos x + x·√(1 − x²)], Ã_G = ¼·[arccos x − x·√(1 − x²)],
    S̃_L = π − arccos x, S̃_G = arccos x, S̃_i = √(1 − x²).

    :param level: безразмерный уровень жидкости h̃ ∈ (0, 1)
    :raises ValueError: если уровень вне интервала (0, 1)
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"Уровень h̃ должен быть в (0, 1), получено {level}")
    x = 2.0 * level - 1.0
    angle = math.acos(x)
    width = math.sqrt(1.0 - x * x)
    return StratifiedGeometry(
        level=level,
        liquid_area=_QUARTER * (math.pi - angle + x * width),
        gas_area=_QUARTER * (angle - x * width),
        liquid_perimeter=math.pi - angle,
        gas_perimeter=angle,
        interface_width=width,
    )


def phase_velocities(
    *,
    geometry: StratifiedGeometry,
    superficial_liquid_velocity: float,
    superficial_gas_velocity: float,
) -> tuple[float, float]:
    """
    Истинные скорости фаз: u_L = v_SL·(π/4)/Ã_L, u_G = v_Sg·(π/4)/Ã_G, м/с.

    :param geometry: геометрия при данном уровне
    :param superficial_liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param superficial_gas_velocity: приведённая скорость газа v_Sg, м/с
    :return: (u_L, u_G)
    """
    pipe_area = math.pi / 4.0
    return (
        superficial_liquid_velocity * pipe_area / geometry.liquid_area,
        superficial_gas_velocity * pipe_area / geometry.gas_area,
    )


def liquid_hydraulic_diameter(
    *, geometry: StratifiedGeometry, diameter: float
) -> float:
    """
    Гидравлический диаметр жидкой фазы: D_L = 4·Ã_L/S̃_L·d, м.

    :param geometry: геометрия при данном уровне
    :param diameter: диаметр трубы d, м
    """
    return (
        _HYDRAULIC_DIAMETER_FACTOR
        * geometry.liquid_area
        / geometry.liquid_perimeter
        * diameter
    )


def liquid_friction_factor(
    *,
    geometry: StratifiedGeometry,
    diameter: float,
    liquid_velocity: float,
    liquid_density: float,
    liquid_viscosity: float,
) -> float:
    """
    Коэффициент трения Фаннинга жидкой фазы: f_L = f_TD(ρ_L·u_L·D_L/μ_L).

    :param geometry: геометрия при данном уровне
    :param diameter: диаметр трубы d, м
    :param liquid_velocity: истинная скорость жидкости u_L, м/с
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    """
    hydraulic_diameter = liquid_hydraulic_diameter(geometry=geometry, diameter=diameter)
    return fanning_friction_taitel_dukler(
        liquid_density * liquid_velocity * hydraulic_diameter / liquid_viscosity
    )


def kelvin_helmholtz_gas_velocity(
    *,
    geometry: StratifiedGeometry,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    angle_rad: float,
) -> float:
    """
    Критическая скорость газа по Кельвину–Гельмгольцу (ур. 11.2.11):
    u_G,crit = (1 − h̃)·[(ρ_L − ρ_G)·g·cos θ·A_G/(ρ_G·S_i)]^0.5, м/с.
    Расслоённое течение устойчиво при u_G < u_G,crit.

    :param geometry: геометрия при данном уровне
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param angle_rad: угол наклона от горизонтали θ, рад
    """
    gas_area = geometry.gas_area * diameter**2
    interface_width = geometry.interface_width * diameter
    return (1.0 - geometry.level) * (
        (liquid_density - gas_density)
        * GRAVITY
        * math.cos(angle_rad)
        * gas_area
        / (gas_density * interface_width)
    ) ** _KH_EXPONENT


def jeffreys_wavy_gas_velocity(
    *,
    liquid_viscosity: float,
    liquid_density: float,
    gas_density: float,
    liquid_velocity: float,
    angle_rad: float,
    sheltering_coefficient: float,
) -> float:
    """
    Скорость газа начала волнообразования по Джеффрису (ур. 11.2.13):
    u_G,wavy = [4·μ_L·(ρ_L − ρ_G)·g·cos θ/(s·ρ_L·ρ_G·u_L)]^0.5, м/с.
    При u_G ≥ u_G,wavy расслоённое течение волновое.

    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_velocity: истинная скорость жидкости u_L, м/с
    :param angle_rad: угол наклона от горизонтали θ, рад
    :param sheltering_coefficient: коэффициент экранирования s (0.01)
    """
    return (
        _JEFFREYS_FACTOR
        * liquid_viscosity
        * (liquid_density - gas_density)
        * GRAVITY
        * math.cos(angle_rad)
        / (sheltering_coefficient * liquid_density * gas_density * liquid_velocity)
    ) ** _KH_EXPONENT


def stratified_residual(
    *,
    level: float,
    superficial_liquid_velocity: float,
    superficial_gas_velocity: float,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    angle_rad: float,
) -> float:
    """
    Невязка уравнения равновесного уровня R(h̃) (ур. 11.2.20–11.2.22), Па/м:
    R = τ_WL·S_L/A_L − τ_WG·S_G/A_G − τ_i·S_i·(1/A_L + 1/A_G) + (ρ_L − ρ_G)·g·sin θ.
    Вывод — в docstring модуля.

    :param level: безразмерный уровень h̃ ∈ (0, 1)
    :param superficial_liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param superficial_gas_velocity: приведённая скорость газа v_Sg, м/с
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param gas_viscosity: вязкость газа μ_G, Па·с
    :param angle_rad: угол наклона от горизонтали θ, рад
    """
    geometry = stratified_geometry(level)
    liquid_velocity, gas_velocity = phase_velocities(
        geometry=geometry,
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
    )
    liquid_diameter = liquid_hydraulic_diameter(geometry=geometry, diameter=diameter)
    gas_diameter = (
        _HYDRAULIC_DIAMETER_FACTOR
        * geometry.gas_area
        / (geometry.gas_perimeter + geometry.interface_width)
        * diameter
    )
    liquid_friction = fanning_friction_taitel_dukler(
        liquid_density * liquid_velocity * liquid_diameter / liquid_viscosity
    )
    gas_friction = fanning_friction_taitel_dukler(
        gas_density * gas_velocity * gas_diameter / gas_viscosity
    )
    liquid_wall_stress = liquid_friction * liquid_density * liquid_velocity**2 / 2.0
    gas_wall_stress = gas_friction * gas_density * gas_velocity**2 / 2.0
    velocity_difference = gas_velocity - liquid_velocity
    interface_stress = (
        gas_friction
        * gas_density
        * velocity_difference
        * abs(velocity_difference)
        / 2.0
    )
    liquid_area = geometry.liquid_area * diameter**2
    gas_area = geometry.gas_area * diameter**2
    return (
        liquid_wall_stress * geometry.liquid_perimeter * diameter / liquid_area
        - gas_wall_stress * geometry.gas_perimeter * diameter / gas_area
        - interface_stress
        * geometry.interface_width
        * diameter
        * (1.0 / liquid_area + 1.0 / gas_area)
        + (liquid_density - gas_density) * GRAVITY * math.sin(angle_rad)
    )


@dataclass(frozen=True, slots=True)
class _GridGeometry:
    """Безразмерная геометрия на сетке уровней (не зависит от скоростей и флюидов)."""

    liquid_area: np.ndarray
    gas_area: np.ndarray
    liquid_perimeter: np.ndarray
    gas_perimeter: np.ndarray
    interface_width: np.ndarray
    liquid_diameter_factor: np.ndarray  # 4·Ã_L/S̃_L
    gas_diameter_factor: np.ndarray  # 4·Ã_G/(S̃_G + S̃_i)


def _build_grid_geometry(levels: np.ndarray) -> _GridGeometry:
    x = 2.0 * levels - 1.0
    angle = np.arccos(x)
    width = np.sqrt(1.0 - x * x)
    liquid_area = _QUARTER * (math.pi - angle + x * width)
    gas_area = _QUARTER * (angle - x * width)
    liquid_perimeter = math.pi - angle
    return _GridGeometry(
        liquid_area=liquid_area,
        gas_area=gas_area,
        liquid_perimeter=liquid_perimeter,
        gas_perimeter=angle,
        interface_width=width,
        liquid_diameter_factor=_HYDRAULIC_DIAMETER_FACTOR
        * liquid_area
        / liquid_perimeter,
        gas_diameter_factor=_HYDRAULIC_DIAMETER_FACTOR * gas_area / (angle + width),
    )


_GRID_GEOMETRY: Final = _build_grid_geometry(_LEVEL_GRID)


def _stratified_residual_grid(
    *,
    superficial_liquid_velocity: float,
    superficial_gas_velocity: float,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    angle_rad: float,
) -> np.ndarray:
    """Векторная версия `stratified_residual` на сетке уровней `_LEVEL_GRID`."""
    geometry = _GRID_GEOMETRY
    pipe_area = math.pi / 4.0
    liquid_velocity = superficial_liquid_velocity * pipe_area / geometry.liquid_area
    gas_velocity = superficial_gas_velocity * pipe_area / geometry.gas_area
    liquid_friction = fanning_friction_taitel_dukler_array(
        liquid_density
        * liquid_velocity
        * geometry.liquid_diameter_factor
        * (diameter / liquid_viscosity)
    )
    gas_friction = fanning_friction_taitel_dukler_array(
        gas_density
        * gas_velocity
        * geometry.gas_diameter_factor
        * (diameter / gas_viscosity)
    )
    liquid_wall_stress = liquid_friction * liquid_density * liquid_velocity**2 / 2.0
    gas_wall_stress = gas_friction * gas_density * gas_velocity**2 / 2.0
    velocity_difference = gas_velocity - liquid_velocity
    interface_stress = (
        gas_friction
        * gas_density
        * velocity_difference
        * np.abs(velocity_difference)
        / 2.0
    )
    # S/A с размерностями: S̃·d/(Ã·d²) = S̃/(Ã·d)
    return (
        liquid_wall_stress
        * geometry.liquid_perimeter
        / (geometry.liquid_area * diameter)
        - gas_wall_stress * geometry.gas_perimeter / (geometry.gas_area * diameter)
        - interface_stress
        * geometry.interface_width
        * (1.0 / geometry.liquid_area + 1.0 / geometry.gas_area)
        / diameter
        + (liquid_density - gas_density) * GRAVITY * math.sin(angle_rad)
    )


def solve_equilibrium_level(
    *,
    superficial_liquid_velocity: float,
    superficial_gas_velocity: float,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    angle_rad: float,
    refine_all_roots: bool = True,
) -> EquilibriumLevel:
    """
    Равновесный уровень жидкости при расслоённом течении (Taitel & Dukler, 1976;
    ур. 11.2.20–11.2.22).

    Смены знака R(h̃) ищутся на равномерной сетке h̃ ∈ [1e-4, 1 − 1e-4] (401 узел),
    каждый корень уточняется методом Брента до 1e-10. Физическое решение —
    наименьший корень (при нескольких корнях устойчив наименьший; Barnea & Taitel).
    Если корней нет, расслоённое течение невозможно и level = None.

    :param superficial_liquid_velocity: приведённая скорость жидкости v_SL, м/с, > 0
    :param superficial_gas_velocity: приведённая скорость газа v_Sg, м/с, > 0
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param gas_viscosity: вязкость газа μ_G, Па·с
    :param angle_rad: угол наклона от горизонтали θ, рад (положительный вверх)
    :param refine_all_roots: уточнять все корни (True) или только наименьший (False);
        при False `all_roots` содержит один наименьший корень. Ускоряет расчёт,
        когда нужен только `level`
    :return: наименьший корень и все найденные корни
    """
    kwargs = {
        "superficial_liquid_velocity": superficial_liquid_velocity,
        "superficial_gas_velocity": superficial_gas_velocity,
        "diameter": diameter,
        "liquid_density": liquid_density,
        "gas_density": gas_density,
        "liquid_viscosity": liquid_viscosity,
        "gas_viscosity": gas_viscosity,
        "angle_rad": angle_rad,
    }
    values = _stratified_residual_grid(**kwargs)
    sign_changes = np.nonzero(values[:-1] * values[1:] <= 0.0)[0]
    if sign_changes.size == 0:
        return EquilibriumLevel(level=None, all_roots=())

    def residual(level: float) -> float:
        return stratified_residual(level=level, **kwargs)

    indices = sign_changes if refine_all_roots else sign_changes[:1]
    roots = tuple(
        _refine_root(
            residual,
            float(_LEVEL_GRID[index]),
            float(_LEVEL_GRID[index + 1]),
            lower_value=float(values[index]),
            upper_value=float(values[index + 1]),
        )
        for index in indices
    )
    return EquilibriumLevel(level=roots[0], all_roots=roots)


def least_residual_level(
    *,
    superficial_liquid_velocity: float,
    superficial_gas_velocity: float,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    angle_rad: float,
) -> float:
    """
    Уровень h̃ на сетке поиска, при котором |R(h̃)| минимален. Нужен для оценки
    геометрии, когда равновесного уровня нет (`solve_equilibrium_level` вернул None).

    Параметры — как у `solve_equilibrium_level`.
    """
    values = _stratified_residual_grid(
        superficial_liquid_velocity=superficial_liquid_velocity,
        superficial_gas_velocity=superficial_gas_velocity,
        diameter=diameter,
        liquid_density=liquid_density,
        gas_density=gas_density,
        liquid_viscosity=liquid_viscosity,
        gas_viscosity=gas_viscosity,
        angle_rad=angle_rad,
    )
    return float(_LEVEL_GRID[int(np.argmin(np.abs(values)))])


def _refine_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    lower_value: float,
    upper_value: float,
) -> float:
    """
    Корень на отрезке со сменой знака методом Брента (допуск _LEVEL_XTOL);
    значения функции на концах уже известны из сканирования сетки.
    """
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    # full_output=False (по умолчанию) возвращает float; стабы scipy это не различают
    return cast(float, brentq(function, lower, upper, xtol=_LEVEL_XTOL))
