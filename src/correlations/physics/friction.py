import math
from typing import Final

import numpy as np

_LAMINAR_MAX_REYNOLDS: Final = 2000.0  # Граница ламинарного режима течения в трубе
_LAMINAR_COEFFICIENT: Final = 64.0  # f = 64 / Re (Дарси–Вейсбах)

# Коэффициенты явной аппроксимации Брькича (2011) уравнения Колбрука
_BRKIC_INNER_COEFFICIENT: Final = 1.816
_BRKIC_LOG_COEFFICIENT: Final = 1.1
_COLEBROOK_ROUGHNESS_DIVISOR: Final = 3.71
_COLEBROOK_SLOPE: Final = 2.0

# Коэффициент трения Фаннинга по Taitel–Dukler (гладкая труба)
_TAITEL_DUKLER_LAMINAR_MAX_REYNOLDS: Final = 2300.0
_TAITEL_DUKLER_LAMINAR_COEFFICIENT: Final = 16.0  # f = 16 / Re
_TAITEL_DUKLER_TURBULENT_COEFFICIENT: Final = 0.046  # f = 0.046·Re^(-0.2)
_TAITEL_DUKLER_TURBULENT_EXPONENT: Final = -0.2


def darcy_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """
    Коэффициент трения Дарси–Вейсбаха (диаграмма Муди).

    Ламинарный режим (Re < 2000): f = 64 / Re.
    Турбулентный режим: явная аппроксимация Брькича уравнения Колбрука.

    :param reynolds: число Рейнольдса, безразмерное, должно быть > 0
    :param relative_roughness: относительная шероховатость ε/d, безразмерная
    :return: коэффициент трения Дарси, безразмерный
    :raises ValueError: если число Рейнольдса не положительное
    """
    if not reynolds > 0.0:
        raise ValueError(f"Число Рейнольдса должно быть > 0, получено {reynolds}")
    if reynolds < _LAMINAR_MAX_REYNOLDS:
        return _LAMINAR_COEFFICIENT / reynolds

    inner = _BRKIC_LOG_COEFFICIENT * reynolds
    s = math.log(
        reynolds / (_BRKIC_INNER_COEFFICIENT * math.log(inner / math.log(1.0 + inner)))
    )
    root_term = -_COLEBROOK_SLOPE * math.log10(
        relative_roughness / _COLEBROOK_ROUGHNESS_DIVISOR
        + _COLEBROOK_SLOPE * s / reynolds
    )
    return 1.0 / root_term**2


def fanning_friction_taitel_dukler(reynolds: float) -> float:
    """
    Коэффициент трения Фаннинга по Taitel–Dukler для гладкой трубы:
    f = 16/Re при Re < 2300, иначе f = 0.046·Re^(−0.2).

    Коэффициент Фаннинга в 4 раза меньше коэффициента Дарси: τ_W = f·ρ·v²/2.

    :param reynolds: число Рейнольдса, безразмерное, должно быть > 0
    :return: коэффициент трения Фаннинга, безразмерный
    :raises ValueError: если число Рейнольдса не положительное
    """
    if not reynolds > 0.0:
        raise ValueError(f"Число Рейнольдса должно быть > 0, получено {reynolds}")
    if reynolds < _TAITEL_DUKLER_LAMINAR_MAX_REYNOLDS:
        return _TAITEL_DUKLER_LAMINAR_COEFFICIENT / reynolds
    return (
        _TAITEL_DUKLER_TURBULENT_COEFFICIENT
        * reynolds**_TAITEL_DUKLER_TURBULENT_EXPONENT
    )


def fanning_friction_taitel_dukler_array(reynolds: np.ndarray) -> np.ndarray:
    """
    Векторная версия `fanning_friction_taitel_dukler` (для сканирования по сетке).

    :param reynolds: массив чисел Рейнольдса, все элементы должны быть > 0
    :return: массив коэффициентов трения Фаннинга, безразмерный
    :raises ValueError: если есть неположительные числа Рейнольдса
    """
    if not np.all(reynolds > 0.0):
        raise ValueError("Все числа Рейнольдса должны быть > 0")
    return np.where(
        reynolds < _TAITEL_DUKLER_LAMINAR_MAX_REYNOLDS,
        _TAITEL_DUKLER_LAMINAR_COEFFICIENT / reynolds,
        _TAITEL_DUKLER_TURBULENT_COEFFICIENT
        * reynolds**_TAITEL_DUKLER_TURBULENT_EXPONENT,
    )
