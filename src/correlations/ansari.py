"""
Механистическая модель Ансари (Ansari et al., 1994) для восходящего течения.

Первоисточник формул: Брилл Дж. П., Мукерджи Х. «Многофазный поток в скважинах»,
раздел 4.2.2 «Метод Анзари и др.», ур. 4.158–4.172 и 4.204–4.235.
Номера уравнений в комментариях и docstring — по этой книге.

Классификация выполняется прямой проверкой критериев в точке (v_SL, v_Sg)
в порядке: дисперсно-пузырьковый → кольцевой → пузырьковый → пробковый.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Final, Literal, cast

import numpy as np
from scipy.optimize import brentq

from correlations.base import (
    ClassificationResult,
    FlowPatternCode,
    IFlowModel,
    check_velocities,
    single_phase_pattern,
)
from correlations.constants import GRAVITY
from correlations.domain.params import FluidParams, PipeParams
from correlations.physics.friction import darcy_friction_factor

# Дисперсно-пузырьковый режим, ур. 4.161
_DISPERSION_BUBBLE_FACTOR: Final = 2.0
_DISPERSION_SURFACE_COEFFICIENT: Final = 0.4
_DISPERSION_DENSITY_EXPONENT: Final = 0.6
_DISPERSION_FRICTION_EXPONENT: Final = 0.4
_DISPERSION_VELOCITY_EXPONENT: Final = 1.2
_DISPERSION_RHS_BASE: Final = 0.725
_DISPERSION_RHS_GAS_COEFFICIENT: Final = 4.15

# Критерий Тёрнера, ур. 4.163
_TURNER_COEFFICIENT: Final = 3.1

# Модель плёнки, ур. 4.206–4.227
_CRITICAL_VELOCITY_FACTOR: Final = 1e4  # ур. 4.209
_ENTRAINMENT_SLOPE: Final = 0.125  # ур. 4.208
_ENTRAINMENT_CRITICAL_OFFSET: Final = 1.5  # ур. 4.208
_MAX_ENTRAINMENT_FRACTION: Final = 0.999  # F_E < 1, иначе X_M² вырождается в 0
_HIGH_ENTRAINMENT_THRESHOLD: Final = 0.9  # ур. 4.221/4.222
_INTERFACIAL_ROUGHNESS_HIGH_ENTRAINMENT: Final = 300.0  # ур. 4.221
_INTERFACIAL_ROUGHNESS_COEFFICIENT: Final = 24.0  # ур. 4.222
_INTERFACIAL_ROUGHNESS_DENSITY_EXPONENT: Final = 1.0 / 3.0  # ур. 4.222
_FILM_HOLDUP_FACTOR: Final = 4.0  # H = 4δ(1 − δ)
_FILM_HOLDUP_EXPONENT: Final = 2.5  # ур. 4.233
_FILM_DELTA_MIN: Final = 1e-6  # Нижняя граница поиска δ
_FILM_DELTA_MAX: Final = 0.4999  # Верхняя граница поиска δ (δ < 0.5)
_FILM_GRID_NODES: Final = 200
_FILM_DELTA_GRID: Final = tuple(
    np.geomspace(_FILM_DELTA_MIN, _FILM_DELTA_MAX, _FILM_GRID_NODES).tolist()
)
_FILM_DELTA_XTOL: Final = 1e-10  # Допуск по δ при уточнении корня

# Неустойчивость плёнки, ур. 4.169
_INSTABILITY_HOLDUP_LINEAR: Final = 1.5
_INSTABILITY_NUMERATOR: Final = 2.0

# Пузырьковый режим, ур. 4.158–4.160
_HARMATHY_COEFFICIENT: Final = 1.53  # ур. 4.160
_BUBBLE_MIN_DIAMETER_COEFFICIENT: Final = 19.01  # ур. 4.158
_BUBBLE_RISE_FACTOR: Final = 0.25  # ур. 4.159
_BUBBLE_LIQUID_DIVISOR: Final = 3.0  # ур. 4.159


class TransitionReason(Enum):
    """
    Причина, по которой получен режим течения.

    Для BUBBLE и SLUG в `reason` указывается причина отказа в кольцевом режиме
    (BELOW_TURNER, NO_FILM_SOLUTION, BLOCKAGE, FILM_INSTABILITY): сам режим
    уже записан в `AnsariResult.pattern`.
    """

    SINGLE_PHASE = auto()
    DISPERSED_BUBBLE = auto()  # выполнен критерий 4.161 и λ_G ≤ порога
    BELOW_TURNER = auto()  # v_Sg < v_Sg,T
    NO_FILM_SOLUTION = auto()
    BLOCKAGE = auto()  # 4.165
    FILM_INSTABILITY = auto()  # 4.169
    ANNULAR = auto()
    BUBBLE = auto()
    SLUG = auto()


@dataclass(frozen=True, slots=True)
class AnsariSettings:
    """
    Настройки модели Ансари.

    Attributes
    ----------
    dispersed_max_gas_fraction: float
        Максимальная расходная доля газа λ_G для дисперсно-пузырькового режима
        (ур. 4.162, Scott & Kouba, ≡ v_Sg ≤ 3.17·v_SL). 0.76 верно именно для Ансари;
        0.52 — значение Taitel/Barnea и Хасан–Кабир.
    blockage_threshold: float
        Порог перекрытия сечения H_LF + λ_LC(1 − 2δ)² (ур. 4.165)
    bubble_min_angle_deg: float
        Минимальный угол от горизонтали, при котором возможен пузырьковый режим.
        70° как в VBA-порте; в литературе (Barnea; Shoham, 2006) встречается 60°
    turner_uses_inclination: bool
        Умножать g на sin θ в критерии Тёрнера (расширение Barnea на наклон).
        В книге (ур. 4.163) sin θ нет; при 75–90° разница меньше 1%
    film_friction_ratio: float
        Отношение f_F/f_SL в X_M² (ур. 4.167); в книге принято равным 1
    annular_criteria: "full" | "turner_only"
        "full" — критерии Тёрнера и Barnea; "turner_only" — только Тёрнер
        (для сравнения с `AnsariVBAModel`)
    """

    dispersed_max_gas_fraction: float = 0.76
    blockage_threshold: float = 0.12
    bubble_min_angle_deg: float = 70.0
    turner_uses_inclination: bool = True
    film_friction_ratio: float = 1.0
    annular_criteria: Literal["full", "turner_only"] = "full"

    def __post_init__(self) -> None:
        if not 0.0 < self.dispersed_max_gas_fraction <= 1.0:
            raise ValueError("dispersed_max_gas_fraction должен быть в (0, 1]")
        # При H_LF ≥ 2/3 знаменатель критерия 4.169 теряет смысл
        if not 0.0 < self.blockage_threshold < 2.0 / 3.0:
            raise ValueError("blockage_threshold должен быть в (0, 2/3)")
        if not self.film_friction_ratio > 0.0:
            raise ValueError("film_friction_ratio должен быть > 0")
        if self.annular_criteria not in ("full", "turner_only"):
            raise ValueError("annular_criteria должен быть 'full' или 'turner_only'")


@dataclass(frozen=True, slots=True)
class FilmState:
    """
    Состояние плёнки; заполняется, только если дошли до расчёта плёнки.

    Attributes
    ----------
    entrainment_fraction: float
        Доля жидкости, унесённой в ядро, F_E (ур. 4.208)
    core_liquid_fraction: float
        Расходная доля жидкости в ядре λ_LC (ур. 4.207)
    core_density: float
        Плотность ядра ρ_C, кг/м³ (ур. 4.206)
    x_m_squared: float
        Параметр Мартинелли X_M² (ур. 4.167)
    y_m: float
        Параметр наклона Y_M (ур. 4.168)
    delta: float | None
        Безразмерная толщина плёнки δ; None при NO_FILM_SOLUTION
    film_holdup: float | None
        Доля сечения под плёнкой H_LF = 4δ(1 − δ)
    blockage_value: float | None
        H_LF + λ_LC(1 − 2δ)² (левая часть ур. 4.165)
    """

    entrainment_fraction: float
    core_liquid_fraction: float
    core_density: float
    x_m_squared: float
    y_m: float
    delta: float | None
    film_holdup: float | None
    blockage_value: float | None


@dataclass(frozen=True, slots=True)
class AnsariResult(ClassificationResult):
    """Результат классификации: режим, причина и (если считалась) плёнка."""

    reason: TransitionReason
    film: FilmState | None = None


def dispersion_prefactor(
    *, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """
    Множитель левой части ур. 4.161, не зависящий от скоростей:
    2·[0.4σ/(Δρ·g)]^0.5 · (ρ_L/σ)^0.6.

    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    """
    density_difference = liquid_density - gas_density
    return (
        _DISPERSION_BUBBLE_FACTOR
        * (
            _DISPERSION_SURFACE_COEFFICIENT
            * surface_tension
            / (density_difference * GRAVITY)
        )
        ** 0.5
        * (liquid_density / surface_tension) ** _DISPERSION_DENSITY_EXPONENT
    )


def dispersion_terms(
    *,
    liquid_velocity: float,
    gas_velocity: float,
    diameter: float,
    relative_roughness: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    prefactor: float,
) -> tuple[float, float]:
    """
    Левая и правая части критерия дробления пузырей (Barnea, ур. 4.161):
    LHS = prefactor · (f_n/(2d))^0.4 · v_m^1.2, RHS = 0.725 + 4.15·λ_G^0.5.

    f_n — коэффициент трения Дарси по no-slip числу Рейнольдса
    Re_n = ρ_n·v_m·d/μ_n. Множитель (f_n/(2d))^0.4 с f Дарси эквивалентен
    (2f_F/d)^0.4 с f Фаннинга в оригинале Barnea.

    :param liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param gas_velocity: приведённая скорость газа v_Sg, м/с
    :param diameter: диаметр трубы d, м
    :param relative_roughness: относительная шероховатость ε/d
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param gas_viscosity: вязкость газа μ_G, Па·с
    :param prefactor: результат `dispersion_prefactor`
    :return: (LHS, RHS)
    """
    mixture_velocity = liquid_velocity + gas_velocity
    gas_fraction = gas_velocity / mixture_velocity
    liquid_fraction = 1.0 - gas_fraction
    no_slip_density = liquid_density * liquid_fraction + gas_density * gas_fraction
    no_slip_viscosity = (
        liquid_viscosity * liquid_fraction + gas_viscosity * gas_fraction
    )
    reynolds = no_slip_density * mixture_velocity * diameter / no_slip_viscosity
    friction = darcy_friction_factor(reynolds, relative_roughness)

    left_side = (
        prefactor
        * (friction / (2.0 * diameter)) ** _DISPERSION_FRICTION_EXPONENT
        * mixture_velocity**_DISPERSION_VELOCITY_EXPONENT
    )
    right_side = _DISPERSION_RHS_BASE + _DISPERSION_RHS_GAS_COEFFICIENT * math.sqrt(
        gas_fraction
    )
    return left_side, right_side


def is_dispersed_bubble(
    *,
    liquid_velocity: float,
    gas_velocity: float,
    diameter: float,
    relative_roughness: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    prefactor: float,
    max_gas_fraction: float,
) -> bool:
    """
    Дисперсно-пузырьковый режим (Barnea, ур. 4.161–4.162): турбулентность дробит
    пузыри (LHS ≥ RHS, см. `dispersion_terms`) и пузыри не слипаются
    (λ_G ≤ max_gas_fraction).

    Параметры — как у `dispersion_terms`, плюс:
    :param max_gas_fraction: порог расходной доли газа λ_G (ур. 4.162)
    """
    gas_fraction = gas_velocity / (liquid_velocity + gas_velocity)
    if gas_fraction > max_gas_fraction:
        return False
    left_side, right_side = dispersion_terms(
        liquid_velocity=liquid_velocity,
        gas_velocity=gas_velocity,
        diameter=diameter,
        relative_roughness=relative_roughness,
        liquid_density=liquid_density,
        gas_density=gas_density,
        liquid_viscosity=liquid_viscosity,
        gas_viscosity=gas_viscosity,
        prefactor=prefactor,
    )
    return left_side >= right_side


def turner_velocity(
    *,
    liquid_density: float,
    gas_density: float,
    surface_tension: float,
    inclination_factor: float = 1.0,
) -> float:
    """
    Минимальная скорость газа для выноса капель, критерий Тёрнера (ур. 4.163):
    v_Sg,T = 3.1·[g·σ·Δρ/ρ_G²]^0.25, м/с.

    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    :param inclination_factor: множитель к g (sin θ или 1.0, если наклон не учитывается)
    """
    return (
        _TURNER_COEFFICIENT
        * (
            GRAVITY
            * inclination_factor
            * surface_tension
            * (liquid_density - gas_density)
            / gas_density**2
        )
        ** 0.25
    )


def _find_first_root(
    function: Callable[[float], float], grid: Sequence[float]
) -> float | None:
    """
    Наименьший корень: первая смена знака на сетке, затем уточнение методом Брента.

    :param function: функция одного аргумента
    :param grid: возрастающая последовательность узлов
    :return: корень или None, если смены знака на сетке нет
    """
    previous_node = grid[0]
    previous_value = function(previous_node)
    for node in grid[1:]:
        value = function(node)
        if previous_value * value <= 0.0:
            # full_output=False (по умолчанию) возвращает float; стабы scipy это не различают
            return cast(
                float, brentq(function, previous_node, node, xtol=_FILM_DELTA_XTOL)
            )
        previous_node, previous_value = node, value
    return None


def solve_film(
    *,
    liquid_velocity: float,
    gas_velocity: float,
    diameter: float,
    relative_roughness: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    surface_tension: float,
    sin_angle: float,
    friction_ratio: float,
) -> FilmState:
    """
    Модель кольцевого течения: находит безразмерную толщину плёнки δ из ур. 4.233
    F(δ) = Y_M − Z(δ)/(H·(1 − H)^2.5) + X_M²/H³ = 0, H = 4δ(1 − δ).

    Берётся наименьший корень на δ ∈ (0, 0.5): при δ → 0 F → +∞, поэтому это
    самая тонкая плёнка. Метод Ньютона из книги (4.234–4.235) не используется:
    он чувствителен к начальному приближению и может сойтись к «толстому» корню.
    Если смены знака нет, плёнка не удерживается: delta, film_holdup и
    blockage_value равны None.

    :param liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param gas_velocity: приведённая скорость газа v_Sg, м/с
    :param diameter: диаметр трубы d, м
    :param relative_roughness: относительная шероховатость ε/d
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param gas_viscosity: вязкость газа μ_G, Па·с
    :param surface_tension: поверхностное натяжение σ, Н/м
    :param sin_angle: синус угла наклона от горизонтали
    :param friction_ratio: отношение f_F/f_SL в X_M² (ур. 4.167)
    """
    critical_velocity = (
        _CRITICAL_VELOCITY_FACTOR
        * gas_velocity
        * gas_viscosity
        / surface_tension
        * math.sqrt(gas_density / liquid_density)
    )  # ур. 4.209
    entrainment_fraction = min(
        max(
            1.0
            - math.exp(
                -_ENTRAINMENT_SLOPE * (critical_velocity - _ENTRAINMENT_CRITICAL_OFFSET)
            ),
            0.0,
        ),
        _MAX_ENTRAINMENT_FRACTION,
    )  # ур. 4.208

    core_velocity = entrainment_fraction * liquid_velocity + gas_velocity  # ур. 4.226
    core_liquid_fraction = entrainment_fraction * liquid_velocity / core_velocity
    core_density = liquid_density * core_liquid_fraction + gas_density * (
        1.0 - core_liquid_fraction
    )  # ур. 4.206
    core_viscosity = liquid_viscosity * core_liquid_fraction + gas_viscosity * (
        1.0 - core_liquid_fraction
    )  # ур. 4.227

    liquid_friction = darcy_friction_factor(
        liquid_density * liquid_velocity * diameter / liquid_viscosity,
        relative_roughness,
    )  # ур. 4.217
    core_friction = darcy_friction_factor(
        core_density * core_velocity * diameter / core_viscosity,
        relative_roughness,
    )  # ур. 4.225
    liquid_gradient = (
        liquid_friction * liquid_density * liquid_velocity**2 / (2.0 * diameter)
    )  # ур. 4.216
    core_gradient = (
        core_friction * core_density * core_velocity**2 / (2.0 * diameter)
    )  # ур. 4.224

    x_m_squared = (
        (1.0 - entrainment_fraction) ** 2
        * friction_ratio
        * liquid_gradient
        / core_gradient
    )  # ур. 4.167
    y_m = GRAVITY * sin_angle * (liquid_density - core_density) / core_gradient  # 4.168

    if entrainment_fraction > _HIGH_ENTRAINMENT_THRESHOLD:
        roughness_slope = _INTERFACIAL_ROUGHNESS_HIGH_ENTRAINMENT  # ур. 4.221
    else:
        roughness_slope = (
            _INTERFACIAL_ROUGHNESS_COEFFICIENT
            * (liquid_density / gas_density) ** _INTERFACIAL_ROUGHNESS_DENSITY_EXPONENT
        )  # ур. 4.222

    def film_holdup_of(delta: float) -> float:
        return _FILM_HOLDUP_FACTOR * delta * (1.0 - delta)

    def film_equation(delta: float) -> float:
        holdup = film_holdup_of(delta)
        return (
            y_m
            - (1.0 + roughness_slope * delta)
            / (holdup * (1.0 - holdup) ** _FILM_HOLDUP_EXPONENT)
            + x_m_squared / holdup**3
        )  # ур. 4.233

    delta = _find_first_root(film_equation, _FILM_DELTA_GRID)
    if delta is None:
        return FilmState(
            entrainment_fraction=entrainment_fraction,
            core_liquid_fraction=core_liquid_fraction,
            core_density=core_density,
            x_m_squared=x_m_squared,
            y_m=y_m,
            delta=None,
            film_holdup=None,
            blockage_value=None,
        )

    film_holdup = film_holdup_of(delta)
    return FilmState(
        entrainment_fraction=entrainment_fraction,
        core_liquid_fraction=core_liquid_fraction,
        core_density=core_density,
        x_m_squared=x_m_squared,
        y_m=y_m,
        delta=delta,
        film_holdup=film_holdup,
        blockage_value=film_holdup + core_liquid_fraction * (1.0 - 2.0 * delta) ** 2,
    )


def is_film_unstable(*, film: FilmState) -> bool:
    """
    Неустойчивость плёнки (ур. 4.169):
    Y_M > (2 − 1.5·H_LF) / (H_LF³·(1 − 1.5·H_LF)) · X_M².

    Достаточно сравнения при фактическом H_LF, поиск δ_min методом Ньютона
    (ур. 4.170–4.172) не нужен: после проверки перекрытия (4.165) H_LF ≤ 0.12,
    а на этом интервале правая часть 4.169 монотонно убывает по H. Поэтому
    «δ_min > δ» (формулировка книги) ⇔ «Y_M ≤ RHS(H_LF)».
    Проверять только после `is_film_blocking`.

    :param film: состояние плёнки с найденным δ
    :raises ValueError: если плёнка не найдена (delta is None)
    """
    holdup = film.film_holdup
    if holdup is None:
        raise ValueError("Критерий неустойчивости требует найденной плёнки")
    right_side = (
        (_INSTABILITY_NUMERATOR - _INSTABILITY_HOLDUP_LINEAR * holdup)
        / (holdup**3 * (1.0 - _INSTABILITY_HOLDUP_LINEAR * holdup))
        * film.x_m_squared
    )
    return film.y_m > right_side


def is_film_blocking(*, film: FilmState, threshold: float) -> bool:
    """
    Перекрытие сечения (ур. 4.165): H_LF + λ_LC·(1 − 2δ)² > threshold.

    :param film: состояние плёнки с найденным δ
    :param threshold: порог (0.12 по Ансари)
    :raises ValueError: если плёнка не найдена (delta is None)
    """
    if film.blockage_value is None:
        raise ValueError("Критерий перекрытия требует найденной плёнки")
    return film.blockage_value > threshold


def slip_velocity(
    *, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """
    Скорость подъёма пузыря по Хармати (ур. 4.160):
    v_s = 1.53·[g·σ·Δρ/ρ_L²]^0.25, м/с.
    """
    return (
        _HARMATHY_COEFFICIENT
        * (
            GRAVITY
            * surface_tension
            * (liquid_density - gas_density)
            / liquid_density**2
        )
        ** 0.25
    )


def minimum_bubble_diameter(
    *, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """
    Минимальный диаметр трубы, при котором возможен пузырьковый режим (ур. 4.158):
    d_min = 19.01·[Δρ·σ/(ρ_L²·g)]^0.5, м.
    """
    return _BUBBLE_MIN_DIAMETER_COEFFICIENT * math.sqrt(
        (liquid_density - gas_density) * surface_tension / (liquid_density**2 * GRAVITY)
    )


def bubble_slug_velocity(
    *, liquid_velocity: float, slip_velocity: float, sin_angle: float
) -> float:
    """
    Граница пузырьковый/пробковый режимы (ур. 4.159):
    v_Sg,B/S = 0.25·v_s·sin θ + v_SL/3, м/с.

    :param liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param slip_velocity: скорость подъёма пузыря v_s, м/с
    :param sin_angle: синус угла наклона от горизонтали
    """
    return (
        _BUBBLE_RISE_FACTOR * slip_velocity * sin_angle
        + liquid_velocity / _BUBBLE_LIQUID_DIVISOR
    )


@dataclass(frozen=True, slots=True)
class _AnnularRejection:
    """Причина отказа в кольцевом режиме и состояние плёнки (если считалась)."""

    reason: TransitionReason
    film: FilmState | None = None


@dataclass(frozen=True, slots=True)
class _Invariants:
    """Величины, не зависящие от (v_SL, v_Sg)."""

    relative_roughness: float
    sin_angle: float
    dispersion_prefactor: float
    turner_velocity: float
    slip_velocity: float
    bubble_possible: bool


class AnsariModel(IFlowModel):
    """
    Модель Ансари для расчёта режима течения в восходящем потоке (75°–90°).

    Классификация — прямая проверка критериев в точке (v_SL, v_Sg):

    1. дисперсно-пузырьковый режим (Barnea, ур. 4.161–4.162);
    2. кольцевой режим: критерий Тёрнера (4.163), затем перекрытие сечения
       плёнкой (4.165) и её устойчивость (4.169);
    3. пузырьковый режим, если он возможен по d и θ (4.158) и
       v_Sg < v_Sg,B/S (4.159);
    4. иначе пробковый (эмульсионный/churn у Ансари входит в пробковый).

    Коэффициент трения везде по Дарси–Вейсбаху (Муди). Упрощённый порт VBA-кода
    (только критерий Тёрнера) доступен как `AnsariVBAModel`.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    :param settings: Настройки модели (по умолчанию AnsariSettings())
    """

    def __init__(
        self,
        pipe: PipeParams,
        fluid: FluidParams,
        *,
        settings: AnsariSettings | None = None,
    ):
        super().__init__(pipe, fluid)
        self.settings = settings if settings is not None else AnsariSettings()

    name = "Ansari"
    angle_range = (75.0, 90.0)

    @cached_property
    def _invariants(self) -> _Invariants:
        """Величины, зависящие только от трубы и флюида (PipeParams/FluidParams неизменяемы)."""
        fluid_kwargs = {
            "liquid_density": self.fluid.density_liquid,
            "gas_density": self.fluid.density_gas,
            "surface_tension": self.fluid.surface_tension,
        }
        angle = self.pipe.angle
        sin_angle = math.sin(math.radians(angle))
        bubble_possible = (
            self.pipe.diameter > minimum_bubble_diameter(**fluid_kwargs)
            and angle >= self.settings.bubble_min_angle_deg
        )
        return _Invariants(
            relative_roughness=self.pipe.roughness / self.pipe.diameter,
            sin_angle=sin_angle,
            dispersion_prefactor=dispersion_prefactor(**fluid_kwargs),
            turner_velocity=turner_velocity(
                **fluid_kwargs,
                inclination_factor=(
                    sin_angle if self.settings.turner_uses_inclination else 1.0
                ),
            ),
            slip_velocity=slip_velocity(**fluid_kwargs),
            bubble_possible=bubble_possible,
        )

    def classify(self, vsl: float, vsg: float) -> AnsariResult:
        """
        Определяет режим течения и причину перехода (для диагностики).

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная или не конечная
        """
        check_velocities(vsl, vsg)
        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return AnsariResult(single_phase, TransitionReason.SINGLE_PHASE)

        invariants = self._invariants
        fluid = self.fluid
        diameter = self.pipe.diameter

        if is_dispersed_bubble(
            liquid_velocity=vsl,
            gas_velocity=vsg,
            diameter=diameter,
            relative_roughness=invariants.relative_roughness,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_viscosity=fluid.viscosity_liquid,
            gas_viscosity=fluid.viscosity_gas,
            prefactor=invariants.dispersion_prefactor,
            max_gas_fraction=self.settings.dispersed_max_gas_fraction,
        ):
            return AnsariResult(
                FlowPatternCode.DISPERSED_BUBBLE, TransitionReason.DISPERSED_BUBBLE
            )

        rejection = self._check_annular(vsl, vsg)
        if isinstance(rejection, AnsariResult):
            return rejection

        bubble_limit = bubble_slug_velocity(
            liquid_velocity=vsl,
            slip_velocity=invariants.slip_velocity,
            sin_angle=invariants.sin_angle,
        )
        if invariants.bubble_possible and vsg < bubble_limit:
            return AnsariResult(
                FlowPatternCode.BUBBLE, rejection.reason, rejection.film
            )
        return AnsariResult(FlowPatternCode.SLUG, rejection.reason, rejection.film)

    def _check_annular(
        self, vsl: float, vsg: float
    ) -> AnsariResult | _AnnularRejection:
        """
        Проверяет критерии кольцевого режима.

        :return: AnsariResult с режимом ANNULAR или причина отказа
        """
        invariants = self._invariants
        fluid = self.fluid

        if vsg < invariants.turner_velocity:
            return _AnnularRejection(TransitionReason.BELOW_TURNER)
        if self.settings.annular_criteria == "turner_only":
            return AnsariResult(FlowPatternCode.ANNULAR, TransitionReason.ANNULAR)

        film = solve_film(
            liquid_velocity=vsl,
            gas_velocity=vsg,
            diameter=self.pipe.diameter,
            relative_roughness=invariants.relative_roughness,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_viscosity=fluid.viscosity_liquid,
            gas_viscosity=fluid.viscosity_gas,
            surface_tension=fluid.surface_tension,
            sin_angle=invariants.sin_angle,
            friction_ratio=self.settings.film_friction_ratio,
        )
        if film.delta is None:
            return _AnnularRejection(TransitionReason.NO_FILM_SOLUTION, film)
        if is_film_blocking(film=film, threshold=self.settings.blockage_threshold):
            return _AnnularRejection(TransitionReason.BLOCKAGE, film)
        if is_film_unstable(film=film):
            return _AnnularRejection(TransitionReason.FILM_INSTABILITY, film)
        return AnsariResult(FlowPatternCode.ANNULAR, TransitionReason.ANNULAR, film)
