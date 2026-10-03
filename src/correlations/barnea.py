"""
Единая механистическая модель Barnea (1987) для переходов между режимами течения
при любом угле наклона трубы (−90°…+90°).

Источники:

* Barnea D. A unified model for predicting flow-pattern transitions for the whole
  range of pipe inclinations // Int. J. Multiphase Flow. 1987. 13(1):1–12;
* Barnea D. Transition from annular flow and from dispersed bubble flow — unified
  models for the whole range of pipe inclinations // IJMF. 1986. 12(5):733–744;
* Barnea D., Brauner N. Holdup of the liquid slug in two phase intermittent flow //
  IJMF. 1985;
* Bratland O. Pipe Flow 2: Multi-phase Flow Assurance, гл. 11 (ур. 11.2.11–11.2.22,
  11.3.6, 11.4.1–11.4.4);
* Брилл Дж. П., Мукерджи Х. «Многофазный поток в скважинах», §4.2.2
  (ур. 4.161, 4.166–4.169).

Диапазон применимости: любые углы от −90° до +90°; труба круглая. Формулы
реализованы по ТЗ, с оригинальными статьями Barnea (1986, 1987) сверка не
выполнялась.

Возвращаемые режимы: STRATIFIED (расслоённый гладкий), STRATIFIED_WAVY (расслоённый
волновой), BUBBLE (пузырьковый), SLUG (прерывистый: пробковый и удлинённые пузыри),
DISPERSED_BUBBLE (дисперсно-пузырьковый), ANNULAR (кольцевой), а также SINGLE_GAS и
SINGLE_LIQUID для однофазных случаев.

Порядок проверок: расслоённое течение (только при |θ| < 90°) → дисперсно-пузырьковое
→ кольцевое → пузырьковое → прерывистое. Коэффициент трения везде Фаннинга по
Taitel–Dukler (`fanning_friction_taitel_dukler`).

Ограничения и неоднозначности источника:

* Модель Ансари использует для перекрытия сечения порог 0.12 с учётом капель в ядре;
  у Barnea порог 0.5·H_LS (≥ 0.24) без учёта капель. Это разные модели, они не
  унифицируются.
* В литературе встречается упрощение `blockage_mode="constant"` с H_LS = 0.48
  (порог 0.24); по умолчанию используется удержание по Barnea & Brauner.
* Для вертикального потока граница кольцевого режима у Barnea может лежать заметно
  правее критерия Тёрнера. Это ожидаемо.
* В Pipe Flow 2 тип коэффициента трения в критерии перехода «расслоённое →
  кольцевое» при нисходящем потоке (11.4.4) не указан; принят Фаннинг (как в
  остальных критериях Taitel–Dukler и Barnea). Требует сверки с Barnea (1982/1987).
* Удержание жидкости в пробке α_GS = ((LHS − 0.725)/4.15)² ограничивается снизу
  нулём после возведения в квадрат (как в ТЗ). При LHS < 0.725 выражение в скобках
  отрицательно, квадрат даёт малое положительное α_GS (не более 0.03) вместо нуля.
  Влияние на порог перекрытия не более 1.5%.
* Для BUBBLE и SLUG в `reason` указывается причина отказа в кольцевом режиме
  (NO_FILM_SOLUTION, BLOCKAGE, FILM_INSTABILITY): сам режим записан в `pattern`.
"""

import math
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
from correlations.domain.params import FluidParams, PipeParams
from correlations.physics.friction import fanning_friction_taitel_dukler
from correlations.physics.stratified import (
    GRAVITY,
    StratifiedGeometry,
    jeffreys_wavy_gas_velocity,
    kelvin_helmholtz_gas_velocity,
    liquid_friction_factor,
    phase_velocities,
    solve_equilibrium_level,
    stratified_geometry,
)

_RIGHT_ANGLE_DEG: Final = 90.0  # При |θ| = 90° расслоённого течения нет
_COS_EPS: Final = 1e-12  # При |cos θ| ниже порога d_CB считается бесконечным

# Дисперсно-пузырьковый режим (Barnea, 1986; ур. 4.161)
_DISPERSED_BASE: Final = 0.725
_DISPERSED_GAS_COEFFICIENT: Final = 4.15
_DISPERSED_SURFACE_EXPONENT: Final = 0.6
_DISPERSED_FRICTION_EXPONENT: Final = -0.4
_BREAKUP_FACTOR: Final = 2.0  # d_CD = 2·[0.4σ/(Δρ·g)]^0.5
_BREAKUP_SURFACE_COEFFICIENT: Final = 0.4
_CREAMING_FACTOR: Final = 3.0 / 8.0  # d_CB = (3/8)·(ρ_L/Δρ)·f_m·v_m²/(g·|cos θ|)

# Кольцевое течение (Barnea, 1986)
_GRADIENT_FACTOR: Final = 2.0  # (dp/dL)_S = 2·f·ρ·v²/d
_FILM_LINEAR_COEFFICIENT: Final = 75.0  # (1 + 75·H) в уравнении плёнки
_FILM_EXPONENT: Final = 1.5  # (1 − H)^1.5, Barnea (1987), ур. 15
_FILM_HOLDUP_MIN: Final = 1e-7  # Нижняя граница поиска H
_FILM_HOLDUP_MAX: Final = 0.999  # Верхняя граница поиска H
_FILM_GRID_NODES: Final = 300
_FILM_GRID: Final = np.geomspace(_FILM_HOLDUP_MIN, _FILM_HOLDUP_MAX, _FILM_GRID_NODES)
_FILM_XTOL: Final = 1e-10  # Допуск по H при уточнении корня
_INSTABILITY_HOLDUP_LINEAR: Final = 1.5  # ур. 4.169
_INSTABILITY_NUMERATOR: Final = 2.0  # ур. 4.169
_INSTABILITY_MAX_HOLDUP: Final = 2.0 / 3.0  # При H ≥ 2/3 знаменатель 4.169 ≤ 0

# Пузырьковый режим (Pipe Flow 2, 11.3.6, 11.4.1)
_BUBBLE_MIN_DIAMETER_COEFFICIENT: Final = 19.0
_BUBBLE_RISE_COEFFICIENT: Final = 1.15
_BUBBLE_LIQUID_DIVISOR: Final = 3.0


class BarneaReason(Enum):
    """
    Критерий, по которому получен режим.

    Для BUBBLE и SLUG в `reason` указывается причина отказа в кольцевом режиме
    (NO_FILM_SOLUTION, BLOCKAGE, FILM_INSTABILITY): сам режим записан в
    `BarneaResult.pattern`.
    """

    SINGLE_PHASE = auto()
    STRATIFIED_SMOOTH = auto()
    STRATIFIED_WAVY = auto()
    DOWNFLOW_ANNULAR = auto()  # расслоённое → кольцевое при θ < 0 (11.4.4)
    DISPERSED_BUBBLE = auto()
    ANNULAR = auto()
    NO_FILM_SOLUTION = auto()
    BLOCKAGE = auto()
    FILM_INSTABILITY = auto()
    BUBBLE = auto()
    SLUG = auto()


@dataclass(frozen=True, slots=True)
class BarneaSettings:
    """
    Настройки модели Barnea.

    Attributes
    ----------
    sheltering_coefficient: float
        Коэффициент экранирования s в критерии волнообразования Джеффриса (ур. 11.2.13)
    downflow_wave_froude: float
        Число Фруда u_L/√(g·h_L), выше которого расслоённое течение при θ < 0
        волновое (11.4.3)
    dispersed_max_gas_fraction: float
        Максимальная расходная доля газа λ_G для дисперсно-пузырькового режима
        (плотная упаковка пузырей)
    blockage_factor: float
        Доля H_LS: кольцевой режим перекрыт, если H > blockage_factor·H_LS
    blockage_mode: "barnea_brauner" | "constant"
        Способ оценки удержания жидкости в пробке H_LS: по Barnea & Brauner (1985)
        или постоянное `constant_slug_holdup`
    constant_slug_holdup: float
        Удержание жидкости в пробке H_LS при blockage_mode="constant"
    bubble_min_angle_deg: float
        Минимальный угол от горизонтали, при котором возможен пузырьковый режим
        (11.4.1; Shoham, 2006)
    """

    sheltering_coefficient: float = 0.01
    downflow_wave_froude: float = 1.5
    dispersed_max_gas_fraction: float = 0.52
    blockage_factor: float = 0.5
    blockage_mode: Literal["barnea_brauner", "constant"] = "barnea_brauner"
    constant_slug_holdup: float = 0.48
    bubble_min_angle_deg: float = 60.0

    def __post_init__(self) -> None:
        if not self.sheltering_coefficient > 0.0:
            raise ValueError("sheltering_coefficient должен быть > 0")
        if not self.downflow_wave_froude > 0.0:
            raise ValueError("downflow_wave_froude должен быть > 0")
        if not 0.0 < self.dispersed_max_gas_fraction <= 1.0:
            raise ValueError("dispersed_max_gas_fraction должен быть в (0, 1]")
        # При H ≥ 2/3 знаменатель критерия неустойчивости плёнки теряет смысл
        if not 0.0 < self.blockage_factor < _INSTABILITY_MAX_HOLDUP:
            raise ValueError("blockage_factor должен быть в (0, 2/3)")
        if self.blockage_mode not in ("barnea_brauner", "constant"):
            raise ValueError(
                "blockage_mode должен быть 'barnea_brauner' или 'constant'"
            )
        if not 0.0 < self.constant_slug_holdup <= 1.0:
            raise ValueError("constant_slug_holdup должен быть в (0, 1]")


@dataclass(frozen=True, slots=True)
class BarneaResult(ClassificationResult):
    """
    Результат классификации.

    Attributes
    ----------
    pattern: FlowPatternCode
        Режим течения (унаследовано от ClassificationResult)
    reason: BarneaReason
        Критерий, по которому получен режим
    level: float | None
        h̃ равновесного расслоённого течения (None, если не считался или не найден)
    film_holdup: float | None
        Доля плёнки H из уравнения плёнки (None, если не считалась или нет решения)
    slug_holdup: float | None
        Удержание жидкости в пробке H_LS (None, если не считалось)
    x_squared: float | None
        Параметр Мартинелли X² (None, если не считался)
    y: float | None
        Параметр наклона Y (None, если не считался)
    """

    reason: BarneaReason
    level: float | None
    film_holdup: float | None
    slug_holdup: float | None
    x_squared: float | None
    y: float | None


def breakup_diameter(
    *, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """
    Критический диаметр пузыря по дроблению/коалесценции (Barnea, 1986):
    d_CD = 2·[0.4·σ/(Δρ·g)]^0.5, м.

    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    """
    return _BREAKUP_FACTOR * math.sqrt(
        _BREAKUP_SURFACE_COEFFICIENT
        * surface_tension
        / ((liquid_density - gas_density) * GRAVITY)
    )


def creaming_diameter(
    *,
    liquid_density: float,
    gas_density: float,
    mixture_friction: float,
    mixture_velocity: float,
    cos_angle: float,
) -> float:
    """
    Критический диаметр пузыря по всплытию к верхней стенке (Barnea, 1986):
    d_CB = (3/8)·(ρ_L/Δρ)·f_m·v_m²/(g·|cos θ|), м. При cos θ ≈ 0 равен бесконечности.

    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param mixture_friction: коэффициент трения Фаннинга смеси f_m
    :param mixture_velocity: скорость смеси v_m, м/с
    :param cos_angle: cos θ
    """
    if abs(cos_angle) < _COS_EPS:
        return math.inf
    return (
        _CREAMING_FACTOR
        * liquid_density
        / (liquid_density - gas_density)
        * mixture_friction
        * mixture_velocity**2
        / (GRAVITY * abs(cos_angle))
    )


def maximum_bubble_diameter(
    *,
    gas_fraction: float,
    mixture_velocity: float,
    diameter: float,
    liquid_density: float,
    surface_tension: float,
    mixture_friction: float,
) -> float:
    """
    Максимальный устойчивый диаметр пузыря в турбулентном потоке (Barnea, 1986):
    d_max = (0.725 + 4.15·λ_G^0.5)·(σ/ρ_L)^0.6·(2·f_m·v_m³/d)^(−0.4), м.

    :param gas_fraction: расходная доля газа λ_G
    :param mixture_velocity: скорость смеси v_m, м/с
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    :param mixture_friction: коэффициент трения Фаннинга смеси f_m
    """
    return (
        (_DISPERSED_BASE + _DISPERSED_GAS_COEFFICIENT * math.sqrt(gas_fraction))
        * (surface_tension / liquid_density) ** _DISPERSED_SURFACE_EXPONENT
        * (_BREAKUP_FACTOR * mixture_friction * mixture_velocity**3 / diameter)
        ** _DISPERSED_FRICTION_EXPONENT
    )


def is_dispersed_bubble(
    *,
    maximum_diameter: float,
    breakup: float,
    creaming: float,
    gas_fraction: float,
    max_gas_fraction: float,
) -> bool:
    """
    Дисперсно-пузырьковый режим (Barnea, 1986): d_max ≤ min(d_CD, d_CB) и
    λ_G ≤ max_gas_fraction. Условие d_max ≤ d_CD эквивалентно ур. 4.161
    Брилла–Мукерджи (с f Дарси = 4·f Фаннинга).

    :param maximum_diameter: d_max, м
    :param breakup: d_CD, м
    :param creaming: d_CB, м
    :param gas_fraction: расходная доля газа λ_G
    :param max_gas_fraction: порог λ_G (плотная упаковка)
    """
    return (
        maximum_diameter <= min(breakup, creaming) and gas_fraction <= max_gas_fraction
    )


def superficial_gradient(
    *, velocity: float, density: float, viscosity: float, diameter: float
) -> float:
    """
    Приведённый градиент давления одной фазы: (dp/dL)_S = 2·f·ρ·v²/d, Па/м,
    f — коэффициент Фаннинга по Re = ρ·v·d/μ.

    :param velocity: приведённая скорость фазы, м/с, > 0
    :param density: плотность фазы, кг/м³
    :param viscosity: вязкость фазы, Па·с
    :param diameter: диаметр трубы d, м
    """
    friction = fanning_friction_taitel_dukler(density * velocity * diameter / viscosity)
    return _GRADIENT_FACTOR * friction * density * velocity**2 / diameter


def film_parameters(
    *,
    liquid_gradient: float,
    gas_gradient: float,
    liquid_density: float,
    gas_density: float,
    sin_angle: float,
) -> tuple[float, float]:
    """
    Параметры уравнения плёнки (Barnea, 1986): X² = (dp/dL)_SL/(dp/dL)_SG,
    Y = Δρ·g·sin θ/(dp/dL)_SG.

    :param liquid_gradient: приведённый градиент жидкости (dp/dL)_SL, Па/м
    :param gas_gradient: приведённый градиент газа (dp/dL)_SG, Па/м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param sin_angle: sin θ
    :return: (X², Y)
    """
    return (
        liquid_gradient / gas_gradient,
        (liquid_density - gas_density) * GRAVITY * sin_angle / gas_gradient,
    )


def film_equation(*, film_holdup: float, x_squared: float, y: float) -> float:
    """
    Невязка уравнения плёнки (Barnea, 1986):
    F(H) = Y − (1 + 75·H)/((1 − H)^1.5·H) + X²/H³ → +∞ при H → 0.

    :param film_holdup: доля плёнки H ∈ (0, 1)
    :param x_squared: параметр Мартинелли X²
    :param y: параметр наклона Y
    """
    return (
        y
        - (1.0 + _FILM_LINEAR_COEFFICIENT * film_holdup)
        / ((1.0 - film_holdup) ** _FILM_EXPONENT * film_holdup)
        + x_squared / film_holdup**3
    )


def solve_film_holdup(*, x_squared: float, y: float) -> float | None:
    """
    Доля плёнки H — наименьший корень уравнения плёнки: первая смена знака на
    логарифмической сетке H ∈ [1e-7, 0.999] (300 узлов), затем уточнение методом
    Брента до 1e-10.

    :param x_squared: параметр Мартинелли X²
    :param y: параметр наклона Y
    :return: H или None, если смены знака нет (плёнка не удерживается)
    """
    holdups = _FILM_GRID
    values = (
        y
        - (1.0 + _FILM_LINEAR_COEFFICIENT * holdups)
        / ((1.0 - holdups) ** _FILM_EXPONENT * holdups)
        + x_squared / holdups**3
    )
    sign_changes = np.nonzero(values[:-1] * values[1:] <= 0.0)[0]
    if sign_changes.size == 0:
        return None
    index = int(sign_changes[0])

    def residual(holdup: float) -> float:
        return film_equation(film_holdup=holdup, x_squared=x_squared, y=y)

    lower, upper = float(holdups[index]), float(holdups[index + 1])
    if residual(lower) == 0.0:
        return lower
    if residual(upper) == 0.0:
        return upper
    # full_output=False (по умолчанию) возвращает float; стабы scipy это не различают
    return cast(float, brentq(residual, lower, upper, xtol=_FILM_XTOL))


def slug_liquid_holdup(
    *,
    mixture_velocity: float,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    surface_tension: float,
    mixture_friction: float,
    max_gas_fraction: float,
) -> float:
    """
    Удержание жидкости в пробке по Barnea & Brauner (1985):
    LHS = 2·[0.4σ/(Δρ·g)]^0.5·(ρ_L/σ)^0.6·(2·f_m/d)^0.4·v_m^1.2,
    α_GS = clip(((LHS − 0.725)/4.15)², 0, max_gas_fraction), H_LS = 1 − α_GS.

    :param mixture_velocity: скорость смеси v_m, м/с
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
    :param mixture_friction: коэффициент трения Фаннинга смеси f_m
    :param max_gas_fraction: верхняя граница α_GS (плотная упаковка)
    """
    left_side = (
        breakup_diameter(
            liquid_density=liquid_density,
            gas_density=gas_density,
            surface_tension=surface_tension,
        )
        * (liquid_density / surface_tension) ** _DISPERSED_SURFACE_EXPONENT
        * (_BREAKUP_FACTOR * mixture_friction / diameter)
        ** (-_DISPERSED_FRICTION_EXPONENT)
        * mixture_velocity**1.2
    )
    gas_holdup = ((left_side - _DISPERSED_BASE) / _DISPERSED_GAS_COEFFICIENT) ** 2
    return 1.0 - min(max(gas_holdup, 0.0), max_gas_fraction)


def is_film_blocking(
    *, film_holdup: float, slug_holdup: float, blockage_factor: float
) -> bool:
    """
    Перекрытие сечения ядром (Barnea, 1986): H > blockage_factor·H_LS.

    :param film_holdup: доля плёнки H
    :param slug_holdup: удержание жидкости в пробке H_LS
    :param blockage_factor: доля H_LS (0.5)
    """
    return film_holdup > blockage_factor * slug_holdup


def is_film_unstable(*, film_holdup: float, x_squared: float, y: float) -> bool:
    """
    Неустойчивость плёнки (ур. 4.169, формулировка Брилла–Мукерджи):
    Y > (2 − 1.5·H)/(H³·(1 − 1.5·H))·X². Определена при H < 2/3; при H ≥ 2/3
    возвращает True (плёнка не может быть устойчивой).

    :param film_holdup: доля плёнки H
    :param x_squared: параметр Мартинелли X²
    :param y: параметр наклона Y
    """
    if film_holdup >= _INSTABILITY_MAX_HOLDUP:
        return True
    right_side = (
        (_INSTABILITY_NUMERATOR - _INSTABILITY_HOLDUP_LINEAR * film_holdup)
        / (film_holdup**3 * (1.0 - _INSTABILITY_HOLDUP_LINEAR * film_holdup))
        * x_squared
    )
    return y > right_side


def bubble_flow_possible(
    *, diameter: float, angle_deg: float, min_angle_deg: float, minimum_diameter: float
) -> bool:
    """
    Возможен ли пузырьковый режим (11.3.6, 11.4.1): d > d_min и θ ≥ min_angle_deg.

    :param diameter: диаметр трубы d, м
    :param angle_deg: угол наклона от горизонтали θ, градусы
    :param min_angle_deg: минимальный угол пузырькового режима, градусы (60)
    :param minimum_diameter: d_min, м
    """
    return diameter > minimum_diameter and angle_deg >= min_angle_deg


@dataclass(frozen=True, slots=True)
class _Invariants:
    """Величины, не зависящие от (v_SL, v_Sg)."""

    angle_rad: float
    sin_angle: float
    cos_angle: float
    breakup_diameter: float
    rise_velocity: float  # U = [g·Δρ·σ/ρ_L²]^0.25
    bubble_possible: bool
    is_inclined: bool  # |θ| < 90°: расслоённое течение возможно


class BarneaModel(IFlowModel):
    """
    Единая модель Barnea для углов от −90° до +90°.

    Порядок проверок: расслоённое течение (|θ| < 90°) → дисперсно-пузырьковое →
    кольцевое → пузырьковое → прерывистое. Диапазон применимости, ограничения и
    неоднозначности источника — в docstring модуля.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    :param settings: Настройки модели (по умолчанию BarneaSettings())
    """

    def __init__(
        self,
        pipe: PipeParams,
        fluid: FluidParams,
        *,
        settings: BarneaSettings | None = None,
    ):
        super().__init__(pipe, fluid)
        self.settings = settings if settings is not None else BarneaSettings()

    name = "Barnea"
    angle_range = (-90.0, 90.0)

    @cached_property
    def _invariants(self) -> _Invariants:
        """Величины, зависящие только от трубы и флюида (PipeParams/FluidParams неизменяемы)."""
        fluid = self.fluid
        angle_deg = self.pipe.angle
        angle_rad = math.radians(angle_deg)
        density_difference = fluid.density_liquid - fluid.density_gas
        minimum_diameter = _BUBBLE_MIN_DIAMETER_COEFFICIENT * math.sqrt(
            density_difference
            * fluid.surface_tension
            / (fluid.density_liquid**2 * GRAVITY)
        )
        return _Invariants(
            angle_rad=angle_rad,
            sin_angle=math.sin(angle_rad),
            cos_angle=math.cos(angle_rad),
            breakup_diameter=breakup_diameter(
                liquid_density=fluid.density_liquid,
                gas_density=fluid.density_gas,
                surface_tension=fluid.surface_tension,
            ),
            rise_velocity=(
                GRAVITY
                * density_difference
                * fluid.surface_tension
                / fluid.density_liquid**2
            )
            ** 0.25,
            bubble_possible=bubble_flow_possible(
                diameter=self.pipe.diameter,
                angle_deg=angle_deg,
                min_angle_deg=self.settings.bubble_min_angle_deg,
                minimum_diameter=minimum_diameter,
            ),
            is_inclined=abs(angle_deg) < _RIGHT_ANGLE_DEG,
        )

    def classify(self, vsl: float, vsg: float) -> BarneaResult:
        """
        Определяет режим течения и критерий перехода (для диагностики).

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная или не конечная
        """
        check_velocities(vsl, vsg)
        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return _empty_result(single_phase, BarneaReason.SINGLE_PHASE)

        invariants = self._invariants
        fluid = self.fluid
        level: float | None = None

        if invariants.is_inclined:
            level, stratified = self._check_stratified(vsl, vsg)
            if stratified is not None:
                return stratified

        diameter = self.pipe.diameter
        mixture_velocity = vsl + vsg
        gas_fraction = vsg / mixture_velocity
        liquid_fraction = 1.0 - gas_fraction
        mixture_density = (
            fluid.density_liquid * liquid_fraction + fluid.density_gas * gas_fraction
        )
        mixture_viscosity = (
            fluid.viscosity_liquid * liquid_fraction
            + fluid.viscosity_gas * gas_fraction
        )
        mixture_friction = fanning_friction_taitel_dukler(
            mixture_density * mixture_velocity * diameter / mixture_viscosity
        )

        max_diameter = maximum_bubble_diameter(
            gas_fraction=gas_fraction,
            mixture_velocity=mixture_velocity,
            diameter=diameter,
            liquid_density=fluid.density_liquid,
            surface_tension=fluid.surface_tension,
            mixture_friction=mixture_friction,
        )
        creaming = creaming_diameter(
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            mixture_friction=mixture_friction,
            mixture_velocity=mixture_velocity,
            cos_angle=invariants.cos_angle,
        )
        if is_dispersed_bubble(
            maximum_diameter=max_diameter,
            breakup=invariants.breakup_diameter,
            creaming=creaming,
            gas_fraction=gas_fraction,
            max_gas_fraction=self.settings.dispersed_max_gas_fraction,
        ):
            return BarneaResult(
                pattern=FlowPatternCode.DISPERSED_BUBBLE,
                reason=BarneaReason.DISPERSED_BUBBLE,
                level=level,
                film_holdup=None,
                slug_holdup=None,
                x_squared=None,
                y=None,
            )

        liquid_gradient = superficial_gradient(
            velocity=vsl,
            density=fluid.density_liquid,
            viscosity=fluid.viscosity_liquid,
            diameter=diameter,
        )
        gas_gradient = superficial_gradient(
            velocity=vsg,
            density=fluid.density_gas,
            viscosity=fluid.viscosity_gas,
            diameter=diameter,
        )
        x_squared, y = film_parameters(
            liquid_gradient=liquid_gradient,
            gas_gradient=gas_gradient,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            sin_angle=invariants.sin_angle,
        )
        film_holdup = solve_film_holdup(x_squared=x_squared, y=y)
        slug_holdup: float | None = None
        rejection = BarneaReason.NO_FILM_SOLUTION
        if film_holdup is not None:
            slug_holdup = self._slug_holdup(mixture_velocity, mixture_friction)
            if is_film_blocking(
                film_holdup=film_holdup,
                slug_holdup=slug_holdup,
                blockage_factor=self.settings.blockage_factor,
            ):
                rejection = BarneaReason.BLOCKAGE
            elif is_film_unstable(film_holdup=film_holdup, x_squared=x_squared, y=y):
                rejection = BarneaReason.FILM_INSTABILITY
            else:
                rejection = BarneaReason.ANNULAR

        if rejection is BarneaReason.ANNULAR:
            pattern = FlowPatternCode.ANNULAR
        elif (
            invariants.bubble_possible
            and vsg
            < (
                vsl
                + _BUBBLE_RISE_COEFFICIENT
                * invariants.rise_velocity
                * invariants.sin_angle
            )
            / _BUBBLE_LIQUID_DIVISOR
        ):
            pattern = FlowPatternCode.BUBBLE
        else:
            pattern = FlowPatternCode.SLUG
        return BarneaResult(
            pattern=pattern,
            reason=rejection,
            level=level,
            film_holdup=film_holdup,
            slug_holdup=slug_holdup,
            x_squared=x_squared,
            y=y,
        )

    def _slug_holdup(self, mixture_velocity: float, mixture_friction: float) -> float:
        """H_LS: по Barnea & Brauner или постоянное (blockage_mode="constant")."""
        settings = self.settings
        if settings.blockage_mode == "constant":
            return settings.constant_slug_holdup
        return slug_liquid_holdup(
            mixture_velocity=mixture_velocity,
            diameter=self.pipe.diameter,
            liquid_density=self.fluid.density_liquid,
            gas_density=self.fluid.density_gas,
            surface_tension=self.fluid.surface_tension,
            mixture_friction=mixture_friction,
            max_gas_fraction=settings.dispersed_max_gas_fraction,
        )

    def _check_stratified(
        self, vsl: float, vsg: float
    ) -> tuple[float | None, BarneaResult | None]:
        """
        Шаг 1: расслоённое течение.

        :return: (уровень h̃ или None, результат STRATIFIED/STRATIFIED_WAVY/ANNULAR
            или None, если расслоённое течение невозможно или неустойчиво)
        """
        invariants = self._invariants
        fluid = self.fluid
        diameter = self.pipe.diameter
        equilibrium = solve_equilibrium_level(
            superficial_liquid_velocity=vsl,
            superficial_gas_velocity=vsg,
            diameter=diameter,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_viscosity=fluid.viscosity_liquid,
            gas_viscosity=fluid.viscosity_gas,
            angle_rad=invariants.angle_rad,
            refine_all_roots=False,
        )
        level = equilibrium.level
        if level is None:
            return None, None

        geometry = stratified_geometry(level)
        liquid_velocity, gas_velocity = phase_velocities(
            geometry=geometry,
            superficial_liquid_velocity=vsl,
            superficial_gas_velocity=vsg,
        )
        critical_gas_velocity = kelvin_helmholtz_gas_velocity(
            geometry=geometry,
            diameter=diameter,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            angle_rad=invariants.angle_rad,
        )
        if gas_velocity >= critical_gas_velocity:
            return level, None

        if invariants.angle_rad < 0.0 and self._is_downflow_annular(
            geometry, liquid_velocity
        ):
            return level, self._stratified_result(
                FlowPatternCode.ANNULAR, BarneaReason.DOWNFLOW_ANNULAR, level
            )

        if self._is_wavy(geometry, liquid_velocity, gas_velocity):
            return level, self._stratified_result(
                FlowPatternCode.STRATIFIED_WAVY, BarneaReason.STRATIFIED_WAVY, level
            )
        return level, self._stratified_result(
            FlowPatternCode.STRATIFIED, BarneaReason.STRATIFIED_SMOOTH, level
        )

    def _is_downflow_annular(
        self, geometry: StratifiedGeometry, liquid_velocity: float
    ) -> bool:
        """Переход «расслоённое → кольцевое» при θ < 0 (11.4.4): u_L² > g·d·cos θ·(1 − h̃)/f_L."""
        friction = liquid_friction_factor(
            geometry=geometry,
            diameter=self.pipe.diameter,
            liquid_velocity=liquid_velocity,
            liquid_density=self.fluid.density_liquid,
            liquid_viscosity=self.fluid.viscosity_liquid,
        )
        return liquid_velocity**2 > (
            GRAVITY
            * self.pipe.diameter
            * self._invariants.cos_angle
            * (1.0 - geometry.level)
            / friction
        )

    def _is_wavy(
        self, geometry: StratifiedGeometry, liquid_velocity: float, gas_velocity: float
    ) -> bool:
        """Волновое течение: критерий Джеффриса (11.2.13) или, при θ < 0, число Фруда (11.4.3)."""
        fluid = self.fluid
        wavy_velocity = jeffreys_wavy_gas_velocity(
            liquid_viscosity=fluid.viscosity_liquid,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_velocity=liquid_velocity,
            angle_rad=self._invariants.angle_rad,
            sheltering_coefficient=self.settings.sheltering_coefficient,
        )
        if gas_velocity >= wavy_velocity:
            return True
        if self._invariants.angle_rad < 0.0:
            froude = liquid_velocity / math.sqrt(
                GRAVITY * geometry.level * self.pipe.diameter
            )
            return froude > self.settings.downflow_wave_froude
        return False

    @staticmethod
    def _stratified_result(
        pattern: FlowPatternCode, reason: BarneaReason, level: float
    ) -> BarneaResult:
        return BarneaResult(
            pattern=pattern,
            reason=reason,
            level=level,
            film_holdup=None,
            slug_holdup=None,
            x_squared=None,
            y=None,
        )


def _empty_result(pattern: FlowPatternCode, reason: BarneaReason) -> BarneaResult:
    """Результат без диагностических величин (однофазные случаи)."""
    return BarneaResult(
        pattern=pattern,
        reason=reason,
        level=None,
        film_holdup=None,
        slug_holdup=None,
        x_squared=None,
        y=None,
    )
