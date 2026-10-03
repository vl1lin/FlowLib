"""
Механистическая модель Taitel–Dukler (1976) для горизонтальных и слабонаклонных труб.

Источники: Taitel Y., Dukler A.E. A model for predicting flow regime transitions in
horizontal and near horizontal gas-liquid flow // AIChE J. 1976. 22(1):47–55;
Bratland O. Pipe Flow 2: Multi-phase Flow Assurance, §11.2 (ур. 11.2.11–11.2.22).

Диапазон применимости: горизонтальные и слабонаклонные трубы, |θ| ≤ 10°.
Модель Taitel–Dukler получена для гладких труб при нормальных условиях
(воздух–вода); шероховатость не учитывается.

Возвращаемые режимы: STRATIFIED (расслоённый гладкий), STRATIFIED_WAVY (расслоённый
волновой), SLUG (прерывистый), ANNULAR (кольцевой), DISPERSED_BUBBLE
(дисперсно-пузырьковый), а также SINGLE_GAS и SINGLE_LIQUID для однофазных случаев.
Модель различает только эти режимы: пробковый режим и режим удлинённых пузырей не
разделяются (оба — SLUG).

Порядок проверок: равновесный уровень жидкости h̃ → критерий Кельвина–Гельмгольца →
гладкое или волновое расслоённое течение (Джеффрис) → кольцевое (низкий уровень)
→ дисперсно-пузырьковое или прерывистое.

Ограничения и неоднозначности источника:

* В Pipe Flow 2 коэффициент в ур. 11.2.19 назван «Дарси–Вейсбаха», но
  τ_W = ½·f·ρ·v² — это определение Фаннинга; с коэффициентом Дарси граница сдвигается
  в 2 раза. Использован Фаннинг (f_TD).
* Порог h̃ = 0.5 — оригинал Taitel–Dukler. Barnea et al. (1980) предлагали 0.35
  (Pipe Flow 2, 11.2.14). Порог — настройка `annular_level_threshold`.
* f_i = f_G (гладкая граница раздела) — допущение оригинальной модели.
* Если равновесного уровня нет (`NO_EQUILIBRIUM`), шаги «кольцевое» и
  «дисперсно-пузырьковое» выполняются на уровне с минимальной |R(h̃)|. Для
  |θ| ≤ 10° такого не должно быть.
* При `distinguish_wavy=False` волновое течение возвращается как STRATIFIED, а
  `reason` сохраняет STRATIFIED_WAVY (критерий Джеффриса сработал).
* Для всех результатов ветвей «кольцевое» и «дисперсно-пузырьковое/прерывистое» при
  отсутствии равновесного уровня `reason` равен NO_EQUILIBRIUM.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Final

from correlations.base import (
    ClassificationResult,
    FlowPatternCode,
    IFlowModel,
    check_velocities,
    single_phase_pattern,
)
from correlations.domain.params import FluidParams, PipeParams
from correlations.physics.stratified import (
    GRAVITY,
    StratifiedGeometry,
    jeffreys_wavy_gas_velocity,
    kelvin_helmholtz_gas_velocity,
    least_residual_level,
    liquid_friction_factor,
    phase_velocities,
    solve_equilibrium_level,
    stratified_geometry,
)

_DISPERSED_FACTOR: Final = 4.0  # Множитель в ур. 11.2.19
_DISPERSED_EXPONENT: Final = 0.5  # Показатель степени в ур. 11.2.19


class TaitelDuklerReason(Enum):
    """
    Критерий, по которому получен режим.

    NO_EQUILIBRIUM относится ко всем результатам шагов «кольцевое» и
    «дисперсно-пузырьковое/прерывистое», полученным на уровне без равновесного решения.
    """

    SINGLE_PHASE = auto()
    STRATIFIED_SMOOTH = auto()
    STRATIFIED_WAVY = auto()
    NO_EQUILIBRIUM = auto()
    ANNULAR_LOW_LEVEL = auto()
    DISPERSED_BUBBLE = auto()
    INTERMITTENT = auto()


@dataclass(frozen=True, slots=True)
class TaitelDuklerSettings:
    """
    Настройки модели Taitel–Dukler.

    Attributes
    ----------
    annular_level_threshold: float
        Порог безразмерного уровня h̃, ниже которого неустойчивое расслоённое течение
        переходит в кольцевое (ур. 11.2.14; Taitel–Dukler: 0.5, Barnea et al. 1980: 0.35)
    sheltering_coefficient: float
        Коэффициент экранирования s в критерии Джеффриса (ур. 11.2.13, Sverdrup & Munk)
    distinguish_wavy: bool
        Различать гладкое и волновое расслоённое течение; при False всё расслоённое
        течение возвращается как STRATIFIED
    """

    annular_level_threshold: float = 0.5
    sheltering_coefficient: float = 0.01
    distinguish_wavy: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.annular_level_threshold < 1.0:
            raise ValueError("annular_level_threshold должен быть в (0, 1)")
        if not self.sheltering_coefficient > 0.0:
            raise ValueError("sheltering_coefficient должен быть > 0")


@dataclass(frozen=True, slots=True)
class TaitelDuklerResult(ClassificationResult):
    """
    Результат классификации.

    Attributes
    ----------
    pattern: FlowPatternCode
        Режим течения (унаследовано от ClassificationResult)
    reason: TaitelDuklerReason
        Критерий, по которому получен режим
    level: float | None
        Безразмерный уровень жидкости h̃ (при NO_EQUILIBRIUM — уровень с минимальной
        невязкой; None для однофазных случаев)
    gas_velocity: float | None
        Истинная скорость газа u_G, м/с
    liquid_velocity: float | None
        Истинная скорость жидкости u_L, м/с
    kh_critical_gas_velocity: float | None
        Критическая скорость газа по Кельвину–Гельмгольцу (ур. 11.2.11), м/с;
        None, если критерий не проверялся
    """

    reason: TaitelDuklerReason
    level: float | None
    gas_velocity: float | None
    liquid_velocity: float | None
    kh_critical_gas_velocity: float | None


def dispersed_bubble_liquid_velocity(
    *,
    geometry: StratifiedGeometry,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_friction: float,
    angle_rad: float,
) -> float:
    """
    Скорость жидкости начала дисперсно-пузырькового режима (ур. 11.2.19):
    u_L,DB = [4·A_G/S_i·g·cos θ/f_L·(1 − ρ_G/ρ_L)]^0.5, м/с.
    При u_L ≥ u_L,DB течение дисперсно-пузырьковое.

    f_L — коэффициент Фаннинга (в Pipe Flow 2 назван «Дарси–Вейсбаха», но
    τ_W = ½·f·ρ·v² — определение Фаннинга).

    :param geometry: геометрия при равновесном уровне
    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_friction: коэффициент трения Фаннинга жидкости f_L
    :param angle_rad: угол наклона от горизонтали θ, рад
    """
    area_over_width = geometry.gas_area * diameter / geometry.interface_width
    return (
        _DISPERSED_FACTOR
        * area_over_width
        * GRAVITY
        * math.cos(angle_rad)
        / liquid_friction
        * (1.0 - gas_density / liquid_density)
    ) ** _DISPERSED_EXPONENT


class TaitelDuklerModel(IFlowModel):
    """
    Модель Taitel–Dukler для горизонтальных и слабонаклонных труб (|θ| ≤ 10°).

    Порядок проверок: равновесный уровень h̃ → критерий Кельвина–Гельмгольца (11.2.11)
    → гладкое или волновое расслоённое течение (Джеффрис, 11.2.13) → кольцевое при
    h̃ < порога (11.2.14) → дисперсно-пузырьковое (11.2.19) или прерывистое.
    Диапазон применимости и ограничения — в docstring модуля.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    :param settings: Настройки модели (по умолчанию TaitelDuklerSettings())
    """

    def __init__(
        self,
        pipe: PipeParams,
        fluid: FluidParams,
        *,
        settings: TaitelDuklerSettings | None = None,
    ):
        super().__init__(pipe, fluid)
        self.settings = settings if settings is not None else TaitelDuklerSettings()

    name = "Taitel-Dukler"
    angle_range = (-10.0, 10.0)

    @cached_property
    def _angle_rad(self) -> float:
        """Угол наклона в радианах (PipeParams неизменяем)."""
        return math.radians(self.pipe.angle)

    def classify(self, vsl: float, vsg: float) -> TaitelDuklerResult:
        """
        Определяет режим течения и критерий перехода (для диагностики).

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная или не конечная
        """
        check_velocities(vsl, vsg)
        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return TaitelDuklerResult(
                pattern=single_phase,
                reason=TaitelDuklerReason.SINGLE_PHASE,
                level=None,
                gas_velocity=None,
                liquid_velocity=None,
                kh_critical_gas_velocity=None,
            )

        fluid = self.fluid
        diameter = self.pipe.diameter
        angle_rad = self._angle_rad
        equilibrium = solve_equilibrium_level(
            superficial_liquid_velocity=vsl,
            superficial_gas_velocity=vsg,
            diameter=diameter,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_viscosity=fluid.viscosity_liquid,
            gas_viscosity=fluid.viscosity_gas,
            angle_rad=angle_rad,
            refine_all_roots=False,
        )
        has_equilibrium = equilibrium.level is not None
        if equilibrium.level is not None:
            level = equilibrium.level
        else:
            level = least_residual_level(
                superficial_liquid_velocity=vsl,
                superficial_gas_velocity=vsg,
                diameter=diameter,
                liquid_density=fluid.density_liquid,
                gas_density=fluid.density_gas,
                liquid_viscosity=fluid.viscosity_liquid,
                gas_viscosity=fluid.viscosity_gas,
                angle_rad=angle_rad,
            )
        geometry = stratified_geometry(level)
        liquid_velocity, gas_velocity = phase_velocities(
            geometry=geometry,
            superficial_liquid_velocity=vsl,
            superficial_gas_velocity=vsg,
        )

        critical_gas_velocity: float | None = None
        if has_equilibrium:
            critical_gas_velocity = kelvin_helmholtz_gas_velocity(
                geometry=geometry,
                diameter=diameter,
                liquid_density=fluid.density_liquid,
                gas_density=fluid.density_gas,
                angle_rad=angle_rad,
            )
            if gas_velocity < critical_gas_velocity:
                pattern, reason = self._stratified_pattern(
                    geometry, liquid_velocity, gas_velocity
                )
                return TaitelDuklerResult(
                    pattern=pattern,
                    reason=reason,
                    level=level,
                    gas_velocity=gas_velocity,
                    liquid_velocity=liquid_velocity,
                    kh_critical_gas_velocity=critical_gas_velocity,
                )

        pattern, reason = self._unstratified_pattern(
            geometry, liquid_velocity, has_equilibrium
        )
        return TaitelDuklerResult(
            pattern=pattern,
            reason=reason,
            level=level,
            gas_velocity=gas_velocity,
            liquid_velocity=liquid_velocity,
            kh_critical_gas_velocity=critical_gas_velocity,
        )

    def _stratified_pattern(
        self, geometry: StratifiedGeometry, liquid_velocity: float, gas_velocity: float
    ) -> tuple[FlowPatternCode, TaitelDuklerReason]:
        """Гладкое или волновое расслоённое течение (критерий Джеффриса, ур. 11.2.13)."""
        fluid = self.fluid
        wavy_velocity = jeffreys_wavy_gas_velocity(
            liquid_viscosity=fluid.viscosity_liquid,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_velocity=liquid_velocity,
            angle_rad=self._angle_rad,
            sheltering_coefficient=self.settings.sheltering_coefficient,
        )
        if gas_velocity < wavy_velocity:
            return FlowPatternCode.STRATIFIED, TaitelDuklerReason.STRATIFIED_SMOOTH
        pattern = (
            FlowPatternCode.STRATIFIED_WAVY
            if self.settings.distinguish_wavy
            else FlowPatternCode.STRATIFIED
        )
        return pattern, TaitelDuklerReason.STRATIFIED_WAVY

    def _unstratified_pattern(
        self,
        geometry: StratifiedGeometry,
        liquid_velocity: float,
        has_equilibrium: bool,
    ) -> tuple[FlowPatternCode, TaitelDuklerReason]:
        """Расслоённое течение неустойчиво или невозможно: кольцевое, дисперсное или прерывистое."""
        fluid = self.fluid
        if geometry.level < self.settings.annular_level_threshold:
            return FlowPatternCode.ANNULAR, (
                TaitelDuklerReason.ANNULAR_LOW_LEVEL
                if has_equilibrium
                else TaitelDuklerReason.NO_EQUILIBRIUM
            )

        friction = liquid_friction_factor(
            geometry=geometry,
            diameter=self.pipe.diameter,
            liquid_velocity=liquid_velocity,
            liquid_density=fluid.density_liquid,
            liquid_viscosity=fluid.viscosity_liquid,
        )
        dispersed_velocity = dispersed_bubble_liquid_velocity(
            geometry=geometry,
            diameter=self.pipe.diameter,
            liquid_density=fluid.density_liquid,
            gas_density=fluid.density_gas,
            liquid_friction=friction,
            angle_rad=self._angle_rad,
        )
        if liquid_velocity >= dispersed_velocity:
            pattern, reason = (
                FlowPatternCode.DISPERSED_BUBBLE,
                TaitelDuklerReason.DISPERSED_BUBBLE,
            )
        else:
            pattern, reason = FlowPatternCode.SLUG, TaitelDuklerReason.INTERMITTENT
        if not has_equilibrium:
            reason = TaitelDuklerReason.NO_EQUILIBRIUM
        return pattern, reason
