"""
Механистическая модель Hasan–Kabir (1988) для восходящего потока в вертикальных и
наклонных скважинах.

Источники:

* Hasan A.R., Kabir C.S. A study of multiphase flow behavior in vertical wells //
  SPE Prod. Eng. 1988;
* Hasan A.R., Kabir C.S. Predicting multiphase flow behavior in a deviated well //
  SPE Prod. Eng. 1988;
* Брилл Дж. П., Мукерджи Х. «Многофазный поток в скважинах», §4.2.2 «Метод Хасана и
  Кабира», ур. 4.160, 4.163, 4.240–4.243, пример 4.10.

Диапазон применимости: восходящий поток, углы от 45° до 90° от горизонтали
(`angle_limit()`). Нижняя граница 45° — консервативный выбор; с оригинальной работой
Hasan & Kabir (1988, deviated wells) сверка не выполнялась, значение без согласования
не менять. Реализовано только определение режима.

Возвращаемые режимы: BUBBLE (пузырьковый), SLUG (пробковый), CHURN (эмульсионный),
DISPERSED_BUBBLE (дисперсно-пузырьковый), ANNULAR (кольцевой), а также SINGLE_GAS и
SINGLE_LIQUID для однофазных случаев.

Порядок проверок: дисперсно-пузырьковый → кольцевой (Тёрнер, опционально критерии
Barnea) → эмульсионный → пузырьковый → пробковый.

Ограничения и неоднозначности источника:

* C_0 (ур. 4.241). В книге условия записаны с «или» и перекрываются
  (d < 0.12 или v_SL > 0.02 → 1.2; d > 0.12 или v_SL < 0.02 → 2.0). Принято: 2.0 только
  при d > 0.12 м и v_SL < 0.02 м/с. Это согласуется с примером 4.10
  (d = 0.1524, v_SL = 1.208 → C_0 = 1.2).
* Эмульсионный режим (ур. 4.243 и текст). По книге переход «пробковый → эмульсионный»
  — по Barnea & Brauner: газосодержание в пробке достигает 0.52 при той же
  турбулентности, что на границе дисперсно-пузырькового режима. Отсюда:
  v_m ≥ v_m,DB при λ_G > 0.52. Использование расходного λ_G вместо истинного
  газосодержания — упрощение.
* Опечатка в примере 4.10: в числовой подстановке ур. 4.243 пропущен показатель 0.08
  у (ρ_L/μ_L), хотя результат 4.401 посчитан с ним.
* Кольцевой режим. По книге переход только по Тёрнеру и не зависит от дебита жидкости.
  В примере 4.10 проверка кольцевого режима пропущена: при v_Sg = 1.173 > v_Sg,T = 0.87
  модель по своей логике даёт ANNULAR, а книга пишет «пробковый». По умолчанию
  (`use_barnea_annular_criteria=False`) реализована логика модели (ANNULAR).
* Флаг `use_barnea_annular_criteria=True` — необязательная модификация для сравнения,
  не часть оригинальной модели Hasan–Kabir. При v_Sg > v_Sg,T дополнительно проверяются
  критерии Barnea в формулировке Ансари (ур. 4.165 и 4.169): перекрытие сечения и
  неустойчивость плёнки. Модель плёнки (F_E, λ_LC, X_M, Y_M, δ; ур. 4.204–4.233)
  импортируется из `correlations/ansari.py` (`solve_film`, `is_film_blocking`,
  `is_film_unstable`), параметры (порог 0.12, f_F/f_SL = 1) — по умолчанию из
  `AnsariSettings`. Если кольцевой режим отвергнут, классификация продолжается
  шагами «эмульсионный» → «пузырьковый» → «пробковый». С флагом пример 4.10 даёт SLUG,
  как в книге. Для этого нужна вязкость газа μ_G из `FluidParams`.
* Критерий Тёрнера без sin θ (как в ур. 4.163). Для 45–90° разница меньше 10%;
  настройка `turner_uses_inclination`.
* Если v_TB ≤ v_s (малые диаметры), пузырьковый режим невозможен: пузыри догоняют пузырь
  Тейлора и сливаются с ним (reason BUBBLE_IMPOSSIBLE_SMALL_PIPE).
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Final

from correlations.ansari import (
    AnsariSettings,
    is_film_blocking,
    is_film_unstable,
    solve_film,
)
from correlations.base import (
    ClassificationResult,
    FlowPatternCode,
    IFlowModel,
    check_velocities,
    single_phase_pattern,
)
from correlations.constants import GRAVITY
from correlations.domain.params import FluidParams, PipeParams

# Скорость подъёма пузыря по Хармати, ур. 4.160
_HARMATHY_COEFFICIENT: Final = 1.53
# Скорость пузыря Тейлора, ур. 4.242
_TAYLOR_COEFFICIENT: Final = 0.35
_TAYLOR_INCLINATION_EXPONENT: Final = 1.2
# Критерий Тёрнера, ур. 4.163
_TURNER_COEFFICIENT: Final = 3.1
# Граница дисперсно-пузырькового режима, ур. 4.243 (СИ)
_DISPERSED_COEFFICIENT: Final = 4.68
_DISPERSED_DIAMETER_EXPONENT: Final = 0.48
_DISPERSED_SURFACE_EXPONENT: Final = 0.6
_DISPERSED_VISCOSITY_EXPONENT: Final = 0.08
_DISPERSED_MIXTURE_EXPONENT: Final = 1.12
# Коэффициент распределения C_0, ур. 4.241
_DISTRIBUTION_COEFFICIENT_LARGE_PIPE: Final = 2.0
_DISTRIBUTION_COEFFICIENT_DEFAULT: Final = 1.2
# Граница пузырьковый/пробковый, ур. 4.240
_BUBBLE_SLUG_DENOMINATOR: Final = 4.0


class HasanKabirReason(Enum):
    """
    Критерий, по которому получен режим.

    BUBBLE_IMPOSSIBLE_SMALL_PIPE — reason для SLUG, когда v_Sg < v_Sg,BS, но
    v_TB ≤ v_s (пузырьковый режим невозможен в малом диаметре).
    """

    SINGLE_PHASE = auto()
    DISPERSED_BUBBLE = auto()
    ANNULAR = auto()
    CHURN = auto()
    BUBBLE = auto()
    SLUG = auto()
    BUBBLE_IMPOSSIBLE_SMALL_PIPE = auto()


class AnnularRejection(Enum):
    """Почему кольцевой режим отвергнут при `use_barnea_annular_criteria=True`."""

    BLOCKAGE = auto()  # перекрытие сечения, ур. 4.165
    FILM_INSTABILITY = auto()  # неустойчивость плёнки, ур. 4.169
    NO_FILM_SOLUTION = auto()  # плёнка не удерживается


@dataclass(frozen=True, slots=True)
class HasanKabirSettings:
    """
    Настройки модели Hasan–Kabir.

    Attributes
    ----------
    dispersed_max_gas_fraction: float
        Максимальная расходная доля газа λ_G дисперсно-пузырькового режима
        (плотная упаковка); выше неё при v_m ≥ v_m,DB течение эмульсионное
    large_diameter_threshold: float
        Диаметр, м, выше которого C_0 = 2.0 при малом v_SL (ур. 4.241)
    low_liquid_velocity_threshold: float
        Приведённая скорость жидкости, м/с, ниже которой C_0 = 2.0 в большой трубе
        (ур. 4.241)
    turner_uses_inclination: bool
        Умножать g на sin θ в критерии Тёрнера; по умолчанию нет, как в ур. 4.163
    use_barnea_annular_criteria: bool
        Учитывать ли критерии Barnea при переходе в кольцевой режим.
        False — только Тёрнер, ур. 4.163 (оригинальная модель, по умолчанию);
        True — Тёрнер + критерии Barnea (перекрытие сечения и устойчивость плёнки) по
        модели плёнки Ансари. Необязательная модификация для сравнения.
    """

    dispersed_max_gas_fraction: float = 0.52
    large_diameter_threshold: float = 0.12
    low_liquid_velocity_threshold: float = 0.02
    turner_uses_inclination: bool = False
    use_barnea_annular_criteria: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.dispersed_max_gas_fraction <= 1.0:
            raise ValueError("dispersed_max_gas_fraction должен быть в (0, 1]")
        if not self.large_diameter_threshold > 0.0:
            raise ValueError("large_diameter_threshold должен быть > 0")
        if not self.low_liquid_velocity_threshold > 0.0:
            raise ValueError("low_liquid_velocity_threshold должен быть > 0")


@dataclass(frozen=True, slots=True)
class HasanKabirResult(ClassificationResult):
    """
    Результат классификации.

    Attributes
    ----------
    pattern: FlowPatternCode
        Режим течения (унаследовано от ClassificationResult)
    reason: HasanKabirReason
        Критерий, по которому получен режим
    distribution_coefficient: float
        C_0 (ур. 4.241)
    bubble_slug_gas_velocity: float
        v_Sg,BS, м/с (ур. 4.240)
    dispersed_mixture_velocity: float
        v_m,DB, м/с (ур. 4.243)
    turner_gas_velocity: float
        v_Sg,T, м/с (ур. 4.163)
    taylor_bubble_velocity: float
        v_TB, м/с (ур. 4.242)
    annular_rejection: AnnularRejection | None
        Причина отказа в кольцевом режиме; только при
        `use_barnea_annular_criteria=True` и v_Sg > v_Sg,T
    film_blockage_value: float | None
        H_LF + λ_LC·(1 − 2δ)², если считалась плёнка
    """

    reason: HasanKabirReason
    distribution_coefficient: float
    bubble_slug_gas_velocity: float
    dispersed_mixture_velocity: float
    turner_gas_velocity: float
    taylor_bubble_velocity: float
    annular_rejection: AnnularRejection | None = None
    film_blockage_value: float | None = None


def bubble_rise_velocity(
    *, liquid_density: float, gas_density: float, surface_tension: float
) -> float:
    """
    Скорость подъёма пузыря в неподвижной жидкости по Хармати (ур. 4.160):
    v_s = 1.53·[g·σ·Δρ/ρ_L²]^0.25, м/с.

    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param surface_tension: поверхностное натяжение σ, Н/м
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


def taylor_bubble_velocity(
    *,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    sin_angle: float,
    cos_angle: float,
) -> float:
    """
    Скорость подъёма пузыря Тейлора (ур. 4.242):
    v_TB = 0.35·[g·d·Δρ/ρ_L]^0.5·(sin θ)^0.5·(1 + cos θ)^1.2, м/с.

    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param sin_angle: sin θ, ≥ 0
    :param cos_angle: cos θ
    """
    return (
        _TAYLOR_COEFFICIENT
        * math.sqrt(
            GRAVITY * diameter * (liquid_density - gas_density) / liquid_density
        )
        * math.sqrt(sin_angle)
        * (1.0 + cos_angle) ** _TAYLOR_INCLINATION_EXPONENT
    )


def turner_gas_velocity(
    *,
    liquid_density: float,
    gas_density: float,
    surface_tension: float,
    inclination_factor: float = 1.0,
) -> float:
    """
    Критерий Тёрнера (ур. 4.163): v_Sg,T = 3.1·[g·σ·Δρ/ρ_G²]^0.25, м/с
    (в книге без sin θ).

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


def dispersed_mixture_velocity(
    *,
    diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    surface_tension: float,
) -> float:
    """
    Минимальная скорость смеси дисперсно-пузырькового режима (ур. 4.243, СИ):
    v_m,DB = {4.68·d^0.48·[g·Δρ/σ]^0.5·(σ/ρ_L)^0.6·(ρ_L/μ_L)^0.08}^(1/1.12), м/с.

    :param diameter: диаметр трубы d, м
    :param liquid_density: плотность жидкости ρ_L, кг/м³
    :param gas_density: плотность газа ρ_G, кг/м³
    :param liquid_viscosity: вязкость жидкости μ_L, Па·с
    :param surface_tension: поверхностное натяжение σ, Н/м
    """
    base = (
        _DISPERSED_COEFFICIENT
        * diameter**_DISPERSED_DIAMETER_EXPONENT
        * math.sqrt(GRAVITY * (liquid_density - gas_density) / surface_tension)
        * (surface_tension / liquid_density) ** _DISPERSED_SURFACE_EXPONENT
        * (liquid_density / liquid_viscosity) ** _DISPERSED_VISCOSITY_EXPONENT
    )
    return base ** (1.0 / _DISPERSED_MIXTURE_EXPONENT)


def distribution_coefficient(
    *,
    diameter: float,
    liquid_velocity: float,
    large_diameter_threshold: float,
    low_liquid_velocity_threshold: float,
) -> float:
    """
    Коэффициент распределения C_0 (ур. 4.241): 2.0 при d > 0.12 м **и**
    v_SL < 0.02 м/с, иначе 1.2 (условия книги с «или» перекрываются; принято «и»).

    :param diameter: диаметр трубы d, м
    :param liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param large_diameter_threshold: порог диаметра, м (0.12)
    :param low_liquid_velocity_threshold: порог скорости жидкости, м/с (0.02)
    """
    if (
        diameter > large_diameter_threshold
        and liquid_velocity < low_liquid_velocity_threshold
    ):
        return _DISTRIBUTION_COEFFICIENT_LARGE_PIPE
    return _DISTRIBUTION_COEFFICIENT_DEFAULT


def bubble_slug_gas_velocity(
    *,
    sin_angle: float,
    distribution_coefficient: float,
    liquid_velocity: float,
    rise_velocity: float,
) -> float:
    """
    Граница пузырьковый/пробковый режимы (ур. 4.240):
    v_Sg,BS = sin θ/(4 − C_0)·(C_0·v_SL + v_s), м/с.

    :param sin_angle: sin θ
    :param distribution_coefficient: C_0
    :param liquid_velocity: приведённая скорость жидкости v_SL, м/с
    :param rise_velocity: скорость подъёма пузыря v_s, м/с
    """
    return (
        sin_angle
        / (_BUBBLE_SLUG_DENOMINATOR - distribution_coefficient)
        * (distribution_coefficient * liquid_velocity + rise_velocity)
    )


@dataclass(frozen=True, slots=True)
class _Invariants:
    """Величины, не зависящие от (v_SL, v_Sg)."""

    sin_angle: float
    rise_velocity: float  # v_s
    taylor_velocity: float  # v_TB
    turner_velocity: float  # v_Sg,T
    dispersed_velocity: float  # v_m,DB
    relative_roughness: float
    film_settings: AnsariSettings


class HasanKabirModel(IFlowModel):
    """
    Модель Hasan–Kabir для восходящего потока (45°–90°).

    Порядок проверок: дисперсно-пузырьковый (v_m ≥ v_m,DB и λ_G ≤ 0.52) → кольцевой
    (v_Sg > v_Sg,T; опционально + критерии Barnea) → эмульсионный (v_m ≥ v_m,DB и
    λ_G > 0.52) → пузырьковый (v_TB > v_s и v_Sg < v_Sg,BS) → пробковый.
    Диапазон применимости, ограничения и неоднозначности источника — в docstring модуля.

    :param pipe: Объект PipeParams, в котором информация о трубе
    :param fluid: Объект FluidParams, в котором информация о свойствах флюидов в потоке
    :param settings: Настройки модели (по умолчанию HasanKabirSettings())
    """

    def __init__(
        self,
        pipe: PipeParams,
        fluid: FluidParams,
        *,
        settings: HasanKabirSettings | None = None,
    ):
        super().__init__(pipe, fluid)
        self.settings = settings if settings is not None else HasanKabirSettings()

    name = "Hasan-Kabir"
    angle_range = (45.0, 90.0)

    @cached_property
    def _invariants(self) -> _Invariants:
        """
        Величины, зависящие только от трубы и флюида (PipeParams/FluidParams
        неизменяемы). Угол уже проверен `IFlowModel.__init__` по `angle_range`
        (45°–90°), поэтому здесь его отдельно не проверяем.
        """
        angle = self.pipe.angle
        fluid = self.fluid
        angle_rad = math.radians(angle)
        sin_angle = math.sin(angle_rad)
        fluid_kwargs = {
            "liquid_density": fluid.density_liquid,
            "gas_density": fluid.density_gas,
        }
        return _Invariants(
            sin_angle=sin_angle,
            rise_velocity=bubble_rise_velocity(
                **fluid_kwargs, surface_tension=fluid.surface_tension
            ),
            taylor_velocity=taylor_bubble_velocity(
                diameter=self.pipe.diameter,
                **fluid_kwargs,
                sin_angle=sin_angle,
                cos_angle=math.cos(angle_rad),
            ),
            turner_velocity=turner_gas_velocity(
                **fluid_kwargs,
                surface_tension=fluid.surface_tension,
                inclination_factor=(
                    sin_angle if self.settings.turner_uses_inclination else 1.0
                ),
            ),
            dispersed_velocity=dispersed_mixture_velocity(
                diameter=self.pipe.diameter,
                **fluid_kwargs,
                liquid_viscosity=fluid.viscosity_liquid,
                surface_tension=fluid.surface_tension,
            ),
            relative_roughness=self.pipe.roughness / self.pipe.diameter,
            film_settings=AnsariSettings(),
        )

    def classify(self, vsl: float, vsg: float) -> HasanKabirResult:
        """
        Определяет режим течения и критерий перехода (для диагностики).

        :param vsl: приведённая скорость жидкости v_SL, м/с
        :param vsg: приведённая скорость газа v_Sg, м/с
        :raises ValueError: если скорость отрицательная или не конечная
        """
        check_velocities(vsl, vsg)
        invariants = self._invariants
        settings = self.settings
        coefficient = distribution_coefficient(
            diameter=self.pipe.diameter,
            liquid_velocity=vsl,
            large_diameter_threshold=settings.large_diameter_threshold,
            low_liquid_velocity_threshold=settings.low_liquid_velocity_threshold,
        )
        bubble_slug_velocity = bubble_slug_gas_velocity(
            sin_angle=invariants.sin_angle,
            distribution_coefficient=coefficient,
            liquid_velocity=vsl,
            rise_velocity=invariants.rise_velocity,
        )

        def result(
            pattern: FlowPatternCode,
            reason: HasanKabirReason,
            rejection: AnnularRejection | None = None,
            blockage_value: float | None = None,
        ) -> HasanKabirResult:
            return HasanKabirResult(
                pattern=pattern,
                reason=reason,
                distribution_coefficient=coefficient,
                bubble_slug_gas_velocity=bubble_slug_velocity,
                dispersed_mixture_velocity=invariants.dispersed_velocity,
                turner_gas_velocity=invariants.turner_velocity,
                taylor_bubble_velocity=invariants.taylor_velocity,
                annular_rejection=rejection,
                film_blockage_value=blockage_value,
            )

        single_phase = single_phase_pattern(vsl, vsg)
        if single_phase is not None:
            return result(single_phase, HasanKabirReason.SINGLE_PHASE)

        mixture_velocity = vsl + vsg
        gas_fraction = vsg / mixture_velocity
        turbulent = mixture_velocity >= invariants.dispersed_velocity
        if turbulent and gas_fraction <= settings.dispersed_max_gas_fraction:
            return result(
                FlowPatternCode.DISPERSED_BUBBLE, HasanKabirReason.DISPERSED_BUBBLE
            )

        rejection: AnnularRejection | None = None
        blockage_value: float | None = None
        if vsg > invariants.turner_velocity:
            if not settings.use_barnea_annular_criteria:
                return result(FlowPatternCode.ANNULAR, HasanKabirReason.ANNULAR)
            rejection, blockage_value = self._check_film(vsl, vsg)
            if rejection is None:
                return result(
                    FlowPatternCode.ANNULAR,
                    HasanKabirReason.ANNULAR,
                    blockage_value=blockage_value,
                )

        if turbulent:
            return result(
                FlowPatternCode.CHURN,
                HasanKabirReason.CHURN,
                rejection,
                blockage_value,
            )

        if vsg < bubble_slug_velocity:
            if invariants.taylor_velocity > invariants.rise_velocity:
                return result(
                    FlowPatternCode.BUBBLE,
                    HasanKabirReason.BUBBLE,
                    rejection,
                    blockage_value,
                )
            return result(
                FlowPatternCode.SLUG,
                HasanKabirReason.BUBBLE_IMPOSSIBLE_SMALL_PIPE,
                rejection,
                blockage_value,
            )
        return result(
            FlowPatternCode.SLUG, HasanKabirReason.SLUG, rejection, blockage_value
        )

    def _check_film(
        self, vsl: float, vsg: float
    ) -> tuple[AnnularRejection | None, float | None]:
        """
        Критерии Barnea в формулировке Ансари (ур. 4.165, 4.169) по модели плёнки Ансари.

        :return: (причина отказа или None, если кольцевой режим устойчив;
            H_LF + λ_LC·(1 − 2δ)², если плёнка найдена)
        """
        invariants = self._invariants
        fluid = self.fluid
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
            friction_ratio=invariants.film_settings.film_friction_ratio,
        )
        if film.delta is None:
            return AnnularRejection.NO_FILM_SOLUTION, None
        if is_film_blocking(
            film=film, threshold=invariants.film_settings.blockage_threshold
        ):
            return AnnularRejection.BLOCKAGE, film.blockage_value
        if is_film_unstable(film=film):
            return AnnularRejection.FILM_INSTABILITY, film.blockage_value
        return None, film.blockage_value
