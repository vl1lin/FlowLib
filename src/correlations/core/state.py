"""Вход и выход расчётов: `FlowState`, `FlowPatternResult`, `CorrelationResult`.

Все величины — в СИ, модели неизменяемы. Нарушение условий входа даёт
`pydantic.ValidationError`.

Источник требований: ТЗ проекта, разделы «Вход FlowState» и «Выход».
"""

import math
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from correlations.core.patterns import FlowPattern

# Относительная точность проверки суммы компонент градиента (ТЗ, «Выход»).
_SUM_RTOL: Final = 1e-10
# Абсолютный допуск только для случая, когда все компоненты ≈ 0.
_SUM_ATOL: Final = 1e-12


class FlowState(BaseModel):
    """Локальное состояние потока в сечении трубы. Все величины в СИ.

    Условия согласованности: ρ_L > ρ_g; d_t < d; ε < d; v_sl + v_sg > 0.
    Однофазные пределы (v_sl = 0 или v_sg = 0) допустимы.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    v_sl_m_s: float = Field(ge=0, description="Приведённая скорость жидкости, м/с")
    v_sg_m_s: float = Field(ge=0, description="Приведённая скорость газа, м/с")
    rho_l_kg_m3: float = Field(gt=0, description="Плотность жидкости, кг/м³")
    rho_g_kg_m3: float = Field(gt=0, description="Плотность газа, кг/м³")
    mu_l_pa_s: float = Field(gt=0, description="Вязкость жидкости, Па·с")
    mu_g_pa_s: float = Field(gt=0, description="Вязкость газа, Па·с")
    sigma_n_m: float = Field(gt=0, description="Поверхностное натяжение, Н/м")
    d_m: float = Field(gt=0, description="Внутр. диаметр трубы (затрубье — обсадной), м")
    d_t_m: float = Field(default=0.0, ge=0, description="Наруж. диаметр НКТ, 0 — труба, м")
    eps_m: float = Field(default=0.0, ge=0, description="Абс. шероховатость, м")
    theta_deg: float = Field(ge=-90, le=90, description="Угол от горизонтали, > 0 — вверх, град")
    p_pa: float = Field(gt=0, description="Давление, Па")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Проверяет согласованность полей: ρ_L > ρ_g, d_t < d, ε < d, v_sl + v_sg > 0."""
        if self.rho_l_kg_m3 <= self.rho_g_kg_m3:
            raise ValueError("плотность жидкости должна быть больше плотности газа")
        if self.d_t_m >= self.d_m:
            raise ValueError("наружный диаметр НКТ d_t должен быть меньше диаметра d")
        if self.eps_m >= self.d_m:
            raise ValueError("шероховатость eps должна быть меньше диаметра d")
        if self.v_sl_m_s + self.v_sg_m_s <= 0:
            raise ValueError("сумма приведённых скоростей должна быть положительной")
        return self


class FlowPatternResult(BaseModel):
    """Результат модуля режима потока.

    Атрибуты:
        pattern: определённый режим.
        details: промежуточные величины (границы, h̃_L, α_s и т.п.); ключи с единицами,
            например `h_l_over_d`, `v_sg_annular_m_s`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: FlowPattern
    details: dict[str, float]


class CorrelationResult(BaseModel):
    """Результат полной корреляции: режим, holdup и градиент давления.

    Атрибуты:
        pattern: режим течения.
        holdup: истинное объёмное содержание жидкости H_L, 0…1.
        dpdl_pa_m: −dp/dL, Па/м (> 0 — давление падает по потоку).
        dpdl_gravity_pa_m: гравитационная составляющая, Па/м.
        dpdl_friction_pa_m: составляющая трения, Па/м.
        dpdl_acceleration_pa_m: составляющая ускорения, Па/м.
        details: промежуточные величины (ключи с единицами).

    Проверка: dpdl = gravity + friction + acceleration с rtol 1e-10.
    Для формы с Ek (BB, ур. 1) ускорение = dpdl − (gravity + friction).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: FlowPattern
    holdup: float = Field(ge=0, le=1)
    dpdl_pa_m: float
    dpdl_gravity_pa_m: float
    dpdl_friction_pa_m: float
    dpdl_acceleration_pa_m: float
    details: dict[str, float]

    @model_validator(mode="after")
    def _check_components_sum(self) -> Self:
        """Проверяет, что полный градиент равен сумме компонент (rtol 1e-10)."""
        total = self.dpdl_gravity_pa_m + self.dpdl_friction_pa_m + self.dpdl_acceleration_pa_m
        if not math.isclose(self.dpdl_pa_m, total, rel_tol=_SUM_RTOL, abs_tol=_SUM_ATOL):
            raise ValueError(
                f"dpdl_pa_m={self.dpdl_pa_m} не равно сумме компонент {total} (rtol {_SUM_RTOL})"
            )
        return self
