"""Общий перечень режимов течения газожидкостной смеси.

Схема: один `StrEnum` для всех корреляций и модулей режима. Значение члена — его имя
в нижнем регистре (`FlowPattern.SLUG == "slug"`), что удобно для сериализации.

Источник перечня: ТЗ проекта, раздел «Выход» (режимы Beggs–Brill, Mukherjee–Brill,
Taitel–Dukler, Barnea, Hasan–Kabir, Ansari).
"""

from enum import StrEnum, auto


class FlowPattern(StrEnum):
    """Режим течения.

    SEGREGATED, INTERMITTENT, DISTRIBUTED, TRANSITION — режимы карты Beggs–Brill (1973).
    Остальные — режимы механистических моделей и карты Mukherjee–Brill (1985).
    SINGLE_PHASE_LIQUID / SINGLE_PHASE_GAS — предельные случаи v_sg = 0 / v_sl = 0.
    """

    SEGREGATED = auto()
    INTERMITTENT = auto()
    DISTRIBUTED = auto()
    TRANSITION = auto()
    BUBBLE = auto()
    DISPERSED_BUBBLE = auto()
    SLUG = auto()
    CHURN = auto()
    ANNULAR = auto()
    STRATIFIED = auto()
    STRATIFIED_SMOOTH = auto()
    STRATIFIED_WAVY = auto()
    ELONGATED_BUBBLE = auto()
    SINGLE_PHASE_LIQUID = auto()
    SINGLE_PHASE_GAS = auto()
