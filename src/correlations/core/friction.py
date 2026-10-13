"""Коэффициенты трения.

Соглашение: ядро всегда использует коэффициент трения Moody/Darcy (f_D, τ_w = f_D·ρv²/8).
Корреляции, работающие с Fanning (Taitel–Dukler, Barnea, Hasan–Kabir; f_F = f_D/4),
переводят значение явно через `fanning_to_darcy` / `darcy_to_fanning` — это единственное
место множителя 4.

Источники:
    * 64/Re — закон Хагена–Пуазёйля для ламинарного течения;
    * Colebrook C.F., 1939 (в форме Colebrook–White);
    * Beggs H.D., Brill J.P., 1973, ур. (13) — гладкая труба;
    * Taitel Y., Dukler A.E., 1976 (AIChE J. 22(1)) — Fanning: 16/Re, 0.046·Re^-0.2;
    * Blasius H., 1913 — Fanning: 0.079·Re^-0.25.

Предусловия (Re > 0, 0 ≤ ε/d < 1) не проверяются: вход проверяет `FlowState`.

Допущение (TODO(question): по решению пользователя; предел совпадает с принятым в Ansari для
ε/d_HF): относительная шероховатость в Colebrook ограничивается сверху значением
`MAX_RELATIVE_ROUGHNESS` = 0.05 — верхней границей диаграммы Moody. При превышении выдаётся
`OutOfRangeWarning`, расчёт идёт при ε/d = 0.05 (полностью шероховатый режим, f ≈ 0.072). Без
предела при ε/d более ~3.3 (например, у газовой фазы расслоённого потока с исчезающим сечением)
у уравнения нет вещественного корня и решатель отказывал.
"""

import math
import warnings
from typing import Final

from correlations.core import solvers
from correlations.core.errors import OutOfRangeWarning

# Критическое число Рейнольдса диаграммы Moody: ниже — ламинарный режим, f = 64/Re
RE_LAMINAR_MAX: Final = 2000.0
# Коэффициенты формулы Colebrook–White: 1/√f = −2·lg(ε/(3.7d) + 2.51/(Re√f))
_COLEBROOK_ROUGHNESS_DIVISOR: Final = 3.7
_COLEBROOK_REYNOLDS_COEFFICIENT: Final = 2.51
# Верхний предел относительной шероховатости: граница диаграммы Moody (Moody L.F., 1944)
MAX_RELATIVE_ROUGHNESS: Final = 0.05
# Скобка и старт для решения относительно x = 1/√f (f от ~1e-6 до ~100; при ε/d ≤ 0.05 x > 3.7)
_COLEBROOK_BRACKET: Final = (0.1, 1.0e3)
_COLEBROOK_START: Final = 1.0
# Beggs–Brill 1973, ур. (13): f = [2·lg(Re/(4.5223·lg Re − 3.8215))]^-2
_BB_SMOOTH_A: Final = 4.5223
_BB_SMOOTH_B: Final = 3.8215
# Taitel–Dukler 1976: f = C·Re^-n; турбулентный C = 0.046, n = 0.2; ламинарный C = 16, n = 1
_TD_TURBULENT_C: Final = 0.046
_TD_TURBULENT_N: Final = 0.2
_TD_LAMINAR_C: Final = 16.0
# Blasius: f = 0.079·Re^-0.25
_BLASIUS_C: Final = 0.079
_BLASIUS_N: Final = 0.25
# Darcy = 4 · Fanning
_DARCY_PER_FANNING: Final = 4.0


def laminar_darcy(re: float) -> float:
    """Коэффициент трения Darcy для ламинарного течения f = 64/Re.

    Параметры:
        re: число Рейнольдса, безразм. (> 0).

    Возвращает:
        Коэффициент трения Darcy, безразм.
    """
    return 64.0 / re


def colebrook(re: float, rel_roughness: float) -> float:
    """Коэффициент трения Darcy по уравнению Colebrook–White.

    Решает 1/√f = −2·lg(ε/(3.7·d) + 2.51/(Re·√f)) относительно x = 1/√f методом
    Ньютона с откатом на бисекцию (`solvers.newton_bracketed`).

    Параметры:
        re: число Рейнольдса, безразм. (> 0).
        rel_roughness: относительная шероховатость ε/d, безразм. (0 — гладкая труба); значения
            выше `MAX_RELATIVE_ROUGHNESS` (0.05) заменяются на этот предел с `OutOfRangeWarning`.

    Возвращает:
        Коэффициент трения Darcy, безразм.

    Исключения:
        ConvergenceError: решатель не сошёлся.

    Предупреждения:
        OutOfRangeWarning: ε/d выше `MAX_RELATIVE_ROUGHNESS`, расчёт выполнен при ε/d = 0.05.
    """
    if rel_roughness > MAX_RELATIVE_ROUGHNESS:
        warnings.warn(
            f"ε/d = {rel_roughness:g} выше предела диаграммы Moody "
            f"({MAX_RELATIVE_ROUGHNESS:g}): коэффициент трения рассчитан при ε/d = "
            f"{MAX_RELATIVE_ROUGHNESS:g}",
            OutOfRangeWarning,
            stacklevel=2,
        )
        rel_roughness = MAX_RELATIVE_ROUGHNESS
    base = rel_roughness / _COLEBROOK_ROUGHNESS_DIVISOR
    c_re = _COLEBROOK_REYNOLDS_COEFFICIENT / re

    def residual(x: float) -> float:
        return x + 2.0 * math.log10(base + c_re * x)

    def derivative(x: float) -> float:
        return 1.0 + 2.0 * c_re / (math.log(10.0) * (base + c_re * x))

    x = solvers.newton_bracketed(residual, derivative, _COLEBROOK_START, _COLEBROOK_BRACKET)
    return 1.0 / (x * x)


def bb_smooth(re: float) -> float:
    """Коэффициент трения Darcy гладкой трубы по Beggs–Brill 1973, ур. (13).

    Параметры:
        re: число Рейнольдса, безразм. (> 0, ориентир Re > 3000).

    Возвращает:
        Коэффициент трения Darcy, безразм.
    """
    log_re = math.log10(re)
    return (2.0 * math.log10(re / (_BB_SMOOTH_A * log_re - _BB_SMOOTH_B))) ** -2


def moody(re: float, rel_roughness: float) -> float:
    """Коэффициент трения Moody (Darcy): 64/Re при Re < 2000, иначе Colebrook.

    Параметры:
        re: число Рейнольдса, безразм. (> 0).
        rel_roughness: относительная шероховатость ε/d, безразм.

    Возвращает:
        Коэффициент трения Darcy, безразм.
    """
    if re < RE_LAMINAR_MAX:
        return laminar_darcy(re)
    return colebrook(re, rel_roughness)


def fanning_taitel_dukler(re: float, *, laminar: bool) -> float:
    """Коэффициент трения Fanning по Taitel–Dukler 1976 (гладкая труба).

    Параметры:
        re: число Рейнольдса, безразм. (> 0).
        laminar: True — ламинарный режим (f = 16/Re), False — турбулентный
            (f = 0.046·Re^-0.2).

    Возвращает:
        Коэффициент трения Fanning (= Darcy/4), безразм. В ядре переводить через
        `fanning_to_darcy`.
    """
    if laminar:
        return _TD_LAMINAR_C / re
    return _TD_TURBULENT_C * re**-_TD_TURBULENT_N


def fanning_blasius(re: float) -> float:
    """Коэффициент трения Fanning по Blasius: f = 0.079·Re^-0.25.

    Применима для гладкой трубы при 4·10³ ≲ Re ≲ 10⁵.

    Параметры:
        re: число Рейнольдса, безразм. (> 0).

    Возвращает:
        Коэффициент трения Fanning (= Darcy/4), безразм.
    """
    return _BLASIUS_C * re**-_BLASIUS_N


def fanning_to_darcy(f_fanning: float) -> float:
    """Переводит коэффициент трения Fanning в Darcy (Moody): f_D = 4·f_F.

    Параметры:
        f_fanning: коэффициент трения Fanning, безразм.

    Возвращает:
        Коэффициент трения Darcy, безразм.
    """
    return _DARCY_PER_FANNING * f_fanning


def darcy_to_fanning(f_darcy: float) -> float:
    """Переводит коэффициент трения Darcy (Moody) в Fanning: f_F = f_D/4.

    Параметры:
        f_darcy: коэффициент трения Darcy, безразм.

    Возвращает:
        Коэффициент трения Fanning, безразм.
    """
    return f_darcy / _DARCY_PER_FANNING
