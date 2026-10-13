"""Собственные скалярные решатели с контролем сходимости.

Схема: `bisect` (надёжный), `brent` (быстрый, скобка), `newton_bracketed` (Newton с
откатом на бисекцию, если шаг выходит из скобки или производная непригодна).
scipy в новом коде не используется. Несходимость и отсутствие смены знака на скобке
дают `ConvergenceError` с параметрами задачи.

Источник: Brent R.P. «Algorithms for Minimization without Derivatives», 1973, гл. 4;
реализация по Press et al. «Numerical Recipes», zbrent.
"""

import math
import sys
from collections.abc import Callable, Iterator
from typing import Final

from correlations.core.errors import ConvergenceError

_DEFAULT_RTOL: Final = 1e-12
_DEFAULT_ATOL: Final = 1e-14
_EPS: Final = sys.float_info.epsilon

ScalarFunction = Callable[[float], float]


def _check_bracket(f_a: float, f_b: float, a: float, b: float) -> None:
    """Проверяет смену знака функции на скобке [a, b].

    Параметры:
        f_a, f_b: значения функции на концах скобки.
        a, b: концы скобки.

    Исключения:
        ConvergenceError: знак на концах одинаков (корень не заключён в скобку).
    """
    if f_a * f_b > 0.0:
        raise ConvergenceError(
            "на скобке нет смены знака",
            {"a": a, "b": b, "f_a": f_a, "f_b": f_b},
        )


def bisect(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    max_iter: int = 200,
) -> float:
    """Находит корень f(x) = 0 методом бисекции на скобке [a, b].

    Параметры:
        f: функция одной переменной.
        a, b: концы скобки, f(a) и f(b) разного знака (или один из них равен 0).
        rtol, atol: относительная и абсолютная точность по x.
        max_iter: максимальное число делений скобки.

    Возвращает:
        Приближение корня.

    Исключения:
        ConvergenceError: нет смены знака или превышено `max_iter`.
    """
    f_a, f_b = f(a), f(b)
    if f_a == 0.0:
        return a
    if f_b == 0.0:
        return b
    _check_bracket(f_a, f_b, a, b)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        if f_mid == 0.0 or 0.5 * abs(b - a) <= atol + rtol * abs(mid):
            return mid
        if f_a * f_mid < 0.0:
            b = mid
        else:
            a, f_a = mid, f_mid
    raise ConvergenceError(
        "бисекция не сошлась за max_iter итераций",
        {"a": a, "b": b, "max_iter": float(max_iter)},
    )


def brent(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    max_iter: int = 100,
) -> float:
    """Находит корень f(x) = 0 методом Брента на скобке [a, b].

    Параметры:
        f: функция одной переменной.
        a, b: концы скобки, f(a) и f(b) разного знака (или один из них равен 0).
        rtol, atol: относительная и абсолютная точность по x.
        max_iter: максимальное число итераций.

    Возвращает:
        Приближение корня.

    Исключения:
        ConvergenceError: нет смены знака или превышено `max_iter`.
    """
    f_a, f_b = f(a), f(b)
    if f_a == 0.0:
        return a
    if f_b == 0.0:
        return b
    _check_bracket(f_a, f_b, a, b)
    c, f_c = b, f_b
    d = e = b - a
    for _ in range(max_iter):
        if (f_b > 0.0 and f_c > 0.0) or (f_b < 0.0 and f_c < 0.0):
            c, f_c = a, f_a
            d = e = b - a
        if abs(f_c) < abs(f_b):
            a, b, c = b, c, b
            f_a, f_b, f_c = f_b, f_c, f_b
        tol = 2.0 * _EPS * abs(b) + 0.5 * (atol + rtol * abs(b))
        xm = 0.5 * (c - b)
        if abs(xm) <= tol or f_b == 0.0:
            return b
        if abs(e) >= tol and abs(f_a) > abs(f_b):
            s = f_b / f_a
            if a == c:
                p, q = 2.0 * xm * s, 1.0 - s
            else:
                q, r = f_a / f_c, f_b / f_c
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, f_a = b, f_b
        b += d if abs(d) > tol else math.copysign(tol, xm)
        f_b = f(b)
    raise ConvergenceError(
        "метод Брента не сошёлся за max_iter итераций",
        {"a": a, "b": b, "max_iter": float(max_iter)},
    )


def _scan_roots(
    f: ScalarFunction,
    lo: float,
    hi: float,
    n_grid: int,
    rtol: float,
    atol: float,
) -> Iterator[float]:
    """Лениво отдаёт корни f на [lo, hi] по возрастанию: сетка + уточнение методом Брента.

    Параметры:
        f: функция одной переменной.
        lo, hi: границы интервала поиска.
        n_grid: число интервалов равномерной сетки.
        rtol, atol: точность уточнения корня.

    Возвращает:
        Итератор корней по возрастанию (корень в узле сетки отдаётся один раз).
    """
    step = (hi - lo) / n_grid
    x_prev = lo
    f_prev = f(lo)
    if f_prev == 0.0:
        yield lo
    for i in range(1, n_grid + 1):
        x = lo + step * i
        f_x = f(x)
        if f_prev * f_x < 0.0:
            yield brent(f, x_prev, x, rtol=rtol, atol=atol)
        elif f_x == 0.0:
            yield x
        x_prev, f_prev = x, f_x


def bracketed_roots(
    f: ScalarFunction,
    lo: float,
    hi: float,
    *,
    n_grid: int = 400,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> list[float]:
    """Находит все корни f(x) = 0 на [lo, hi], заключённые в интервалы сетки со сменой знака.

    Два корня внутри одного интервала сетки (чётная кратность) не обнаруживаются —
    сетку нужно брать достаточно частой.

    Параметры:
        f: функция одной переменной.
        lo, hi: границы интервала поиска.
        n_grid: число интервалов равномерной сетки.
        rtol, atol: точность уточнения корня.

    Возвращает:
        Корни по возрастанию (пустой список, если корней нет).
    """
    return list(_scan_roots(f, lo, hi, n_grid, rtol, atol))


def smallest_root(
    f: ScalarFunction,
    lo: float,
    hi: float,
    *,
    n_grid: int = 400,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> float | None:
    """Находит наименьший корень f(x) = 0 на [lo, hi] (вопросы 1, 9 ТЗ: «берём наименьший»).

    Параметры:
        f: функция одной переменной.
        lo, hi: границы интервала поиска.
        n_grid: число интервалов равномерной сетки.
        rtol, atol: точность уточнения корня.

    Возвращает:
        Наименьший корень или None, если смены знака на интервале нет.
    """
    return next(_scan_roots(f, lo, hi, n_grid, rtol, atol), None)


def newton_bracketed(
    f: ScalarFunction,
    df: ScalarFunction,
    x0: float,
    bracket: tuple[float, float],
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    max_iter: int = 100,
) -> float:
    """Находит корень f(x) = 0 методом Ньютона с откатом на бисекцию.

    Скобка обновляется на каждой итерации; шаг Ньютона, выходящий за текущую скобку
    (или при нулевой/неконечной производной), заменяется серединой скобки.

    Параметры:
        f: функция одной переменной.
        df: её производная.
        x0: начальное приближение (вне скобки заменяется серединой).
        bracket: скобка (a, b), f(a) и f(b) разного знака (или один из них равен 0).
        rtol, atol: относительная и абсолютная точность по x (по размеру шага).
        max_iter: максимальное число итераций.

    Возвращает:
        Приближение корня.

    Исключения:
        ConvergenceError: нет смены знака на скобке или превышено `max_iter`.
    """
    a, b = sorted(bracket)
    f_a, f_b = f(a), f(b)
    if f_a == 0.0:
        return a
    if f_b == 0.0:
        return b
    _check_bracket(f_a, f_b, a, b)
    x = x0 if a < x0 < b else 0.5 * (a + b)
    for _ in range(max_iter):
        f_x = f(x)
        if f_x == 0.0:
            return x
        if f_a * f_x < 0.0:
            b = x
        else:
            a, f_a = x, f_x
        slope = df(x)
        x_new = x - f_x / slope if slope != 0.0 and math.isfinite(slope) else math.nan
        if not (a < x_new < b):
            x_new = 0.5 * (a + b)
        if abs(x_new - x) <= atol + rtol * abs(x_new):
            return x_new
        x = x_new
    raise ConvergenceError(
        "Ньютон не сошёлся за max_iter итераций",
        {"a": a, "b": b, "x": x, "max_iter": float(max_iter)},
    )
