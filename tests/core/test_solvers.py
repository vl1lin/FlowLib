"""Тесты `core/solvers.py` — собственные решатели (scipy в новом коде не используется).

Ожидаемый API (`from correlations.core import solvers`), `f: Callable[[float], float]`:
    bisect(f, a, b, *, rtol=1e-12, atol=1e-14, max_iter=200) -> float
    brent(f, a, b, *, rtol=1e-12, atol=1e-14, max_iter=100) -> float
    newton_bracketed(f, df, x0, bracket, *, rtol=1e-12, atol=1e-14, max_iter=100) -> float
        Newton; шаг, выходящий из `bracket` или не уменьшающий скобку, заменяется бисекцией.
Несходимость / нет смены знака на скобке -> `ConvergenceError` с параметрами задачи.
"""

import math
from collections.abc import Callable

import pytest

from correlations.core import solvers
from correlations.core.errors import ConvergenceError

SQRT2 = math.sqrt(2.0)


def square_minus_two(x: float) -> float:
    return x * x - 2.0


class CountingFunction:
    """Обёртка, считающая число вызовов функции."""

    def __init__(self, f: Callable[[float], float]) -> None:
        self._f = f
        self.calls = 0

    def __call__(self, x: float) -> float:
        self.calls += 1
        return self._f(x)


@pytest.mark.parametrize("solver", [solvers.bisect, solvers.brent])
def test_bracketing_solvers_find_sqrt2(solver: Callable[..., float]) -> None:
    assert solver(square_minus_two, 0.0, 2.0) == pytest.approx(SQRT2, abs=1e-10)


@pytest.mark.parametrize("solver", [solvers.bisect, solvers.brent])
def test_bracketing_solvers_accept_reversed_sign_function(solver: Callable[..., float]) -> None:
    assert solver(lambda x: 2.0 - x * x, 0.0, 2.0) == pytest.approx(SQRT2, abs=1e-10)


@pytest.mark.parametrize("solver", [solvers.bisect, solvers.brent])
def test_root_at_bracket_end_returned_exactly(solver: Callable[..., float]) -> None:
    assert solver(lambda x: x - 1.0, 1.0, 3.0) == 1.0
    assert solver(lambda x: x - 3.0, 1.0, 3.0) == 3.0


@pytest.mark.parametrize("solver", [solvers.bisect, solvers.brent])
def test_no_sign_change_raises_convergence_error(solver: Callable[..., float]) -> None:
    with pytest.raises(ConvergenceError) as info:
        solver(square_minus_two, 2.0, 3.0)
    assert {"a", "b", "f_a", "f_b"} <= set(info.value.params)
    assert info.value.params["a"] == 2.0
    assert info.value.params["b"] == 3.0


def test_bisect_max_iter_exceeded_raises() -> None:
    with pytest.raises(ConvergenceError) as info:
        solvers.bisect(square_minus_two, 0.0, 2.0, rtol=1e-15, atol=1e-300, max_iter=3)
    assert info.value.params["max_iter"] == 3


def test_brent_max_iter_exceeded_raises() -> None:
    with pytest.raises(ConvergenceError) as info:
        solvers.brent(square_minus_two, 0.0, 2.0, rtol=1e-15, atol=1e-300, max_iter=2)
    assert info.value.params["max_iter"] == 2


def test_brent_needs_fewer_evaluations_than_bisect() -> None:
    counted_bisect = CountingFunction(square_minus_two)
    counted_brent = CountingFunction(square_minus_two)
    solvers.bisect(counted_bisect, 0.0, 2.0)
    solvers.brent(counted_brent, 0.0, 2.0)
    assert counted_brent.calls < counted_bisect.calls


def test_solution_tolerance_is_respected() -> None:
    x = solvers.brent(square_minus_two, 0.0, 2.0, rtol=1e-6, atol=1e-6)
    assert abs(x - SQRT2) <= 1e-5


def test_newton_converges_quadratically() -> None:
    counted = CountingFunction(square_minus_two)
    x = solvers.newton_bracketed(counted, lambda x: 2.0 * x, 1.0, (0.0, 2.0))
    assert x == pytest.approx(SQRT2, abs=1e-12)
    assert counted.calls <= 12


def cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def cbrt_derivative(x: float) -> float:
    return (1.0 / 3.0) * abs(x) ** (-2.0 / 3.0) if x != 0.0 else math.inf


def test_newton_falls_back_to_bisection_when_step_leaves_bracket() -> None:
    """Для f = ∛x чистый Newton расходится (x -> −2x); с откатом на бисекцию корень найден."""
    x = solvers.newton_bracketed(cbrt, cbrt_derivative, 1.0, (-1.0, 2.0))
    assert x == pytest.approx(0.0, abs=1e-9)


def test_newton_zero_derivative_falls_back_to_bisection() -> None:
    # f = x³ − 1 при x0 = 0: f' = 0 — шаг Newton невозможен
    x = solvers.newton_bracketed(lambda x: x**3 - 1.0, lambda x: 3.0 * x * x, 0.0, (-2.0, 3.0))
    assert x == pytest.approx(1.0, abs=1e-10)


def test_newton_invalid_bracket_raises() -> None:
    with pytest.raises(ConvergenceError) as info:
        solvers.newton_bracketed(square_minus_two, lambda x: 2.0 * x, 3.0, (2.0, 4.0))
    assert {"a", "b"} <= set(info.value.params)


# --- поиск корней на сетке (нужен TD/Barnea/Ansari: «берём наименьший корень», вопросы 1, 9 ТЗ) ---
def cubic(x: float) -> float:
    return (x - 0.2) * (x - 0.5) * (x - 0.9)


def test_bracketed_roots_finds_all_roots_in_ascending_order() -> None:
    roots = solvers.bracketed_roots(cubic, 0.0, 1.0)
    assert roots == pytest.approx([0.2, 0.5, 0.9], abs=1e-10)
    assert roots == sorted(roots)


def test_smallest_root_returns_first_root() -> None:
    assert solvers.smallest_root(cubic, 0.0, 1.0) == pytest.approx(0.2, abs=1e-10)


def test_smallest_root_respects_interval() -> None:
    assert solvers.smallest_root(cubic, 0.3, 1.0) == pytest.approx(0.5, abs=1e-10)


def test_no_roots_gives_none_and_empty_list() -> None:
    assert solvers.smallest_root(lambda x: x * x + 1.0, -1.0, 1.0) is None
    assert solvers.bracketed_roots(lambda x: x * x + 1.0, -1.0, 1.0) == []


def test_root_exactly_on_grid_node_is_found_once() -> None:
    roots = solvers.bracketed_roots(lambda x: x - 0.5, 0.0, 1.0, n_grid=10)
    assert roots == pytest.approx([0.5], abs=1e-12)
    assert len(roots) == 1


def test_close_roots_need_fine_enough_grid() -> None:
    def f(x: float) -> float:
        return (x - 0.500) * (x - 0.503)

    assert len(solvers.bracketed_roots(f, 0.0, 1.0, n_grid=1000)) == 2
