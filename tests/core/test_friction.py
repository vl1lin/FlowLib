"""Тесты `core/friction.py` — коэффициенты трения.

Соглашение ТЗ: ядро везде использует Moody/Darcy; Fanning (TD, Barnea, HK) явно
переводится в Darcy множителем 4 в одном месте (`fanning_to_darcy`).

Ожидаемый API (`from correlations.core import friction`):
    laminar_darcy(re) -> float                         64/Re (Darcy)
    colebrook(re, rel_roughness) -> float              Darcy, уравнение Colebrook–White
    bb_smooth(re) -> float                             Darcy, Beggs–Brill 1973, ур. (13)
    moody(re, rel_roughness) -> float                  64/Re при Re < 2000, иначе Colebrook
    fanning_taitel_dukler(re, *, laminar) -> float     Fanning: 16/Re или 0.046·Re^-0.2
    fanning_blasius(re) -> float                       Fanning: 0.079·Re^-0.25
    fanning_to_darcy(f) -> float                       4·f
    darcy_to_fanning(f) -> float                       f/4
"""

import math
import warnings

import pytest

from correlations.core import friction
from correlations.core.errors import OutOfRangeWarning

# --- Эталоны ---------------------------------------------------------------------------
# reference: ТЗ, «Элементные эталоны», BB: Re_ns / f_ns Colebrook / f_ns гладкая (13)
R1_RE_NS = 3.152e5
R1_F_COLEBROOK = 0.01551
R1_F_SMOOTH_13 = 0.01434
R1_REL_ROUGHNESS = 18.288e-6 / 0.1524  # ε/d для R1 = 1.2e-4


def test_laminar_darcy() -> None:
    # reference: определение f = 64/Re (расчёт агента)
    assert friction.laminar_darcy(1000.0) == pytest.approx(0.064, rel=1e-12)
    assert friction.laminar_darcy(100.0) == pytest.approx(0.64, rel=1e-12)


def test_colebrook_matches_tz_reference_r1() -> None:
    f = friction.colebrook(R1_RE_NS, R1_REL_ROUGHNESS)
    assert f == pytest.approx(R1_F_COLEBROOK, rel=1e-3)


@pytest.mark.parametrize("re", [4.0e3, 1.0e5, 3.152e5, 1.0e7])
@pytest.mark.parametrize("rel_rough", [0.0, 1.0e-5, 1.2e-4, 1.0e-3, 5.0e-2])
def test_colebrook_satisfies_its_own_equation(re: float, rel_rough: float) -> None:
    # reference: Colebrook–White: 1/√f = −2·log10(ε/(3.7d) + 2.51/(Re·√f)) (расчёт агента)
    f = friction.colebrook(re, rel_rough)
    lhs = 1.0 / math.sqrt(f)
    rhs = -2.0 * math.log10(rel_rough / 3.7 + 2.51 / (re * math.sqrt(f)))
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_colebrook_grows_with_roughness() -> None:
    values = [friction.colebrook(1.0e5, r) for r in (0.0, 1.0e-4, 1.0e-3, 1.0e-2)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_bb_smooth_matches_tz_reference_r1() -> None:
    assert friction.bb_smooth(R1_RE_NS) == pytest.approx(R1_F_SMOOTH_13, rel=1e-3)


def test_moody_laminar_branch() -> None:
    assert friction.moody(1000.0, 1.0e-4) == pytest.approx(0.064, rel=1e-12)


def test_moody_turbulent_branch_equals_colebrook() -> None:
    assert friction.moody(R1_RE_NS, R1_REL_ROUGHNESS) == pytest.approx(
        friction.colebrook(R1_RE_NS, R1_REL_ROUGHNESS), rel=1e-12
    )


# --- верхняя граница относительной шероховатости ---
# reference: решение пользователя — ограничить ε/d физичным пределом вместо ConvergenceError;
# предел 0.05 — верхняя граница диаграммы Moody (то же значение уже принято в Ansari, ε/d_HF ≤ 0.05)
def test_max_relative_roughness_value() -> None:
    assert friction.MAX_RELATIVE_ROUGHNESS == 0.05


def test_colebrook_at_the_limit_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        friction.colebrook(1.0e5, friction.MAX_RELATIVE_ROUGHNESS)


@pytest.mark.parametrize("rel_roughness", [0.051, 0.3, 10.0, 1.0e6])
def test_colebrook_caps_roughness_with_warning(rel_roughness: float) -> None:
    """ε/d выше предела: f считается при ε/d = 0.05, выдаётся OutOfRangeWarning, ошибки нет."""
    capped = friction.colebrook(1.0e5, friction.MAX_RELATIVE_ROUGHNESS)
    with pytest.warns(OutOfRangeWarning, match="ε/d"):
        f = friction.colebrook(1.0e5, rel_roughness)
    assert f == pytest.approx(capped, rel=1e-12)


def test_colebrook_capped_value_is_physical() -> None:
    """Полностью шероховатый режим при ε/d = 0.05: f ≈ 0.072 (Moody), 0 < f < 0.1."""
    f = friction.colebrook(1.0e8, friction.MAX_RELATIVE_ROUGHNESS)
    assert f == pytest.approx(0.072, rel=0.03)


def test_moody_caps_roughness_in_turbulent_branch() -> None:
    capped = friction.moody(1.0e5, friction.MAX_RELATIVE_ROUGHNESS)
    with pytest.warns(OutOfRangeWarning):
        assert friction.moody(1.0e5, 5.0) == pytest.approx(capped, rel=1e-12)


def test_moody_laminar_branch_ignores_roughness_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert friction.moody(1000.0, 5.0) == pytest.approx(0.064, rel=1e-12)


def test_fanning_taitel_dukler_turbulent() -> None:
    # reference: TD 1976, f = 0.046·Re^-0.2; при Re = 1e5 ровно 0.0046 (расчёт агента)
    assert friction.fanning_taitel_dukler(1.0e5, laminar=False) == pytest.approx(0.0046, rel=1e-12)


def test_fanning_taitel_dukler_laminar() -> None:
    # reference: TD 1976, f = 16/Re (расчёт агента)
    assert friction.fanning_taitel_dukler(100.0, laminar=True) == pytest.approx(0.16, rel=1e-12)


def test_fanning_blasius() -> None:
    # reference: Blasius, f = 0.079·Re^-0.25; при Re = 1e4 ровно 0.0079 (расчёт агента)
    assert friction.fanning_blasius(1.0e4) == pytest.approx(0.0079, rel=1e-12)


@pytest.mark.parametrize("f", [0.004, 0.0155, 0.16])
def test_fanning_darcy_conversion_roundtrip(f: float) -> None:
    assert friction.fanning_to_darcy(f) == pytest.approx(4.0 * f, rel=1e-15)
    assert friction.darcy_to_fanning(friction.fanning_to_darcy(f)) == pytest.approx(f, rel=1e-14)


def test_fanning_laminar_converted_equals_darcy_laminar() -> None:
    """Moody/Fanning: 4·(16/Re) = 64/Re — ловит потерянный/лишний множитель 4."""
    for re in (50.0, 500.0, 1900.0):
        converted = friction.fanning_to_darcy(friction.fanning_taitel_dukler(re, laminar=True))
        assert converted == pytest.approx(friction.laminar_darcy(re), rel=1e-12)


@pytest.mark.parametrize("re", [2.0e4, 1.0e5, 5.0e5])
def test_fanning_turbulent_converted_close_to_smooth_darcy(re: float) -> None:
    """Moody/Fanning: 4·f_TD(Re) ≈ Colebrook гладкой трубы (в пределах 6%).

    Потерянный множитель 4 дал бы расхождение в 4 раза.
    """
    darcy = friction.fanning_to_darcy(friction.fanning_taitel_dukler(re, laminar=False))
    assert darcy == pytest.approx(friction.colebrook(re, 0.0), rel=0.06)


@pytest.mark.parametrize("re", [2.0e4, 1.0e5])
def test_fanning_blasius_converted_close_to_smooth_darcy(re: float) -> None:
    """Moody/Fanning: 4·f_Blasius(Re) ≈ Colebrook гладкой трубы (в пределах 8%).

    Область применимости Blasius — Re ≲ 1e5 (при Re = 5e5 расхождение ≈ 10% по физике
    формулы, а не из-за множителя 4), поэтому точки выше 1e5 не берутся.
    """
    blasius = friction.fanning_to_darcy(friction.fanning_blasius(re))
    assert blasius == pytest.approx(friction.colebrook(re, 0.0), rel=0.08)
