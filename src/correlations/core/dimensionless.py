"""Безразмерные числа (пересечение №1 ТЗ): N_Lv, N_gv, N_L, N_d, N_FR, Re.

Все формулы — в СИ. Предусловия (положительность аргументов) не проверяются:
вход проверяет `FlowState`.

Источники:
    * Duns H., Ros N.C.J., 1963 — N_Lv, N_gv, N_d, N_L;
    * Mukherjee H., Brill J.P., 1985 (IJMF 11(3)) — N_L с g (в JERT опечатка без g);
    * Beggs H.D., Brill J.P., 1973 — N_FR = v_m²/(g·d), N_Lv.
"""

from dataclasses import dataclass

from correlations.core.constants import GRAVITY_M_S2


@dataclass(frozen=True, slots=True)
class DunsRosNumbers:
    """Числа Данса–Роса.

    Атрибуты:
        n_lv: число скорости жидкости N_Lv, безразм.
        n_gv: число скорости газа N_gv, безразм.
        n_l: число вязкости жидкости N_L, безразм.
    """

    n_lv: float
    n_gv: float
    n_l: float


def froude_number(v_m_m_s: float, d_m: float) -> float:
    """Число Фруда смеси N_FR = v_m²/(g·d) (Beggs–Brill 1973).

    Параметры:
        v_m_m_s: скорость смеси v_sl + v_sg, м/с.
        d_m: диаметр трубы, м.

    Возвращает:
        N_FR, безразм.
    """
    return v_m_m_s**2 / (GRAVITY_M_S2 * d_m)


def duns_ros(
    v_sl_m_s: float,
    v_sg_m_s: float,
    rho_l_kg_m3: float,
    sigma_n_m: float,
    mu_l_pa_s: float,
) -> DunsRosNumbers:
    """Числа Данса–Роса в СИ.

    N_Lv = v_sl·(ρ_L/(g·σ))^0.25, N_gv = v_sg·(ρ_L/(g·σ))^0.25,
    N_L = μ_L·(g/(ρ_L·σ³))^0.25 (Mukherjee–Brill 1985, с g).

    Параметры:
        v_sl_m_s: приведённая скорость жидкости, м/с.
        v_sg_m_s: приведённая скорость газа, м/с.
        rho_l_kg_m3: плотность жидкости, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.
        mu_l_pa_s: вязкость жидкости, Па·с.

    Возвращает:
        `DunsRosNumbers` (N_Lv, N_gv, N_L).
    """
    velocity_scale = (rho_l_kg_m3 / (GRAVITY_M_S2 * sigma_n_m)) ** 0.25
    n_l = mu_l_pa_s * (GRAVITY_M_S2 / (rho_l_kg_m3 * sigma_n_m**3)) ** 0.25
    return DunsRosNumbers(
        n_lv=v_sl_m_s * velocity_scale,
        n_gv=v_sg_m_s * velocity_scale,
        n_l=n_l,
    )


def diameter_number(d_m: float, rho_l_kg_m3: float, sigma_n_m: float) -> float:
    """Диаметральное число Данса–Роса N_d = d·√(ρ_L·g/σ).

    Параметры:
        d_m: диаметр трубы, м.
        rho_l_kg_m3: плотность жидкости, кг/м³.
        sigma_n_m: поверхностное натяжение, Н/м.

    Возвращает:
        N_d, безразм.
    """
    return d_m * (rho_l_kg_m3 * GRAVITY_M_S2 / sigma_n_m) ** 0.5


def reynolds_number(rho_kg_m3: float, v_m_s: float, d_m: float, mu_pa_s: float) -> float:
    """Число Рейнольдса Re = ρ·v·d/μ.

    Параметры:
        rho_kg_m3: плотность, кг/м³.
        v_m_s: скорость, м/с.
        d_m: характерный (гидравлический) диаметр, м.
        mu_pa_s: динамическая вязкость, Па·с.

    Возвращает:
        Re, безразм.
    """
    return rho_kg_m3 * v_m_s * d_m / mu_pa_s
