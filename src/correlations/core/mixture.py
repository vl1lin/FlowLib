"""Свойства газожидкостной смеси (пересечение №2 ТЗ).

Два способа смешения:
    * без проскальзывания (no-slip): по истинной доле жидкости λ_L = v_sl/v_m;
    * с проскальзыванием (slip): по истинному объёмному содержанию жидкости H_L.
Смешение плотности и вязкости линейное (Beggs–Brill 1973, ур. 3–4, 14; Mukherjee–Brill 1985).

Предусловия (v_sl + v_sg > 0, 0 ≤ H_L ≤ 1) не проверяются: вход проверяет `FlowState`.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoSlipMixture:
    """Смесь без проскальзывания.

    Атрибуты:
        lam: доля жидкости без проскальзывания λ_L = v_sl/v_m, безразм.
        rho_ns_kg_m3: плотность ρ_ns = λ_L·ρ_L + (1 − λ_L)·ρ_g, кг/м³.
        mu_ns_pa_s: вязкость μ_ns = λ_L·μ_L + (1 − λ_L)·μ_g, Па·с.
        v_m_m_s: скорость смеси v_sl + v_sg, м/с.
    """

    lam: float
    rho_ns_kg_m3: float
    mu_ns_pa_s: float
    v_m_m_s: float


@dataclass(frozen=True, slots=True)
class SlipMixture:
    """Смесь с проскальзыванием.

    Атрибуты:
        rho_s_kg_m3: плотность ρ_s = H_L·ρ_L + (1 − H_L)·ρ_g, кг/м³.
        mu_s_pa_s: вязкость μ_s = H_L·μ_L + (1 − H_L)·μ_g, Па·с.
    """

    rho_s_kg_m3: float
    mu_s_pa_s: float


def no_slip(
    v_sl_m_s: float,
    v_sg_m_s: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    mu_l_pa_s: float,
    mu_g_pa_s: float,
) -> NoSlipMixture:
    """Рассчитывает свойства смеси без проскальзывания.

    Параметры:
        v_sl_m_s: приведённая скорость жидкости, м/с.
        v_sg_m_s: приведённая скорость газа, м/с.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        mu_l_pa_s, mu_g_pa_s: вязкости жидкости и газа, Па·с.

    Возвращает:
        `NoSlipMixture` (λ_L, ρ_ns, μ_ns, v_m).
    """
    v_m = v_sl_m_s + v_sg_m_s
    lam = v_sl_m_s / v_m
    return NoSlipMixture(
        lam=lam,
        rho_ns_kg_m3=lam * rho_l_kg_m3 + (1.0 - lam) * rho_g_kg_m3,
        mu_ns_pa_s=lam * mu_l_pa_s + (1.0 - lam) * mu_g_pa_s,
        v_m_m_s=v_m,
    )


def slip(
    holdup: float,
    rho_l_kg_m3: float,
    rho_g_kg_m3: float,
    mu_l_pa_s: float,
    mu_g_pa_s: float,
) -> SlipMixture:
    """Рассчитывает свойства смеси по истинному содержанию жидкости.

    Параметры:
        holdup: истинное объёмное содержание жидкости H_L, 0…1.
        rho_l_kg_m3, rho_g_kg_m3: плотности жидкости и газа, кг/м³.
        mu_l_pa_s, mu_g_pa_s: вязкости жидкости и газа, Па·с.

    Возвращает:
        `SlipMixture` (ρ_s, μ_s).
    """
    return SlipMixture(
        rho_s_kg_m3=holdup * rho_l_kg_m3 + (1.0 - holdup) * rho_g_kg_m3,
        mu_s_pa_s=holdup * mu_l_pa_s + (1.0 - holdup) * mu_g_pa_s,
    )
