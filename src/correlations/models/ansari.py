"""Корреляция Ansari et al. для подъёма в скважине (полная корреляция, ICorrelation).

Схема расчёта:
    BUBBLE: H_L из 1.53·[g·σ·Δρ/ρ_L²]^0.25·H_L^0.5 = v_sg/(1 − H_L) − 1.2·v_m (22); ρ_TP, μ_TP по
      H_L (23)–(24);
        −dp/dL = ρ_TP·g·sinθ + f·ρ_TP·v_m²/(2d) (25)–(27), f — Moody по Re = ρ_TP·v_m·d/μ_TP (v_TP
          = v_m, (18));
    DISPERSED_BUBBLE: то же при H_L = λ_L (no-slip, (16)–(19));
    SLUG: v_TB (34), H_gLS (38), v_gLS (35), Ā (40), H_LTB — корень (39), v_LTB (37), v_LLS (32),
      v_gTB (33), β (29),
        L_LS = 30·d (48), L_TB (49); L_c (44)–(47) ≥ L_TB → развивающаяся пробка: L_TB* (54)–(57),
          β* = L_TB*/(L_TB* + L_LS),
        H_LTBA (64); градиент: элевация (60) или (62), трение (65) (β или β*); ускорения нет;
        H_L = (1 − β)·H_LLS + β·H_LTB (развивающаяся: (1 − β*)·H_LLS + β*·H_LTBA);
    ANNULAR: плёнка (96), φ_C² = Z/(1 − 2δ)^5 (99), −dp/dL = φ_C²·(dp/dL)_SC + ρ_C·g·sinθ (102);
        H_L = 4δ(1 − δ) + λ_LC·(1 − 2δ)².
    CHURN рассчитывается как SLUG (Ansari: churn входит в slug).

Источник: Ansari A.M., Sylvester N.D., Sarica C., Shoham O., Brill J.P., 1994, SPE Production &
  Facilities 9(2),
145–152, ур. (1)–(102). Опечатки (ТЗ): (18) v_TP = v_m; (32) v_TB; (40) Ā = H_gLS·(v_TB − v_gLS) +
  v_m
(буквальная форма — флаг vo_shoham_a_as_published, H_LTB = 0.1188 вместо 0.1298 для R2); (85)
  (ρ_L/ρ_g)^(1/3).

Расхождение сверх таблицы «Опечатки» (сообщено в отчёте): в (56) напечатано b = [v_sg − v_gLS·(2 −
  H_LLS)]/v_TB·L_LS;
для R2 это даёт комплексные корни (54). Из (52)–(53) следует множитель (1 − H_LLS); принято по
  умолчанию как вывод из (52)–(53). Напечатанный вариант — флаг `eq56_as_published` (корня нет →
  полностью развитая пробка).
Уравнение (54) получено возведением в квадрат уравнения a·L − 2c·√L − b = 0, поэтому берётся
  больший корень (вопрос 3 ТЗ).
Допущение (вопрос 3 ТЗ): v_NLTB в (45) определяется из баланса (32) с H_NLTB (46).
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import Field

from correlations.core import bubbles, dimensionless, friction, gradient, mixture, solvers
from correlations.core.constants import GRAVITY_M_S2
from correlations.core.errors import ConvergenceError, UnsupportedPatternError
from correlations.core.interfaces import IFlowModel
from correlations.core.options import BaseOptions
from correlations.core.patterns import FlowPattern
from correlations.core.state import CorrelationResult, FlowState
from correlations.flow_models.ansari_fp import (
    AnsariFlowModel,
    AnsariFlowOptions,
    solve_annular_film,
)
from correlations.models._common import assemble_gradient, single_phase_correlation_result

# Профиль скорости смеси (34), (35), (20): центральная скорость = 1.2·v_m
_CENTERLINE_FACTOR: Final = 1.2
# (38): H_gLS = v_sg/(0.425 + 2.65·v_m)
_SLUG_VOID_A: Final = 0.425
_SLUG_VOID_B: Final = 2.65
# (36)–(37): v_LTB = 9.916·[g·d·(1 − √H_gTB)]^0.5
_FALLING_FILM_COEFFICIENT: Final = 9.916
# (48): L_LS = C'·d, C' = 30
_SLUG_LENGTH_FACTOR: Final = 30.0
# (45): δ_N = [3/4·d·v_NLTB·μ_L·(1 − H_NLTB)/(g·Δρ)]^(1/3)
_NUSSELT_FACTOR: Final = 0.75
# Сетки поиска корней
_HLTB_LO: Final = 1e-9
_HLTB_HI: Final = 1.0 - 1e-12
_HLTB_GRID: Final = 2000
_NUSSELT_LO: Final = 1e-6
_NUSSELT_HI: Final = 0.4999
_NUSSELT_GRID: Final = 4000
_BUBBLE_HOLDUP_LO: Final = 1e-12
_BUBBLE_HOLDUP_HI: Final = 1.0 - 1e-12
_BUBBLE_HOLDUP_GRID: Final = 2000


class AnsariOptions(BaseOptions):
    """Флаги корреляции Ansari."""

    developing_slug: bool = Field(
        default=True,
        description=(
            "Учёт развивающейся пробки (44)–(64). Выключать для сверки с Примером 4.9 и PIPESIM "
            "(если там этой ветки нет). Источник: Ansari et al. 1994, ур. 44–64."
        ),
    )
    vo_shoham_a_as_published: bool = Field(
        default=False,
        description=(
            "Ā по напечатанному (40) вместо согласованного с (32), (35). Только буквальное "
            "воспроизведение статьи; даёт H_LTB = 0.119 вместо 0.130 для R2. "
            "Источник: Ansari et al. 1994, ур. 40."
        ),
    )
    bridging_with_delta_min: bool = Field(
        default=True,
        description=(
            "В критерии (12) берётся δ_min (текст статьи); False — δ из (96) (Пример 4.9). "
            "False физически ближе к Barnea. "
            "Источник: Ansari et al. 1994, ур. 8, 12 (вопрос 1 ТЗ)."
        ),
    )
    bubble_slug_sin_theta: bool = Field(
        default=False,
        description=(
            "0.25·v_s·sinθ в (2), как у Barnea [4]. Включать для наклонных скважин. "
            "Источник: Ansari et al. 1994, ур. 2; Barnea 1987, [4]."
        ),
    )
    eq56_as_published: bool = Field(
        default=False,
        description=(
            "b в (56) по напечатанному (2 − H_LLS) вместо (1 − H_LLS) из вывода (52)–(53). Для R2 "
            "вещественного корня (54) нет, развивающаяся пробка не определяется и расчёт идёт по "
            "полностью развитой пробке. Только буквальное воспроизведение статьи. "
            "Источник: Ansari et al. 1994, ур. 52–57."
        ),
    )
    ff_over_fsl_unity: bool = Field(
        default=False,
        description=(
            "f_F/f_SL = 1 без итерации по Re плёнки. Быстрый расчёт; Пример 4.9. "
            "Источник: Ansari et al. 1994, ур. 74–78."
        ),
    )


@dataclass(frozen=True, slots=True)
class SlugUnit:
    """Пробковая ячейка (развитое течение), ур. (29)–(43), (48), (49).

    Атрибуты:
        v_tb: скорость пузыря Тейлора v_TB (34), м/с.
        h_gls: газосодержание в пробке H_gLS (38).
        h_lls: содержание жидкости в пробке H_LLS = 1 − H_gLS.
        v_gls: скорость газа в пробке v_gLS (35), м/с.
        a_tilde: параметр Ā (40), м/с.
        h_ltb: содержание жидкости в зоне пузыря Тейлора H_LTB — корень (39).
        v_ltb: скорость плёнки v_LTB (37), м/с.
        v_lls: скорость жидкости в пробке v_LLS (32), м/с.
        v_gtb: скорость газа в пузыре v_gTB (33), м/с.
        beta: доля пузыря в ячейке β = L_TB/L_SU (29).
        l_ls_m: длина жидкой пробки L_LS = 30·d (48), м.
        l_tb_m: длина пузыря Тейлора L_TB (49), м.
    """

    v_tb: float
    h_gls: float
    h_lls: float
    v_gls: float
    a_tilde: float
    h_ltb: float
    v_ltb: float
    v_lls: float
    v_gtb: float
    beta: float
    l_ls_m: float
    l_tb_m: float


@dataclass(frozen=True, slots=True)
class DevelopingSlug:
    """Параметры развивающейся пробки, ур. (50)–(57), (64).

    Атрибуты:
        l_tb_star_m: длина пузыря Тейлора L_TB*, м.
        beta_star: β* = L_TB*/(L_TB* + L_LS).
        h_ltba: среднее содержание жидкости в зоне пузыря H_LTBA (64).
    """

    l_tb_star_m: float
    beta_star: float
    h_ltba: float


def solve_slug_unit(state: FlowState, *, vo_shoham_a_as_published: bool = False) -> SlugUnit:
    """Решает модель пробкового течения: v_TB, H_gLS, H_LTB (корень (39)) и производные величины.

    Параметры:
        state: состояние потока (СИ, двухфазное).
        vo_shoham_a_as_published: использовать Ā по напечатанному ур. (40) (typo-флаг).

    Возвращает:
        `SlugUnit`.

    Исключения:
        ConvergenceError: уравнение (39) не имеет корня на (0, 1).
    """
    rho_l, rho_g, d = state.rho_l_kg_m3, state.rho_g_kg_m3, state.d_m
    v_sg = state.v_sg_m_s
    v_m = state.v_sl_m_s + v_sg
    v_tb = _CENTERLINE_FACTOR * v_m + bubbles.taylor_rise_velocity(
        d, rho_l, rho_g, coef=bubbles.TAYLOR_COEFFICIENT_ANSARI
    )  # (34)
    h_gls = v_sg / (_SLUG_VOID_A + _SLUG_VOID_B * v_m)  # (38)
    h_lls = 1.0 - h_gls
    v_inf = bubbles.harmathy_rise_velocity(rho_l, rho_g, state.sigma_n_m)
    v_bubble_in_slug = v_inf * math.sqrt(h_lls)  # (21), n' = 0.5
    v_gls = _CENTERLINE_FACTOR * v_m + v_bubble_in_slug  # (35)
    if vo_shoham_a_as_published:  # (40) как напечатано
        a_tilde = h_gls * v_tb + h_lls * (v_m - h_gls * v_inf * math.sqrt(1.0 - h_gls))
    else:  # typo fix: Ā = H_gLS·(v_TB − v_gLS) + v_m (ТЗ, «Опечатки», Ansari (40))
        a_tilde = h_gls * (v_tb - v_gls) + v_m
    falling = _FALLING_FILM_COEFFICIENT * math.sqrt(GRAVITY_M_S2 * d)

    def residual_39(h_ltb: float) -> float:
        return (
            falling * math.sqrt(1.0 - math.sqrt(1.0 - h_ltb)) * h_ltb
            - v_tb * (1.0 - h_ltb)
            + a_tilde
        )

    h_ltb = solvers.smallest_root(residual_39, _HLTB_LO, _HLTB_HI, n_grid=_HLTB_GRID)
    if h_ltb is None:
        raise ConvergenceError(
            "уравнение (39) для H_LTB не имеет корня на (0, 1)",
            {"v_sl_m_s": state.v_sl_m_s, "v_sg_m_s": v_sg, "d_m": d},
        )
    h_gtb = 1.0 - h_ltb
    v_ltb = _FALLING_FILM_COEFFICIENT * math.sqrt(
        GRAVITY_M_S2 * d * (1.0 - math.sqrt(h_gtb))
    )  # (37)
    v_lls = v_tb - (v_tb + v_ltb) * h_ltb / h_lls  # (32)
    v_gtb = v_tb - (v_tb - v_gls) * h_gls / h_gtb  # (33)
    beta = (v_sg - v_gls * h_gls) / (v_gtb * h_gtb - v_gls * h_gls)  # (29)
    l_ls = _SLUG_LENGTH_FACTOR * d
    return SlugUnit(
        v_tb=v_tb,
        h_gls=h_gls,
        h_lls=h_lls,
        v_gls=v_gls,
        a_tilde=a_tilde,
        h_ltb=h_ltb,
        v_ltb=v_ltb,
        v_lls=v_lls,
        v_gtb=v_gtb,
        beta=beta,
        l_ls_m=l_ls,
        l_tb_m=l_ls * beta / (1.0 - beta),  # (49)
    )


def cap_length(state: FlowState, unit: SlugUnit) -> float:
    """Длина «головной» части пузыря L_c (ур. 44–47) при толщине плёнки Нуссельта.

    δ_N находится из (45)–(46), v_NLTB — из баланса (32) с H_NLTB (вопрос 3 ТЗ).

    Параметры:
        state: состояние потока.
        unit: решение пробковой ячейки.

    Возвращает:
        L_c, м.

    Исключения:
        ConvergenceError: толщина плёнки Нуссельта не найдена.
    """
    d = state.d_m
    delta_rho = state.rho_l_kg_m3 - state.rho_g_kg_m3
    v_m = state.v_sl_m_s + state.v_sg_m_s

    def nusselt_state(delta_over_d: float) -> tuple[float, float, float]:
        h_n = 1.0 - (1.0 - 2.0 * delta_over_d) ** 2  # (46)
        v_n = (unit.v_tb - unit.v_lls) * unit.h_lls / h_n - unit.v_tb  # (32) при H_NLTB
        return h_n, v_n, delta_over_d

    def residual_45(delta_over_d: float) -> float:
        h_n, v_n, _ = nusselt_state(delta_over_d)
        if v_n <= 0.0:
            return -delta_over_d
        delta_n = (
            _NUSSELT_FACTOR * d * v_n * state.mu_l_pa_s * (1.0 - h_n) / (GRAVITY_M_S2 * delta_rho)
        ) ** (1.0 / 3.0)
        return delta_n / d - delta_over_d

    delta_over_d = solvers.smallest_root(
        residual_45, _NUSSELT_LO, _NUSSELT_HI, n_grid=_NUSSELT_GRID
    )
    if delta_over_d is None:
        raise ConvergenceError(
            "толщина плёнки Нуссельта (45)–(46) не найдена",
            {"v_sl_m_s": state.v_sl_m_s, "v_sg_m_s": state.v_sg_m_s, "d_m": d},
        )
    h_n, _, _ = nusselt_state(delta_over_d)
    v_ng = unit.v_tb - (unit.v_tb - unit.v_gls) * unit.h_gls / (1.0 - h_n)  # (47)
    return (unit.v_tb + v_ng * (1.0 - h_n) / h_n - v_m / h_n) ** 2 / (2.0 * GRAVITY_M_S2)  # (44)


def developing_slug_unit(
    state: FlowState, unit: SlugUnit, *, eq56_as_published: bool = False
) -> DevelopingSlug | None:
    """Параметры развивающейся пробки: L_TB* (54)–(57), β*, H_LTBA (64).

    Уравнение (54) — квадратное для L; оно получено возведением в квадрат
    a·L − 2c·√L − b = 0 (из (52)–(53)), поэтому берётся больший положительный корень.
    (56): напечатано (2 − H_LLS); по умолчанию принято (1 − H_LLS) по выводу из (52)–(53)
    (см. модуль), напечатанный вариант включается флагом `eq56_as_published`.

    Параметры:
        state: состояние потока.
        unit: решение пробковой ячейки.
        eq56_as_published: брать множитель (2 − H_LLS) из (56) как напечатано; для R2 корня нет.

    Возвращает:
        `DevelopingSlug` или None, если вещественного положительного корня нет.
    """
    a = 1.0 - state.v_sg_m_s / unit.v_tb  # (55)
    h_factor = 2.0 - unit.h_lls if eq56_as_published else 1.0 - unit.h_lls  # (56)
    b = (state.v_sg_m_s - unit.v_gls * h_factor) / unit.v_tb * unit.l_ls_m
    c = (unit.v_tb - unit.v_lls) / math.sqrt(2.0 * GRAVITY_M_S2) * unit.h_lls  # (57)
    if a <= 0.0:
        return None
    p = (-2.0 * a * b - 4.0 * c**2) / a**2  # (54)
    q = b**2 / a**2
    discriminant = p**2 - 4.0 * q
    if discriminant < 0.0:
        return None
    l_star = (-p + math.sqrt(discriminant)) / 2.0
    if l_star <= 0.0:
        return None
    h_ltba = 2.0 * (unit.v_tb - unit.v_lls) * unit.h_lls / math.sqrt(2.0 * GRAVITY_M_S2 * l_star)
    return DevelopingSlug(
        l_tb_star_m=l_star,
        beta_star=l_star / (l_star + unit.l_ls_m),
        h_ltba=h_ltba,
    )


def bubble_holdup(state: FlowState) -> float:
    """Истинное содержание жидкости пузырькового режима по ур. (22).

    1.53·[g·σ·Δρ/ρ_L²]^0.25·H_L^0.5 = v_sg/(1 − H_L) − 1.2·v_m (n' = 0.5).

    Параметры:
        state: состояние потока (СИ, двухфазное).

    Возвращает:
        H_L на (0, 1).

    Исключения:
        ConvergenceError: корня на (0, 1) нет.
    """
    v_inf = bubbles.harmathy_rise_velocity(state.rho_l_kg_m3, state.rho_g_kg_m3, state.sigma_n_m)
    v_m = state.v_sl_m_s + state.v_sg_m_s

    def residual_22(h_l: float) -> float:
        return v_inf * math.sqrt(h_l) - state.v_sg_m_s / (1.0 - h_l) + _CENTERLINE_FACTOR * v_m

    h_l = solvers.smallest_root(
        residual_22, _BUBBLE_HOLDUP_LO, _BUBBLE_HOLDUP_HI, n_grid=_BUBBLE_HOLDUP_GRID
    )
    if h_l is None:
        raise ConvergenceError(
            "уравнение (22) для H_L пузырькового режима не имеет корня на (0, 1)",
            {"v_sl_m_s": state.v_sl_m_s, "v_sg_m_s": state.v_sg_m_s},
        )
    return h_l


_BUBBLE_PATTERNS: Final = (FlowPattern.BUBBLE, FlowPattern.DISPERSED_BUBBLE)
_SLUG_PATTERNS: Final = (FlowPattern.SLUG, FlowPattern.CHURN)


class Ansari:
    """Корреляция Ansari (ICorrelation).

    Атрибуты:
        name: "ansari".
    """

    name: ClassVar[str] = "ansari"

    def __init__(
        self,
        options: AnsariOptions | None = None,
        flow_model: IFlowModel | None = None,
    ) -> None:
        """Создаёт корреляцию.

        Параметры:
            options: настройки (по умолчанию — поведение статьи).
            flow_model: модуль режима; по умолчанию `AnsariFlowModel` с тремя флагами режима.
        """
        self._options = options if options is not None else AnsariOptions()
        if flow_model is None:
            flow_options = AnsariFlowOptions(
                bridging_with_delta_min=self._options.bridging_with_delta_min,
                bubble_slug_sin_theta=self._options.bubble_slug_sin_theta,
                ff_over_fsl_unity=self._options.ff_over_fsl_unity,
            )
            flow_model = AnsariFlowModel(flow_options)
        self._flow_model = flow_model

    @property
    def options(self) -> AnsariOptions:
        """Настройки корреляции."""
        return self._options

    @property
    def flow_model(self) -> IFlowModel:
        """Используемый модуль режима."""
        return self._flow_model

    def calc(self, state: FlowState) -> CorrelationResult:
        """Рассчитывает режим, holdup и градиент давления.

        Параметры:
            state: состояние потока (СИ).

        Возвращает:
            `CorrelationResult`; ускорение всегда 0; details зависят от режима (см. модуль).

        Исключения:
            UnsupportedPatternError: модуль режима вернул режим вне BUBBLE, DISPERSED_BUBBLE,
                SLUG, CHURN, ANNULAR.
            ConvergenceError: не найдено решение (22), (39), (45) или (96).
        """
        single = single_phase_correlation_result(state)
        if single is not None:
            return single
        pattern = self._flow_model.predict(state).pattern
        if pattern in _BUBBLE_PATTERNS:
            holdup, grav, fric, details = self._bubble(
                state, dispersed=pattern is _BUBBLE_PATTERNS[1]
            )
        elif pattern in _SLUG_PATTERNS:
            holdup, grav, fric, details = self._slug(state)
        elif pattern is FlowPattern.ANNULAR:
            holdup, grav, fric, details = self._annular(state)
        else:
            raise UnsupportedPatternError(pattern, "ansari")
        parts = assemble_gradient(grav, fric, 0.0, include_acceleration=False)
        return CorrelationResult(
            pattern=pattern,
            holdup=holdup,
            dpdl_pa_m=parts.dpdl_pa_m,
            dpdl_gravity_pa_m=parts.gravity_pa_m,
            dpdl_friction_pa_m=parts.friction_pa_m,
            dpdl_acceleration_pa_m=parts.acceleration_pa_m,
            details=details,
        )

    def _bubble(
        self, state: FlowState, *, dispersed: bool
    ) -> tuple[float, float, float, dict[str, float]]:
        """BUBBLE / DISPERSED_BUBBLE: (H_L, гравитация, трение, details)."""
        v_m = state.v_sl_m_s + state.v_sg_m_s
        holdup = state.v_sl_m_s / v_m if dispersed else bubble_holdup(state)
        mix = mixture.slip(
            holdup,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        )
        re_tp = dimensionless.reynolds_number(
            mix.rho_s_kg_m3, v_m, state.d_m, mix.mu_s_pa_s
        )  # (28)
        f_darcy = friction.moody(re_tp, state.eps_m / state.d_m)
        fric = gradient.friction_gradient(f_darcy, mix.rho_s_kg_m3, v_m, state.d_m)  # (27)
        grav = gradient.gravity_gradient(mix.rho_s_kg_m3, math.radians(state.theta_deg))  # (26)
        details = {
            "h_l": holdup,
            "rho_tp_kg_m3": mix.rho_s_kg_m3,
            "mu_tp_pa_s": mix.mu_s_pa_s,
            "re_tp": re_tp,
            "f_darcy": f_darcy,
        }
        return holdup, grav, fric, details

    def _slug(self, state: FlowState) -> tuple[float, float, float, dict[str, float]]:
        """SLUG / CHURN: (H_L, гравитация, трение, details) по ур. (29)–(65)."""
        unit = solve_slug_unit(
            state, vo_shoham_a_as_published=self._options.vo_shoham_a_as_published
        )
        l_c = cap_length(state, unit)
        developing = (
            developing_slug_unit(state, unit, eq56_as_published=self._options.eq56_as_published)
            if self._options.developing_slug and l_c >= unit.l_tb_m
            else None
        )
        v_m = state.v_sl_m_s + state.v_sg_m_s
        slug = mixture.slip(
            unit.h_lls,
            state.rho_l_kg_m3,
            state.rho_g_kg_m3,
            state.mu_l_pa_s,
            state.mu_g_pa_s,
        )  # (61)
        re_ls = dimensionless.reynolds_number(
            slug.rho_s_kg_m3, v_m, state.d_m, slug.mu_s_pa_s
        )  # (66)
        f_ls = friction.moody(re_ls, state.eps_m / state.d_m)
        sin_theta = math.sin(math.radians(state.theta_deg))
        details = {
            "v_tb_m_s": unit.v_tb,
            "h_gls": unit.h_gls,
            "h_lls": unit.h_lls,
            "v_gls_m_s": unit.v_gls,
            "a_tilde_m_s": unit.a_tilde,
            "h_ltb": unit.h_ltb,
            "v_ltb_m_s": unit.v_ltb,
            "v_lls_m_s": unit.v_lls,
            "v_gtb_m_s": unit.v_gtb,
            "beta": unit.beta,
            "l_tb_m": unit.l_tb_m,
            "l_c_m": l_c,
            "developing": 1.0 if developing is not None else 0.0,
            "rho_ls_kg_m3": slug.rho_s_kg_m3,
            "mu_ls_pa_s": slug.mu_s_pa_s,
            "re_ls": re_ls,
            "f_ls": f_ls,
        }
        if developing is None:
            beta = unit.beta
            rho_bubble = state.rho_g_kg_m3
            holdup = (1.0 - beta) * unit.h_lls + beta * unit.h_ltb
        else:
            beta = developing.beta_star
            rho_bubble = mixture.slip(
                developing.h_ltba,
                state.rho_l_kg_m3,
                state.rho_g_kg_m3,
                state.mu_l_pa_s,
                state.mu_g_pa_s,
            ).rho_s_kg_m3  # (63)
            holdup = (1.0 - beta) * unit.h_lls + beta * developing.h_ltba
            details["l_tb_star_m"] = developing.l_tb_star_m
            details["beta_star"] = beta
            details["h_ltba"] = developing.h_ltba
        grav = (
            ((1.0 - beta) * slug.rho_s_kg_m3 + beta * rho_bubble) * GRAVITY_M_S2 * sin_theta
        )  # (60), (62)
        fric = gradient.friction_gradient(f_ls, slug.rho_s_kg_m3, v_m, state.d_m) * (
            1.0 - beta
        )  # (65)
        return holdup, grav, fric, details

    def _annular(self, state: FlowState) -> tuple[float, float, float, dict[str, float]]:
        """ANNULAR: (H_L, гравитация, трение, details) по ур. (96), (99), (102)."""
        film = solve_annular_film(state, ff_over_fsl_unity=self._options.ff_over_fsl_unity)
        if film is None:
            raise ConvergenceError(
                "уравнение плёнки (96) не имеет корня",
                {"v_sl_m_s": state.v_sl_m_s, "v_sg_m_s": state.v_sg_m_s, "d_m": state.d_m},
            )
        delta = film.delta
        phi_c_sq = film.z / (1.0 - 2.0 * delta) ** 5  # (99)
        fric = phi_c_sq * film.dp_sc_pa_m
        grav = gradient.gravity_gradient(film.rho_c_kg_m3, math.radians(state.theta_deg))
        holdup = 4.0 * delta * (1.0 - delta) + film.lam_lc * (1.0 - 2.0 * delta) ** 2
        details = {
            "f_e": film.f_e,
            "lam_lc": film.lam_lc,
            "rho_c_kg_m3": film.rho_c_kg_m3,
            "delta": delta,
            "x_m_sq": film.x_m_sq,
            "y_m": film.y_m,
            "z": film.z,
            "phi_c_sq": phi_c_sq,
            "dp_sc_pa_m": film.dp_sc_pa_m,
        }
        return holdup, grav, fric, details
