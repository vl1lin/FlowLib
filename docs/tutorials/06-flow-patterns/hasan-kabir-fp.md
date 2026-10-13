# Модель режимов Hasan–Kabir (`hasan_kabir_fp`)

Механистическая модель режимов для вертикального подъёма в трубе и затрубье.

## Источник

Hasan A.R., Kabir C.S., 1988, SPE Production Engineering 3(2), 263–272 (с Erratum 1989), ур.
(1)–(13).

## Режимы

BUBBLE, DISPERSED_BUBBLE, SLUG, CHURN, ANNULAR.

## Схема расчёта

Обозначения: r = d_t/d (0 для трубы), C0 = 1.2 + 0.371·r. Проверки идут в указанном порядке.

1. **DISPERSED_BUBBLE.** Скорость смеси больше границы дисперсного режима v_M,DB (ур. 6) и
   газосодержание f_g,bubbly = v_gs/(C0·v_M + v∞) меньше 0.52.
2. **BUBBLE.** v_gs < α·(C0·v_Ls + v∞)/(1 − α·C0), α = 0.25 (ур. 1–2).
3. **ANNULAR.** v_gs > 3.1·[σ·g·Δρ/ρ_g²]^0.25 (ур. 13).
4. Иначе — граница slug/churn:
   - ρ_L·v_Ls² > 3300 lbm/(ft·s²) — CHURN и `OutOfRangeWarning`;
   - при включённом `porteous_low_pressure` и p < 1013 кПа: v_M ≥ 0.103·√(g·d·Δρ/ρ_g)
     (ур. 7) — CHURN, иначе SLUG;
   - иначе граница Hewitt–Roberts (ур. 8, 9): ρ_g·v_gs² выше границы — CHURN, иначе SLUG.

## Флаги

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `porteous_low_pressure` | `False` | При p < 1013 кПа граница slug/churn по Porteous (7) вместо карты Hewitt–Roberts; для низконапорных скважин и лабораторных данных |

## Расхождения и допущения

- Исправление опечатки: в ур. (6) ρ_M заменено на ρ_L (Erratum 1989), реализовано в
  `core.transitions`.
- Порядок проверок (а)–(г) принят как указано в списке; при ρ_L·v_Ls² > 3300 lbm/(ft·s²) вне
  DISPERSED_BUBBLE и ANNULAR принят CHURN с предупреждением.

## Ключи `details`

`f_g_bubbly`, `v_inf_m_s`, `v_m_dispersed_m_s`, `v_gs_bubble_slug_m_s`, `v_gs_annular_m_s`,
`hr_*` (граница Hewitt–Roberts в lbm/(ft·s²)), `v_m_porteous_m_s`.

## Совместимость

Родная модель режима корреляции [Hasan–Kabir](../05-correlations/hasan-kabir.md); режимы
полностью совместимы и с корреляцией [Ansari](../05-correlations/ansari.md).
