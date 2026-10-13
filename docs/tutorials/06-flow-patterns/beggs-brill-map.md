# Карта режимов Beggs–Brill (`beggs_brill_map`)

Эмпирическая карта режимов для горизонтального потока. Режим **не зависит от угла**: карта
построена для горизонтали, а по режиму корреляция выбирает коэффициенты истинного содержания
жидкости.

## Источники

- Beggs H.D., Brill J.P., 1973, JPT 25(5), ур. 9–10 — оригинальная карта.
- Brill J.P., Mukherjee H., «Multiphase Flow in Wells», 1999, ур. 4.110–4.119 — уточнённая карта
  (допустимое использование монографии как источника формул).

## Режимы

SEGREGATED, INTERMITTENT, DISTRIBUTED; при `revised_map` — ещё TRANSITION.

## Схема расчёта

1. λ_L = v_sl/v_m и число Фруда N_FR = v_m²/(g·d).
2. Оригинальная карта: X = ln λ, границы L1 (ур. 9) и L2 (ур. 10):
   - N_FR < L1 — SEGREGATED;
   - N_FR > L1 и N_FR > L2 — DISTRIBUTED;
   - L1 < N_FR < L2 — INTERMITTENT.
3. Уточнённая карта (`revised_map`): границы L1…L4 с переходной зоной TRANSITION.

## Флаги

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `revised_map` | `False` | Карта L1–L4 с переходной зоной вместо оригинальной; убирает скачок H_L на границе segregated/intermittent; для сверки с PIPESIM «BB Revised» |

## Ключи `details`

`lambda_l`, `n_fr`, `l1_froude`, `l2_froude`; при `revised_map` дополнительно `l3_froude`,
`l4_froude`.

## Совместимость

Режимы этой карты умеет считать только корреляция [Beggs–Brill](../05-correlations/beggs-brill.md).
