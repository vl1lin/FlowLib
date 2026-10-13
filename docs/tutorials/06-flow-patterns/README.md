# 6. Определение режима потока

Модель режима потока (`IFlowModel`) по состоянию потока `FlowState` определяет режим течения.
Модель можно использовать отдельно (карта режимов) или внедрить в корреляцию (см.
[раздел 5](../05-correlations/README.md)).

Метод модели — `predict(state)`. Он возвращает `FlowPatternResult`:

- `pattern` — режим течения (`FlowPattern`);
- `details` — словарь промежуточных величин: границы режимов, безразмерные числа, равновесный
  уровень жидкости и т.п. Имена ключей содержат единицы (например, `h_l_over_d`,
  `v_sg_annular_m_s`). Набор ключей зависит от ветви расчёта: если расчёт завершился на раннем
  шаге, поздние величины в словаре отсутствуют.

> **О слове «карта».** В названиях моделей и статей (`beggs_brill_map`, «карта режимов
> Mukherjee–Brill» и т.п.) «карта» означает набор критериев перехода между режимами. Модель
> отвечает на вопрос «какой режим в этой точке?» для одного состояния потока. Библиотека **не
> строит** графики и карты в координатах (v_sl, v_sg): для этого нужен внешний код, который
> вызывает `predict` на сетке значений и рисует результат.

Однофазные пределы обрабатываются всеми моделями одинаково: `v_sg = 0` — `SINGLE_PHASE_LIQUID`,
`v_sl = 0` — `SINGLE_PHASE_GAS`; словарь `details` в этом случае пуст.

## Модели

| Модель | Имя в реестре | Для каких труб | Режимы | Флаги |
|---|---|---|---|---|
| [Beggs–Brill (карта)](beggs-brill-map.md) | `beggs_brill_map` | Горизонтальная карта, режим не зависит от угла | SEGREGATED, INTERMITTENT, DISTRIBUTED, TRANSITION | `revised_map` |
| [Mukherjee–Brill](mukherjee-brill-fp.md) | `mukherjee_brill_fp` | Наклонные трубы любого угла | BUBBLE, SLUG, ANNULAR, STRATIFIED | нет |
| [Hasan–Kabir](hasan-kabir-fp.md) | `hasan_kabir_fp` | Вертикальный подъём, труба и затрубье | BUBBLE, DISPERSED_BUBBLE, SLUG, CHURN, ANNULAR | `porteous_low_pressure` |
| [Ansari](ansari-fp.md) | `ansari_fp` | Вертикальный и наклонный подъём | BUBBLE, DISPERSED_BUBBLE, SLUG, ANNULAR | `bridging_with_delta_min`, `bubble_slug_sin_theta`, `ff_over_fsl_unity` |
| [Taitel–Dukler](taitel-dukler.md) | `taitel_dukler` | Горизонтальные и близкие к ним трубы | STRATIFIED_SMOOTH, STRATIFIED_WAVY, ANNULAR, DISPERSED_BUBBLE, INTERMITTENT | `laminar_switch`, `use_moody_friction` |
| [Barnea](barnea.md) | `barnea` | Единая модель для всех углов, −90°…90° | DISPERSED_BUBBLE, BUBBLE, SLUG, CHURN, ANNULAR, STRATIFIED_SMOOTH, STRATIFIED_WAVY, ELONGATED_BUBBLE | `fm_noslip_reynolds`, `elongated_bubble_horizontal_only`, `use_moody_friction`, `split_churn`, `friction_ratio_as_published` |

Приоритет проекта — механистические модели: Barnea, Taitel–Dukler, Hasan–Kabir и Ansari. Карты
Beggs–Brill и Mukherjee–Brill — эмпирические.

## Перечень режимов (`FlowPattern`)

Один перечень на все модели; значение режима — его имя в нижнем регистре (удобно для
сериализации: `FlowPattern.SLUG == "slug"`).

| Режим | Откуда | Описание |
|---|---|---|
| `SEGREGATED` | Beggs–Brill | Раздельное течение (расслоённое, волновое, кольцевое) |
| `INTERMITTENT` | Beggs–Brill, Taitel–Dukler | Прерывистое течение (пробковое, снарядное) |
| `DISTRIBUTED` | Beggs–Brill | Распределённое течение (пузырьковое, туманное) |
| `TRANSITION` | Beggs–Brill (`revised_map`) | Переходная зона между SEGREGATED и INTERMITTENT |
| `BUBBLE` | Mukherjee–Brill, Hasan–Kabir, Ansari, Barnea | Пузырьковый режим |
| `DISPERSED_BUBBLE` | Hasan–Kabir, Ansari, Barnea, Taitel–Dukler | Мелкодисперсный пузырьковый режим |
| `SLUG` | Mukherjee–Brill, Hasan–Kabir, Ansari, Barnea | Пробковый режим |
| `CHURN` | Hasan–Kabir, Barnea | Пульсирующий (churn) |
| `ANNULAR` | Все, кроме карты Beggs–Brill | Кольцевой режим |
| `STRATIFIED` | Mukherjee–Brill | Расслоённый режим |
| `STRATIFIED_SMOOTH` | Taitel–Dukler, Barnea | Расслоённый с гладкой границей |
| `STRATIFIED_WAVY` | Taitel–Dukler, Barnea | Расслоённый с волнистой границей |
| `ELONGATED_BUBBLE` | Barnea | Вытянутые пузыри (горизонтальный случай) |
| `SINGLE_PHASE_LIQUID` | Все | Течёт только жидкость (v_sg = 0) |
| `SINGLE_PHASE_GAS` | Все | Течёт только газ (v_sl = 0) |

## Как выбрать модель

| Задача | Подходящая модель |
|---|---|
| Вертикальная скважина | `hasan_kabir_fp` или `ansari_fp` |
| Наклонная труба любого угла | `barnea` или `mukherjee_brill_fp` |
| Горизонтальная труба | `taitel_dukler` или `barnea` |
| Сверка с эмпирическими методами | `beggs_brill_map`, `mukherjee_brill_fp` |
| Нужен режим для корреляции | Родная модель режима корреляции (см. [раздел 5](../05-correlations/README.md)) |

Результаты разных моделей в одной точке могут не совпадать: карты построены на разных данных и
допущениях. Сравнение результатов — тема отдельного исследования.
