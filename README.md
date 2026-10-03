# Correlations

Библиотека корреляций для определения режима течения газожидкостной смеси в
трубах (bubble, slug, churn, annular, stratified и т.д.) по механистическим и
эмпирическим моделям из промысловой литературы.

## Статус

Проект в разработке. Публичный API собран в `correlations/__init__.py`, но
автоматических тестов пока нет — модели проверялись только вручную, по
исходным статьям и книге Брилла–Мукерджи «Многофазный поток в скважинах».

## Модели

| Модель                 | Класс               | Диапазон углов от горизонтали |
| ----------------------- | ------------------- | ------------------------------ |
| Beggs–Brill             | `BeggsBrillModel`   | −90°…90°                       |
| Mukherjee–Brill         | `MukherjeeBrillModel` | −90°…90°                     |
| Barnea (unified)        | `BarneaModel`       | −90°…90°                       |
| Taitel–Dukler           | `TaitelDuklerModel` | −10°…10°                       |
| Ansari                  | `AnsariModel`       | 75°…90°                        |
| Ansari (VBA-порт)       | `AnsariVBAModel`    | 75°…90°                        |
| Hasan–Kabir             | `HasanKabirModel`   | 45°…90°                        |

`AnsariVBAModel` сохранён только для воспроизводимости старых расчётов
(упрощённый критерий перехода в кольцевой режим); для новых задач
использовать `AnsariModel`.

## Установка

```bash
uv sync
```

## Быстрый старт

```python
from correlations import AnsariModel, FluidParams, PipeParams

pipe = PipeParams(diameter=0.1, roughness=1e-5, angle=90.0)
fluid = FluidParams(
    density_liquid=850.0,
    density_gas=15.0,
    viscosity_liquid=2e-3,
    viscosity_gas=1.5e-5,
    surface_tension=0.02,
)

model = AnsariModel(pipe, fluid)
result = model.classify(vsl=0.5, vsg=3.0)
print(result.pattern)  # FlowPatternCode.SLUG / BUBBLE / ANNULAR / ...
print(result.reason)   # причина перехода — для диагностики
```

Каждая модель принимает `PipeParams` и `FluidParams` и реализует общий
интерфейс `IFlowModel.classify(vsl, vsg) -> ClassificationResult`. Угол трубы
проверяется при создании модели: `ValueError`, если он вне `angle_range`
модели.

## Структура

- `base.py` — базовый интерфейс `IFlowModel`, `FlowPatternCode`,
  `ClassificationResult`;
- `constants.py` — универсальные физические константы (`GRAVITY`,
  `SINGLE_PHASE_VELOCITY_EPS`); эмпирические коэффициенты конкретных
  корреляций остаются в модулях моделей;
- `domain/params.py` — входные параметры (`PipeParams`, `FluidParams`,
  `SystemParams`);
- `physics/` — формулы, общие для нескольких моделей (трение, расслоённое
  течение);
- остальные модули — по одной механистической/эмпирической модели каждый.
