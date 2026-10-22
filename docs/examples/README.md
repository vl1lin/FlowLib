# Примеры использования

Самодостаточные скрипты на публичном API `correlations`. Запуск из корня репозитория:

```bash
uv run python docs/examples/01_quickstart.py
```

Свойства флюидов в примерах заданы константами (`_fluids.py`, СИ): воздух–вода при 0.1 МПа и
условная нефть–газ при 10 МПа. Подготовка свойств по давлению и температуре (PVT) в библиотеку
не входит. Графики тоже: примеры печатают таблицы.

| № | Файл | О чём | Раздел руководства |
|---|---|---|---|
| 1 | [`01_quickstart.py`](01_quickstart.py) | `FlowState`, `get_correlation`, `calc`, разбор результата | [3](../tutorials/03-input-data.md), [5](../tutorials/05-correlations/README.md) |
| 2 | [`02_compare_correlations.py`](02_compare_correlations.py) | Все четыре корреляции на одном состоянии | [4](../tutorials/04-models-overview.md) |
| 3 | [`03_flow_models.py`](03_flow_models.py) | Модели режима отдельно и внедрение в корреляцию, `UnsupportedPatternError` | [6](../tutorials/06-flow-patterns/README.md) |
| 4 | [`04_options_and_describe.py`](04_options_and_describe.py) | Флаги: `describe_*`, передача, проверка | [5](../tutorials/05-correlations/README.md) |
| 5 | [`05_validation_and_errors.py`](05_validation_and_errors.py) | `ValidationError`, `ConvergenceError`, `OutOfRangeWarning` | [2](../tutorials/02-validation-and-errors.md) |
| 6 | [`06_well_profile.py`](06_well_profile.py) | Профиль давления в скважине по участкам | [7](../tutorials/07-pressure-gradient.md) |
| 7 | [`07_sweep_table.py`](07_sweep_table.py) | Таблица «угол × скорость газа» | [7](../tutorials/07-pressure-gradient.md) |

Примеры проверяются тестом [`tests/test_examples.py`](../../tests/test_examples.py): каждый
скрипт должен завершаться с кодом 0.
