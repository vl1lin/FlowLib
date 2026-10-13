"""Тесты `core/options.py::BaseOptions` (ТЗ, «Флаги настроек»).

Ожидаемый API: `from correlations.core.options import BaseOptions`.
Свойства флагов конкретных моделей (описание, влияние) проверяются на этапах 4–5.
"""

import pytest
from pydantic import Field, ValidationError

from correlations.core.options import BaseOptions


class DemoOptions(BaseOptions):
    """Опции-образец для проверки базового класса."""

    flag_a: bool = Field(default=False, description="Что меняет A; когда включать; источник.")
    flag_b: bool = Field(default=True, description="Что меняет B; когда включать; источник.")


def test_defaults() -> None:
    opts = DemoOptions()
    assert opts.flag_a is False
    assert opts.flag_b is True


def test_describe_returns_name_to_description() -> None:
    assert DemoOptions.describe() == {
        "flag_a": "Что меняет A; когда включать; источник.",
        "flag_b": "Что меняет B; когда включать; источник.",
    }


def test_unknown_flag_rejected() -> None:
    with pytest.raises(ValidationError):
        DemoOptions.model_validate({"flag_z": True})


@pytest.mark.parametrize("value", ["True", "true", 1, 0], ids=["Str", "str", "int1", "int0"])
def test_strict_rejects_non_bool(value: object) -> None:
    with pytest.raises(ValidationError):
        DemoOptions.model_validate({"flag_a": value})


def test_model_validate_with_flags() -> None:
    opts = DemoOptions.model_validate({"flag_a": True})
    assert opts.flag_a is True


def test_frozen() -> None:
    opts = DemoOptions()
    with pytest.raises(ValidationError):
        opts.flag_a = True  # type: ignore[misc]  # проверка frozen


# Проектные решения агента (ТЗ говорит «description обязателен», «только bool») — на подтверждение:
def test_field_without_description_is_rejected_at_class_creation() -> None:
    with pytest.raises(TypeError):

        class NoDescription(BaseOptions):  # pyright: ignore[reportUnusedClass]
            flag: bool = False


def test_non_bool_field_is_rejected_at_class_creation() -> None:
    with pytest.raises(TypeError):

        class NotBool(BaseOptions):  # pyright: ignore[reportUnusedClass]
            level: int = Field(default=1, description="Не флаг.")
