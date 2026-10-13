"""Базовый класс флагов настроек корреляций.

Каждая модель имеет собственный `<Name>Options(BaseOptions)` только с полями `bool`;
значение по умолчанию соответствует поведению оригинальной статьи. Описание каждого
флага (что меняет, когда включать, источник) обязательно и доступно через `describe()`.

Источник требований: ТЗ проекта, раздел «Флаги настроек».
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseOptions(BaseModel):
    """Неизменяемый набор булевых флагов; неизвестные флаги и не-bool значения отклоняются.

    При создании подкласса проверяется, что каждое поле имеет тип `bool` и непустое
    `description`; иначе выбрасывается `TypeError`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Проверяет поля подкласса: тип `bool` и непустое описание.

        Параметры:
            **kwargs: аргументы создания класса (передаются pydantic дальше).

        Исключения:
            TypeError: поле не `bool` или без описания.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if field.annotation is not bool:
                raise TypeError(f"{cls.__name__}.{name}: флаг должен иметь тип bool")
            if not (field.description or "").strip():
                raise TypeError(f"{cls.__name__}.{name}: у флага обязательно непустое description")

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Возвращает описания флагов.

        Возвращает:
            Словарь «имя флага -> описание» (что меняет, когда включать, источник).
        """
        return {name: field.description or "" for name, field in cls.model_fields.items()}
