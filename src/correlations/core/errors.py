"""Исключения и предупреждения библиотеки.

Входные данные проверяются pydantic (`pydantic.ValidationError`); здесь — только
несходимость решателей и выход за область применимости корреляции.
"""

from collections.abc import Mapping

from correlations.core.patterns import FlowPattern


class ConvergenceError(Exception):
    """Итерационный решатель не сошёлся.

    Атрибуты:
        params: параметры задачи (имя -> значение), при которых решение не найдено;
            нужны для диагностики и воспроизведения.
    """

    def __init__(self, message: str, params: Mapping[str, float]) -> None:
        """Создаёт исключение.

        Параметры:
            message: описание проблемы.
            params: параметры задачи (имя -> значение) на момент отказа.
        """
        self.message = message
        self.params: dict[str, float] = dict(params)
        super().__init__(message, self.params)

    def __str__(self) -> str:
        """Возвращает сообщение вместе с параметрами задачи."""
        if not self.params:
            return self.message
        details = ", ".join(f"{name}={value:g}" for name, value in self.params.items())
        return f"{self.message} ({details})"


class OutOfRangeWarning(UserWarning):
    """Расчёт выполнен вне области применимости корреляции.

    Например, формула Steen–Wallis при Re плёнки < 3000. Выдаётся через `warnings.warn`.

    Атрибуты:
        context: числовые величины, связанные с предупреждением (например, `theta_deg`). Позволяют
            держать текст сообщения постоянным: стандартный фильтр Python тогда показывает
            предупреждение один раз на место вызова, а не на каждое значение величины.
    """

    def __init__(self, message: str, context: Mapping[str, float] | None = None) -> None:
        """Создаёт предупреждение.

        Параметры:
            message: текст предупреждения.
            context: числовые величины (имя -> значение); копируется. По умолчанию пусто.
        """
        super().__init__(message)
        self.context: dict[str, float] = dict(context) if context is not None else {}


class UnsupportedPatternError(ValueError):
    """Модуль режима вернул режим, который корреляция не умеет обрабатывать.

    Атрибуты:
        pattern: режим, не поддерживаемый корреляцией.
        source: имя корреляции (или пояснение), отказавшейся его обработать.
    """

    def __init__(self, pattern: FlowPattern, source: str) -> None:
        """Создаёт исключение.

        Параметры:
            pattern: неподдерживаемый режим.
            source: имя корреляции или пояснение.
        """
        self.pattern = pattern
        self.source = source
        super().__init__(f"{source}: режим {pattern.value} не поддерживается корреляцией")
