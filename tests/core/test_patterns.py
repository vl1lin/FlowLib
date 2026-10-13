"""Тесты `core/patterns.py::FlowPattern` (ТЗ, раздел «Выход»: один общий StrEnum)."""

from enum import StrEnum

from correlations.core.patterns import FlowPattern

# reference: ТЗ, «Выход», перечень FlowPattern
EXPECTED_MEMBERS = {
    "SEGREGATED",
    "INTERMITTENT",
    "DISTRIBUTED",
    "TRANSITION",
    "BUBBLE",
    "DISPERSED_BUBBLE",
    "SLUG",
    "CHURN",
    "ANNULAR",
    "STRATIFIED",
    "STRATIFIED_SMOOTH",
    "STRATIFIED_WAVY",
    "ELONGATED_BUBBLE",
    "SINGLE_PHASE_LIQUID",
    "SINGLE_PHASE_GAS",
}


def test_is_str_enum() -> None:
    assert issubclass(FlowPattern, StrEnum)


def test_members_match_spec_exactly() -> None:
    assert {m.name for m in FlowPattern} == EXPECTED_MEMBERS


def test_values_are_unique() -> None:
    values = [m.value for m in FlowPattern]
    assert len(values) == len(set(values))


def test_members_behave_as_strings() -> None:
    assert isinstance(FlowPattern.SLUG, str)
    assert FlowPattern("slug") is FlowPattern.SLUG  # значение = имя в нижнем регистре
    assert FlowPattern.SLUG == "slug"
