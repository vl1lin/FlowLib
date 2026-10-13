"""Тесты публичного API `correlations` (этап 6).

Ожидаемый `correlations/__init__.py`: экспортирует (через `__all__`)
    реестр: get_correlation, get_flow_model, list_correlations, list_flow_models,
        describe_correlation, describe_flow_model;
    типы: FlowState, FlowPattern, FlowPatternResult, CorrelationResult, BaseOptions,
        IFlowModel, ICorrelation;
    ошибки: ConvergenceError, OutOfRangeWarning, UnsupportedPatternError;
    корреляции и их опции: BeggsBrill(+Options), MukherjeeBrill(+Options), HasanKabir(+Options),
        Ansari(+Options);
    модели режима и их опции: BeggsBrillMap, MukherjeeBrillFlowModel, TaitelDuklerModel,
        BarneaModel, HasanKabirFlowModel, AnsariFlowModel (+ `*Options`).
Импорт `correlations` не должен тянуть старые плоские модули и `ansari_vba`.
README.md в корне: для каждой корреляции и модели режима таблица флагов
(имена флагов из `describe()` присутствуют в тексте README).
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import correlations
from correlations import registry

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_EXPORTS = {
    "get_correlation",
    "get_flow_model",
    "list_correlations",
    "list_flow_models",
    "describe_correlation",
    "describe_flow_model",
    "FlowState",
    "FlowPattern",
    "FlowPatternResult",
    "CorrelationResult",
    "BaseOptions",
    "IFlowModel",
    "ICorrelation",
    "ConvergenceError",
    "OutOfRangeWarning",
    "UnsupportedPatternError",
    "BeggsBrill",
    "BeggsBrillOptions",
    "MukherjeeBrill",
    "MukherjeeBrillOptions",
    "HasanKabir",
    "HasanKabirOptions",
    "Ansari",
    "AnsariOptions",
    "BeggsBrillMap",
    "BeggsBrillMapOptions",
    "MukherjeeBrillFlowModel",
    "MukherjeeBrillFlowOptions",
    "TaitelDuklerModel",
    "TaitelDuklerOptions",
    "BarneaModel",
    "BarneaOptions",
    "HasanKabirFlowModel",
    "HasanKabirFlowOptions",
    "AnsariFlowModel",
    "AnsariFlowOptions",
}


def test_all_contains_expected_exports() -> None:
    assert set(correlations.__all__) >= EXPECTED_EXPORTS


def test_all_names_exist_and_unique() -> None:
    assert len(correlations.__all__) == len(set(correlations.__all__))
    for name in correlations.__all__:
        assert hasattr(correlations, name), name


def test_exports_are_same_objects_as_in_modules() -> None:
    assert correlations.get_correlation is registry.get_correlation
    assert correlations.FlowState is importlib.import_module("correlations.core.state").FlowState
    assert (
        correlations.BeggsBrill
        is importlib.import_module("correlations.models.beggs_brill").BeggsBrill
    )


def test_top_level_usage(aw_state) -> None:  # type: ignore[no-untyped-def]
    model = correlations.get_correlation("mukherjee_brill")
    result = model.calc(aw_state(0.1, 1.0, 0.0508, 45.0))
    assert isinstance(result, correlations.CorrelationResult)


def test_import_does_not_load_legacy_modules() -> None:
    code = (
        "import sys, correlations;"
        "legacy = ['correlations.ansari', 'correlations.barnea', 'correlations.beggs_brill',"
        " 'correlations.hasan_kabir', 'correlations.mukherjee_brill', 'correlations.taitel_dukler',"
        " 'correlations.base', 'correlations.domain', 'correlations.physics',"
        " 'correlations.ansari_vba'];"
        "loaded = [m for m in legacy if m in sys.modules];"
        "print(','.join(loaded)); sys.exit(1 if loaded else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_py_typed_marker_present() -> None:
    assert (Path(correlations.__file__).parent / "py.typed").is_file()


@pytest.fixture(scope="module")
def readme_text() -> str:
    path = ROOT / "README.md"
    assert path.is_file(), "README.md отсутствует"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", registry.list_correlations())
def test_readme_documents_correlation_flags(name: str, readme_text: str) -> None:
    assert f"`{name}`" in readme_text
    for flag in registry.describe_correlation(name):
        assert f"`{flag}`" in readme_text, f"{name}.{flag} не описан в README"


@pytest.mark.parametrize("name", registry.list_flow_models())
def test_readme_documents_flow_model_flags(name: str, readme_text: str) -> None:
    assert f"`{name}`" in readme_text
    for flag in registry.describe_flow_model(name):
        assert f"`{flag}`" in readme_text, f"{name}.{flag} не описан в README"


def test_readme_shows_registry_usage(readme_text: str) -> None:
    assert "get_correlation(" in readme_text
    assert "get_flow_model(" in readme_text
