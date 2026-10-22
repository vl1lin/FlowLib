"""Модели режима потока: отдельный расчёт и подстановка в корреляцию.

Запуск: uv run python docs/examples/03_flow_models.py
"""

from _fluids import OIL_GAS, make_state, setup_console

from correlations import UnsupportedPatternError, get_correlation, get_flow_model, list_flow_models


def main() -> None:
    """Определяет режим всеми моделями, затем подставляет модели режима в корреляции."""
    state = make_state(OIL_GAS, v_sl_m_s=0.3, v_sg_m_s=2.0, d_m=0.1, theta_deg=90.0)

    print("1. Режим потока по каждой модели")
    for name in list_flow_models():
        flow = get_flow_model(name).predict(state)
        print(f"  {name:<20}{flow.pattern.name:<18}(промежуточных величин: {len(flow.details)})")

    print("\n2. Hasan–Kabir с разными моделями режима")
    # Модель режима передаётся именем или готовым экземпляром.
    for flow_model in ("hasan_kabir_fp", "barnea", get_flow_model("taitel_dukler")):
        correlation = get_correlation("hasan_kabir", flow_model=flow_model)
        result = correlation.calc(state)
        print(
            f"  {correlation.flow_model.name:<20}{result.pattern.name:<18}"
            f"H_L={result.holdup:.3f}  -dp/dL={result.dpdl_pa_m:.0f} Па/м"
        )

    print("\n3. Несовместимая пара корреляция + модель режима")
    try:
        get_correlation("beggs_brill", flow_model="barnea").calc(state)
    except UnsupportedPatternError as error:
        print(f"  UnsupportedPatternError: {error}")


if __name__ == "__main__":
    setup_console()
    main()
