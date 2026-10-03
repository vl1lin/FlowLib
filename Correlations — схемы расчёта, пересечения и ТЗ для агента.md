# Correlations — схемы расчёта, пересечения и ТЗ для агента

Oct 3, 2026 · @Sergio

## Сводка

Библиотека получает 4 полные корреляции (`ICorrelation`: режим + holdup + dp/dL) и 4 модуля режима (`IFlowModel`), построенных на общем ядре из \~12 переиспользуемых функций. Основа — оригиналы статей; монография Брилла–Мукерджи используется только как источник примеров и сверки, её формулы в эталон не берутся без сверки с оригиналом.

| Модуль | Тип | Источник | Что выдаёт |
| --- | --- | --- | --- |
| Beggs & Brill (1973) | ICorrelation | JPT 1973 | режим, H\_L, dp/dL |
| Mukherjee & Brill (1985) | ICorrelation | JERT 1985 + IJMF 1985 | режим, H\_L, dp/dL |
| Hasan & Kabir (1988) | ICorrelation | SPEPE 1988 + Erratum 1989 | режим, H\_L, dp/dL (труба и затрубье) |
| Ansari et al. (1994) | ICorrelation | SPE PF 1994 | режим, H\_L, dp/dL |
| Taitel & Dukler (1976) | IFlowModel | AIChE J 1976 | режим (+ h\_L/D для расслоенного) |
| Barnea (1987) | IFlowModel | IJMF 1987 | режим (+ h\_L/D, α\_s как детали) |
| Mukherjee & Brill flow patterns | IFlowModel | IJMF 1985 | режим (используется MB-корреляцией) |
| Beggs & Brill map | IFlowModel | JPT 1973 (+ revised-карта флагом) | режим (используется BB-корреляцией) |

Главные находки:

- Пример 4.7 (BB) и Пример 4.8 (MB) пересчитаны независимо и воспроизводятся: 4588 Па/м и 4734 Па/м. Для BB это требует флагов `payne_correction`, `rough_pipe_friction`, `clip_negative_c` и выключенного ускорения.
- По оригиналу Hasan–Kabir 1988 точка Примера 3.2 — **кольцевой** режим (v\_gs = 1,173 > 0,871 м/с), а не пробковый, как в Примере 4.10 монографии. Пример 4.10 построен на поздней наклонной версии модели и для оригинала эталоном не является.
- По оригиналу Ansari для точки Примера 3.2 L\_c = 3,17 м > L\_TB = 1,84 м, т.е. пробка «развивающаяся». Пример 4.9 этот шаг пропускает; совпадение с ним возможно только при `developing_slug=False`.
- Barnea Fig. 6 получен; логика восстановлена (раздел «Схемы расчёта»). В блок-схеме «Condition \[4\]» в ветке L — опечатка, правильно \[14\].
- Знак угла: у Taitel–Dukler α > 0 — **нисходящий** поток, у всех остальных θ > 0 — восходящий. В библиотеке единое соглашение θ > 0 вверх; модуль TD пересчитывает α = −θ внутри.

## Архитектура

Два интерфейса и общее ядро: корреляция = модуль режима + замыкания holdup/трения + сборка градиента. Любой блок заменяем без правки остальных.

### Структура пакета

```
src/correlations/
  __init__.py           # публичный API: get_correlation, модели, FlowState, FlowPattern, исключения
  core/                 # общее ядро, без знаний о конкретных корреляциях
    state.py            # pydantic: FlowState (вход), FlowPatternResult, CorrelationResult
    patterns.py         # FlowPattern(StrEnum)
    options.py          # BaseOptions (pydantic, только bool)
    errors.py           # ConvergenceError, OutOfRangeWarning
    friction.py         # laminar, colebrook, bb_smooth(13), blasius_fanning(0.046/16)
    dimensionless.py    # N_Lv, N_gv, N_L, N_d, N_FR, Re, λ
    mixture.py          # ρ_ns, μ_ns, ρ_s, μ_s
    bubbles.py          # Harmathy v∞, Taylor v∞T (0.35 / 0.345 / Wallis C2)
    stratified.py       # геометрия по h̃ и по δ, решатель уравнения TD (7)
    transitions.py      # общие критерии (раздел «Пересечения»)
    annular.py          # Wallis-фактор, v_crit, entrainment (Wallis, Steen–Wallis)
    gradient.py         # сборка dp/dL с Ek
    solvers.py          # Newton / bisection / brent с контролем сходимости
  flow_models/          # IFlowModel
    taitel_dukler.py, barnea.py, mukherjee_brill_fp.py, beggs_brill_map.py
  models/               # ICorrelation
    beggs_brill.py, mukherjee_brill.py, hasan_kabir.py, ansari.py
  registry.py           # get_correlation("beggs_brill", payne_correction=True)
tests/                  # зеркально пакету + tests/reference/*.json
```

Соотнесение с текущей структурой (CLAUDE.md): файлы `taitel-dukler.py`, `mukherdge-brill.py`, `hasan-khabir` (без расширения), `physics.pay` нельзя импортировать в Python (дефис, расширение). Переименование и перенос в `core/`, `flow_models/`, `models/` — изменение вне задачи, агент обязан сначала предложить план и получить подтверждение. `answer_vba.py` не трогать без отдельного указания.

### Интерфейсы

```python
from typing import ClassVar, Protocol, runtime_checkable

@runtime_checkable
class IFlowModel(Protocol):
    name: ClassVar[str]
    @property
    def options(self) -> BaseOptions: ...
    def predict(self, state: FlowState) -> FlowPatternResult: ...

@runtime_checkable
class ICorrelation(Protocol):
    name: ClassVar[str]
    @property
    def options(self) -> BaseOptions: ...
    @property
    def flow_model(self) -> IFlowModel: ...   # инъекция через конструктор
    def calc(self, state: FlowState) -> CorrelationResult: ...

# конструктор любой модели:
# BeggsBrill(options: BeggsBrillOptions | None = None,
#            flow_model: IFlowModel | None = None) -> None
```

### Вход `FlowState` (pydantic, СИ, неизменяемый)

```python
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator

class FlowState(BaseModel):
    """Локальное состояние потока в сечении трубы. Все величины в СИ."""
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    v_sl_m_s: float = Field(ge=0, description="Приведённая скорость жидкости, м/с")
    v_sg_m_s: float = Field(ge=0, description="Приведённая скорость газа, м/с")
    rho_l_kg_m3: float = Field(gt=0, description="Плотность жидкости, кг/м³")
    rho_g_kg_m3: float = Field(gt=0, description="Плотность газа, кг/м³")
    mu_l_pa_s: float = Field(gt=0, description="Вязкость жидкости, Па·с")
    mu_g_pa_s: float = Field(gt=0, description="Вязкость газа, Па·с")
    sigma_n_m: float = Field(gt=0, description="Поверхностное натяжение, Н/м")
    d_m: float = Field(gt=0, description="Внутр. диаметр трубы (затрубье — обсадной), м")
    d_t_m: float = Field(default=0.0, ge=0, description="Наруж. диаметр НКТ, 0 — труба, м")
    eps_m: float = Field(default=0.0, ge=0, description="Абс. шероховатость, м")
    theta_deg: float = Field(ge=-90, le=90, description="Угол от горизонтали, > 0 — вверх, град")
    p_pa: float = Field(gt=0, description="Давление, Па")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        # rho_l > rho_g; d_t < d; v_sl + v_sg > 0; eps < d
        ...
        return self
```

Нарушение любого условия → `pydantic.ValidationError`. Производные величины (v\_m, λ, θ в радианах) — `@computed_field` или функции `core/`, не поля ввода.

### Выход

- `FlowPatternResult(BaseModel, frozen)`: `pattern: FlowPattern`, `details: dict[str, float]` (все промежуточные величины: границы, h̃\_L, α\_s и т.п.; ключи с единицами, например `h_l_over_d`, `v_sg_annular_m_s`).
- `CorrelationResult(BaseModel, frozen)`: `pattern: FlowPattern`, `holdup: float = Field(ge=0, le=1)`, `dpdl_pa_m` = −dp/dL \[Па/м\] (> 0 — давление падает по потоку), `dpdl_gravity_pa_m`, `dpdl_friction_pa_m`, `dpdl_acceleration_pa_m`, `details: dict[str, float]`. `model_validator` проверяет сумму компонент (rtol 1e-10). Для формы с Ek (BB eq. 1) ускорение = dpdl − (grav + fric).
- `FlowPattern(StrEnum)` — один общий enum: `SEGREGATED, INTERMITTENT, DISTRIBUTED, TRANSITION` (BB), `BUBBLE, DISPERSED_BUBBLE, SLUG, CHURN, ANNULAR, STRATIFIED, STRATIFIED_SMOOTH, STRATIFIED_WAVY, ELONGATED_BUBBLE, SINGLE_PHASE_LIQUID, SINGLE_PHASE_GAS`.

### Флаги настроек

```python
class BaseOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Имя флага -> описание (что меняет, когда включать, источник)."""
        return {name: f.description or "" for name, f in cls.model_fields.items()}

class BeggsBrillOptions(BaseOptions):
    payne_correction: bool = Field(
        default=False,
        description="H_L x 0.924 при θ>0 (не ниже λ), x 0.685 при θ<0. "
                    "Включать для скважин и трубопроводов: оригинал завышает H_L. "
                    "Источник: Payne et al. 1979.",
    )
    ...
```

- Каждая модель имеет свой `<Name>Options(BaseOptions)` только с полями `bool`; значение по умолчанию = поведение оригинала статьи.
- `strict=True`: строки `"True"`, числа 0/1 не принимаются; неизвестный флаг (`extra="forbid"`) → `pydantic.ValidationError`.
- Передача: `BeggsBrill(BeggsBrillOptions(payne_correction=True))` или `get_correlation("beggs_brill", payne_correction=True)`; registry вызывает `OptionsCls.model_validate(flags)`.
- `description` обязателен у каждого поля (что меняет, когда включать, источник); тест проверяет непустое описание. Тот же текст выдаёт `describe()` и README.
- Числовые константы (C\_L = 0.8, s = 0.01, C' = 30 и т.д.) — `Final`-константы модуля со ссылкой на уравнение, не флаги.
- Полный список флагов — в разделе «Схемы расчёта» у каждой модели.

### Общие правила

- Типизация полная, `uv run pyright` без ошибок (режим strict для `src/correlations`). Скалярная физика — `float`; массивы, если появятся, — `numpy.typing.NDArray[np.float64]`.
- Решатели — собственные в `core/solvers.py` (Newton с откатом на бисекцию/brent). Если понадобится scipy — без типовых заглушек он ломает strict-pyright; добавление зависимости (`uv add`, `scipy-stubs`) только с подтверждения.
- Однофазные пределы: `v_sg = 0` → `SINGLE_PHASE_LIQUID`, H\_L = 1; `v_sl = 0` → `SINGLE_PHASE_GAS`, H\_L = 0; градиент — однофазный (Colebrook). Модели режима в этих случаях не вызываются.
- Коэффициент трения в ядре всегда Moody/Darcy; модели, работающие с Fanning (TD, Barnea, HK), явно делят на 4 в одном месте. Главный источник ошибок, проверяется тестами.
- Исключения: вход — `pydantic.ValidationError`; несходимость — `ConvergenceError` (с параметрами); выход за область применимости — `warnings.warn(..., OutOfRangeWarning)` (например, Steen–Wallis при Re плёнки < 3000).
- Docstrings: в начале каждого модуля корреляции — источник и схема расчёта (из раздела «Схемы расчёта»); у каждой функции — что делает, параметры с единицами, что возвращает, источник «Автор год, ур. N».
- Коэффициенты и критерии переходов меняются только по таблице «Опечатки». Любое другое расхождение кода с источником — сообщить в отчёте, не исправлять молча.

## Пересечения корреляций

Пересечений 14; каждое реализуется одной функцией ядра с параметрами, различия между авторами передаются аргументами, а не копиями кода.

| # | Общий элемент | Где используется | Различия между авторами | Функция ядра |
| --- | --- | --- | --- | --- |
| 1 | Числа Данса–Роса N\_Lv, N\_gv, N\_L | BB (N\_Lv в C), MB (все три) | MB: N\_L с g (в JERT опечатка без g) | `dimensionless.duns_ros()` |
| 2 | No-slip λ, ρ\_ns, μ\_ns, Re\_ns | BB, MB, Ansari (dispersed bubble), HK (через λ\_LC) | нет | `mixture.no_slip()` |
| 3 | Коэффициент трения Moody: 64/Re + Colebrook | MB, Ansari, HK, BB (флаг), Barnea | BB по умолчанию — гладкая формула (13); HK приложение — Blasius 0.079Re^−0.25 (Fanning) в кольцевом; TD — 0.046Re^−0.2 / 16/Re (Fanning) | `friction.*` |
| 4 | Скорость всплытия пузырька Harmathy 1.53\[gσΔρ/ρ\_L²\]^0.25 | Barnea \[3\]/\[4\], HK (3), Ansari (3), Ansari bubble (21) с H\_L^0.5 | HK eq. 17 — 1.50 (флаг); Ansari — множитель H\_L^0.5 в пузырьковом и в пробке | `bubbles.harmathy(coef=1.53)` |
| 5 | Скорость пузыря Тейлора C·√(gdΔρ/ρ\_L) | HK (C2 = 0.345 или Wallis eq. 23), Ansari (0.35), затрубье HK (30) | коэффициент; HK — d\_c − d\_t в затрубье | `bubbles.taylor(c2)` |
| 6 | Переход пузырьковый → пробковый при α = 0.25 | Barnea \[4\], Ansari (2), HK (2) | Barnea/Ansari: C0 = 1 → v\_sg = v\_sl/3 + 0.25v∞·sinβ (у Ansari без sin); HK: C0 = 1.2 → 0.429v\_sl + 0.357v∞ | `transitions.bubble_slug(alpha=0.25, c0, use_sin)` |
| 7 | Минимальный диаметр для пузырькового режима D > 19\[Δρσ/(ρ\_L²g)\]^0.5 | Barnea \[1\] (19), Ansari (19.01) | коэффициент 19 / 19.01 | `transitions.d_min(coef)` |
| 8 | Переход в дисперсно-пузырьковый (Taitel 1980 / Barnea 1986) | Barnea \[5\]–\[7\], Ansari (4), HK (6) | Barnea: d\_C = min(d\_CD, d\_CB), Fanning f, 3/5 и −2/5; Ansari: только d\_CD, Moody f (f/2d); HK: явная формула v\_M^1.12 (вертикаль, Taitel 1980) | `transitions.dispersed_bubble_barnea()`, `..._taitel1980()` |
| 9 | Предел плотной упаковки пузырьков | Barnea \[8\] α = 0.52, HK α = 0.52, Ansari (5) α = 0.76 | значение α | `transitions.max_packing(alpha)` |
| 10 | Переход в кольцевой v\_sg = 3.1\[σgΔρ/ρ\_g²\]^0.25 | HK (13), Ansari (6) | нет | `transitions.annular_turner()` |
| 11 | Устойчивость плёнки и перемыкание ядра (Barnea 1986) | Barnea \[15\]–\[17\], Ansari (7)–(15), (96) | Barnea: α\_L ≥ 0.24 (=0.5·R\_sm), X,Y обычные; Ansari: H\_LF > 0.12, X\_M, Y\_M модифицированные, учёт уноса | `annular.film_stability(...)` |
| 12 | Фактор межфазного трения Уоллиса 1 + 300δ̃ ≈ 1 + 75α\_L | Barnea \[15\], Ansari Z (84), HK (38)–(39) | Ansari переключает на Whalley–Hewitt при F\_E < 0.9 | `annular.wallis_z()` |
| 13 | Унос жидкости: v\_crit = 10⁴ v\_sg μ\_g/σ (ρ\_g/ρ\_L)^0.5 | Ansari (71)–(72) Wallis exp, HK (34)–(36) Steen–Wallis | формула E(v\_crit) | `annular.v_crit()`, `entrainment_wallis()`, `entrainment_steen_wallis()` |
| 14 | Геометрия расслоенного потока и равновесный уровень | TD (7)–(16), Barnea \[9\]–\[14\], MB (18)–(25) | TD/Barnea — через h̃\_L; MB — через угол δ, H\_L задаётся корреляцией. Знак Y: TD — α > 0 вниз, Barnea — β > 0 вверх (в уравнении (7) +4Y при β-соглашении) | `stratified.geometry(h)`, `stratified.delta_from_holdup()`, `stratified.equilibrium_level(X, Y)` |

Дополнительно общие:

- Ускорение через Ek = ρ\_s v\_m v\_sg / p (BB eq. 1, MB Δp\_a, HK кольцевой eq. 33 с ρ\_c v\_g² / p) — `gradient.with_kinetic_term(grav, fric, ek)`.
- Критерий Кельвина–Гельмгольца с C2 = 1 − h̃\_L: TD (25) и Barnea \[9\] — одна функция.
- Критерий SS/SW по Джеффрису: TD (30) и Barnea \[20\] — одна функция, s = 0.01.
- Барнеа и Ansari используют один и тот же критерий дисперсного режима в разной записи. Тест: при d\_CB > d\_CD и одинаковом f (Fanning·4 = Moody) обе записи дают одинаковую границу (rtol 1e-9).

## Схемы расчёта

Каждая схема — то, что агент обязан реализовать; номера уравнений — по оригинальным статьям. Флаги: имя, значение по умолчанию (= оригинал), что меняет и когда включать.

### Beggs & Brill (1973) — ICorrelation

1. λ = v\_sl/v\_m, G\_m, N\_FR = v\_m²/(gd), N\_Lv, ρ\_ns, μ\_ns, Re\_ns (14).
2. Режим — `BeggsBrillMap`: X = ln λ; L1 (9), L2 (10). N\_FR < L1 → SEGREGATED; N\_FR > L1 и > L2 → DISTRIBUTED; L1 < N\_FR < L2 → INTERMITTENT.
3. H\_L(0) = A λ^α N\_FR^β (19–21, коэффициенты Table 1); ограничение H\_L(0) ≥ λ.
4. C (22–25): θ > 0 — по режиму (distributed → C = 0); θ < 0 — одна формула (25); θ = 0 → ψ = 1.
5. ψ = 1 + C\[sin(1.8θ) − sin³(1.8θ)/3\]; H\_L(θ) = H\_L(0)ψ, ограничение 0 ≤ H\_L ≤ 1.
6. y = λ/H\_L²; S по (16), при 1 < y < 1.2 — S = ln(2.2y − 1.2), при y = 1 — S = 0; f\_tp = f\_ns·e^S.
7. f\_ns — гладкая формула (13).
8. −dp/dL по (1): \[ρ\_tp g sinθ + f\_tp ρ\_ns v\_m²/(2d)\] / (1 − ρ\_tp v\_m v\_sg/p).

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `revised_map` | False | Карта L1–L4 с переходной зоной (монография 4.110–4.119), H\_L в зоне — интерполяция A = (L3 − N\_FR)/(L3 − L2) между segregated и intermittent | Сверка с PIPESIM «BB Revised»; убирает скачок H\_L на границе segregated/intermittent |
| `payne_correction` | False | H\_L × 0.924 при θ > 0 (не ниже λ), × 0.685 при θ < 0 | Скважины и трубопроводы: оригинал завышает H\_L и dp; Пример 4.7 |
| `rough_pipe_friction` | False | f\_ns по Colebrook с ε вместо гладкой (13) | Реальные (шероховатые) трубы; Пример 4.7 |
| `clip_negative_c` | False | C < 0 → C = 0 (ψ ≥ 1 при подъёме) | Совместимость с монографией и PIPESIM; Пример 4.7 |
| `include_acceleration` | True | Знаменатель 1 − Ek в (1) | Выключать для сверки с примерами монографии и при Ek → 1 (высокоскоростной газ) |

### Beggs & Brill map — IFlowModel

Шаг 2 выше, выделен в модуль; флаг `revised_map` тот же. Выход: pattern + L1…L4 в details.

### Mukherjee & Brill (1985) — ICorrelation

1. N\_Lv, N\_gv, N\_L = μ\_L\[g/(ρ\_L σ³)\]^0.25.
2. Режим — `MukherjeeBrillFlowModel` (ниже).
3. H\_L по (10): θ ≥ 0 — набор «uphill and horizontal»; θ < 0 и STRATIFIED — набор «downhill stratified»; θ < 0 и прочие — «downhill other». Степень у C3 — sin²θ.
4. BUBBLE/SLUG: −dp/dL = \[f\_ns ρ\_s v\_m²/(2d) + ρ\_s g sinθ\]/(1 − Ek), Ek = ρ\_s v\_m v\_sg/p, f\_ns — Moody (Colebrook) по Re\_ns.
5. ANNULAR: H\_R = λ/H\_L, f\_R из Table 2 (вне 0.01…10 → 1.0), f = f\_R·f\_ns; −dp/dL = \[f ρ\_ns v\_m²/(2d) + ρ\_s g sinθ\]/(1 − Ek).
6. STRATIFIED: δ из H\_L = (δ − sinδ)/2π (20) (решатель, старт 0.001); h\_L/d (18); D\_EL, D\_Eg (24–25); P = πd, P\_g = (1 − δ/2π)P, P\_L = P − P\_g; v\_L = v\_sl/H\_L, v\_g = v\_sg/(1 − H\_L); f\_L, f\_g — Moody по Re с D\_E; τ\_wL, τ\_wg; −dp/dL = τ\_wg P\_g/A\_g + ρ\_g g sinθ (15, τ\_i = 0). Ускорение = 0.

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `include_acceleration` | True | 1 − Ek в BUBBLE/SLUG/ANNULAR | Выключать для сверки с Примером 4.8 |
| `stratified_combined_momentum` | False | Уравнение (17) (сумма фаз, без τ\_i) вместо (15) | Трубы большого диаметра (рекомендация авторов); граница диаметра не задана |
| `wall_shear_as_published` | False | τ\_w = f ρ v²/2 (как напечатано: γv²/2g при Moody f) вместо f ρ v²/8 | Только для воспроизведения кода, реализованного «буквально» (сверка с PIPESIM) |
| `annular_fr_log_interp` | False | Интерполяция Table 2 по lg H\_R вместо линейной по H\_R | Таблица снята с графика в лог-шкале (Fig. 4); даёт более гладкую f\_R |

### Mukherjee & Brill flow patterns — IFlowModel

N\_gvSM \[2\]. Если N\_gv > N\_gvSM → ANNULAR. Иначе при θ > 0: N\_LvBS \[1\]; N\_Lv > N\_LvBS → BUBBLE, иначе SLUG. При θ ≤ 0: N\_gvBS \[3\], N\_LvST \[4\]. Если |θ| > 30°: N\_gv > N\_gvBS → (N\_Lv > N\_LvST → SLUG, иначе STRATIFIED), иначе BUBBLE. Если |θ| ≤ 30°: N\_Lv > N\_LvST → (N\_gv > N\_gvBS → SLUG, иначе BUBBLE), иначе STRATIFIED. Флагов нет. В \[4\] член −0.017·N\_gv линейный — так в обоих оригиналах, не исправлять.

### Hasan & Kabir (1988) — ICorrelation

Модель для вертикального подъёма; при θ ≠ 90° гравитация берётся с sinθ и выдаётся предупреждение. r = d\_t/d (0 для трубы).

1. v∞ = 1.53\[gσΔρ/ρ\_L²\]^0.25 (3).
2. Режим, порядок: (а) v\_M > v\_M,DB по (6) и f\_g,bubbly < 0.52 → DISPERSED\_BUBBLE; (б) v\_gs < 0.429v\_Ls + 0.357v∞ → BUBBLE; (в) v\_gs > 3.1\[σgΔρ/ρ\_g²\]^0.25 (13) → ANNULAR; (г) иначе Hewitt–Roberts (8)/(9) в единицах lbm/(ft·s²) (ρv²\_SI / 1.488): ρ\_g v\_gs² выше границы → CHURN, иначе SLUG; при ρ\_L v\_Ls² > 3300 → CHURN + предупреждение.
3. BUBBLE/DB: f\_g = v\_gs/(C0 v\_M + v∞), C0 = 1.2 + 0.371r; ρ\_M (21); трение f ρ\_M v\_M²/(2d), f — Moody по Re = ρ\_L v\_M d/μ\_L.
4. SLUG: f\_g = v\_gs/(C1 v\_M + v∞T); труба: C1 = 1.2, v∞T = 0.345√(gdΔρ/ρ\_L) (28); затрубье: C1 = 1.182 + 0.90r (29), v∞T = (0.30 + 0.22r)√(g(d − d\_t)Δρ/ρ\_L) (30). Трение f ρ\_L(1 − f\_g) v\_M²/(2d) (31).
5. CHURN: как SLUG, но C1 = 1.15 + 0.90r (Приложение).
6. ANNULAR: v\_crit = v\_gs μ\_g/σ·(ρ\_g/ρ\_L)^0.5·10⁴; E по (35)/(36), 0 ≤ E ≤ 1; ρ\_c (37b); x = ρ\_g v\_gs/(ρ\_g v\_gs + E ρ\_L v\_Ls); X (42); f\_g = (1 + X^0.8)^−0.378 (40); v\_g = v\_gs/f\_g; f\_c = 0.079 Re\_g^−0.25 \[1 + 75(1 − f\_g)\] (Fanning), Re\_g = ρ\_g v\_gs d/μ\_g; −dp/dL по (33) с 4·f\_c при переводе в Moody. H\_L = (1 − f\_g) + f\_g(1 − f\_gc).

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `porteous_low_pressure` | False | При p < 1013 кПа граница slug/churn по Porteous (7) v\_M = 0.103√(gdΔρ/ρ\_g) | Низконапорные скважины, лабораторные данные (так рекомендует текст статьи) |
| `wallis_c2` | False | C2 по Wallis (23)–(26) вместо 0.345 | Вязкая нефть или малый диаметр: N\_f < 300 или N\_Eo < 100 |
| `rise_coef_150` | False | 1.50 вместо 1.53 в v∞ для пузырькового H\_L (ур. 17) | Буквальное воспроизведение ур. 17 |
| `churn_as_slug` | False | Эмульсионный режим считается с C1 пробкового | Рекомендация Govier–Aziz из текста; сглаживает скачок slug/churn |
| `include_acceleration` | True | Знаменатель 1 − ρ\_c v\_g²/p в (33) | Выключать для оценки вклада ускорения |

### Ansari et al. (1994) — ICorrelation

1. Режим (порядок по Примеру 4.9, в статье не задан): (а) DB: левая часть (4) ≥ 0.725 + 4.15(v\_sg/v\_m)^0.5 (f — Moody по Re\_ns) и v\_sg < 3.17v\_sl (5) → DISPERSED\_BUBBLE; (б) ANNULAR: v\_sg > (6) и решение плёнки (шаг 4): δ < δ\_min из (8) и H\_LF + λ\_LC·A\_C/A ≤ 0.12 (12); (в) d > d\_min и v\_sg < 0.25v\_s + 0.333v\_sl (2) → BUBBLE; (г) иначе SLUG (churn входит в slug).
2. BUBBLE: DB — no-slip ρ\_TP, μ\_TP; пузырьковый — H\_L из (22) (корень на (0, 1)); −dp/dL = ρ\_TP g sinθ + f ρ\_TP v\_m²/(2d), f — Moody по Re\_TP. Ускорение = 0.
3. SLUG: v\_TB (34), H\_gLS (38), v\_gLS (35); Ā = H\_gLS(v\_TB − v\_gLS) + v\_m; H\_LTB — корень (39) (Newton, старт 0.15, корень единственный на (0, 1)); v\_LTB (37), v\_LLS (32), v\_gTB (33), β (29); L\_LS = 30d, L\_TB (49). Проверка развития: L\_c (44)–(47) ≥ L\_TB → развивающаяся пробка: L\_TB\* (54)–(57), β\* = L\_TB\*/(L\_TB\* + L\_LS), H\_LTBA (64), гравитация (62); иначе (60). Трение (65) с β или β\*. H\_L = (1 − β)H\_LLS + βH\_LTB (развивающаяся: H\_LTBA).
4. ANNULAR: F\_E (71)–(72), ограничение ≥ 0; λ\_LC (70), ρ\_C, μ\_C, v\_SC (89); (dp/dL)\_SC, (dp/dL)\_SL — Moody; f\_F по Re\_F (74) с d\_HF (76) — итерация вместе с δ; X\_M, Y\_M (9)–(10); Z (84)/(85); δ — корень (96); φ\_C² (99); −dp/dL = φ\_C²(dp/dL)\_SC + ρ\_C g sinθ (102). H\_L = 4δ(1 − δ) + λ\_LC(1 − 2δ)².

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `developing_slug` | True | Учёт развивающейся пробки (44)–(64) | Выключать для сверки с Примером 4.9 и PIPESIM (если там этой ветки нет) |
| `vo_shoham_a_as_published` | False | Ā по напечатанному (40) вместо согласованного с (32), (35) | Только буквальное воспроизведение статьи; даёт H\_LTB = 0.119 вместо 0.130 |
| `bridging_with_delta_min` | True | В критерии (12) берётся δ\_min (текст статьи); False — δ из (96) (Пример 4.9) | См. открытый вопрос; False — физически ближе к Barnea |
| `bubble_slug_sin_theta` | False | 0.25v\_s·sinθ в (2), как у Barnea \[4\] | Наклонные скважины |
| `ff_over_fsl_unity` | False | f\_F/f\_SL = 1 без итерации | Быстрый расчёт; Пример 4.9 |

### Taitel & Dukler (1976) — IFlowModel

1. α = −θ (α > 0 — вниз). (dP/dx)^S для фаз: Fanning, C = 0.046, n = 0.2 (турб.) / C = 16, n = 1 (лам.); X (8), Y (9), F (26), K (31), T (37).
2. h̃\_L — корень (7) на (0, 1); при нескольких корнях — наименьший (допущение, см. вопросы). Ламинарность каждой фазы — по Re с фактической скоростью и гидравлическим диаметром (6).
3. (25) F²·\[ũ\_G² (dÃ\_L/dh̃)/(C2² Ã\_G)\] < 1, C2 = 1 − h̃\_L → расслоенный: (30) выполнено → STRATIFIED\_WAVY, иначе STRATIFIED\_SMOOTH (s = 0.01).
4. Иначе h̃\_L < 0.5 → ANNULAR; иначе (36) выполнено → DISPERSED\_BUBBLE, иначе INTERMITTENT.
5. details: h̃\_L, H\_L,strat = Ã\_L/Ã, X, Y, F, K, T.

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `laminar_switch` | True | Выбор C, n по фактическому Re фазы | False — только турбулентный расчёт, как Fig. 4 (сверка с картой) |
| `use_moody_friction` | False | (dP/dx)^S и f по Colebrook с ε вместо гладких 0.046Re^−0.2 | Шероховатые трубы (предложение авторов в конце статьи) |

### Barnea (1987) — IFlowModel

Логика по Fig. 6, θ = β > 0 вверх.

1. DB: α\_ns = v\_sg/v\_m; d\_max = 0.725 + 4.15(v\_sg/v\_m)^0.5^(3/5)(2f\_M v\_m³/D)^(−2/5); d\_C = min(d\_CD (6), d\_CB (7)), при cosβ = 0 → d\_C = d\_CD. Если d\_max < d\_C и α\_ns < 0.52 → DISPERSED\_BUBBLE.
2. A: h̃\_L — равновесный уровень (общий решатель, X, Y по \[12\]); если \[9\] < 1 (устойчиво) → проверка L: если \[14\] не выполнено → STRATIFIED, далее \[20\] или \[23\] выполнено → STRATIFIED\_WAVY, иначе STRATIFIED\_SMOOTH. Если \[9\] ≥ 1 или \[14\] выполнено → шаг 3. При |β| = 90° расслоенный режим невозможен (cosβ = 0).
3. J: α\_L — корень \[15\]; \[16\] выполнено или α\_L ≥ 0.24 \[17\] → шаг 4, иначе ANNULAR.
4. Пузырьковый возможен, если D > D\_crit \[1\] и |β| ≥ β\_crit \[2\]. Да: v\_sl > U\_LS по \[4\] (α < 0.25) → BUBBLE, иначе → шаг 6. Нет → шаг 5.
5. α\_s по \[18\] с d\_C (внутренняя скобка < 0 → α\_s = 0); α\_s = 0 и β = 0 → ELONGATED\_BUBBLE.
6. α\_s ≤ 0.52 → SLUG, иначе CHURN.
7. f\_LS/f\_L в \[14\] — отношение коэффициентов одной модели трения (гладкая труба даёт (D̃\_L Ũ\_L)^−n из статьи).

| Флаг | По умолч. | Что меняет | Когда включать |
| --- | --- | --- | --- |
| `fm_noslip_reynolds` | False | f\_M по no-slip Re вместо Re = ρ\_L v\_m D/μ\_L | Высокое газосодержание; согласование с Ansari |
| `elongated_bubble_horizontal_only` | True | ELONGATED\_BUBBLE только при β = 0 (как в статье) | False — выделять EB при любом угле |
| `use_moody_friction` | True | f для X, Y, Z, f\_M — Colebrook с ε; False — гладкие Fanning-формулы TD | False — сверка с Fig. 1–2 (гладкие лабораторные трубы) |
| `split_churn` | True | Выделять CHURN по α\_s > 0.52 | False — CHURN отдаётся как SLUG (для моделей, где churn не считается отдельно) |

## Опечатки и принятые исправления

Все исправления — по правилу «подозрение на опечатку = опечатка»; агент реализует исправленную форму, в docstring указывает исходную.

| Источник | Место | Напечатано | Принято | Обоснование |
| --- | --- | --- | --- | --- |
| Barnea 1987 (md) | \[5\] | (σ/ρ\_L)^(3/2), (…)^(−2/3) | (σ/ρ\_L)^(3/5), (…)^(−2/5) | Согласование с \[18\] и Ansari (4) |
| Barnea 1987 (md) | \[15\] | (1 − α\_L)^(3/2) | (1 − α\_L)^(5/2) | Ansari (96) / монография 4.166 |
| Barnea 1987 | Fig. 6, ветка L | «Condition \[4\]» | \[14\] | \[4\] — критерий пузырькового режима |
| Barnea 1987 | \[19\] | ρ cosβ | g cosβ | TD (29) |
| Taitel–Dukler | (25) | ũ\_G | ũ\_G² | (23), Barnea \[9\] |
| Mukherjee–Brill JERT | N\_L | μ\_L\[1/(ρ\_L σ³)\]^0.25 | μ\_L\[g/(ρ\_L σ³)\]^0.25 | IJMF 1985 |
| Mukherjee–Brill JERT | (10) | непарные скобки | общепринятая форма | IJMF/Пример 4.8 |
| Mukherjee–Brill IJMF | \[3\] | sin 0 | sinθ | JERT (8) |
| Hasan–Kabir | (6), Приложение | ρ\_M | ρ\_L | Erratum 1989 |
| Hasan–Kabir | (23) | exp(3.37 N\_Eo/m) | exp((3.37 − N\_Eo)/m) | Wallis 1969; иначе C2 < 0 |
| Hasan–Kabir | (26b) | 69 N\_f^0.35 | 69 N\_f^−0.35 | сшивка с (26a), (26c) |
| Hasan–Kabir | (30) | g(d\_t/d\_c) | g(d\_c − d\_t) | Erratum 1989 |
| Hasan–Kabir | (37b) | v\_gs + E v\_gs | v\_gs + E v\_Ls | (37a) |
| Ansari | (18) | v\_TP = v\_M v\_SL + v\_Sg | v\_TP = v\_m | смысл, монография 4.175 |
| Ansari | (32) | v\_TB cc | v\_TB | монография 4.188 |
| Ansari | (40) | Ā (несогласован с (32), (35)) | Ā = H\_gLS(v\_TB − v\_gLS) + v\_m | вывод из (32), (35); воспроизводит Пример 4.9. Буквальная форма — флаг |
| Ansari | (85) | (ρ\_g/ρ\_g)^(1/3) | (ρ\_L/ρ\_g)^(1/3) | монография 4.222 |
| Монография | Табл. 4.2 | a = 0.855 | 0.845 | BB Table 1; Пример 4.7 |
| Монография | (4.135) | sin³θ | sin²θ | MB (10); Пример 4.8 |
| Монография | Пример 4.7 | ρ\_s = (433.14)(0.53)+… | (762.64)(0.53)+… | итог 448.52 верен |
| Монография | Пример 4.9 | (dp/dL)\_SC и (dp/dL)\_SL перепутаны; 2.881 | правильные подписи; 2.381 | числа проверки кольцевого режима в эталон не брать |

Монография не используется как источник формул: её версия Hasan–Kabir — другая модель (наклонная, с C0 = 2.0 при d > 0.12 м), её описание Ansari опускает развивающуюся пробку.

## Тесты и эталонные значения

Три уровня: юнит-тест на каждую функцию ядра, тест на каждый элемент модели (режим, H\_L, трение, градиент), 2–3 сквозных теста на модель (пример из книги/статьи + пересчёт + PIPESIM). Эталоны хранятся в `tests/reference/*.json` с полями `source`, `flags`, `inputs`, `expected`, `rtol`.

### Наборы входных данных

- **R1** (Пример 3.2, вариант Примеров 4.7–4.8): v\_sl = 1.21, v\_sg = 1.176 м/с; ρ\_L = 762.64, ρ\_g = 94.19 кг/м³; μ\_L = 0.97·10⁻³, μ\_g = 0.016·10⁻³ Па·с; σ = 8.41·10⁻³ Н/м; d = 0.1524 м; ε = 18.288·10⁻⁶ м; θ = 90°; p = 117.13 бар.
- **R2** (вариант Примеров 4.9–4.10): то же, но v\_sl = 1.208, v\_sg = 1.173, ρ\_L = 761.7, ρ\_g = 94.1, ε = 1.83·10⁻⁵.
- **AW**: воздух–вода 25 °C, 0.1 МПа: ρ\_L = 997, ρ\_g = 1.18, μ\_L = 0.89·10⁻³, μ\_g = 1.8·10⁻⁵, σ = 0.072; d = 0.051 или 0.0254 м.

### Сквозные эталоны

Источник «пересчёт» — независимый расчёт Claude по схеме выше (Colebrook вместо диаграммы Муди); до сверки с PIPESIM пометка `provisional`.

| Модель | Набор | Флаги | Режим | H\_L | −dp/dL, Па/м | rtol | Источник |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BB | R1 | payne, rough, clip\_c = True; accel = False | INTERMITTENT | 0.5305 | 4588 (книга 4580) | 1% | Пример 4.7 + пересчёт |
| BB | R1 | по умолчанию | INTERMITTENT | 0.5659 | 4804 | 0.1% | пересчёт |
| MB | R1 | accel = False | SLUG | 0.5603 | 4734 (книга 4727) | 0.5% | Пример 4.8 + пересчёт |
| Ansari | R2 | developing\_slug = False, ff\_over\_fsl\_unity = True | SLUG | — | 4912 (книга 4921) | 1% | Пример 4.9 + пересчёт |
| HK 1988 | R2 | по умолчанию | ANNULAR | 0.5744 | 4186 | 0.1% | пересчёт, provisional |

### Элементные эталоны (R1/R2)

| Модель | Величина | Значение | rtol |
| --- | --- | --- | --- |
| BB | N\_FR / N\_Lv | 3.808 / 11.86 | 1e-3 |
| BB | L1 / L2 (оригинал) | 0.1018 / 33.11 | 1e-3 |
| BB | H\_L(0) / C | 0.5741 / −0.0481 | 1e-3 |
| BB | ψ при θ = 90°, без clip | 0.9856 | 1e-3 |
| BB | Re\_ns / f\_ns Colebrook / f\_ns гладкая (13) | 3.152·10⁵ / 0.01551 / 0.01434 | 1e-3 |
| BB | y / S / f\_tp (флаги Примера 4.7) | 1.802 / 0.3871 / 0.02284 | 1e-3 |
| MB | N\_L / N\_gv | 0.01176 / 11.53 | 1e-3 |
| MB | N\_gvSM / N\_LvBS | 350.7 / 18.38 | 1e-3 |
| Ansari | DB: левая / правая часть (4) | 1.162 / 3.638 | 1e-3 |
| Ansari | v\_sg кольцевой (6) / d\_min | 0.8706 м/с / 0.01888 м | 1e-3 |
| Ansari | F\_E / λ\_LC / ρ\_C | 0.5475 / 0.3605 / 334.8 | 1e-3 |
| Ansari | X\_M² / Y\_M / δ (96) | 0.2214 / 71.13 / 0.1311 | 1e-3 |
| Ansari | H\_LF + λ\_LC A\_C/A | 0.652 (> 0.12 → не кольцевой) | 1e-3 |
| Ansari | v\_TB / H\_gLS / v\_gLS / Ā | 3.2578 / 0.17417 / 2.9944 / 2.4269 | 1e-4 |
| Ansari | H\_LTB / v\_LTB / v\_LLS / v\_gTB | 0.12983 / 3.1424 / 2.2516 / 3.2051 | 1e-3 |
| Ansari | β / L\_TB / L\_c | 0.2873 / 1.843 м / 3.17 м | 1e-3 / 1e-3 / 5e-2 |
| Ansari | H\_LTB при `vo_shoham_a_as_published` | 0.1188 | 1e-3 |
| HK | v∞ / v\_gs,BS / v\_M,DB / v\_gs,A | 0.1510 / 0.5721 / 4.405 / 0.8706 | 1e-3 |
| HK | Hewitt–Roberts: ρ\_Lv\_Ls² / ρ\_gv\_gs² / граница (lbm/ft·s²) | 747.0 / 87.0 / 25.93 | 1e-3 |
| HK | v∞T (0.345) | 0.3949 м/с | 1e-3 |
| HK | E / ρ\_c / X / f\_g (кольцевой) | 0.5666 / 340.1 / 2.143 / 0.6740 | 1e-3, provisional |

### Контрольные точки из текстов статей

| Модель | Проверка | Ожидание | rtol |
| --- | --- | --- | --- |
| HK (4) | AW, неподвижная вода: v\_gs переход bubble/slug | 0.088 м/с (пересчёт 0.0891) | 3% |
| HK (13) | AW: v\_gs переход в кольцевой | ≈ 15 м/с (пересчёт 14.70) | 5% |
| HK (6) | AW, d = 0.0508: v\_M переход в DB | 3.6 м/с (пересчёт 3.54) | 3% |
| HK | AW, d = 0.0508: N\_f | 35 000 (пересчёт 35 700) | 3% |
| TD | X при h̃\_L = 0.5, Y = 0, турб.–турб. | 1.6 (пересчёт 1.584) | 2% |
| Barnea \[1\] | AW: D\_crit | 0.0515 м; d = 0.0254 → пузырькового нет | 1e-3 |
| MB \[3\] | θ = 0: N\_gvBS не зависит от N\_Lv | равенство при N\_Lv = 1 и 100 | 1e-12 |
| MB \[2\] | N\_gvSM не зависит от θ | равенство при θ = −90…90 | 1e-12 |
| BB (16) | y = 1 → f\_tp/f\_ns = 1; λ → 0 → S → 0 | точно | 1e-12 |
| Ядро | Barnea \[5\] и Ansari (4) дают одну границу при f\_Moody = 4f\_Fanning, d\_C = d\_CD | равенство | 1e-9 |

### Карты режимов Barnea (Fig. 1–2, AW) — ориентировочные точки

Точки выбраны в глубине областей по экспериментальным символам; координаты сняты с рисунка на глаз, перед фиксацией сверить с OLGA/PIPESIM.

| d, см | β | U\_GS, м/с | U\_LS, м/с | Ожидаемый режим |
| --- | --- | --- | --- | --- |
| 5.1 | 0 | 0.1 | 0.003 | STRATIFIED\_SMOOTH |
| 5.1 | 0 | 10 | 0.003 | STRATIFIED\_WAVY |
| 5.1 | 0 | 80 | 0.01 | ANNULAR |
| 5.1 | 0 | 1 | 6 | DISPERSED\_BUBBLE |
| 5.1 | 0 | 0.1 | 0.5 | ELONGATED\_BUBBLE |
| 5.1 | −30 | 1 | 0.01 | STRATIFIED\_WAVY |
| 5.1 | 30 | 1 | 0.1 | SLUG (intermittent) |
| 2.54 | 90 | 0.1 | 0.1 | SLUG |
| 2.54 | 90 | 6 | 0.01 | CHURN |
| 2.54 | 90 | 30 | 0.01 | ANNULAR |
| 2.54 | −90 | 1 | 0.01 | ANNULAR |
| 2.54 | −90 | 1 | 8 | DISPERSED\_BUBBLE |

Точки 5.1 см при β = ±80…90° в пузырьковой области в тесты не включать: D\_crit = 5.15 см ≈ 5.1 см, результат зависит от σ.

### Общие тесты свойств (для всех ICorrelation)

- Однофазные пределы: v\_sg → 0 даёт однофазный Colebrook-градиент жидкости (rtol 1e-6), v\_sl → 0 — газа.
- 0 ≤ H\_L ≤ 1 (гарантирует `CorrelationResult`), H\_L ≥ λ для подъёма там, где это следует из модели.
- dpdl = grav + fric + acc (rtol 1e-10, проверяет `model_validator`).
- Каждый флаг меняет результат хотя бы в одной точке (тест на «мёртвые» флаги) и имеет непустой `description`.
- Валидация pydantic: неизвестный флаг, флаг-строка `"True"`, отрицательная плотность, ρ\_L ≤ ρ\_g, d\_t ≥ d, θ вне \[−90, 90\] → `pydantic.ValidationError`; модели frozen (присваивание → `ValidationError`).
- `isinstance(model, ICorrelation)` / `IFlowModel` — True для всех зарегистрированных моделей.

## ТЗ для агента

Цель: реализовать пакет `correlations` по разделам «Архитектура» и «Схемы расчёта» строго в порядке «тест → код», источник формул — только оригиналы статей из проекта и таблица «Опечатки».

### Порядок работ

Каждый этап: сначала тесты (падают), затем код (тесты проходят), затем отчёт агента.

0. План реструктуризации `src/correlations` (переименование файлов с дефисами, `physics.pay`, перенос в `core/`, `flow_models/`, `models/`, место папки тестов) → **ждать подтверждения**.
1. `core/state.py`, `patterns.py`, `options.py`, `errors.py` (pydantic-модели, enum, исключения).
2. `core/friction.py`, `dimensionless.py`, `mixture.py`, `solvers.py`.
3. `core/bubbles.py`, `transitions.py`, `annular.py`, `stratified.py`, `gradient.py` — все 14 пересечений.
4. IFlowModel: `beggs_brill_map`, `mukherjee_brill_fp`, `taitel_dukler`, `barnea`.
5. ICorrelation: `beggs_brill`, `mukherjee_brill`, `hasan_kabir`, `ansari`.
6. `registry.py`, `__init__.py` (публичный API), README с таблицей флагов из `describe()`.

Цикл каждого этапа (по CLAUDE.md):

1. Агент **сам пишет тесты** этапа по разделу «Тесты и эталонные значения» и оригиналам статей. Эталонные числа берутся только из таблиц этого документа; любое новое эталонное значение помечается `# reference: <источник>` и выносится в отчёт.
2. Запуск: тесты должны падать (кода ещё нет).
3. **Стоп**: показать список тестов (файл, что проверяет, эталон, rtol) и ждать подтверждения.
4. После подтверждения — код только в файлах этапа.
5. `uv run python check.py` (pytest -q, ruff check --fix, ruff format, pyright) — всё зелёное, тесты не ослаблять.
6. Отчёт по форме CLAUDE.md.

### Требования к коду

- Окружение: Windows, PowerShell, uv. Перед `uv add` проверить `Get-Command python`; при глобальном интерпретаторе — `.venv/Scripts/activate.ps1`. Зависимости: pydantic ≥ 2 (обязательно), прочие — только с подтверждения.
- Валидация входа, опций и результатов — pydantic v2 (`BaseModel`, `Field`, `model_validator`, `ConfigDict(frozen=True, extra="forbid", strict=True)`). Ручные `if … raise ValueError` для входных данных не писать.
- Полная типизация, `uv run pyright` без ошибок; никаких `Any`, `# type: ignore` без комментария с причиной.
- Единицы — СИ, в именах полей при неоднозначности (`velocity_m_s`, `dpdl_pa_m`).
- Docstring модуля корреляции: источник + схема расчёта. Docstring функции: что делает, параметры с единицами, что возвращает, источник «Автор год, ур. N». Исправленные опечатки — пометка `# typo fix: напечатано …` со ссылкой на таблицу «Опечатки».
- Никаких «магических» констант в теле функций — `Final`-константы модуля со ссылкой на уравнение.
- Ветки режимов не дублируют формулы ядра: дубликат формулы из таблицы «Пересечения» — дефект.
- Не использовать формулы из монографии там, где есть оригинал; исключение — `revised_map` BB (источник прямо указать).
- Расхождение кода с источником сверх таблицы «Опечатки» — сообщить в отчёте, не исправлять молча.
- Нерешённые места из «Открытых вопросов» — по умолчанию из таблицы, пометка `# TODO(question N)`.
- Работать только внутри `src/correlations` (и `pyproject.toml`); без попутного рефакторинга; не помечать тесты skip/xfail и не ослаблять rtol; без `--no-verify`, commit и push — только по прямой просьбе.

### Критерии приёмки

- [ ] `uv run python check.py` проходит: pytest, ruff check, ruff format, pyright — без ошибок.
- [ ] Покрытие core ≥ 95%, моделей ≥ 90%.
- [ ] Сквозные эталоны BB, MB, Ansari воспроизводят примеры книги в пределах rtol из таблицы.
- [ ] Элементные эталоны совпадают с rtol 1e-3.
- [ ] Каждая ветка блок-схем (BB, MB-FP, TD, Barnea Fig. 6, HK, Ansari) покрыта хотя бы одним тестом.
- [ ] У каждого флага: `description`, тест влияния, значение по умолчанию = оригинал.
- [ ] Все входы/опции/результаты — pydantic-модели; некорректный ввод даёт `ValidationError`.
- [ ] Нет расхождения Moody/Fanning: тест на каждую функцию, принимающую коэффициент трения.
- [ ] Отчёт по форме CLAUDE.md: «Сделано», «Не сделано / отложено», «Известные проблемы»; при падении тестов — какие, краткий traceback, причина.

## Открытые вопросы

До ответа агент реализует указанное умолчание и помечает `TODO(question N)`.

| N | Вопрос | Умолчание |
| --- | --- | --- |
| 1 | Ansari, (12): δ\_min или δ из (96)? Для R2 уравнение (8) даёт два корня δ\_min (0.052 и 0.208) — какой брать? | δ\_min, наименьший корень; на R2 режим SLUG при любом выборе |
| 2 | Ansari: порядок проверок в статье не задан. Берём порядок Примера 4.9 (DB → кольцевой → пузырьковый → пробковый)? | да |
| 3 | Ansari, развивающаяся пробка: v\_NLTB в (45) явно не определён; принимаю баланс (32) с H\_NLTB. Корень (54) — больший положительный? | да / больший |
| 4 | HK: порядок проверок (DB → bubbly → annular → slug/churn) и что делать при ρ\_Lv\_Ls² > 3300 lbm/(ft·s²) вне DB и annular | порядок как указан; CHURN + предупреждение |
| 5 | HK, кольцевой: x «с учётом уноса» — x = G\_g/(G\_g + E·G\_L)? Скорость в (33) — v\_g = v\_gs/f\_g? | да / да |
| 6 | HK 1988 — модель для вертикали. При θ ≠ 90° считать с sinθ и предупреждать или запрещать? | считать + предупреждение |
| 7 | Barnea \[2\]: d — диаметр пузырька (берём d\_CD?), γ из диапазона 1.1–1.5 — какое значение? | d = d\_CD, γ = 1.3 |
| 8 | Barnea: Re для f\_M — по свойствам жидкости или no-slip? | жидкость, флаг `fm_noslip_reynolds` |
| 9 | TD/Barnea: при подъёме уравнение (7) и \[15\] могут иметь несколько корней — берём наименьший? | наименьший |
| 10 | MB, расслоенный: подтверждаете τ\_w = f\_Moody ρv²/8 по умолчанию? | да, буквальная форма — флагом |
| 11 | PIPESIM-эталоны: какие варианты BB (Original/Revised) и какие флаги им соответствуют — сверить после ваших расчётов | — |

Вопросы по CLAUDE.md:

- **12.** CLAUDE.md ограничивает работу модулем `correlations`. Где лежат тесты: `tests/` в корне (нужно ваше разрешение) или `src/correlations/tests/`? Умолчание: `tests/` в корне после подтверждения.
- **13.** Переименовать файлы с дефисами и `physics.pay` (иначе они не импортируются) и перенести в `core/`, `flow_models/`, `models/`? Умолчание: да, отдельным этапом 0 после подтверждения.
- **14.** Текущий код в `ansari.py`, `beggs_brill.py` и др. — переписываем под новую схему или сохраняем как эталон для сравнения? Умолчание: переписываем, старые результаты используем как дополнительную сверку.
