# Лабораторна робота №1. Побудова простої трансформерної моделі

Реалізовано обидва варіанти: Варіант 1 — класифікація spam/ham на spam.csv через Transformer Encoder; Варіант 2 — авторегресивне продовження статей BBC через Transformer Decoder. Навчання виконується з нуля. Baseline, tuning і фінальне оцінювання — окремі ручні комірки Jupyter. Навчання обмежено максимальною кількістю епох і validation-based early stopping. Попередні локальні результати збережено; новий протокол пише в окремі каталоги `early_stopping/`.

## Початкові матеріали й дані

У каталозі були `spam.csv`, три референсні notebooks, README-заготовка та порожні каталоги. Кореневий `pyproject.toml` вимагає Python ≥3.11 і не містить залежностей; їх додано локально в `requirements.txt`, без змін інших лабораторних.

CSV збережено без змін у `data/raw/spam.csv`: UTF-8, колонки `Category` і `Message`, 5572 записи, 4825 ham (86.59%) і 747 spam (13.41%), пропусків немає, 415 точних повторів тексту. Це короткі SMS за змістом, хоча користувач назвав набір Spam Emails; методичка допускає обидва типи. Цей набір використовується тільки у Варіанті 1.

Для Варіанта 1 regex-токенізація: lowercase, слова та пунктуація. Видаляються порожні тексти, дублікати токенізованих текстів і всі записи із суперечливими мітками. Після очищення: 5156 записів (4515 ham, 641 spam). Стратифікований split 70/15/15, seed=42: 3609 train, 773 validation, 774 test. Словник будується тільки за train (min_frequency=2, максимум 8000; фактично 3202 токени). PAD=0, UNK=1, BOS=2, EOS=3. Максимальна довжина визначається 95-м перцентилем train із верхньою межею 96; для поточних даних — 42. Глобальний 95-й перцентиль — 43, максимум — 224; глобальні статистики не налаштовують модель.

Варіант 2 використовує наданий локальний BBC News у `data/raw/bbc-text.csv`. Файл переміщено з `data/bbc-text.csv` без зміни байтів (SHA-256 перевірено). Це довші зв’язні статті, придатні для контекстного прогнозування наступного токена. UTF-8; схема `category,text`; 2225 статей, 0 пропусків, 0 пошкоджених CSV-рядків, 99 точних повторів, 2126 унікальних текстів. Raw категорії: sport 511, business 510, politics 417, tech 401, entertainment 386. Категорії — лише metadata для EDA та стратифікації, не входи/цілі моделі.

Довжини raw BBC (слова / regex-токени): мінімум 90 / 97; медіана 337 / 369; середнє 390,30 / 424,80; p95 736,4 / 802,6; максимум 4492 / 4811. Очищення зберігає перший точний текст, без нормалізації для dedup і без семантичного видалення. SHA-256 тексту — стабільний document ID. Після dedup документи стратифіковано за category: 70/15/15, seed=42, **до створення вікон**. Перетин ID та точних текстів спричиняє помилку; кожне вікно зберігає ID свого документа.

| BBC split | Документи | Токени тексту | Вікна | UNK, % |
|---|---:|---:|---:|---:|
| train | 1488 | 635644 | 10669 | 2,8140 |
| validation | 319 | 132119 | 2230 | 4,3945 |
| test | 319 | 135638 | 2284 | 4,4781 |

BBC vocabulary: той самий lowercase regex для слів і пунктуації, min_frequency=2, max_size=12000; фактично **12000** разом із PAD/UNK/BOS/EOS. Побудова тільки за train. Train UNK для cap 8000 / 12000 / 16000 становить 5,17% / 2,81% / 1,51%; обрано 12000 як компроміс покриття та розміру вихідного шару. Held-out слова не додаються до словника. Оцінки UNK validation/test — опис даних, не критерій вибору параметрів.

BBC window length=64, stride=64: компактний attention для 4 GB VRAM, без цілого документа в одному вході та без повторних цільових токенів. Сусідні вікна мають один спільний граничний токен для shift; останнє доповнюється PAD. Усі слова статті та EOS залишаються цілями. Це 10669 train-вікон, а не надмірно перекритий корпус. Кількості отримано підготовкою даних без навчання; фактичну VRAM і тривалість експериментів ще не виміряно.

Оригінальні notebooks переміщено в `notebooks/reference/` без редагування. `TransformerEncoderExample` використовує синтетичні числові входи; `TEncoderClassifier` демонструє pretrained tokenizer і pooling першого токена; `CausalLM` використовує синтетичний корпус та сталу нульову memory. Вони залишаються навчальними матеріалами, а не фінальними експериментами.

Переглянуто notebooks зі [студентського проєкту Computational Intelligence](https://github.com/Eclipse-XS/Theory-and-Methods-of-Computational-Intelligence) та [Dental X-ray](https://github.com/Eclipse-XS/dental-xray-pathology-detection): збережено пояснення перед етапами, явні параметри, pathlib, табличні результати, Matplotlib і поділ validation/test. Фінальні пояснення написано українською, Python-ідентифікатори — англійською.

## Структура

```text
lab01_transformer/
├── README.md
├── requirements.txt
├── data/raw/spam.csv                  # Варіант 1, локальний оригінал
├── data/raw/bbc-text.csv              # Варіант 2, локальний оригінал
├── notebooks/
│   ├── 01_spam_classification.ipynb
│   ├── 02_text_generation.ipynb
│   └── reference/
│       ├── TransformerEncoderExample.ipynb
│       ├── TEncoderClassifier.ipynb
│       └── CausalLM.ipynb
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   ├── positional_encoding.py
│   │   ├── reproducibility.py
│   │   ├── checkpoint.py
│   │   ├── training.py
│   │   ├── experiments.py
│   │   ├── inspection.py
│   │   └── visualization.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── inspection.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── generation/
│       ├── __init__.py
│       ├── data.py
│       ├── corpus_inspection.py
│       ├── model.py
│       ├── inspection.py
│       ├── train.py
│       ├── evaluate.py
│       └── generate.py
├── tests/
│   ├── test_lab01.py
│   ├── test_training.py
│   ├── test_inspection.py
│   └── test_bbc.py
└── outputs/                          # створюється вміст під час ручних запусків
```

Наявні `.gitkeep` збережено. Обидва `data/raw/*.csv` і результати виключаються чинним `.gitignore`; для іншої машини CSV потрібно перенести окремо. Нових датасетів код не завантажує.

## Середовище

Команди PowerShell із кореня репозиторію:

```powershell
.\.venv\Scripts\python.exe -m pip install -r labs/lab01_transformer/requirements.txt
.\.venv\Scripts\python.exe -m pytest labs/lab01_transformer/tests -q
```

У PyCharm/Jupyter оберіть `.venv` як kernel. Notebooks знаходять Lab root при запуску з кореня репозиторію, каталогу лабораторної або `notebooks/`.

Поточне середовище має torch 2.14.0+cu130; `torch.cuda.is_available()` повертає True, GPU — NVIDIA GeForce RTX 3050 Laptop GPU (4 GB). Пристрій обирається автоматично, CPU fallback збережено. Перевірки цього доопрацювання виконано на CPU без навчання; наявність CUDA не є вимірюванням GPU-продуктивності. Нових залежностей для early stopping не додано. NLTK обчислює BLEU зі списків токенів без завантаження корпусів.

## Архітектури і вимоги

| Вимога | Реалізація |
|---|---|
| Реальні дані, очищення, train/validation/test, train-only vocabulary | `classification/data.py`, `common/text.py`; обидва notebooks, розділи 3–4 |
| Seed Python/NumPy/CPU/CUDA, device info | `common/reproducibility.py` |
| Синусоїдальне PE, спільне для обох моделей | `common/positional_encoding.py`; теплові карти у розділах 7 (classification) та 6 (generation) |
| Encoder: Embedding + PE + 1–2 layers + Linear | `classification/model.py`, типовий один шар |
| Padding mask і masked mean pooling | `SpamClassifier.forward`, `masked_mean`; PAD не входить у pooling |
| Балансування | `classification/data.py::class_weights`, weighted cross-entropy; без oversampling |
| Accuracy / precision / recall / F1, spam=1 | `classification/evaluate.py`, `common/training.py` |
| Decoder: Embedding + PE + 1–5 layers + Linear | `generation/model.py`, типовий один шар |
| Shift на один токен і розбиття документів до вікон | `generation/data.py::split_corpus`, `LanguageModelDataset` |
| Causal mask і відсутність витоку через memory | `generation/model.py`: верхня трикутна bool mask і стала нульова memory |
| Авторегресивна генерація | `generation/generate.py`: greedy, rolling context, EOS, заборона PAD/BOS |
| Token loss, perplexity, ігнорування PAD | `common/training.py`, `generation/evaluate.py` |
| BLEU / ROUGE-L із реальними продовженнями | `generation/evaluate.py::evaluate_continuations` |
| Training loop, AdamW, gradient clipping, early stopping | `common/training.py`; зрозумілі task wrappers у `*/train.py` |
| Checkpoint найкращої validation-епохи, save/load | `common/checkpoint.py`, `common/training.py` |
| Tuning d_model/nhead, перевірка подільності | `common/experiments.py`, `common/positional_encoding.py::validate_config` |
| Порівняння 64/2, 64/4, 128/2, 128/4 | `run_grid`: решта settings і порядок batch фіксовані |
| Графіки, таблиці, confusion matrix, приклади | обидва notebooks; `common/visualization.py` |
| Автоматичний тест причинності та інші sanity checks | `tests/test_lab01.py` |

Classifier: тексти → regex → train vocabulary → embedding/PE → encoder → masked mean → Linear(2). Ваги train-класів приблизно 0.5710 / 4.0189; epoch loss нормалізується сумою ваг, а не числом batch. Позитивний клас — spam. Найкращий checkpoint має максимальний validation F1.

Generator: статті BBC → split → train vocabulary → BOS/text/EOS → shifted windows → decoder → Linear(vocabulary). Цілі не повторюються між вікнами, документи не зливаються. Cross-attention memory має форму `[batch, 1, d_model]` і містить сталі нулі; вона не залежить від майбутніх токенів. Causal self-attention забороняє верхній трикутник. PAD ігнорується в loss, perplexity = exp(сумарний NLL / число непорожніх цілей); при loss ≥709 повертається infinity. Вибір моделі — мінімальний validation token cross-entropy.

BLEU — corpus BLEU-4, NLTK smoothing method1; ROUGE-L — середнє token LCS F1 без stemming. Шкала обох — 0–1. Протокол: seed=42, до 64 held-out документів із ≥64 токенами, контекст 32, еталонне продовження 32, greedy максимум 32 нових токенів. Еталони не перетворюються на UNK, тому OOV не створює штучних збігів. Метрики характеризують held-out статті BBC підвибірки; вони не еквівалентні perplexity і мають обмеження одного еталона.

## Політика навчання й вибору моделі

Обидва notebooks мають явний блок конфігурації: `SEED=42`, `BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=5`, `LEARNING_RATE=1e-3`, `MODEL_CONFIG`. Компактну архітектуру збережено: d_model=64, nhead=2, один шар, feed-forward=256, dropout=0.1. Словник будується лише за train. Для classification max_len визначається train p95; для BBC задано компактне вікно 64.

30 — верхня межа, а не обов'язковий бюджет. П'ять послідовних епох без покращення завершують навчання. Для classification покращення — **строге зростання validation F1 spam**; для generation — **строге зменшення validation token cross-entropy**. Рівність не скидає patience; перша скінченна оцінка завжди зберігається. Окремий min_delta не введено: правило суворого порівняння просте й не відкидає фактичний найкращий результат. Нечислові/нескінченні validation-метрики спричиняють явну помилку до запису checkpoint. BLEU/ROUGE-L не обчислюються для early stopping.

Таке обмеження дозволяє навчатися довше за початкові 5 епох, але припиняє непотрібні епохи за validation-сигналом. Це початкова політика для локального експерименту, а не гарантія збіжності чи оптимального узагальнення. Не задано мінімальної кількості епох.

Baseline навчає одну модель для перевірки динаміки loss і validation-метрик. Grid порівнює 64/2, 64/4, 128/2, 128/4 з однаковими max_epochs, patience, learning rate, split, seed і порядком batch. Фактична кількість епох може відрізнятися. Comparison містить параметри, кількість ваг, `best_epoch`, `epochs_trained`, `stopped_early`, метрики найкращої validation-епохи, час і шлях до checkpoint.

Фінальний вибір порівнює збережений baseline і виконані grid-конфігурації за validation F1/max або loss/min. Baseline не обирається автоматично. Test loader не передається навчанню або grid; він створюється функцією фінального оцінювання зі словником та max_len відновленого checkpoint лише після ручного виклику фінальної комірки.

## Організація notebooks і межі функцій

Обидва notebooks мають послідовність: пояснення → виклик змістовної операції → таблиця/графік → інтерпретація. Орієнтиром були notebooks Dentex: `00_prepare_dataset`, `01_eda`, `02_preprocessing`, `03_baseline`, `06_fcos_refinement`, `07_final_evaluation`, `08_error_analysis`. Їхній код не копіювався.

У notebooks залишено конфігурацію, явні split/vocabulary/DataLoader, ваги класів, побудову моделі, виклики навчання, grid, вибір checkpoint і невеликі навчальні приклади. Token → id та input → shifted target показано окремими таблицями. Більшість execution-комірок має 2–6 рядків; імпорти та конфігурація довші, оскільки залишаються явними.

- `common/inspection.py`: SMS-аудит та split summary для classification, спільне читання persisted history/summary і таблиця checkpoint metadata.
- `generation/corpus_inspection.py`: BBC-аудит, графік категорій/довжин, кількості документів/вікон і UNK кожного split.
- `generation/data.py`: strict CSV schema, точний dedup, document split, provenance вікон та dataset metadata.
- `classification/inspection.py`: фактичні форми embedding, PE, encoder, pooling і logits; звірка зі звичайним forward без зміни ваг або training mode після виклику.
- `generation/inspection.py`: preview зсуву, форми decoder/memory/logits і причинна інваріантність при зміні, видаленні або padding suffix.
- `common/visualization.py`: графіки даних, PE, causal mask, історії, grid і confusion matrix. Функції приймають дані й явний output path, повертають закритий для автоматичного показу `Figure`; notebook використовує один `display(figure)`.
- `evaluate_final_classifier` та `evaluate_final_generation` у відповідних `evaluate.py`: компонують незмінені метрики, створюють тільки test loader, зберігають ті самі артефакти й повертають метрики та приклади. Вибір моделі залишається поза ними.

Шляхи централізовано: `LAB_ROOT`, `DATA_PATH`, `OUTPUT_DIR`, `BASELINE_DIR`, `BASELINE_CHECKPOINT`, `EXPERIMENTS_DIR`, `FIGURES_DIR`. `VOCAB_CONFIG`, classification `MAX_LENGTH_CAP`, generation `WINDOW_LENGTH`, `WINDOW_STRIDE` і `CONTINUATION_CONFIG` задаються поруч із training settings. `GRID_CONFIGS` — спільна константа з чотирма незміненими конфігураціями. `partial(create_loaders, ...)` зберігає параметри підготовки loaders; runner передає seed явно за ім'ям.

Архітектури та `forward`, правило early stopping, tokenizer, loss/метрики й autoregressive loop збережено. BBC має власне розбиття документів; serializer отримав необов’язкові dataset metadata та перевірку сумісності. Окремого experiment framework або wrappers навколо кожного рядка немає.

## Порядок ручного виконання

Запуск конкретної комірки Jupyter є єдиним механізмом керування; перемикачів виконання немає. Перед навчанням, grid, демонстрацією генерації та test є блок **Ручний запуск**. Не використовуйте Run All, якщо не плануєте виконувати весь експеримент разом із фінальним test.

Спочатку `01_spam_classification.ipynb`:

1. Виконати імпорти, параметри та підготовку даних; перевірити device, split, довжини, ваги train-класів.
2. Виконати sanity-check архітектури та PE. `inspect_classifier_pipeline` показує фактичні форми `[B,L] → [B,L,d_model] → [B,d_model] → [B,2]` і звіряє logits зі звичайним forward; hooks або tracing немає.
3. Вручну виконати комірку baseline. Дочекатися early stopping або max_epochs.
4. Прочитати baseline history і `summary.json` через `read_training_results`, переглянути validation-метрики та криві фактичних епох. Об'єкт результату навчання в пам'яті не потрібен.
5. Вручну виконати grid; переглянути comparison. Для проміжного аналізу тільки baseline можна пропустити grid разом із його comparison-коміркою; повна лабораторна включає tuning.
6. Виконати вибір і відновлення найкращої validation-моделі; переглянути конфігурацію та best epoch.
7. Лише після завершення tuning одноразово виконати фінальну test-комірку. Переглянути метрики, confusion matrix, predictions; записати висновки.

Далі `02_text_generation.ipynb`:

1. Виконати підготовку корпусу, split, vocabulary, shifted windows та sanity-check моделі.
2. Переглянути PE/causal mask і виконати перевірку незмінності prefix logits.
3. Вручну навчити baseline; завантажити його checkpoint і переглянути loss/perplexity та summary.
4. У розділі 9 вручну показати три validation-приклади context/reference/generated (32+32 токени) з baseline.
5. Вручну виконати grid, переглянути comparison і відновити обрану за validation loss модель.
6. Лише після model selection виконати фінальні test perplexity, BLEU, ROUGE-L та таблицю продовжень. Записати висновки.

Після перезапуску kernel повторіть підготовчі комірки й завантаження persisted artifacts; навчати моделі повторно не потрібно. За відсутності checkpoint вибір завершується зрозумілим `FileNotFoundError`. Загальний `load_checkpoint` зберігає сумісність із legacy-файлами без `best_epoch`. BBC notebook завжди передає `expected_dataset_metadata`: старий checkpoint без BBC metadata буде відхилено, навіть якщо його вручну скопіювати в BBC-каталог. Перевіряються dataset ID, SHA-256 raw CSV і vocabulary, tokenizer, vocabulary settings, max_len/stride та split policy/seed; model config і training config також зберігаються.

Новий `OUTPUT_DIR` — `outputs/classification/early_stopping/` або `outputs/generation/bbc/early_stopping/`. Попередні артефакти в батьківських каталогах не змішуються з новим протоколом. Повторний запуск у тому самому OUTPUT_DIR перезаписує відповідну історію/checkpoint; для іншого протоколу змініть OUTPUT_DIR. За нестачі VRAM зменшуйте batch size однаково для порівнюваних конфігурацій.

## Артефакти після запусків

Для кожної задачі під `outputs/classification/early_stopping/` або `outputs/generation/bbc/early_stopping/`:

- `baseline/checkpoints/best.pt`, `baseline/metrics/history.csv`, `baseline/metrics/summary.json`;
- `experiments/d64_h2/…`, `d64_h4/…`, `d128_h2/…`, `d128_h4/…` і `experiments/comparison.csv`;
- `figures/`: дані, PE, історія, comparison; також confusion matrix або causal mask;
- `metrics/test.json`; для classification — `metrics/predictions.csv`;
- для generation — `samples/test_continuations.csv`.

Checkpoint містить `state_dict`, model config, vocabulary, tokenizer version, label mapping лише для classification, BBC dataset metadata для generation, max_epochs/patience та інші training settings, validation-метрики й `best_epoch`. Зберігаються ваги найкращої, а не останньої епохи. Об'єкт моделі в пам'яті після fit відповідає останній епосі, тому для подальшого оцінювання notebooks явно завантажують checkpoint. Повний Python-об'єкт не серіалізується.

`history.csv` закінчується на фактичній останній епосі без доповнення до 30. `summary.json` містить best_epoch, epochs_trained, stopped_early, stop_reason (`patience` або `max_epochs`), best_validation_metric, training_config, кількість параметрів і час. Якщо patience вичерпано саме на max_epochs, stop_reason — patience, а stopped_early — False, бо верхню межу вже досягнуто.

## Перевірки й обмеження

Виконані CPU structural tests охоплюють vocabulary/PAD/UNK, розбиття без спільних входів, padding/pooling, PE з парною і непарною розмірністю, shift вікон, форми tensors, причинність decoder з 1/2/5 шарами, greedy context/EOS, perplexity з PAD, ROUGE-L, незмінені OOV-еталони, checkpoint roundtrip обох моделей та schema/syntax невиконаних notebooks. Prefix logits порівнюються для однакового prefix і різних suffix, короткого prefix та prefix із правим padding; tolerances atol=1e-6, rtol=1e-5.

Команди перевірки:

```powershell
.\.venv\Scripts\python.exe -m pytest labs/lab01_transformer/tests -q
.\.venv\Scripts\python.exe -m compileall -q labs/lab01_transformer/src labs/lab01_transformer/tests
```

Для early stopping додано тести зі scripted validation-метриками та підміною `run_epoch`: вони перевіряють maximize/minimize, reset/exhaustion patience, рівність, max_epochs, нечислові метрики, ваги саме найкращої епохи, історію та summary. Grid перевіряється з підміною fit без навчання; окремо перевіряється сумісність старих checkpoints. Синтетичні значення існують тільки в unit tests і не є результатами лабораторної.

Структурні тести додатково перевіряють реальні форми inspection API, незмінність ваг і відновлення training mode, виявлення навмисно некаузальної моделі, відсутність подвійного показу Figure та компонування фінального оцінювання. В останніх тестах самі обчислювачі метрик підмінено: реального test-оцінювання немає.

Попередня перевірка структурного refactor до міграції BBC: **50 passed in 25.16s**, без warnings; `compileall` і `git diff --check` — без помилок. Обидва notebooks валідні за JSON/schema та синтаксисом. Виконано тільки 12 підготовчих комірок classification і 14 generation на CPU, до першої ручної комірки. Шляхи перевірено з кореня репозиторію, Lab root і `notebooks/`. Графіки реальних даних, PE та causal mask перевірено в тимчасовому QA-каталозі. SHA-256 підтвердив незмінність 42 файлів даних, референсів і збережених результатів, а також незмінність файлів моделей, early stopping, checkpoint serialization, tokenizer і generation loop.

Під час цього доопрацювання baseline, grid, фінальний test, повні notebooks і тривала генерація не запускалися. Попередні користувацькі артефакти залишено без змін. На момент структурного refactor notebooks були збережені без outputs. Після цього користувач виконав обидва notebooks; поточні outputs та числові висновки збережено.

Міграція BBC додає тести schema/dedup, train-only vocabulary, неперетину документів і provenance вікон, незалежності від категорій, повного покриття shifted targets та відхилення несумісних checkpoints. Легкі підготовчі комірки — розділи 2–7; ручні baseline/демонстрація/grid/final test — розділи 8/9/10/12. Розділ 11 обирає checkpoint тільки за validation CE. Старі результати генерації залишено в історичних каталогах без змін.

Перевірка міграції BBC: **57 passed in 23.19s**; `compileall` та `git diff --check` — exit 0. Обидва notebooks пройшли JSON/schema/syntax; виконано 12 підготовчих комірок classification та 14 generation на CPU, до першої ручної комірки. Новий BBC EDA-графік перевірено візуально; QA-файли записано тільки в тимчасовий каталог. SHA-256 підтвердив незмінність 51 захищеного файла (classification, дані, референси, старі outputs, модель генерації, tokenizer та autoregressive loop) і байтову тотожність переміщеного BBC CSV. Пошук `spam|ham|SMS|Message|Category` у generation-коді та notebook не знайшов збігів. Нових навчених BBC-моделей або test-метрик не створено.

## Фактичні результати поточного протоколу

Обидва notebooks виконано користувачем. Post-training аудит звірив історії, summary, comparison, metadata checkpoints та збережені test-передбачення без повторного навчання або model evaluation. Нижче наведено тільки `classification/early_stopping/` та `generation/bbc/early_stopping/`; старі п'ятиепохові результати й попередній корпус генерації не включено.

| Варіант | Обрана модель (d_model/heads/layers) | Критерій validation | Best / виконано епох | Фінальний test |
|---|---|---|---|---|
| Spam classification | 128/4/1 | F1=0,902174, максимум | 12 / 17 | accuracy=0,963824; precision=0,833333; recall=0,885417; F1=0,858586 |
| BBC generation | 128/2/1 | CE=5,631608, мінімум; PPL=279,110503 | 7 / 12 | CE=5,623352; PPL=276,815826; BLEU-4=0,0049112482; ROUGE-L=0,1064453125 |

Усі десять поточних baseline/grid запусків зупинилися за patience=5; best_epoch, validation-метрики та кількість епох узгоджені з history/checkpoint. Classification confusion matrix: `[[661,17],[11,85]]` (actual/predicted: ham, spam); зростання validation loss при малому train loss вказує на перенавчання. BBC CE поліпшилась, але тексти зациклюються: UNK є у 63/64 test-продовжень (705/2048 токенів). Це обмеження фактичної якості генерації, попри коректний каузальний протокол. Детальний аналіз — у фінальних розділах обох notebooks.

Збережені execution counts показують вибір checkpoint перед final test; у коді test не бере участі в early stopping або ранжуванні. Самі notebooks не є повним журналом усіх запусків kernel, тому абсолютну кількість минулих повторних test-викликів за ними встановити неможливо. Артефакти експериментів під час аудиту не перезаписувалися.

Post-training перевірка: **55 passed, 2 failed in 25.44s** (`python -B -m pytest labs/lab01_transformer/tests -q -p no:cacheprovider`). Обидві невдачі — застаріла вимога `test_notebooks_valid_unexecuted` у `tests/test_lab01.py:208`, яка забороняє execution counts/outputs і суперечить збереженню фактично виконаних notebooks. Тести не змінювалися, outputs не очищалися. JSON/schema/syntax обох notebooks валідні; IDE-поле `jetTransient` перенесено всередину стандартного output metadata без зміни видимих результатів. `compileall` та `git diff --check` — успішні. SHA-256 підтвердив незмінність усіх експериментальних файлів, даних, src і tests; змінено тільки фінальні Markdown-висновки, службове metadata notebooks та README.
