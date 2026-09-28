# Лабораторна №4 — Stable Diffusion, Варіант 1

**Виконано:** локальна text-to-image генерація, 13 унікальних PNG, 7 фігур, виконаний notebook без помилок, звіт і відповіді на 10 контрольних запитань. Мета — порівняти seed, steps, CFG, negative prompt та стиль, змінюючи один фактор за раз.

## Модель і середовище

- `stable-diffusion-v1-5/stable-diffusion-v1-5`, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- Методичка називає `runwayml/stable-diffusion-v1-5`; цей ID через API перенаправляється на зазначений еквівалент SD 1.5. Це не інше сімейство моделей.
- `StableDiffusionPipeline`, штатний `PNDMScheduler`, 512×512, batch 1, safety checker збережено.
- Windows; Python **3.13.5**, torch **2.14.0+cu130**, CUDA runtime **13.0**.
- Diffusers **0.40.0**, Transformers **5.17.0**, Accelerate **1.15.0**, Safetensors **0.8.0**.
- NVIDIA GeForce RTX 3050 Laptop GPU, **4 ГБ VRAM**, RAM 15,19 GiB.
- **FP16 + model CPU offload + SDPA (`AttnProcessor2_0`)**. Без xFormers, attention slicing, VAE slicing/tiling. OOM не було.

SD 1.5 обрано через 4 ГБ VRAM і дозвіл методички. Повні перевірені версії, GPU/RAM/disk preflight і scheduler config: [environment.json](outputs/metadata/environment.json). Torch/CUDA не перевстановлювалися; додано лише відсутні бібліотеки й їхні залежності.

## Запуск у Windows / PyCharm

Команди з кореня репозиторію. Виберіть наявний `.venv/Scripts/python.exe` у PyCharm. На іншій машині спочатку потрібен робочий CUDA-enabled torch; requirements навмисно не задає нову збірку torch. Python 3.11 не тестувався.

```powershell
# Перевірити існуючу збірку й захистити її від заміни resolver-ом.
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -c "import torch,pathlib; p=pathlib.Path('labs/lab04_stable_diffusion/outputs/metadata'); p.mkdir(parents=True,exist_ok=True); (p/'torch-constraint.txt').write_text('torch=='+torch.__version__+'\n')"
.\.venv\Scripts\python.exe -m pip install -c labs/lab04_stable_diffusion/outputs/metadata/torch-constraint.txt -r labs/lab04_stable_diffusion/requirements.txt
.\.venv\Scripts\python.exe -m labs.lab04_stable_diffusion.src.run
.\.venv\Scripts\python.exe -m pytest labs/lab04_stable_diffusion/tests -q
```

Основний навчальний матеріал: [notebooks/01_text_to_image.ipynb](notebooks/01_text_to_image.ipynb), **12 виконаних code cells**, українські пояснення, результати вбудовані. Його можна виконати зверху вниз замість CLI. Повторний запуск перевіряє metadata, SHA-256 та розміри PNG; якщо всі результати придатні, ваги взагалі не завантажуються. Відсутні випадки генеруються після єдиного завантаження pipeline. Помилка baseline зупиняє виконання; PNG та JSON кожного попереднього успішного запуску збережені.

Не змішувати різні версії середовища в одному порівнянні. Повний model ID/revision, scheduler config, dtype і controls входять до identity. Зміна середовища вимагає повторної перевірки сумісності; між різними GPU/версіями побітова тотожність не гарантована. Для кожної генерації створюється **новий CPU generator**, а не один глобальний спожитий generator.

## Матриця

| Серія | Значення | Нові PNG | Повторне використання |
|---|---|---:|---|
| Baseline | seed 42, steps 30, CFG 7.5, negative=None | 1 | — |
| Seed | 42, 123, 777, 2026 | 3 | baseline |
| Steps | 20, 30, 50 | 2 | baseline для 30 |
| CFG | 5, 7.5, 10 | 2 | baseline для 7.5 |
| Negative prompt | None / configured, seeds 42 і 123 | 2 | baseline і seed 123 |
| Styles, друга сцена | watercolor, isometric, photorealistic | 3 | — |

Primary scene — futuristic city; друга — cabin/lake/pines. Точні prompts централізовано в `src/config.py`. Scheduler, checkpoint, 512×512 сталі для всіх. Контроль незмінних факторів перевірено тестами. Scheduler comparison: **OPTIONAL — NOT EXECUTED**.

## Результати

Seed помітно змінив композицію та палітру. 20/30/50 кроків зберегли основний мотив, але змінили мощення й фасади; однозначного покращення при 50 не встановлено. CFG 10 зробив башти округлими та сцену більш стилізованою. Negative prompt змінив композицію, але псевдонаписи залишилися. Стилі змінили фактуру і реалізм; `isometric` не забезпечив строгої ізометричної проєкції.

Час 20/30/50 кроків: **6,76 / 18,88 / 13,95 с**. 30-крокова baseline — перший inference, тому містить ефект прогрівання. Це по одному спостереженню, а не коректний benchmark швидкості. Завантаження моделі виключено з часу inference. Повний аналіз — у [report/report.md](report/report.md).

## Файли

- `src/`: immutable config, матриця, pipeline, генерація/metadata/resume, фігури, CLI.
- `tests/`: офлайн тести контролів, унікальності, імен, metadata/resume та notebook; модель не завантажують.
- [outputs/metadata/generations.csv](outputs/metadata/generations.csv): 13 рядків; окремі JSON — повний запис кожного PNG.
- `outputs/images/{baseline,seed,steps,cfg,negative_prompt,styles}/`: 13 PNG.
- `outputs/figures/`: 6 візуальних порівнянь і runtime_by_steps.png.
- [report/report.md](report/report.md): звіт із кодом, фактичними результатами й параметрами.
- [report/defense_notes.md](report/defense_notes.md): усі 10 відповідей, включно з LoRA.
- [report/screenshot_checklist.md](report/screenshot_checklist.md): лише потрібні GUI screenshots, якщо викладач їх окремо вимагає.

`outputs/` ігнорується чинною кореневою політикою Git; PNG/CSV існують локально й не потрапляють у commit автоматично. Вбудовані результати notebook доступні без ваг. Для передачі повного звіту зі зовнішніми зображеннями потрібно передавати також outputs. Ваги залишаються в стандартному Hugging Face cache, не в лабораторній.

## Перевірки і межі

`pytest labs/lab04_stable_diffusion/tests`: **13 passed**. `compileall`: PASS. Notebook schema/syntax/execution/error audit: PASS. `git diff --check`: PASS. Метадані всіх 13 PNG перевірено за конфігураціями, SHA-256, розмірами й існуванням. Візуально оглянуто всі порівняння. Lab 1/2/3 та кореневі файли не змінено; початкові незакомічені зміни Lab 3 збережено. Commit/push не виконувались.

Оцінювання якісне та обмежене малою серією. Пам'ять RAM під час preflight була майже зайнята; offload і прогрівання впливають на час. LoRA розглянуто теоретично, навчання не виконувалось. Варіант 2 не реалізовувався.

## Відповідність методичці

| Вимога | Статус | Доказ |
|---|---|---|
| Середовище | PASS | environment.json; CUDA доступна |
| SD 1.5 | PASS | model_id і revision у кожному sidecar |
| StableDiffusionPipeline | PASS | фактичний pipeline_class |
| Baseline 1–2 | PASS | 1 baseline PNG |
| 3–5 seeds | PASS | 42, 123, 777, 2026; seed_comparison.png |
| Steps 20/30/50 | PASS | steps_comparison.png |
| CFG 5/7.5/10 | PASS | cfg_comparison.png |
| Scheduler comparison | OPTIONAL — NOT EXECUTED | PNDM фіксований для всіх випадків |
| Negative prompt 2–3 приклади | PASS | 2 пари, seeds 42 і 123 |
| Друга сцена | PASS | cabin/lake/pines у стильовій серії |
| Три стилі | PASS | watercolor, isometric, photorealistic |
| Параметри кожного зображення | PASS | 13 JSON sidecars і generations.csv |
| Зрозумілі імена | PASS | seed, steps, CFG у назві PNG |
| Код і результати у звіті | PASS | фрагмент коду, PNG, таблиця нижче |
| Контрольні запитання | PASS | defense_notes.md: усі 10 |
| Notebook | PASS | 12 виконаних code cells, без error outputs |
| Звіт | PASS | report/report.md із реальними PNG та кодом |
| Defense notes | PASS | 10 відповідей українською |
