# Лабораторна робота №5. Робота з мультимодальними моделями

**Варіант 1 — Visual Question Answering з LLaVA.** Виконано 10 базових питань і контрольовані експерименти temperature, max_new_tokens та перефразування. Мета — дослідити спільну обробку зображення і тексту, якість відповідей, галюцинації та узгодженість. Навчання не проводилося.

## Модель і середовище

`llava-hf/llava-interleave-qwen-0.5b-hf`, revision `1090956dd1c79bc93ae98dcf395590369435ec91`. `LlavaForConditionalGeneration`, Hugging Face `ImageTextToTextPipeline` з підкласом для вимірювання continuation/EOS. SigLIP Vision → MLP-проєктор → Qwen2 (Qwen1.5-0.5B-Chat). Повна модель: **864 031 264 параметри**; 0.5B позначає мовну основу. Processor: 384×384. Компактний checkpoint обрано для GPU 4 ГБ.

Windows, Python 3.13.5, PyTorch 2.14.0+cu130, Transformers 5.17.0, Accelerate 1.15.0, Pillow 12.3.0, CUDA 13.0, RTX 3050 Laptop 4 ГБ. Виконано FP16 на cuda:0, SDPA, batch_size=1, без квантизації/offload. Пакети та CUDA не змінювалися.

## Запуск

З кореня репозиторію, PowerShell:

```powershell
# Перевірка чинного CUDA-середовища
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Лише якщо в іншому середовищі бракує залежностей; torch тут не встановлюється
.\.venv\Scripts\python.exe -m pip install -r labs/lab05_multimodal/requirements.txt
# Inference або відновлення з відповідного кешу
.\.venv\Scripts\python.exe -m labs.lab05_multimodal.src.run
# Звіт/графік після ручного оцінювання
.\.venv\Scripts\python.exe -m labs.lab05_multimodal.src.reporting
# CPU-тести без завантаження LLaVA
.\.venv\Scripts\python.exe -m pytest labs/lab05_multimodal/tests
```

Відкрити [notebooks/01_llava_vqa.ipynb](notebooks/01_llava_vqa.ipynb) у PyCharm/Jupyter, вибрати Python із `.venv`, виконати Run All. Notebook уже виконаний та містить реальні зображення й таблиці. Корінь знаходиться від поточного каталогу; користувацьких абсолютних шляхів у коді немає. За повного кешу модель не завантажується; інакше завантажується один раз і генерує тільки відсутні відповіді.

Перший запуск у новому checkout потребує інтернету для ваг (~1,73 ГБ) і місця у стандартному HF cache. Повторні запуски використовують уже завантажені файли. VQA не звертається до веб-пошуку, RAG чи зовнішнього OCR. Lab 2–4 не є runtime-залежностями.

## Вхід і матриця

[primary_image.png](assets/input/primary_image.png) — копія photorealistic-результату Lab 4, 512×512; оригінал не змінено. [source.json](assets/input/source.json) містить походження й SHA-256 `1fc881e5282f86a05f3abb98201985881a331a5241cd404bf9e2b560b5bb08c5`. Питання визначено після огляду пікселів. Модель отримує RGB та питання зі спільною grounding-інструкцією; prompt генератора не передається.

| Експеримент | Умови | Рядків | Нових inference |
|---|---|---:|---:|
| Baseline | 3 описові + 3 факти + 3 аналітичні + 1 контроль; greedy, 64 токени | 10 | 10 |
| Temperature | analysis_01; sampling; 0.2/0.7/1.0; top_p=0.9; seed=42 щоразу; 64 токени | 3 | 3 |
| Max tokens | desc_01; greedy; 24/64/128 | 3 | 2 |
| Перефразування | fact_01; 3 рівнозначні формулювання; greedy, 64 токени | 3 | 2 |
| Разом | Без зайвого добутку параметрів | 19 | 17 |

Greedy не отримує temperature/top_p/seed. Перший fact_01 є smoke test і baseline. Повторно використано baseline для ліміту 64 і першого перефразування; додаткових warm-up-запитів немає.

## Фактичні результати

- Факти baseline: **3 CORRECT, 0 PARTIALLY_CORRECT, 0 INCORRECT**; це не загальна accuracy моделі.
- Контроль координат: `[0.542,0.544,0.856,0.731]` замість визнання невизначеності; HALLUCINATED.
- Опис: основні об’єкти впізнано, матеріал даху wooden не підтверджено, частина просторових деталей пропущена.
- Аналіз: одне пояснення підтримане ознаками, два часткові/тавтологічні.
- Temperature 0.2/0.7/1.0: однаковий висновок про спокійну воду, обґрунтований відбиттями; 19/23/23 токени. Погіршення за вищої температури тут немає.
- Ліміти 24/64/128: фактично 24/48/48 токенів. Перший текст обірвано, два інші однакові й завершені EOS.
- Перефразування: **left / center / Left**. Друге формулювання спричинило помилку.

Ручну оцінку виконав асистент, зіставляючи відповіді із зображенням; це не незалежна експертна оцінка. Один синтетичний кадр, три факти та один seed на температуру не дозволяють узагальнювати частоту помилок. Затримки не є benchmark; перший inference містить warm-up.

## Файли та відтворюваність

- [report/report.md](report/report.md): звіт із відповідями, оцінками, кодом та висновками.
- [report/defense_notes.md](report/defense_notes.md): усі п’ять контрольних запитань.
- [report/manual_evaluation.json](report/manual_evaluation.json): оцінки з run_id, точною відповіддю та image SHA.
- [report/methodology_audit.md](report/methodology_audit.md): відповідність вимогам і перевірки.
- [report/repository_audit.md](report/repository_audit.md): початковий аудит.
- `src/`: config, завантаження, централізований inference, матриця, оцінювання, CLI, звіт.
- `outputs/responses/<run_id>.json`: результат атомарно зберігається одразу після inference.
- `outputs/responses/responses.jsonl`: 17 унікальних відповідей; CSV — 19 рядків порівнянь.
- `outputs/metadata/`: manifest, середовище, model config, evaluation.csv, журнали та validation.
- `outputs/figures/token_budget.png`: графік фактичної довжини відповіді.

Підпис кешу включає SHA зображення, model/revision, питання, grounding, generation config і середовище. Нові умови потребують нової оцінки. За зміни експерименту переглянути manual_evaluation.json та висновки reporting.py/notebook. Seed не гарантує побітового збігу на іншому GPU/стеку.

`outputs/` ігнорується Git; для передачі повного звіту слід включити ці локальні файли. Вбудовані результати notebook доступні без ваг. Model cache і virtualenv не копіювалися в лабораторну. Commit/push не виконувалися. Залишаються усний захист та, якщо потрібні GUI-знімки, [screenshot_checklist.md](report/screenshot_checklist.md).
