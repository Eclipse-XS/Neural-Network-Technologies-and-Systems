# Coursework Knowledge Assistant — Lab 7

Варіант 1: локальний інтерактивний асистент для аналізу звітів із нейромережевих технологій. Об’єднує **Lab 3: LangChain + Redis RAG** та **Lab 6: OpenAI Agents SDK + tools + SQLiteSession**. LM Studio виконує inference; код попередніх лабораторних не імпортується.

Користувач → CLI → Agent/Runner ↔ LM Studio. Агент має три інструменти: пошук фрагментів у Redis через MiniLM, каталог джерел та AST-калькулятор. SQLiteSession зберігає діалог. Пошук не викликає другу LLM. Повна схема й аналіз — у [звіті](report/report.md).

## Швидкий запуск із кореня репозиторію

Перевірене середовище: Windows, Python 3.13.5, torch 2.14.0+cu130; Docker Desktop із Linux containers, LM Studio із наявною tool-capable chat-моделлю. На інших системах setup.ps1 замінюється звичайним venv із сумісним PyTorch. Python ≥3.11 очікується за залежностями, але фактично перевірено лише 3.13.5.

```powershell
# Створює Lab 7 .venv, захищаючи наявні torch, transformers, openai та SDK.
.\labs\lab07_final_project\setup.ps1 -Python .\.venv\Scripts\python.exe
docker compose -f labs/lab07_final_project/docker-compose.yml up -d
docker compose -f labs/lab07_final_project/docker-compose.yml ps
```

У LM Studio завантажте наявну chat-модель і ввімкніть Local Server. У цьому запуску виявлено `mistralai/ministral-3-3b` на `http://localhost:1234/v1`; інша модель повинна пройти preflight із native function calling. API key хмарного OpenAI не потрібний.

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
$env:LM_STUDIO_MODEL = "mistralai/ministral-3-3b"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
.\labs\lab07_final_project\.venv\Scripts\python.exe labs/lab07_final_project/app.py
```

Перший запуск на новій машині потребує ваг невеликого embedding encoder у Hugging Face cache. У поточному середовищі вони вже були. Одноразове завантаження, якщо кеш порожній:

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
```

Після цього застосунок завантажує embeddings із `local_files_only=True`. `.env.example` — довідка, файл автоматично не читається: значення експортуються в shell/PyCharm. Індекс із відповідним fingerprint повторно використовується; автоматичне додавання дублікатів відсутнє. Redis URL за замовчуванням — `redis://localhost:6380`, окремо від Lab 3 на 6379.

## CLI та експерименти

Команди: `/help`, `/sources`, `/new`, `/session`, `/exit`. `/new` зберігає стару історію та створює новий ID. `--session ID` відновлює сесію; `--debug` показує імена tools, latency та перевірку citation IDs.

```powershell
# Індексація та 8 retrieval-запитів
.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab07_final_project.src.experiments --retrieval-only
# Повна серія, або перевірене повторне читання сумісного batch
.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab07_final_project.src.experiments
# Свідомо нові inference-запуски з новими сесіями
.\labs\lab07_final_project\.venv\Scripts\python.exe -m labs.lab07_final_project.src.experiments --force
```

Для нової серії потрібно заново вручну оцінити її run IDs у `experiments/reviews.json`. Evaluator навмисно не переносить оцінки старих відповідей на нові й не підміняє review пошуком ключових слів.

Відкрийте [виконаний notebook](notebooks/01_final_project.ipynb) у PyCharm/Jupyter з interpreter Lab 7. Він перевіряє хеші реальних run/trace артефактів і fingerprint конфігурації, повторно відкриває Redis та виконує один свіжий retrieval. Усі 15 генерацій не повторюються. За відсутніх outputs спочатку виконайте серію й оцінювання.

## Структура

```text
app.py                         інтерактивний CLI
src/config.py                  налаштування та system instructions
src/knowledge_base.py           corpus → chunks → embeddings → Redis
src/tools.py, agent.py          функції, локальна LLM, SQLite, traces
src/experiments.py              фіксовані retrieval/agent/baseline запуски
src/evaluation.py, reporting.py оцінювання й артефакти
assets/knowledge/              чотири незмінені звіти + manifest
experiments/                   запити, сценарії, ручні reviews за run_id
notebooks/                     виконана демонстрація українською
tests/                         offline corpus/tool/memory/notebook тести
report/                        звіт, аудит методички, захист, скриншоти
outputs/                       локальні згенеровані докази, ігноруються Git
```

## Фактичний результат

- 4 документи, 121 chunk, 900 символів/overlap 120; MiniLM CPU, normalized, 384 dimensions.
- Redis Stack: COSINE, FLAT, FLOAT32; FT.INFO підтвердив кількість, повторний запуск додав 0.
- Source Hit@5: **6/7**, 8 запитів загалом. Це наявність звіту, не гарантія знаходження факту.
- **15/15** основних запусків завершили Runner; 10 сценаріїв включають 11 assistant turns і 4 plain baseline.
- Ручні оцінки 11 assistant turns: **3 CORRECT, 3 PARTIAL, 4 INCORRECT, 1 NOT_SCORABLE**. Це не універсальна accuracy.
- Пам’ять зберегла Lab 6 у follow-up; нова сесія попросила уточнення. Сам початковий модельний факт у memory-сценарії не знайдено.
- Multi: три search_knowledge, calculate пропущено, число Lab 5 не знайдено; фінального правильного порівняння немає. Калькулятор окремо перевірений реальним SDK preflight.
- Plain baseline вигадала температуру +18°C; асистент відмовився її визначати. Інші порівняння не дають підстав оголошувати повну перемогу RAG.

**Обмеження:** 76/121 chunks перевищують 256-token limit MiniLM й обрізаються для embedding. Повний текст лишається у Redis. Англомовний encoder на українських звітах дає слабкий retrieval. Звіти також містять цитати хибних відповідей, які модель може переплутати з фактами. Валідні citation IDs не гарантують підтримки тверджень. GUI-скриншоти залишені ручними за [checklist](report/screenshot_checklist.md).

## Докази та перевірки

`outputs/runs/runs.jsonl` — усі реальні відповіді, включно з невдачами, preflight і CLI; `batch.json` відокремлює основні 15. `traces/<run_id>.json` — спостережувані виклики; `summary.csv` і `baseline_comparison.csv` — ручні оцінки та порівняння. `retrieval/retrieval_results.csv`, `metadata/environment.json`, `index.json`, `embedding.json`, `chunks.json` фіксують середовище й пошук. `sessions/assistant_memory.sqlite` — локальна пам’ять. Outputs і віртуальні середовища не додаються до Git.

```powershell
.\labs\lab07_final_project\.venv\Scripts\python.exe -m pytest labs/lab07_final_project/tests -q
.\labs\lab07_final_project\.venv\Scripts\python.exe -m compileall -q labs/lab07_final_project/src labs/lab07_final_project/app.py
git diff --check
```

Фінальна перевірка: **27 passed**, compileall і git diff --check успішні; notebook виконав **18 кодових комірок без error outputs**. Тести не потребують LM Studio, Docker, Redis чи GPU. Окремі живі перевірки виконано під час експериментів і CLI-демо. Початковий Git state, hashes попередніх робіт і кінцевий аудит лежать у outputs/metadata. Перевірено 100 файлів попередніх лабораторних: змін немає; кореневі Python-пакети також незмінні. Комітів і push не виконувалось.
