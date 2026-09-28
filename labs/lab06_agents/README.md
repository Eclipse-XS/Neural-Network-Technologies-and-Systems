# Lab 6 — локальний агентний асистент

**Варіант 3: OpenAI Agents SDK.** Аналіз реальних звітів Lab 3–5, арифметика та контекст діалогу. Експерименти реально виконані через LM Studio; помилки наведено разом з успіхами у [звіті](report/report.md).

Python 3.13.5, openai-agents 0.22.3, openai 3.17.0. Фактично виявлена модель — `mistralai/ministral-3-3b`, endpoint `http://localhost:1234/v1`, Chat Completions, temperature 0, tool_choice auto, max_turns 10. Model ID визначається через `/v1/models`.

## Архітектура

`User → Agent / Runner → LM Studio`; Runner працює з SQLiteSession і чотирма function tools. LLM обирає текст або function call, SDK виконує Python-функції. Окремого Planner чи мультиагентної системи немає.

| Tool | Призначення |
|---|---|
| list_workspace_files | Імена та розміри дозволених звітів |
| search_workspace | Literal case-insensitive пошук із номерами рядків |
| read_workspace_file | Обмежене читання фрагмента |
| calculate | Арифметика через AST |

`assets/workspace/` містить байтові копії трьох звітів і SHA-256 manifest. Агент не читає інші лабораторні каталоги. Історія зберігається в `outputs/sessions/agent_memory.sqlite`; session_id ізольовані. Це пам'ять діалогу, не векторна база.

## Запуск

З кореня репозиторію у PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r labs/lab06_agents/requirements.txt
lms server start
lms ls
lms ps
.\.venv\Scripts\python.exe -m labs.lab06_agents.src.preflight
.\.venv\Scripts\python.exe -m labs.lab06_agents.src.experiments
```

Якщо lms відсутній у PATH: LM Studio → Developer → Start Server на порту 1234. Потрібна локальна chat-модель зі structured tool calling. Перший запит може завантажити її через JIT loading; інакше використайте LM Studio або `lms load <local-model-id>`.

Налаштування наведені у `.env.example`. Файл .env автоматично не завантажується: використовуйте змінні середовища PowerShell. Реальний OpenAI API key не потрібний. Cloud tracing вимкнене програмно. Проксі та redirects вимкнені; base_url дозволяє тільки loopback HTTP.

Нова серія створює свіжі ID сесій, додає JSONL та зберігає артефакти після кожного запуску. Помилки endpoint/model — у startup_error.json; помилки Runner і часткові trajectories — у runs/traces. Загальний timeout Runner — 600 секунд. Fallback на cloud відсутній.

## Експерименти

Сценарії: пряма відповідь, два lookup, cross-file synthesis, multi-step counts, псевдонім та ізоляція, невідомі погодні дані, контрольна пара without/with tools, сума та відсоток. Expected facts читає тільки evaluator.

Початкова серія виявила неправильний filename, пропущений calculator call і непідтверджені припущення. Після уточнення загальних інструкцій виконано повторну серію. Еталон: 13 генерацій Lab 4, 17 унікальних inference Lab 5, різниця 4. Правильність оцінюється окремо від SUCCESS. Результати — у report/report.md та report/manual_evaluation.json.

## Артефакти

- [Виконаний notebook](notebooks/01_openai_agents_sdk.ipynb) завантажує справжні logs і перевіряє hashes; режим явно зазначено.
- `outputs/runs/runs.jsonl`: усі завершені запуски.
- `outputs/traces/`: запити функцій, аргументи, результати та помилки.
- `outputs/metadata/`: версії, models, preflight, batch IDs, session items, початковий Git-стан.
- [Звіт](report/report.md), [захист](report/defense_notes.md), [аудит](report/methodology_audit.md), [скріншоти — чекліст](report/screenshot_checklist.md).

Outputs і SQLite ignored за політикою репозиторію. Notebook містить outputs для перегляду без сервера; для повторного виконання клітинок потрібні logs. Нова серія потребує нових ручних оцінок за run_id. `python -m labs.lab06_agents.src.reporting` перебудовує звіт і невиконаний notebook після оцінювання; після цього його слід виконати у Jupyter/nbclient.

## Перевірки та межі

```powershell
.\.venv\Scripts\python.exe -m pytest labs/lab06_agents/tests -q
.\.venv\Scripts\python.exe -m compileall labs/lab06_agents/src
git diff --check
```

Pytest не звертається до LM Studio. Інструменти не мають shell, web, write чи unrestricted Python. Доступ дозволено тільки до report copies. Звіти є даними, не інструкціями. Literal search не є semantic search; модель може пропустити факт, цитату, калькулятор або неправильно синтезувати прочитане. Скріншоти GUI залишаються ручним кроком.
