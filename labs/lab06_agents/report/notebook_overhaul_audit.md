# Аудит практичного notebook ЛР6

Дата перевірки: 2026-10-30. Baseline: гілка `codex/labs-3-4-portfolio-upgrade`, HEAD `01ec07f9ce7394365b22e0308fe38e01c30790a8`.

## Початковий стан

Початковий notebook мав 32 комірки, з них 15 code cells. Він завантажував готові records через `load_verified()` і показував їх функцією `show_run(...)`. У notebook не було видимого створення `Agent`, `SQLiteSession` або прямого `await Runner.run(...)`. Тому він був виконаним REPLAY-звітом, а не практичною демонстрацією створення й запуску агента.

До початку роботи робоче дерево вже містило незакомічені зміни `assets/workspace/lab03_report.md`, `lab04_report.md` і `manifest.json`. Вони не створені цим overhaul і не відновлювалися. Через зміну manifest hash поточний workspace більше не відповідає hash історичної серії; notebook чесно маркує її `REPLAY_UNVERIFIED_WORKSPACE_MISMATCH`.

## Реалізована структура

1. Теорія, workflow, режими LIVE/REPLAY, imports і фактичні версії.
2. `/v1/models` та реальний Chat Completions preflight з runtime і без виведення key.
3. Видиме створення `Agent` з локальним `OpenAIChatCompletionsModel` і `ModelSettings`.
4. Реалізація, JSON schema, validation, error handling і прямі unit-проби tools.
5. Прямий `await Runner.run(...)`, trace hooks, extraction calls і окреме LIVE-збереження.
6. Контроль without/with tools.
7. `SQLiteSession`: два turns, recall та незалежна session B.
8. Multi-step execution і observable trace.
9. Контрольований вплив instructions.
10. Фактичні failure modes і способи контролю.
11. Provenance та окрема директорія `outputs/notebook_live/`.
12. Pandas/Matplotlib аналіз.
13. Висновки та матриця PASS/BLOCKED/FAIL/NOT_RUN.

Повна матриця не запускається автоматично. `LAB06_RUN_FULL_LIVE=1` вмикає baseline/tools, memory, multi-step та instruction variants. Endpoint preflight і один простий live agent smoke увімкнені за замовчуванням.

## Фактичний LIVE результат

- Endpoint: `http://localhost:1234/v1` — PASS.
- Модель: `mistralai/ministral-3-3b`.
- Direct response: `READY` — PASS.
- SDK Runner smoke: PASS як завершення виконання.
- Run ID: `2c5799b47f02497c8bada9a06d665f20`.
- Autonomous calls: `list_workspace_files` → `search_workspace`.
- Tool call count: 2.
- Отриманий checkpoint: `stable-diffusion-v1-5` з посиланням на `lab04_report.md`.
- Обмеження відповіді: модель додала в final text фрагмент, схожий на невиконаний `read_workspace_file` call. У trace третього call немає. Це реальна помилка форматування локальної моделі, її не можна подавати як виконаний tool call.

## Перевірки

| Перевірка | Статус | Результат |
|---|---|---|
| nbformat schema | PASS | 27 cells |
| Syntax усіх code cells | PASS | 13/13 |
| CPU REPLAY notebook | PASS із warning | історичний hash mismatch явно маркований |
| Executed notebook | PASS | execution_count 1–13, error outputs 0 |
| pytest | PASS | 33 passed, 1 skipped |
| Tool unit tests | PASS | входить у pytest |
| SQLiteSession persistence/isolation | PASS | входить у pytest |
| compileall | PASS | `src/` і builder |
| Endpoint preflight | PASS | models + direct completion |
| Live agent smoke | PASS | Runner і два фактичні calls |
| Full live matrix | NOT_RUN | потребує явного opt-in |
| Новий live multi-step integration | NOT_RUN | код присутній; історичний multi-step доступний лише як REPLAY |
| Historical evidence verification проти поточного workspace | FAIL | manifest був змінений до overhaul |
| Secret exposure check | PASS | API key не збережений і не надрукований |
| `git diff --check` для ЛР6 | PASS | лише line-ending warnings |

Не можна оголошувати FULL PASS: дві обов'язкові live-перевірки залишаються `NOT_RUN`, а historical workspace verification має `FAIL` через наявний drift.

## Запуск у PyCharm/Jupyter

1. У корені репозиторію створити interpreter:
   `uv venv .venv-lab06 --python 3.13`.
2. Встановити залежності:
   `uv pip install --python .venv-lab06\Scripts\python.exe -r labs\lab06_agents\requirements.txt`.
3. У LM Studio завантажити chat model зі structured tool calling і запустити local server на `http://localhost:1234/v1`.
4. У PyCharm вибрати `.venv-lab06\Scripts\python.exe` як interpreter/kernel.
5. Відкрити `notebooks/01_openai_agents_sdk.ipynb` і виконати Run All. За замовчуванням виконається preflight, один Runner smoke та REPLAY.
6. Для повної live-матриці перед запуском задати environment variable `LAB06_RUN_FULL_LIVE=1` і перезапустити kernel.
7. Не переносити manual evaluation старих run IDs на нові outputs; нові відповіді оцінюються окремо.

## Оцінка відповідності

Поточна оцінка: **88/100**.

Notebook структурно покриває Agent, tools, Runner, memory, multi-step, instruction comparison, evaluation, provenance та visualization. Реально перевірені endpoint і автономний tool-calling smoke. Бали знято за невиконану нову full live matrix, невиконаний новий multi-step integration і неможливість підтвердити історичну серію проти зміненого workspace. Після відновлення узгодженого workspace та окремого запуску повної матриці оцінка може зрости; саме наявність коду не є доказом успішного live виконання.
