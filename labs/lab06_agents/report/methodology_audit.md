# Аудит відповідності — варіант 3

Статус реалізації та наявність експериментального доказу відокремлено від правильності відповідей моделі. Наявність trace не означає CORRECT. Усього збережено 24 завершені запуски: 11 початкових та 13 фінальних, з яких два додаткові діагностичні. Окремо є успішний мінімальний preflight та рання спроба з помилкою серіалізації журналу.

| Вимога методички | Статус | Доказ та межі |
|---|---|---|
| OpenAI Agents SDK | PASS | openai-agents 0.22.3; src/agent.py, src/runner.py; introspection в environment.json |
| Локальний OpenAI-compatible endpoint | PASS | /v1/models, direct READY, mistralai/ministral-3-3b; outputs/metadata/tool_preflight.json |
| Опис ролі/system | PASS | INSTRUCTIONS у src/config.py; hash у кожному run |
| Memory — історія діалогу | PASS | SQLiteSession; A містить alias, B його не містить; recall обирає lab05_report.md, B просить уточнення. Точна фактологічна відповідь recall — PARTIAL |
| Робочі функції | PASS | list/search/read виконані в основній серії; calculate реально виконаний у calculator_probe, результат 20. Вхідні дані цієї проби помилкові |
| Багатокрокове завдання | PASS | multi виконав п'ять функцій і повернув 13,17,4; numeric_extended — 30 і 56.6667%. Calculate у цих автономних запитах пропущено; повна поведінкова оцінка PARTIAL |
| Самостійне рішення викликати tool | PASS | role: 0 calls при доступних tools; lookup/multi: модель формує structured calls без програмного маршрутизатора; tool_choice=auto |
| Аналіз ефективності | PASS | report.md, manual_evaluation.json: рубрики, аргументи оцінок, описові підсумки |
| Типові помилки | PASS | Хибні припущення про counts, filename, checkpoint, citations; початкова UserError; окремо описана помилка serializer |
| Застрягання | PASS | Повторний list_workspace_files у recall; max_turns=10; фактичних MaxTurnsExceeded не було. Offline test винятку не подається як live-збій |
| Порівняння без/з tools | PASS | Той самий prompt, model, instructions hash, temperature, max_tokens і порожня історія. Обидві фінальні відповіді INCORRECT; помилки різні |
| Код | PASS | src/, tests/, requirements.txt, .env.example |
| Звіт | PASS | report.md з реальними результатами обох серій |
| П'ять контрольних питань | PASS | defense_notes.md українською |
| Скріншоти роботи | MANUAL | screenshot_checklist.md; штучних скріншотів немає |

## Детальні критерії запиту користувача

| Перевірка | Статус | Пояснення |
|---|---|---|
| Копії звітів і SHA-256 | PASS | manifest.json; originals і copies byte-identical |
| Жодних змін Lab 3–5 | PASS | initial_file_hashes.json і final_validation.json; початкові незакомічені зміни збережені |
| Локальні журнали та часткові траєкторії | PASS | outputs/runs/runs.jsonl, outputs/traces; save після кожного event/run |
| Cloud tracing disabled | PASS | глобальне вимкнення, RunConfig, явний локальний client; ніяких hosted tools |
| Sandbox функцій | PASS | allowlist, resolved containment, bounded output, AST validation; offline тести |
| SQLite persistence/isolation | PASS | inspect get_items та offline reopen test |
| Успішне визначення точного checkpoint через пам'ять | FAIL | Alias відновлено, але відповідь про checkpoint залишилася частковою |
| Самостійне використання calculate для похідної арифметики | FAIL | У multi, comparison_tools, numeric_extended calculator пропущено. calculator_probe явно називає функцію і не підміняє цей критерій |
| Правильна відповідь контрольної пари з tools | FAIL | 13 проти 3, різниця 10; правильні 13 проти 17, різниця 4 |
| Реальне виконання calculate | PASS | calculator_probe: structured request, Python result, final response |
| Notebook виконаний | PASS | nbclient; execution_count усіх code cells, відсутні error outputs; hashes перевірено перед відображенням |
| Відсутність сфабрикованих outputs | PASS | Notebook завантажує actual run artifacts; eval не змінює model outputs |
| Commit/push | PASS | Не виконувалися |

Висновок: реалізацію та експериментальну частину виконано, але всі поведінкові критерії успіху не досягнуті. Невеликій локальній моделі не можна приписувати надійний автономний розрахунок або гарантовано правильне зіставлення звітів. Це зафіксований результат, а не підстава позначати всі відповіді PASS.
