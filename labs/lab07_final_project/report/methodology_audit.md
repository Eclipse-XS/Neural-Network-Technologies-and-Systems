# Аудит відповідності методичці

Джерело вимог: «ЛАБ 7.docx», Варіант 1. Текст документа прочитано як методичні умови, а вкладений Pasted text.txt — як запит на реалізацію. Нижче **PASS означає реалізовану й виконану вимогу**, а не безпомилковість кожної відповіді моделі. Поведінкові невдачі наведено окремо.

| Вимога | Статус | Доказ |
|---|---|---|
| Інтерактивний асистент | PASS | app.py; реальний piped-stdin діалог outputs/runs/cli_demo.txt, /help, /sources, /new, /session |
| Структурована пам’ять | PASS | SQLiteSession; memory scenario: два ходи з одним ID, isolation — інший ID; metadata/memory_*.json |
| Текстові запити | PASS | CLI input; user_turns у scenarios.json; збережені user_input |
| Векторна база документів | PASS | lab07-redis, FT.INFO у metadata/index.json: 121 HASH-вектор |
| Індексація документів | PASS | 4 SHA-verified копії звітів; 121 chunk; ідемпотентне повторне використання |
| RAG retrieval | PASS | LangChain RedisVectorStore; retrieval_results.csv, 8 запитів, Source Hit@5=6/7 |
| Відповіді на основі зовнішніх знань | PASS | lab4: справжня цитата chunk-61ffcc82588fbd5f; факт сімейства SD v1.5 підтверджено; є невдалі інші сценарії |
| Agent functions/actions | PASS | search_knowledge, list_knowledge_sources; calculate у реальному SDK preflight |
| Локальна OpenAI-compatible LLM | PASS | mistralai/ministral-3-3b, localhost:1234/v1, Chat Completions; chat/tool preflight |
| LangChain або LlamaIndex | PASS | langchain-core 1.6.4, text-splitters 1.1.2, huggingface 1.2.2, redis 0.2.6 |
| Агентний механізм | PASS | OpenAI Agents SDK 0.22.3 Agent/Runner, tool_choice=auto, finite max_turns |
| Не менше двох попередніх компонентів | PASS | Lab 3 RAG + Lab 6 tools/memory, власні модулі Lab 7 |
| Формування embeddings пояснено | PASS | report.md §11; MiniLM CPU normalized FLOAT32 384; описано 76 truncated chunks |
| Індексацію пояснено | PASS | report.md §9–13; metadata, fingerprint, namespace, FT.INFO |
| Вибір інструментів пояснено | PASS | report.md §17; жодного сценарного routing чи chain-of-thought журналу |
| Порівняння з простою LLM | PASS | 4 реальні контрольовані пари; baseline_comparison.csv, notebook §17 |
| Програмний код | PASS | app.py та src/; runtime не імпортує labs 3–6 |
| Звіт | PASS | report/report.md, 29 розділів, реальні відповіді та ручні оцінки |
| Контрольні запитання | PASS | defense_notes.md: усі 5 офіційних питань |
| Скриншоти | MANUAL | screenshot_checklist.md; користувач дозволив залишити GUI-знімки ручними |

## Додаткові вимоги запиту

| Перевірка | Статус | Доказ або межа |
|---|---|---|
| Не змінювати попередні роботи | PASS | initial_git_status.txt, previous_lab_hashes.json, final_audit.json; збережено початкові незакомічені зміни |
| Поточна Redis-інтеграція | PASS | langchain-redis, без deprecated Community Redis |
| Автономний no-tool turn | PASS | role, нуль function calls |
| Багатоджерельний сценарій | PASS | cross і multi виконані; синтез частково/повністю хибний |
| Автономний RAG + calculate у multi | FAIL | Три search_knowledge, calculate пропущено; правильне порівняння не отримане |
| Працездатність calculate | PASS | preflight: (21+7)/4=7 через function call і відповідь моделі |
| Context follow-up | PASS | другий memory-turn обирає source_id=lab06 без Lab 6 у user input |
| Ізоляція сесії | PASS | нова сесія просить уточнення; offline SQLite test |
| Невідома інформація | PASS | assistant не вигадує outdoor temperature; plain baseline вигадує +18°C |
| Перевірка цитат | PASS | автоматичне членство IDs у trace + ручна перевірка підтримки тверджень; FAIL-відповіді не виправлялись |
| Відсутність expected-answer leakage | PASS | Runner отримує лише user_turns; labels використовує evaluator |
| Виконаний notebook | PASS | notebooks/01_final_project.ipynb; execution counts, без error outputs; notebook_execution.txt |
| Offline тести і compileall | PASS | metadata/pytest_final.txt, final_audit.json |
| Без хмарного OpenAI | PASS | явний loopback client, RunConfig tracing_disabled, відсутні hosted tools; runtime не має shell/web/read-file tool |
| Повторне використання результатів | PASS | fingerprint конфігурації, SHA-256 кожного run/trace; notebook перевіряє перед читанням |

## Межі результату

Методика реалізації та експерименти виконані. Якість моделі не прирівнюється до статусу інфраструктури. У 11 assistant turns: 3 CORRECT, 3 PARTIAL, 4 INCORRECT, 1 NOT_SCORABLE; 15/15 основних Runner calls завершились без transport errors. Не доведено надійне автономне виконання всіх багатокрокових завдань. MiniLM source-hit і валідний citation ID не є метриками правильності відповіді.

Артефакти не сфабриковано й відповіді не редаговано. Журнали preflight/CLI відокремлені від основного batch. Знімки GUI не створено; їх не позначено PASS.
