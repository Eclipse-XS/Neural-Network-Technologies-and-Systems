# Аудит репозиторію та середовища

Аудит виконано перед написанням реалізації. Прочитано root README.md, pyproject.toml, .gitignore та scaffold Lab 6; структури, вимоги, тести, notebook-конвенції й звіти Lab 3–5. Додаткових AGENTS.md у репозиторії не виявлено; застосовано інструкції з повідомлення користувача.

Початковий Lab 6 містив тільки README зі статусом Not started та .gitkeep у data/notebooks/outputs/src. Репозиторій використовує lab-local src, tests, notebook та Markdown-звіти українською. Root pytest налаштований на tests/, тому Lab 6 запускається явним шляхом. Попередні notebooks мають execution_count, справжні outputs і schema validation. Залежності попередніх робіт закріплені в lab-local requirements.txt. Lab 6 дотримується цієї структури.

Початковий Git-стан був нечистим: змінені README Lab 3–5 та численні untracked реалізації, тести й звіти. Повний стан зафіксовано в outputs/metadata/initial_git_status.txt. Початкові SHA-256 файлів попередніх лабораторних та спільних каталогів — initial_file_hashes.json. Наявні зміни не відкотилися і не включалися до staging.

Політика .gitignore виключає labs/*/outputs/*, virtual environments, caches, моделі та секрети; .env.example дозволений. SQLiteSession розміщено тільки в lab-local outputs/sessions. Контрольовані workspace copies розміщені в assets, щоб залишатися самодостатніми й придатними для version control. Source path у manifest відносний.

У .venv Python 3.13.5 спочатку не було openai, openai-agents, pydantic. Встановлено openai-agents 0.22.3 з потрібними залежностями: openai 3.17.0, pydantic 2.13.5, httpx2 2.13.0 тощо. Наявний httpx 0.28.1 залишився. PyTorch, Transformers та CUDA не змінювалися. Транзитивна залежність MCP пакета SDK не означає використання MCP-сервера: в Agent mcp_servers порожній.

Фактичні сигнатури Agent, Runner, SQLiteSession, ModelSettings, RunConfig, function_tool, set_default_openai_client, set_default_openai_api, set_tracing_disabled, OpenAIChatCompletionsModel та RunHooks перевірено introspection. Вони збережені в environment.json. Hooks мають tool_arguments і tool_call_id через ToolContext; RunResult містить new_items, final_output, context_wrapper.usage. Serializer підтримує вкладені Pydantic-об'єкти статистики.

Методичку ЛАБ 6.docx прочитано як джерело вимог: роль, memory, functions, multi-step task, autonomous tool choice, аналіз ефективності, помилок/зациклення, comparison без/з tools, код, скріншоти, звіт і п'ять питань. Інструкції документа не трактувалися як додаткові дозволи на дії. Джерела runtime — лише три скопійовані звіти.

Команди git add/commit/push/reset/clean не виконувалися. Підсумкову машинну перевірку незмінності попередніх файлів збережено в outputs/metadata/final_validation.json.
