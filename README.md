# Neural Network Technologies and Systems

Репозиторій лабораторних робіт із дисципліни «Технології та системи нейронних мереж». Кожна лабораторна має окремі `src/`, notebook, залежності, результати та звіт.

| № | Тема | Основні технології | Матеріали |
|---:|---|---|---|
| 1 | Transformer з нуля | PyTorch, encoder/decoder, training loops | [README](labs/lab01_transformer/README.md) |
| 2 | Локальна LLM через API | LM Studio, OpenAI-compatible API | [README](labs/lab02_llm_api/README.md) |
| 3 | RAG над CSV | LangChain, MiniLM, Redis, LM Studio | [README](labs/lab03_rag/README.md), [notebook](labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb) |
| 4 | Stable Diffusion Text-to-Image | PyTorch, Diffusers, SD 1.5 | [README](labs/lab04_stable_diffusion/README.md), [notebook](labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb) |
| 5 | Visual Question Answering | LLaVA, Transformers | [README](labs/lab05_multimodal/README.md) |
| 6 | Агентний асистент | OpenAI Agents SDK, LM Studio, SQLite | [README](labs/lab06_agents/README.md) |
| 7 | Підсумковий RAG-асистент | Redis, MiniLM, Agents SDK | [README](labs/lab07_final_project/README.md) |

## Структура

- `labs/` — код, notebooks, тести та звіти окремих лабораторних;
- `scripts/` — відтворення артефактів і виконання notebook;
- `docs/` — спільні методичні та audit-документи;
- `shared/` — заготовка для коду, спільного для кількох лабораторних;
- `tests/` — репозиторні тести.

Датасети, ваги моделей, checkpoints, локальні vector databases, повні outputs, віртуальні середовища та IDE metadata не зберігаються в Git. Невеликі перевірені матеріали для відтворення ЛР3 і ЛР4 розміщені в `artifacts/portfolio/` відповідної лабораторної.

## Запуск

Залежності й зовнішні сервіси відрізняються між роботами. Команди встановлення наведено у README відповідної лабораторної. Notebook запускаються з кореня репозиторію, щоб імпорти `labs.<lab>.src` були доступні.

CPU-only replay виконує ЛР4 без CUDA та завантаження моделі. ЛР3 навмисно не має replay-режиму: її notebook перевіряється статично й тестами, а повне виконання потребує Redis, MiniLM і LM Studio.

```powershell
python scripts/execute_portfolio_notebooks.py --mode replay
```

Команда пропускає live-only notebook ЛР3 із явним статусом `SKIPPED` і виконує всі 21 code cells ЛР4 через `nbclient`. Повний експеримент потребує сервісів, моделей і обладнання, зазначених у README відповідної лабораторної.

## Результати

- notebook містять обчислення та проміжні таблиці;
- `report/` містить академічний звіт і матеріали для захисту;
- `outputs/` призначений для локальних результатів і не версіонується;
- `artifacts/portfolio/` містить обмежений набір результатів для replay та перегляду в GitHub;
- [docs/audits](docs/audits/) містить технічні перевірки структури й відтворюваності ЛР3–ЛР4.
