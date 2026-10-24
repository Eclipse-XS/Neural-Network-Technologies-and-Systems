# Фактичні режими виконання notebooks ЛР3 і ЛР4

Цей документ описує поточний HEAD. Старі підрахунки 16/16 для ЛР3 і 13/13 для ЛР4 стосувалися попередньої replay-структури та більше не є актуальними.

| Notebook | Code cells | Підтримуваний режим | Поточні збережені outputs | Offline CI |
|---|---:|---|---|---|
| ЛР3 `01_redis_csv_rag.ipynb` | 16 | `live-only` | 16 історичних live outputs, 0 error outputs | notebook не виконується; перевіряються nbformat, AST, contracts, routing і метрики |
| ЛР4 `01_text_to_image.ipynb` | 21 | `replay` або `full` | 21 replay outputs, 0 error outputs | усі 21 cells виконуються через `nbclient` |

## Чому режими різні

ЛР3 повинна демонструвати реальний `CSVLoader → MiniLM → Redis → LM Studio` pipeline. Offline replay цього notebook створював би хибне враження, що Redis retrieval і LLM generation були виконані. Тому `scripts/execute_portfolio_notebooks.py --mode replay` явно пропускає ЛР3. Повне виконання доступне лише як `--mode full --lab 3` за наявності CSV, Redis, MiniLM cache/download і LM Studio.

ЛР4 має чітко відокремлений replay: він не виконує diffusion inference, а перевіряє 13 історичних PNG, metadata, controls і SHA-256 та повторно будує таблиці й galleries. Статус кожного використаного зображення — `HISTORICAL_REPLAY`.

## Перевірка

```powershell
python scripts/execute_portfolio_notebooks.py --mode replay
```

Очікуваний результат поточного HEAD:

```text
labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb: SKIPPED mode=replay reason=live-only
labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb: EXECUTED mode=replay code_cells=21 errors=0
```

Builders створюють структуру notebooks, але не мають права підробляти execution counts. `nbclient` встановлює execution counts і outputs лише під час фактичного виконання.
