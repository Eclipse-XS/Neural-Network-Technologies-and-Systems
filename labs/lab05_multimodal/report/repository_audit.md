# Аудит початкового стану Lab 5

До overhaul код уже мав коректний pipeline, контрольовану матрицю 19/17, ручні
оцінки й виконаний notebook. Основні дефекти були іншими:

- `outputs/` ігнорувався, тому новий clone не мав JSON-доказів, model config і metadata;
- практична частина notebook переважно показувала готові таблиці через `display()`;
- архітектура була описана надто стисло й без наочної двогілкової схеми;
- не було окремих temperature, error-taxonomy та input-inspection візуалізацій;
- Run All залежав від локального runtime cache і викликав orchestration inference;
- console output не розрізняв inference, cache hit і повторне використання;
- звіт посилався на ignored PNG та metadata, відсутні після clone.

Історичний локальний manifest перевірено: 19 рядків, 17 унікальних run ID, один image
SHA, один model revision. З нього без зміни відповідей сформовано компактний
`evidence/verified_runs.json`; environment і model config перенесено окремо. Інші
лабораторні та сторонні локальні файли не змінювалися: робота ведеться в окремому
worktree від `origin/main`.
