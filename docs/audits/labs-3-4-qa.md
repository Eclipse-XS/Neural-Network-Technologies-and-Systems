# Технічний аудит лабораторних робіт 3 і 4

Документ фіксує перевірені дані, виправлення та межі відтворення. Він не є оцінкою роботи в балах.

## ЛР3: дані та retrieval

- Корпус: 3 CSV, 185 рядків, 24 вихідні колонки.
- Категорії: 100 суконь, 51 пара спортивного взуття, 34 пари сережок.
- Виявлено 32 пропущені знижки, 22 записи без спостережуваного рейтингу та 29 конфліктів price/discount formula.
- 17 із 185 serialized records перевищують ліміт MiniLM у 256 word pieces.
- Semantic baseline на семи labeled queries: Hit@5 `0,8571`, MRR `0,6905`, mean Recall@5 `0,5734`.
- Hybrid router: Hit@5 `1,0000`, MRR `1,0000`, mean Recall@5 `0,8829`.
- Holdout із п'яти labeled queries: Hit@5 `1,0000`, MRR `1,0000`, Recall@5 `0,9111`.

Початкова реалізація hybrid retrieval помилково передавала evaluation object до inference helper. Через це старі hybrid MRR `0,9048` і Recall@5 `0,8194` були невалідними. Поточний `src/hybrid.py` приймає question, catalog, optional semantic candidates і `k`; labels використовує лише `evaluate_id_lists()` після retrieval. Тести змінюють query ID, expected fields і relevance labels та перевіряють незмінність ranked results.

Обмеження: intent grammar вузька й англомовна, labels є corpus-derived proxies, Redis latency має по одному спостереженню, live Redis і LM Studio під час replay не викликаються. Покращення retrieval не доводить покращення відповідей LLM.

## ЛР4: генерації та metadata

- 13 унікальних PNG, 512×512.
- Model revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- Seeds 42, 123, 777, 2026; steps 20, 30, 50; CFG 5, 7,5, 10.
- Дві negative-prompt пари та три style prompts.
- Scheduler: PNDM; FP16, model CPU offload, PyTorch SDPA.
- Для кожного PNG перевіряються параметри, dimensions і SHA-256.

Euler і DDIM не запускались. CLIPScore, FID, repeated warmed timing і LoRA training не виконувались. 30-step baseline тривав 18,88 с, але був першим запуском; наступні 30-step observations становили приблизно 9–10 с. Ці значення показують warm-up sensitivity, а не стабільну продуктивність.

## Notebook і replay

Notebook generation відокремлено від execution. `scripts/execute_portfolio_notebooks.py` запускає реальний Python kernel через `nbclient`, зберігає outputs і завершується помилкою за наявності error cell або відсутнього execution count.

Replay використовує committed row-level evidence:

- ЛР3 повторно обчислює EDA, PCA visualization, routing, Hit@5/MRR/Recall@5, citation checks і dashboard;
- ЛР4 звіряє 13 PNG із manifest та SHA-256, після чого будує galleries і performance plots;
- historical Redis results, LLM answers, GPU metadata і PNG не називаються новим inference.

Full mode зберігає виклики MiniLM, Redis, LM Studio та Stable Diffusion pipeline, але потребує відповідних сервісів, моделей і GPU.

## Перевірки

Актуальні перевірки виконуються GitHub Actions і локально:

```powershell
python -m compileall -q labs/lab03_rag/src labs/lab04_stable_diffusion/src
python scripts/execute_portfolio_notebooks.py --mode replay
python -m pytest labs/lab03_rag/tests -q
python -m pytest labs/lab04_stable_diffusion/tests -q
```

Behavioral notebook tests перевіряють helper calls, програмні galleries/charts, evaluation boundary та відсутність experiment steps, що лише показують готовий PNG. Live service integration перевіряється окремо й не вважається виконаною під час offline replay.
