# Порівняння виконуваності notebook ЛР3 і ЛР4

Порівняння виконано між commit `2dd184e` та перебудованими notebook. Кількість code cells сама по собі не є показником якості: після декомпозиції важливо, які операції вони виконують. `До` оцінено за джерелом notebook у Git; `Після` — за виконаним replay, behavioral tests і outputs поточного kernel session.

| Показник | До | Після |
|---|---:|---:|
| Code cells | ЛР3: 16; ЛР4: 13 | ЛР3: 15; ЛР4: 12 |
| Experimental steps with actual computation | ЛР3: 7 із 15 обчислювальних STEP; ЛР4: 3 із 12 | ЛР3: 15 із 15; ЛР4: 12 із 12 |
| Pre-rendered-only steps | ЛР3: 4; ЛР4: 8 | 0 в обох notebook |
| Programmatically generated charts | ЛР3: 0; ЛР4: 0 | ЛР3: 3; ЛР4: 2 |
| Programmatically generated galleries | ЛР3: не застосовується; ЛР4: 0 | ЛР3: не застосовується; ЛР4: 7, включно з baseline і двома negative-prompt pairs |
| Recomputed evaluation metrics | ЛР3: готовий `baseline_vs_hybrid.csv`; ЛР4: готовий `performance_summary.csv` | ЛР3: Hit@5/MRR/Recall@5 для semantic і hybrid із row-level ranks; ЛР4: performance summary із current manifest |
| Meaningful intermediate tables | ЛР3: 9; ЛР4: 6 | ЛР3: 18; ЛР4: 18 |
| Explanation depth | ЛР3: 636 Markdown-слів; ЛР4: 626 | ЛР3: 779; ЛР4: 760; додано inputs, outputs, provenance, обмеження і статуси |
| Clean-clone execution | Replay існував, але основні серії залишалися artifact viewers | Replay виконує всі code cells через `nbclient`; окремо перевіряється в тимчасовому clone |

## Технічні докази

- ЛР3 викликає `eda_tables`, `plot_catalog_eda`, `dataframe_to_documents`, `evaluate_semantic_and_hybrid`, `answer_audit` і `plot_retrieval_dashboard`. PCA scatter створюється з `embedding_pca.csv`; notebook прямо вказує, що 384-dimensional inference у replay не запускався.
- ЛР3 full branch викликає `load_csv_documents`, `create_embeddings`, `embedding_audit`, `redis_preflight`, `build_store`, `run_retrieval`, `create_llm` і `generate_answer`.
- ЛР3 один раз перетинає evaluation boundary: router не отримує relevance labels, а `evaluate_id_lists` застосовується після формування ranked IDs.
- ЛР4 формує кожну серію через `experiment_groups`, показує `GenerationSpec`, викликає `resolve_group`, виводить records і будує gallery через `plot_generation_gallery`.
- ЛР4 replay resolver зіставляє controls, відкриває кожен PNG та звіряє SHA-256. Full resolver делегує `generate_image`; pipeline завантажується через `load_pipeline`.
- Behavioral tests відхиляють повернення до більшості viewer-only steps. Вони перевіряють helper calls, усі шість experiment groups, programmatic galleries, SHA-256 та відсутність `display(Image(filename=...)`.
- Обидва notebook зберігають execution counts, outputs, metadata `portfolio_execution.mode=replay` і не містять error outputs після реального запуску kernel.

Підрахунок `experimental steps with actual computation` виключає суто теоретичні STEP та вимагає transformation, validation, model/service call, metric computation або programmatic visualization. Відображення готового PNG чи готової aggregate table не зараховано.
