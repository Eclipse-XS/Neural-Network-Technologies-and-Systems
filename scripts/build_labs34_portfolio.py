"""Build deterministic public artifacts and educational notebooks for Labs 3 and 4."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.lab03_rag.src.analysis import catalog_summary, evaluate_id_lists, load_catalog, load_manual_review
from labs.lab03_rag.src.hybrid import route_and_retrieve

LAB3 = ROOT / "labs/lab03_rag"
LAB4 = ROOT / "labs/lab04_stable_diffusion"
P3 = LAB3 / "artifacts/portfolio"
P4 = LAB4 / "artifacts/portfolio"
STYLE = {"figure.facecolor": "white", "axes.facecolor": "#F7F9FC", "axes.grid": True,
         "grid.alpha": .25, "axes.titleweight": "bold", "font.size": 10}
plt.rcParams.update(STYLE)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def lab3_artifacts():
    frame = load_catalog(LAB3 / "data/raw/csv")
    P3.mkdir(parents=True, exist_ok=True)
    source_columns = [column for column in frame.columns if not column.endswith("_numeric") and column not in {
        "rating_observed", "description_chars", "price_formula_delta_inr", "price_conflict"}]
    frame[source_columns].to_csv(P3 / "catalog_replay.csv", index=False, encoding="utf-8-sig")
    # Publish the small, verified evidence required by CPU-only portfolio replay.
    shutil.copy2(LAB3 / "outputs/embedding_audit.json", P3 / "embedding_audit.json")
    shutil.copy2(LAB3 / "outputs/index_info.json", P3 / "index_info.json")
    shutil.copy2(LAB3 / "outputs/retrieval/results.csv", P3 / "semantic_results.csv")
    shutil.copy2(LAB3 / "outputs/rag/final_answers.csv", P3 / "historical_semantic_answers.csv")
    (P3 / "dataset_summary.json").write_text(
        json.dumps(catalog_summary(frame), ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    frame.category.value_counts().plot.bar(ax=axes[0, 0], color="#3568A8")
    axes[0, 0].set(title="Products by category", xlabel="Category", ylabel="Products, count")
    axes[0, 1].hist(frame.final_price_numeric, bins=24, color="#2A9D8F")
    axes[0, 1].set(title="Listed final price", xlabel="Price, INR", ylabel="Products, count")
    observed = frame.loc[frame.rating_observed, "rating_numeric"]
    axes[0, 2].hist(observed, bins=np.arange(0, 5.25, .25), color="#E9C46A")
    axes[0, 2].set(title="Observed ratings only", xlabel="Rating, stars", ylabel="Products, count")
    missing = frame[["discount", "seller_name", "videos", "variations", "what_customers_said"]].eq("").sum()
    missing.sort_values().plot.barh(ax=axes[1, 0], color="#8D99AE")
    axes[1, 0].set(title="Missing selected fields", xlabel="Missing values, count", ylabel="Field")
    axes[1, 1].hist(frame.discount_numeric.dropna(), bins=20, color="#F4A261")
    axes[1, 1].set(title="Stated discounts", xlabel="Discount, %", ylabel="Products, count")
    frame.groupby("category").description_chars.plot.box(ax=axes[1, 2])
    axes[1, 2].set(title="Description length", xlabel="Category", ylabel="Characters")
    savefig(P3 / "eda_dashboard.png")

    conflicts = frame.loc[frame.price_conflict, ["product_id", "title", "category", "initial_price_numeric",
                                                  "discount_numeric", "final_price_numeric", "price_formula_delta_inr",
                                                  "source_file", "row_id"]].copy()
    conflicts.to_csv(P3 / "price_conflicts.csv", index=False, encoding="utf-8-sig")
    largest = conflicts.nlargest(12, "price_formula_delta_inr").sort_values("price_formula_delta_inr")
    plt.figure(figsize=(9, 5))
    plt.barh(largest.product_id.astype(str), largest.price_formula_delta_inr, color="#D1495B")
    plt.title("Largest price–discount inconsistencies")
    plt.xlabel("Absolute formula deviation, INR")
    plt.ylabel("Product ID")
    savefig(P3 / "price_conflicts.png")

    summary = pd.read_csv(LAB3 / "outputs/retrieval/summary.csv")
    summary.to_csv(P3 / "baseline_retrieval.csv", index=False, encoding="utf-8-sig")
    queries_payload = json.loads((LAB3 / "queries.json").read_text(encoding="utf-8"))
    queries = queries_payload["queries"]
    baseline_ids = (pd.read_csv(LAB3 / "outputs/retrieval/results.csv")
                    .sort_values(["query_id", "rank"]).groupby("query_id").product_id
                    .apply(lambda s: [str(x) for x in s]).to_dict())
    hybrid_ids = {k: list(v) for k, v in baseline_ids.items()}
    structured_rows, routing_rows = [], []
    for query in queries:
        inference = route_and_retrieve(frame, query["question"], baseline_ids.get(query["id"], []), 5)
        exact = inference.rows
        routing_rows.append({"query_id": query["id"], "question": query["question"],
                             "strategy": inference.intent.strategy,
                             "conditions": json.dumps(inference.diagnostics["conditions"], ensure_ascii=False),
                             "returned_ids": json.dumps(inference.product_ids),
                             "diagnostics": json.dumps(inference.diagnostics, ensure_ascii=False)})
        if inference.intent.strategy == "unsupported":
            hybrid_ids[query["id"]] = []
        else:
            ids = inference.product_ids
            hybrid_ids[query["id"]] = ids + [x for x in baseline_ids.get(query["id"], []) if x not in ids]
            structured_rows.extend({"query_id": query["id"], **row} for row in exact.to_dict("records"))
    base_eval = evaluate_id_lists(queries, baseline_ids, 5).assign(configuration="semantic_baseline")
    hybrid_eval = evaluate_id_lists(queries, hybrid_ids, 5).assign(configuration="hybrid_router")
    comparison = pd.concat([base_eval, hybrid_eval], ignore_index=True)
    comparison.to_csv(P3 / "baseline_vs_hybrid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(structured_rows).to_csv(P3 / "structured_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(routing_rows).to_csv(P3 / "routing_diagnostics.csv", index=False, encoding="utf-8-sig")
    holdout = json.loads((LAB3 / "holdout_queries.json").read_text(encoding="utf-8"))["queries"]
    holdout_ids, holdout_routes = {}, []
    for query in holdout:
        inference = route_and_retrieve(frame, query["question"], k=5)
        holdout_ids[query["id"]] = inference.product_ids
        holdout_routes.append({"query_id": query["id"], "question": query["question"],
                               "strategy": inference.intent.strategy,
                               "conditions": json.dumps(inference.diagnostics["conditions"], ensure_ascii=False),
                               "returned_ids": json.dumps(inference.product_ids),
                               "diagnostics": json.dumps(inference.diagnostics, ensure_ascii=False)})
    holdout_eval = evaluate_id_lists(holdout, holdout_ids, 5)
    holdout_eval.to_csv(P3 / "holdout_evaluation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(holdout_routes).to_csv(P3 / "holdout_routing.csv", index=False, encoding="utf-8-sig")
    review = load_manual_review(LAB3 / "outputs/rag/manual_review.json")
    review.to_csv(P3 / "error_taxonomy.csv", index=False, encoding="utf-8-sig")

    plot = comparison[comparison.query_id != "q08"].pivot(index="query_id", columns="configuration", values="recall_at_k")
    plot.plot.bar(figsize=(9, 5), color=["#577590", "#43AA8B"])
    plt.title("Recall@5: semantic baseline vs hybrid router")
    plt.xlabel("Evaluation query")
    plt.ylabel("Recall@5")
    plt.ylim(0, 1.05)
    plt.legend(title="Configuration")
    savefig(P3 / "baseline_vs_hybrid.png")

    results = pd.read_csv(LAB3 / "outputs/retrieval/results.csv")
    q01 = results[results.query_id == "q01"].sort_values("rank")
    plt.figure(figsize=(9, 4))
    plt.barh(q01["product_id"].astype(str), q01.cosine_distance, color="#277DA1")
    plt.gca().invert_yaxis()
    plt.title("Q01 semantic retrieval Top-5")
    plt.xlabel("Cosine distance; lower is closer")
    plt.ylabel("Product ID")
    savefig(P3 / "q01_top5_distances.png")

    # PCA uses the persisted, verified embedding vectors without contacting a model server.
    from labs.lab03_rag.src.data import load_csv_documents
    from labs.lab03_rag.src.documents import build_documents
    from labs.lab03_rag.src.config import MODEL_NAME, MODEL_REVISION
    import hashlib
    documents = build_documents(load_csv_documents(LAB3 / "data/raw/csv"))
    vectors = []
    for doc in documents:
        key = hashlib.sha256((MODEL_NAME + MODEL_REVISION + "cpu:normalized:v1" + doc.page_content).encode()).hexdigest()
        vectors.append(np.load(LAB3 / "outputs/embeddings" / f"{key}.npy", allow_pickle=False))
    matrix = np.asarray(vectors)
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    pca = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1],
                        "category": [d.metadata["category"] for d in documents],
                        "product_id": [d.metadata["product_id"] for d in documents]})
    pca.to_csv(P3 / "embedding_pca.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 6))
    for category, group in pca.groupby("category"):
        plt.scatter(group.pc1, group.pc2, label=category, alpha=.75, s=30)
    plt.title("PCA projection of 384-dimensional MiniLM embeddings")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(title="Category")
    savefig(P3 / "embedding_pca.png")


def diagram(path: Path, labels: list[str], title: str):
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    xs = np.linspace(.05, .95, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=.45", "facecolor": "#EAF2F8", "edgecolor": "#3568A8"}, fontsize=10)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b-.035, .5), xytext=(a+.035, .5), xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "->", "color": "#444"})
    ax.set_title(title, pad=18)
    savefig(path)


def lab4_artifacts():
    P4.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LAB4 / "outputs/metadata/environment.json", P4 / "historical_environment.json")
    for source in (LAB4 / "outputs/figures").glob("*.png"):
        shutil.copy2(source, P4 / source.name)
    baseline = next((LAB4 / "outputs/images/baseline").glob("*.png"))
    shutil.copy2(baseline, P4 / "baseline.png")
    manifest = pd.read_csv(LAB4 / "outputs/metadata/generations.csv", encoding="utf-8-sig")
    public_paths = []
    for row in manifest.itertuples():
        source = LAB4 / row.image_path
        target = P4 / "images" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        public_paths.append(target.relative_to(LAB4).as_posix())
    manifest["public_image_path"] = public_paths
    manifest.to_csv(P4 / "generations.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(LAB4 / "outputs/metadata/visual_observations.json", P4 / "visual_observations.json")
    from labs.lab04_stable_diffusion.src.evaluation import performance_summary, quality_rubric, scheduler_experiment_plan
    performance_summary(manifest).to_csv(P4 / "performance_summary.csv", index=False, encoding="utf-8-sig")
    scheduler_experiment_plan().to_csv(P4 / "scheduler_status.csv", index=False, encoding="utf-8-sig")
    quality_rubric(LAB4 / "outputs/metadata/visual_observations.json").to_csv(
        P4 / "quality_rubric.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    colors = manifest.experiment.map({"baseline": "#264653", "seed": "#2A9D8F", "steps": "#E9C46A",
                                      "cfg": "#F4A261", "negative_prompt": "#E76F51", "styles": "#577590"})
    plt.scatter(manifest.num_inference_steps, manifest.generation_time_seconds, c=colors, s=65)
    for row in manifest.itertuples():
        plt.annotate(row.case_id, (row.num_inference_steps, row.generation_time_seconds), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.title("Verified inference runtime observations")
    plt.xlabel("Inference steps")
    plt.ylabel("Wall-clock time, s")
    savefig(P4 / "performance_dashboard.png")
    diagram(P4 / "stable_diffusion_architecture.png",
            ["Prompt", "Tokenizer", "CLIP encoder", "Text embeddings", "U-Net + scheduler", "Latents", "VAE decoder", "RGB image"],
            "Stable Diffusion 1.5 inference architecture")
    diagram(P3 / "rag_architecture.png",
            ["CSV", "Documents", "MiniLM embeddings", "Redis index", "Retriever", "Grounded prompt", "Local LLM", "Answer"],
            "RAG architecture used in Laboratory 3")
    diagram(P3 / "evaluation_integrity_flow.png",
            ["User question", "Intent parser", "Strategy", "Retrieved IDs", "Evaluation boundary", "Ground-truth labels", "Metrics"],
            "Leakage-free inference and independent evaluation")


def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())


def lab3_notebook():
    cells = [md("""# Лабораторна робота №3: RAG із Redis Vector Similarity та CSV

Варіант 3. Цей notebook є самодостатнім навчальним маршрутом від сирих CSV до перевірки retrieval і grounded-відповідей. Фактичний висновок: vector similarity корисний для семантичного пошуку, але точні ID, глобальні числові умови та арифметика мають оброблятися детерміновано."""),
    md("""## STEP 01 — Вступ до RAG

LLM генерує текст із параметричних знань, але не має гарантованого доступу до локального каталогу. RAG складається з retrieval, augmentation і generation. Це не fine-tuning: ваги моделі не змінюються; змінюється контекст запиту. Архітектура нижче відповідає фактичній реалізації."""),
    code("""from pathlib import Path
import sys, json, platform, os
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p/'labs/lab03_rag').is_dir())
sys.path.insert(0, str(ROOT))
LAB = ROOT/'labs/lab03_rag'
MODE = os.getenv('PORTFOLIO_MODE', 'replay').lower()
if MODE not in {'replay', 'full'}: raise ValueError('PORTFOLIO_MODE must be replay or full')
EVIDENCE = LAB/'artifacts/portfolio' if MODE == 'replay' else LAB/'outputs'
print('REPLAYED FROM VERIFIED EVIDENCE' if MODE == 'replay' else 'FULL EXPERIMENTAL EXECUTION')
display(Image(filename=str(LAB/'artifacts/portfolio/rag_architecture.png')))"""),
    md("""## STEP 02 — Перевірка середовища

Таблиця не припускає доступність сервісів. Статус Redis і LM Studio перевіряється окремо; відсутність сервісу не перетворюється на PASS."""),
    code("""from labs.lab03_rag.src.config import Settings, MODEL_NAME
settings = Settings()
environment = pd.DataFrame([
 {'component':'Python','value':sys.version.split()[0]}, {'component':'OS','value':platform.platform()},
 {'component':'Embedding model','value':MODEL_NAME}, {'component':'Dataset','value':str(settings.data_dir)},
 {'component':'Cache','value':str(settings.output_dir/'embeddings')}, {'component':'Output','value':str(settings.output_dir)}])
display(environment)"""),
    md("""## STEP 03 — Завантаження даних

Три вихідні CSV читаються без зміни значень. `final_price` додатково перетворюється на числову похідну для аналізу. Нульовий rating із нульовою кількістю оцінок позначається як «оцінка не спостерігалася», а не як справжня оцінка 0."""),
    code("""from labs.lab03_rag.src.analysis import load_catalog, load_replay_catalog, catalog_summary
catalog = (load_replay_catalog(LAB/'artifacts/portfolio/catalog_replay.csv') if MODE == 'replay'
           else load_catalog(settings.data_dir))
display(pd.DataFrame([catalog_summary(catalog)]))
display(catalog.groupby('category').head(2)[['source_file','row_id','product_id','title','product_description','final_price','discount','rating']])
assert len(catalog) == 185 and catalog.product_id.is_unique"""),
    md("""## STEP 04 — Exploratory Data Analysis

Графіки показують обсяг категорій, ціни, лише спостережувані рейтинги, пропуски, знижки й довжини описів. Окремий графік не доводить причинності; він виявляє структуру й проблеми даних."""),
    code("display(Image(filename=str(LAB/'artifacts/portfolio/eda_dashboard.png')))"),
    code("""conflicts = pd.read_csv(LAB/'artifacts/portfolio/price_conflicts.csv')
display(conflicts.sort_values('price_formula_delta_inr', ascending=False).head(10))
display(Image(filename=str(LAB/'artifacts/portfolio/price_conflicts.png')))
assert len(conflicts) == int(catalog.price_conflict.sum())"""),
    md("""## STEP 05 — CSVLoader і побудова Document

Один товар дорівнює одному `Document`, бо рядок є атомарною бізнес-сутністю. Довільний character chunking міг би розділити ціну, ID та опис. `source_file` і `row_id` забезпечують provenance."""),
    code("""from labs.lab03_rag.src.data import load_csv_documents
from labs.lab03_rag.src.documents import build_documents, dataframe_to_documents
if MODE == 'replay':
    loaded = None
    documents = dataframe_to_documents(catalog)
    loader_preview = 'REPLAY: original CSVLoader output is not regenerated; the frozen row is adapted with the production Document contract.'
else:
    loaded = load_csv_documents(settings.data_dir)
    documents = build_documents(loaded)
    loader_preview = loaded[0].page_content[:800]
display(pd.DataFrame([{'stage':'raw CSV row','preview':catalog.iloc[0].to_dict()},
                      {'stage':'CSVLoader page_content','preview':loader_preview},
                      {'stage':'serialized Document','preview':documents[0].page_content[:800]},
                      {'stage':'metadata','preview':{k:v for k,v in documents[0].metadata.items() if k!='fields_json'}}]))
assert len(documents) == 185 == len({d.metadata['record_id'] for d in documents})"""),
    md("""## STEP 06 — Embedding model

`all-MiniLM-L6-v2` відображає текст у 384-вимірний unit-normalized вектор. Cosine similarity дорівнює $\\frac{a\\cdot b}{\\|a\\|\\|b\\|}$; для normalized vectors це dot product. Модель має межу 256 word pieces: довші записи обрізаються, тому truncation треба вимірювати, а не приховувати."""),
    code("""path = LAB/'artifacts/portfolio/embedding_audit.json' if MODE == 'replay' else settings.output_dir/'embedding_audit.json'
audit = json.loads(path.read_text(encoding='utf-8'))
display(pd.DataFrame([{k:v for k,v in audit.items() if k!='token_lengths'}]))
lengths = pd.Series(audit['token_lengths'], name='word_pieces')
display(lengths.describe().to_frame().T)
assert audit['dimension'] == 384 and audit['truncated_rows'] == int((lengths > audit['max_seq_length']).sum())"""),
    md("""## STEP 07 — Візуалізація векторного простору

Кожна точка — один товар. PCA стискає 384 виміри до двох і зберігає лише частину дисперсії; близькість на площині не є доказом retrieval quality."""),
    code("display(Image(filename=str(LAB/'artifacts/portfolio/embedding_pca.png')))"),
    md("""## STEP 08 — Redis vector index

Індекс: `FLOAT32`, 384 виміри, `COSINE`, `FLAT`. FLAT виправданий малим корпусом. Реальний `FT.INFO` збережено як evidence; live-перевірка виконується лише при запущеному Redis Stack."""),
    code("""path = LAB/'artifacts/portfolio/index_info.json' if MODE == 'replay' else settings.output_dir/'index_info.json'
index_info = json.loads(path.read_text(encoding='utf-8'))
display(pd.DataFrame([{'index':index_info['index'],'documents':index_info['document_count'],
                       'algorithm':index_info['vector']['algorithm'],'dimension':index_info['vector']['dim'],
                       'metric':index_info['vector']['distance_metric']}]))
assert index_info['document_count'] == 185"""),
    md("""## STEP 09 — Semantic retrieval

Для cosine distance менше значення означає більшу близькість. Rank сам по собі не доводить фактичну релевантність; його треба порівнювати з ground truth."""),
    code("""path = LAB/'artifacts/portfolio/semantic_results.csv' if MODE == 'replay' else settings.output_dir/'retrieval/results.csv'
retrieval = pd.read_csv(path)
display(retrieval[retrieval.query_id=='q01'][['rank','product_id','category','cosine_distance','source_file','row_id','preview']])
display(Image(filename=str(LAB/'artifacts/portfolio/q01_top5_distances.png')))"""),
    md("""## STEP 10 — Оцінювання retrieval

$Hit@k$ відповідає лише на питання, чи знайдено хоча б один релевантний елемент. $Recall@k=|R_k\\cap G|/|G|$. $MRR=1/r$ для рангу першого релевантного результату. Якщо релевантних товарів 9, то за `k=5` максимальний Recall@5 дорівнює 5/9, а не 1."""),
    code("""comparison = pd.read_csv(LAB/'artifacts/portfolio/baseline_vs_hybrid.csv')
display(comparison)
display(Image(filename=str(LAB/'artifacts/portfolio/baseline_vs_hybrid.png')))
display(Image(filename=str(LAB/'artifacts/portfolio/evaluation_integrity_flow.png')))
assert comparison.query_id.nunique() == 8"""),
    md("""## STEP 11 — Побудова grounded prompt

System instruction забороняє непідтверджені твердження; retrieved records відокремлені як untrusted data; цитата має формат `[source.csv:row]`."""),
    code("""from labs.lab03_rag.src.rag import build_prompt
from labs.lab03_rag.src.hybrid import route_and_retrieve
from labs.lab03_rag.src.documents import dataframe_to_documents
q = next(q for q in json.loads((LAB/'queries.json').read_text(encoding='utf-8'))['queries'] if q['id']=='q04')
chain = route_and_retrieve(catalog, q['question'], k=5)
q_documents = dataframe_to_documents(chain.rows)
q_prompt = build_prompt(q['question'], q_documents)
display(pd.DataFrame([{'question':q['question'], 'strategy':chain.intent.strategy,
                       'product_ids':chain.product_ids, 'record_ids':[d.metadata['record_id'] for d in q_documents]}]))
display(pd.DataFrame([{'block':'system','content':q_prompt[0].content},
                      {'block':'user + exact retrieved context','content':q_prompt[1].content}]))
print('Prompt constructed from the same q04 hybrid retrieval. LLM NOT EXECUTED in replay mode.')"""),
    md("""## STEP 12 — Оцінювання відповіді LLM

Retrieval correctness, field correctness, citation validity, numerical correctness, unsupported claims і abstention оцінюються окремо. Простий збіг числа не доводить правильну атрибуцію."""),
    code("""path = LAB/'artifacts/portfolio/historical_semantic_answers.csv' if MODE == 'replay' else settings.output_dir/'rag/final_answers.csv'
answers = pd.read_csv(path)
reviews = pd.read_csv(LAB/'artifacts/portfolio/error_taxonomy.csv')
print('HISTORICAL SEMANTIC BASELINE; NOT A NEW HYBRID LLM RUN')
display(answers[['query_id','status','answer','manual_review']].merge(reviews,on='query_id'))"""),
    md("""## STEP 13 — Error analysis

Вісім кейсів демонструють різні failure modes: суперечливі поля, семантику пропусків, помилку одиниць, неповне multi-row retrieval, хибне використання similarity для глобального мінімуму, арифметичну помилку, unsupported claim і правильну відмову."""),
    code("display(reviews[['query_id','failure_category','review']])"),
    md("""## STEP 14 — Improvement experiments

Hybrid router отримує лише текст запиту, каталог і semantic candidates. Він витягує Product ID регулярним parser, розпізнає category/attribute/numeric intent, виконує min/max aggregation або повертає unsupported. `relevant_product_ids`, `expected_fields` і ID тестового кейсу не входять до inference API. Ground truth з'являється лише після отримання ranked IDs."""),
    code("""routing = pd.read_csv(LAB/'artifacts/portfolio/routing_diagnostics.csv')
display(routing[['query_id','question','strategy','conditions','returned_ids']])
structured = pd.read_csv(LAB/'artifacts/portfolio/structured_results.csv')
display(structured[['query_id','product_id','title','category','final_price_numeric','discount_numeric','rating_numeric','source_file','row_id']])"""),
    md("""## STEP 15 — Підсумкова evaluation dashboard

Known set використовує незмінені вісім запитів. Окремий holdout містить нові формулювання, створені після фіксації правил router; його малий розмір не дає статистичної значущості. Latency baseline є одиничним спостереженням і не є чистим Redis benchmark."""),
    code("""dashboard = comparison.groupby('configuration').agg(
    scored=('hit_at_k','count'), hit_at_5=('hit_at_k','mean'), mrr=('mrr','mean'), recall_at_5=('recall_at_k','mean'))
display(dashboard)
holdout = pd.read_csv(LAB/'artifacts/portfolio/holdout_evaluation.csv')
display(holdout)
display(holdout[holdout.expected_count > 0][['hit_at_k','mrr','recall_at_k','coverage']].mean().to_frame('holdout_mean'))
display(reviews.failure_category.value_counts().rename_axis('failure_category').to_frame('queries'))"""),
    md("""## STEP 16 — Висновки та обмеження

Redis забезпечує швидкий vector search і metadata filters, але cosine similarity не виконує SQL-агрегації й не гарантує multi-row coverage. Leakage-free router покращує exact, attribute і numeric tasks, але його правила навмисно обмежені. Для production потрібні query routing, deterministic numeric tools, citation validation, human-labeled relevance, повторні latency runs, access control, monitoring і versioned data contracts. LM Studio недоступний під час цього rebuild, тому старі підтверджені відповіді збережено; нові LLM твердження не вигадані.""")]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}})
    nbf.write(nb, LAB3 / "notebooks/01_redis_csv_rag.ipynb")


def lab4_notebook():
    cells = [md("""# Лабораторна робота №4: Stable Diffusion Text-to-Image

Варіант 1. Notebook пояснює latent diffusion і документує 13 реальних SD 1.5 генерацій. Фактичний висновок: seed, steps, CFG і prompt змінюють результат, але 13 одиничних зразків не дають статистичного доказу абсолютної якості."""),
    md("""## STEP 01 — Вступ до Generative AI

VAE стискає та реконструює дані, GAN навчає generator проти discriminator, diffusion навчається передбачати шум. Stable Diffusion виконує denoising у latent space, а text-to-image conditioning задає напрям через CLIP embeddings."""),
    md("""## STEP 02 — Forward і reverse diffusion

Forward training process: $x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$, де $\\epsilon\\sim\\mathcal N(0,I)$. Мета noise prediction: $\\mathbb E\\|\\epsilon-\\epsilon_\\theta(x_t,t,c)\\|_2^2$. Під час inference не додається шум до готової фотографії: sampling починається з latent noise і виконує reverse denoising."""),
    md("""## STEP 03 — Архітектура Stable Diffusion

Tokenizer готує token IDs; CLIP text encoder створює conditioning; U-Net прогнозує шум; cross-attention пов’язує text і latent features; scheduler задає sampling update; VAE decoder перетворює latent у RGB. Classifier-Free Guidance: $\\hat\\epsilon=\\epsilon_u+w(\\epsilon_c-\\epsilon_u)$."""),
    code("""from pathlib import Path
import sys, json, platform, os
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p/'labs/lab04_stable_diffusion').is_dir())
sys.path.insert(0, str(ROOT)); LAB=ROOT/'labs/lab04_stable_diffusion'
MODE=os.getenv('PORTFOLIO_MODE','replay').lower()
if MODE not in {'replay','full'}: raise ValueError('PORTFOLIO_MODE must be replay or full')
print('REPLAYED FROM VERIFIED EVIDENCE' if MODE == 'replay' else 'FULL EXPERIMENTAL EXECUTION')
display(Image(filename=str(LAB/'artifacts/portfolio/stable_diffusion_architecture.png')))"""),
    md("""## STEP 04 — Середовище і модель

SD 1.5 обрано через 4 GB VRAM; SDXL суттєво важчий. FP16, model CPU offload і PyTorch SDPA зменшують тиск на VRAM. Таблиця нижче походить із збереженого виконання, а не з поточного хоста."""),
    code("""path=LAB/'artifacts/portfolio/historical_environment.json' if MODE=='replay' else LAB/'outputs/metadata/environment.json'
env=json.loads(path.read_text(encoding='utf-8'))
print('Historical environment evidence; not the current replay host.')
display(pd.DataFrame([{'field':k,'value':v} for k,v in env.items() if k!='versions']))"""),
    md("""## STEP 05 — Завантаження pipeline

`StableDiffusionPipeline.from_pretrained` фіксує model ID і revision. Після завантаження записуються pipeline class, scheduler, dtype, attention processors, load time й memory state. Notebook повторно використовує лише артефакти, чия identity та SHA-256 збігаються."""),
    code("""manifest=pd.read_csv(LAB/'artifacts/portfolio/generations.csv',encoding='utf-8-sig')
display(manifest[['model_id','revision','pipeline_class','scheduler','dtype','device_strategy','model_load_seconds']].drop_duplicates())
assert len(manifest)==13 and manifest.artifact_id.nunique()==13"""),
    md("""## STEP 06 — Baseline generation

Контроль: seed 42, 30 steps, CFG 7.5, PNDM, 512×512. Один результат підтверджує працездатність pipeline, але не узагальнену якість."""),
    code("""baseline=manifest[manifest.case_id=='baseline'].iloc[0]
display(baseline[['prompt','negative_prompt','seed','num_inference_steps','guidance_scale','scheduler','generation_time_seconds']].to_frame('value'))
display(Image(filename=str(LAB/'artifacts/portfolio/baseline.png'),width=512))"""),
    md("## STEP 07 — Seed experiments\n\nSeed 42, 123, 777 і 2026 змінюють початковий latent noise. Інші controls сталі; тому різницю можна приписувати seed у межах цієї серії."),
    code("display(Image(filename=str(LAB/'artifacts/portfolio/seed_comparison.png')))"),
    md("## STEP 08 — Inference steps\n\n20, 30 і 50 steps порівнюються з фіксованими prompt, seed та CFG. Runtime значення одиничні; baseline на 30 steps включає warm-up effect, тому статистичного ranking швидкості немає."),
    code("""display(Image(filename=str(LAB/'artifacts/portfolio/steps_comparison.png')))
display(Image(filename=str(LAB/'artifacts/portfolio/runtime_by_steps.png')))"""),
    md("## STEP 09 — Classifier-Free Guidance\n\nCFG 5, 7.5 і 10 змінює вагу conditional prediction. Вище значення не гарантує природнішого або кращого зображення й може посилювати saturation чи geometry artifacts."),
    code("display(Image(filename=str(LAB/'artifacts/portfolio/cfg_comparison.png')))"),
    md("## STEP 10 — Negative prompt\n\nДві контрольовані пари відрізняються лише negative prompt. Він змінює всю denoising trajectory, тому це не локальний фільтр і не гарантія видалення тексту чи артефактів."),
    code("""display(Image(filename=str(LAB/'artifacts/portfolio/negative_prompt_seed42_comparison.png')))
display(Image(filename=str(LAB/'artifacts/portfolio/negative_prompt_seed123_comparison.png')))"""),
    md("## STEP 11 — Style через prompting\n\nWatercolor, isometric і photorealistic — prompt conditioning, не справжній neural style transfer і не fine-tuning. Однаковий seed за різних prompt не фіксує геометрію."),
    code("display(Image(filename=str(LAB/'artifacts/portfolio/styles_comparison.png')))"),
    md("## STEP 12 — Scheduler analysis\n\nPNDM підтверджений sidecar-метаданими. Euler і DDIM не виконані: немає перевірених зображень або timings. Однакова кількість steps не означає однакову sampling trajectory."),
    code("display(pd.read_csv(LAB/'artifacts/portfolio/scheduler_status.csv'))"),
    md("## STEP 13 — Оцінювання якості\n\nCLIPScore не обчислювався, тому його немає. Замість вигаданої метрики використано ручний rubric: prompt/style adherence, artifact severity, composition consistency та documented limitations. Це якісний аналіз, не абсолютний score."),
    code("""rubric=pd.read_csv(LAB/'artifacts/portfolio/quality_rubric.csv')
display(rubric)"""),
    md("## STEP 14 — Performance profiling\n\nWall-clock runtime і peak CUDA memory записувалися під час inference. Без повторів середнє та стандартне відхилення не інтерпретуються як стабільні estimates. CPU offload trade-off: менше VRAM, більше transfer overhead."),
    code("""display(pd.read_csv(LAB/'artifacts/portfolio/performance_summary.csv'))
display(Image(filename=str(LAB/'artifacts/portfolio/performance_dashboard.png')))"""),
    md("## STEP 15 — Experiment comparison dashboard\n\nManifest є машинозчитуваним зв’язком між prompt, controls, runtime, image path і SHA-256."),
    code("display(manifest[['case_id','experiment','seed','num_inference_steps','guidance_scale','scheduler','generation_time_seconds','peak_cuda_memory_mb','image_path']])"),
    md("""## STEP 16 — LoRA

LoRA заморожує основні ваги $W$ і навчає low-rank update $\\Delta W=BA$, де rank значно менший за розмір $W$. Це дешевше за full fine-tuning і може адаптувати стиль або концепт. У цій роботі LoRA не навчалася й не підключалася; жоден результат не позначено як LoRA-generated."""),
    md("""## STEP 17 — Висновки та обмеження

Seed найбільше змінює композицію; steps створюють trade-off часу й деталей; CFG не має монотонного зв’язку з якістю; negative prompt не є гарантією; style prompting змінює і стиль, і геометрію. Обмеження: SD 1.5, 512×512, 4 GB VRAM, одна генерація на конфігурацію, PNDM-only verified evidence, відсутність CLIPScore та repeated warm runs. Збережені 13 генерацій не втрачено.""")]
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}})
    nbf.write(nb, LAB4 / "notebooks/01_text_to_image.ipynb")


if __name__ == "__main__":
    lab3_artifacts(); lab4_artifacts()
    # The executable notebooks have their own focused builder. Keeping this call
    # here prevents artifact refreshes from restoring the obsolete viewer-only notebooks.
    from scripts.rebuild_executable_notebooks import lab3, lab4
    lab3(); lab4()
