"""Build the single-mode live Lab 03 notebook. Execution counts come only from Jupyter."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb"


def md(title, body):
    return nbf.v4.new_markdown_cell(f"## {title}\n\n{body}")


def code(source):
    return nbf.v4.new_code_cell(source.strip())


nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "lab03_execution": {"mode": "live-only", "status": "UNEXECUTED"},
})
nb.cells = [
    nbf.v4.new_markdown_cell("""# Лабораторна робота №3: Grounded RAG із CSV, Redis і локальною LLM

Варіант 3. Єдиний шлях виконання: `CSV → CSVLoader → Document → MiniLM → Redis FLAT/COSINE → retrieval → grounded prompt → LM Studio → audit`. Notebook не має replay-режиму й не читає історичні відповіді. Недоступний сервіс спричиняє явну помилку з діагностикою."""),
    md("STEP 01 — Environment and configuration", "Конфігурація є незмінною `Settings`; усі шляхи визначаються від кореня репозиторію."),
    code("""
from pathlib import Path
import json, platform, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
cwd = Path.cwd().resolve()
REPO_ROOT = next((p for p in (cwd, *cwd.parents) if (p/'labs/lab03_rag/src').is_dir()), None)
if REPO_ROOT is None: raise FileNotFoundError('Repository root not found')
sys.path.insert(0, str(REPO_ROOT))
LAB_ROOT = REPO_ROOT/'labs/lab03_rag'
print({'python':sys.version.split()[0], 'platform':platform.platform(), 'repo_root':str(REPO_ROOT), 'execution':'live-only'})
"""),
    md("STEP 02 — Dependencies and preflight", "Redis Search, MiniLM та LM Studio перевіряються до експерименту. Немає автоматичного переходу на збережені результати."),
    code("""
from labs.lab03_rag.src.config import Settings, MODEL_NAME
from labs.lab03_rag.src.analysis import load_catalog, catalog_summary
from labs.lab03_rag.src.data import audit_corpus, load_csv_documents
from labs.lab03_rag.src.documents import build_documents, dataframe_to_documents
from labs.lab03_rag.src.embeddings import create_embeddings, embedding_audit
from labs.lab03_rag.src.redis_store import redis_preflight, build_store
from labs.lab03_rag.src.retrieval import retrieve
from labs.lab03_rag.src.evaluation import load_queries, run_retrieval, run_answers, verify_answer, analyse_current_answers
from labs.lab03_rag.src.llm import list_models, create_llm
from labs.lab03_rag.src.rag import build_prompt, generate_answer, validate_citations
from labs.lab03_rag.src.hybrid import route_and_retrieve
from labs.lab03_rag.src.notebook_helpers import eda_tables, plot_catalog_eda, plot_pca_coordinates, evaluate_semantic_and_hybrid, plot_retrieval_dashboard
settings=Settings()
redis_client, redis_status=redis_preflight(settings.redis_url)
available_models=list_models(settings)
if settings.model not in available_models: raise RuntimeError(f'Required model {settings.model!r} not loaded; available={available_models}')
display(pd.DataFrame([{'component':'Redis Search','status':redis_status['status'],'value':settings.redis_url},
 {'component':'embedding','status':'configured','value':MODEL_NAME},
 {'component':'LM Studio','status':'ready','value':settings.model}]))
"""),
    md("STEP 03 — CSV audit", "Три зафіксовані CSV перевіряються на схему, кількість рядків, пропуски, дублікати та SHA-256."),
    code("""
audit=audit_corpus(settings.data_dir)
display(pd.DataFrame([{k:v for k,v in x.items() if k not in {'samples','missing','dtypes','columns'}} for x in audit]))
catalog=load_catalog(settings.data_dir); summary=catalog_summary(catalog)
assert len(catalog)==185 and catalog.product_id.is_unique
display(pd.DataFrame([summary])); display(catalog[['source_file','row_id','product_id','title','category']].head())
"""),
    md("STEP 04 — Exploratory Data Analysis", "EDA обчислюється з поточного каталогу. Нульовий рейтинг без оцінок не трактується як спостережувана оцінка; listed values не виправляються."),
    code("""
eda=eda_tables(catalog)
display(eda['duplicates']); display(eda['categories']); display(eda['prices']); display(eda['missing'].head(10))
fig_eda=plot_catalog_eda(catalog); display(fig_eda); plt.close(fig_eda)
display(catalog.loc[catalog.price_conflict,['product_id','initial_price_numeric','discount_numeric','final_price_numeric','price_formula_delta_inr']].head(10))
"""),
    md("STEP 05 — CSVLoader and Documents", "`CSVLoader` реально читає кожний файл. Один CSV-рядок стає одним `Document`; provenance зберігається як `source_file`, `row_id`, `record_id`, `product_id`."),
    code("""
loaded_rows=load_csv_documents(settings.data_dir); documents=build_documents(loaded_rows)
assert len(documents)==185==len({d.metadata['record_id'] for d in documents})
required={'source_file','row_id','record_id','product_id'}
assert all(required <= d.metadata.keys() for d in documents)
display(pd.DataFrame([{'record_id':d.metadata['record_id'],'product_id':d.metadata['product_id'],'preview':d.page_content[:300]} for d in documents[:3]]))
"""),
    md("STEP 06 — MiniLM embeddings", "Поточні документи кодуються `all-MiniLM-L6-v2`. Перевіряються 384 компоненти, скінченність і одинична L2-норма; локальний content-addressed cache не замінює inference."),
    code("""
embeddings=create_embeddings(settings.output_dir/'embeddings', local_files_only=False)
embedding_report=embedding_audit(embeddings, documents)
(settings.output_dir/'embedding_audit.json').write_text(json.dumps(embedding_report,ensure_ascii=False,indent=2),encoding='utf-8')
display(pd.DataFrame([{k:v for k,v in embedding_report.items() if k!='token_lengths'}]))
display(pd.Series(embedding_report['token_lengths'],name='word_pieces').describe().to_frame().T)
"""),
    md("STEP 07 — Embedding visualization", "PCA є лише двовимірною проєкцією поточної 384-вимірної матриці й не використовується як доказ retrieval quality."),
    code("""
matrix=np.asarray(embeddings.embed_documents([d.page_content for d in documents]),dtype=np.float32)
centered=matrix-matrix.mean(axis=0); _,_,vt=np.linalg.svd(centered,full_matrices=False); xy=centered@vt[:2].T
pca=pd.DataFrame({'pc1':xy[:,0],'pc2':xy[:,1],'category':[d.metadata['category'] for d in documents],'product_id':[d.metadata['product_id'] for d in documents]})
fig_pca=plot_pca_coordinates(pca,'current 384-D embeddings'); display(fig_pca); plt.close(fig_pca)
"""),
    md("STEP 08 — Redis vector indexing", "Корпус-специфічний namespace і стабільні IDs роблять повторний запуск ідемпотентним. Сторонні індекси не видаляються."),
    code("""
store,index_report=build_store(documents,embeddings,settings.redis_url)
(settings.output_dir/'index_info.json').write_text(json.dumps(index_report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
display(pd.DataFrame([{'index':index_report['index'],'documents':index_report['document_count'],'inserted_this_run':index_report['inserted'],'algorithm':index_report['vector']['algorithm'],'dimension':index_report['vector']['dim'],'metric':index_report['vector']['distance_metric']}]))
assert index_report['document_count']==185 and int(index_report['vector']['dim'])==384
"""),
    md("STEP 09 — Similarity retrieval", "Усі вісім запитів виконуються через Redis. Таблиця містить rank, cosine distance та provenance кожного результату."),
    code("""
queries=load_queries(settings.data_dir)
retrieval,retrieval_summary,live_pairs=run_retrieval(store,queries,settings.output_dir,k=5)
display(retrieval[['query_id','rank','cosine_distance','product_id','record_id','category']])
"""),
    md("STEP 10 — Retrieval metrics", "Labels перетинають inference boundary лише після retrieval. Hit@5, MRR і Recall@5 не вимірюють правильність відповіді LLM."),
    code("""
display(retrieval_summary[['query_id','kind','returned','hit_at_k','first_relevant_rank','reciprocal_rank','recall_at_k','query_elapsed_ms']])
semantic_metric_summary=pd.DataFrame([{'configuration':'semantic_baseline','Hit@5':retrieval_summary.hit_at_k.mean(),'MRR':retrieval_summary.reciprocal_rank.mean(),'Recall@5':retrieval_summary.recall_at_k.mean()}])
display(semantic_metric_summary)
"""),
    md("STEP 11 — End-to-end RAG demonstration", "Для q04 окремо показано сирі semantic результати Redis і structured exact-ID результат router. До LLM передаються обидва потрібні записи; відповідь не заготовлена."),
    code("""
q04=next(q for q in queries if q['id']=='q04')
semantic_pairs=live_pairs['q04']; semantic_docs=[d for d,_ in semantic_pairs]
hybrid=route_and_retrieve(catalog,q04['question'],[d.metadata['product_id'] for d in semantic_docs],k=5)
chain_docs=dataframe_to_documents(hybrid.rows); messages=build_prompt(q04['question'],chain_docs)
display(pd.DataFrame([{'source':'Redis semantic','rank':i,'product_id':d.metadata['product_id'],'record_id':d.metadata['record_id'],'distance':score} for i,(d,score) in enumerate(semantic_pairs,1)]))
display(pd.DataFrame([{'source':'structured exact-ID','product_id':d.metadata['product_id'],'record_id':d.metadata['record_id']} for d in chain_docs]))
display(pd.DataFrame([{'role':m.type,'content':m.content} for m in messages]))
llm=create_llm(settings); generation_started=time.perf_counter(); chain_answer,_=generate_answer(llm,q04['question'],chain_docs); generation_ms=(time.perf_counter()-generation_started)*1000
chain_verification=verify_answer(chain_answer,chain_docs,q04['expected_fields'])
print(chain_answer); display(pd.DataFrame([{'model':settings.model,'strategy':hybrid.intent.strategy,'redis_retrieval_ms':semantic_docs[0].metadata['query_elapsed_ms'],'generation_ms':generation_ms,'citation_valid':not chain_verification['unknown_citations'],'fields_matched':sum(c['value_present'] and c['expected_citation_present'] for c in chain_verification['field_checks']),'fields_expected':len(chain_verification['field_checks'])}]))
"""),
    md("STEP 12 — Live answer and citation evaluation", "Вісім відповідей генеруються зараз із Redis context. Lexical field match не доводить entailment; тому semantic completeness лишається `MANUAL REVIEW REQUIRED`, якщо її не можна довести контрактними перевірками."),
    code("""
answers=run_answers(llm,queries,live_pairs,settings.output_dir,settings.model)
answer_view=answers.rename(columns={'query_id':'Query','retrieved_count':'Retrieved','status':'LLM Status','citation_valid':'Citation Valid','fields_matched':'Fields Matched','completeness':'Completeness','total_latency_ms':'Latency'})
display(answer_view[['Query','Retrieved','LLM Status','Citation Valid','Fields Matched','fields_expected','Completeness','Latency']])
display(answers[['query_id','retrieved_context','answer']])
"""),
    md("STEP 13 — Current RAG error analysis", "Класифікація виводиться лише з поточних retrieval, generation і verification результатів. `Unsupported claim / hallucination` не присвоюється без підтвердженого entailment-аудиту."),
    code("""
failure_analysis=analyse_current_answers(answers)
display(failure_analysis)
print(f'Completed generations: {(answers.status=="completed").sum()}/8; citation-valid: {answers.citation_valid.sum()}/8; manual semantic review remains required.')
"""),
    md("STEP 14 — Semantic vs Hybrid comparison", "Router не отримує query ID, expected IDs або expected fields. Обидві конфігурації оцінюються одним evaluation harness після inference."),
    code("""
metrics,routing,structured_rows,hybrid_ids=evaluate_semantic_and_hybrid(catalog,queries,retrieval,k=5)
metric_summary=metrics.groupby('configuration').agg(Hit_at_5=('hit_at_k','mean'),MRR=('mrr','mean'),Recall_at_5=('recall_at_k','mean'))
display(routing[['query_id','strategy','returned_ids','conditions']]); display(metric_summary)
"""),
    md("STEP 15 — Evaluation dashboard", "Dashboard будується з поточного `metrics`. Вісім curated queries не дають підстав для production-level accuracy claims."),
    code("""
fig_dashboard=plot_retrieval_dashboard(metrics); display(fig_dashboard); plt.close(fig_dashboard)
display(metric_summary); display(failure_analysis[['query_id','categories']])
"""),
    md("STEP 16 — Conclusions", "Висновок формується з об'єктів поточної kernel session, а не з історичних артефактів."),
    code("""
conclusion=pd.DataFrame([{'documents':index_report['document_count'],'redis_index':index_report['index'],'idempotent_inserted':index_report['inserted'],'llm_answers_completed':int((answers.status=='completed').sum()),'citation_valid':int(answers.citation_valid.sum()),'semantic_Hit@5':semantic_metric_summary.iloc[0]['Hit@5'],'semantic_MRR':semantic_metric_summary.iloc[0]['MRR'],'semantic_Recall@5':semantic_metric_summary.iloc[0]['Recall@5']}])
display(conclusion)
print('Limits: curated labels are incomplete; lexical field checks are not semantic entailment; one latency observation per query is descriptive only; manual review is required for unsupported-claim detection.')
"""),
]
TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, TARGET)
print(TARGET)
