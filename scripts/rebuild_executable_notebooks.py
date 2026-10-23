"""Build the Lab 3/4 executable educational notebooks deterministically."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def write(path, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    nbf.write(notebook, path)


def lab3():
    cells = [
        md("""# Лабораторна робота №3: executable RAG із CSV та Redis

Notebook відтворює весь шлях `CSV → Document → embedding → index → retrieval → prompt → evaluation`. `FULL EXPERIMENT` виконує моделі й зовнішні сервіси. `PORTFOLIO REPLAY` не імітує inference: він переобчислює статистики, маршрутизацію, метрики та графіки з версіонованих доказів. Кожен результат має статус `LIVE COMPUTATION`, `RECOMPUTED FROM EVIDENCE` або `HISTORICAL INFERENCE`."""),
        md("""## STEP 01 — Bootstrap і обчислювальний контракт

Корінь репозиторію визначається за структурним маркером, а не абсолютним шляхом. Режим задає `PORTFOLIO_MODE`. У replay заборонено називати збережену LLM-відповідь новим inference; у full відсутність Redis, MiniLM або LM Studio є помилкою preflight, а не мовчазним fallback."""),
        code("""from pathlib import Path
import json, os, platform, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

cwd = Path.cwd().resolve()
REPO_ROOT = next((p for p in (cwd, *cwd.parents) if (p / 'labs/lab03_rag/src').is_dir()), None)
if REPO_ROOT is None: raise FileNotFoundError('Repository root not found')
sys.path.insert(0, str(REPO_ROOT))
LAB_ROOT = REPO_ROOT / 'labs/lab03_rag'
PORTFOLIO = LAB_ROOT / 'artifacts/portfolio'
MODE = os.getenv('PORTFOLIO_MODE', 'replay').lower()
if MODE not in {'full', 'replay'}: raise ValueError('PORTFOLIO_MODE must be full or replay')
print('FULL EXPERIMENT' if MODE=='full' else 'REPLAYED FROM VERIFIED EVIDENCE')
print({'step':'01', 'mode':MODE, 'repo_root':str(REPO_ROOT),
       'status':'LIVE COMPUTATION' if MODE == 'full' else 'RECOMPUTED FROM EVIDENCE'})"""),
        md("""## STEP 02 — Imports, configuration і preflight

`Settings` централізує URL сервісів, директорії та model ID. Preflight відокремлює локальне Python-середовище від зовнішніх залежностей. Replay потребує лише committed evidence; full перевіряє Redis Stack перед індексацією, а LM Studio — перед generation."""),
        code("""from labs.lab03_rag.src.config import Settings, MODEL_NAME
from labs.lab03_rag.src.analysis import load_catalog, load_replay_catalog, catalog_summary
from labs.lab03_rag.src.data import load_csv_documents
from labs.lab03_rag.src.documents import build_documents, dataframe_to_documents
from labs.lab03_rag.src.embeddings import create_embeddings, embedding_audit
from labs.lab03_rag.src.evaluation import run_retrieval
from labs.lab03_rag.src.hybrid import route_and_retrieve
from labs.lab03_rag.src.llm import create_llm, list_models
from labs.lab03_rag.src.notebook_helpers import *
from labs.lab03_rag.src.rag import build_prompt, generate_answer
from labs.lab03_rag.src.redis_store import build_store, index_info, redis_preflight

settings = Settings()
preflight = [{'component':'Python','value':sys.version.split()[0], 'required_now':True},
             {'component':'OS','value':platform.platform(), 'required_now':False},
             {'component':'mode','value':MODE, 'required_now':True},
             {'component':'embedding model','value':MODEL_NAME, 'required_now':MODE=='full'},
             {'component':'Redis URL','value':settings.redis_url, 'required_now':MODE=='full'},
             {'component':'LM Studio URL','value':settings.base_url, 'required_now':MODE=='full'}]
display(pd.DataFrame(preflight)); print('Preflight configuration constructed; network checks occur at their owning steps.')"""),
        md("""## STEP 03 — Фактичне завантаження та валідація каталогу

Full читає три сирі CSV через production loader. Replay читає frozen row-level snapshot, а потім заново перетворює числові поля й обчислює integrity statistics. Вхід: рядки товарів; вихід: один `DataFrame` зі provenance `source_file/row_id`."""),
        code("""started = time.perf_counter()
catalog = load_catalog(settings.data_dir) if MODE == 'full' else load_replay_catalog(PORTFOLIO/'catalog_replay.csv')
summary = catalog_summary(catalog)
assert len(catalog) == 185 and catalog.product_id.astype(str).is_unique
assert set(catalog.category) == {'dresses','earrings','sports-shoes'}
display(pd.DataFrame([summary])); display(catalog[['source_file','row_id','product_id','title','category','final_price']].head())
print({'step':'03','rows':len(catalog),'elapsed_ms':round((time.perf_counter()-started)*1000,2),
       'status':computation_status(MODE)})"""),
        md("""## STEP 04 — EDA, обчислена з поточного DataFrame

Пропуски рахуються після нормалізації порожніх рядків; дублікати перевіряються окремо для повних записів та Product ID. Ціновий histogram не доводить причинності. Rating `0` із `ratings_count=0` вилучається з розподілу як неспостережуване значення. Графік нижче створюється зараз, не читається з PNG."""),
        code("""eda = eda_tables(catalog)
display(eda['duplicates']); display(eda['categories']); display(eda['prices'])
display(eda['missing'].head(10))
eda_figure = plot_catalog_eda(catalog); display(eda_figure); plt.close(eda_figure)
conflicts = catalog.loc[catalog.price_conflict, ['product_id','initial_price_numeric','discount_numeric',
                                                  'final_price_numeric','price_formula_delta_inr']]
display(conflicts.nlargest(10, 'price_formula_delta_inr'))
print({'status':computation_status(MODE),'price_conflicts':len(conflicts)})"""),
        md("""## STEP 05 — CSVLoader, row-to-text і `Document`

Один рядок є атомарною бізнес-сутністю, тому character chunking не застосовується. У full `CSVLoader` реально читає CSV, після чого `build_documents` формує content і metadata. Replay викликає той самий serialization contract через `dataframe_to_documents`. Metadata зберігає citation provenance, а не підміняє текст."""),
        code("""if MODE == 'full':
    csvloader_rows = load_csv_documents(settings.data_dir)
    documents = build_documents(csvloader_rows)
    raw_preview = csvloader_rows[0].page_content[:700]
else:
    csvloader_rows = None
    documents = dataframe_to_documents(catalog)
    raw_preview = 'CSVLoader not rerun: source CSV is unavailable in clean replay.'
doc = documents[0]
display(pd.DataFrame([{'representation':'CSVLoader output','preview':raw_preview},
                      {'representation':'Document.page_content','preview':doc.page_content[:700]},
                      {'representation':'Document.metadata','preview':{k:v for k,v in doc.metadata.items() if k!='fields_json'}}]))
assert len(documents)==len(catalog)==len({d.metadata['record_id'] for d in documents})
print({'documents':len(documents),'status':computation_status(MODE)})"""),
        md("""## STEP 06 — MiniLM embeddings: inference або доказовий audit

У full `create_embeddings` завантажує pinned MiniLM і `embedding_audit` кодує всі `Document`. Перевіряються 384 dimensions, норми та truncation відносно max sequence length. Replay не заявляє запуск моделі: він перевіряє audit, створений live inference, та повторно рахує описову статистику token lengths."""),
        code("""if MODE == 'full':
    embeddings = create_embeddings(settings.output_dir/'embeddings', local_files_only=False)
    embedding_report = embedding_audit(embeddings, documents)
    embedding_path = settings.output_dir/'embedding_audit.json'
    embedding_path.write_text(json.dumps(embedding_report, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    embeddings = None
    embedding_report = json.loads((PORTFOLIO/'embedding_audit.json').read_text(encoding='utf-8'))
lengths = pd.Series(embedding_report['token_lengths'], name='word_pieces')
display(pd.DataFrame([{k:v for k,v in embedding_report.items() if k!='token_lengths'}])); display(lengths.describe().to_frame().T)
assert embedding_report['dimension']==384
assert embedding_report['truncated_rows']==int((lengths > embedding_report['max_seq_length']).sum())
print({'model_executed_now':MODE=='full','status':computation_status(MODE)})"""),
        md("""## STEP 07 — PCA векторного простору

У full координати мають бути обчислені з поточної 384-dimensional matrix. У replay доступні лише зафіксовані координати, отримані раніше з verified embeddings; ми перевіряємо їхню схему та кількість і будуємо нову scatter-візуалізацію. Двовимірна відстань не замінює retrieval evaluation."""),
        code("""if MODE == 'full':
    matrix = np.asarray(embeddings.embed_documents([d.page_content for d in documents]))
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    pca = pd.DataFrame({'pc1':xy[:,0],'pc2':xy[:,1],
                        'category':[d.metadata['category'] for d in documents],
                        'product_id':[d.metadata['product_id'] for d in documents]})
    pca_provenance = 'PCA computed now from 384-D embeddings'
else:
    pca = pd.read_csv(PORTFOLIO/'embedding_pca.csv', dtype={'product_id':str})
    pca_provenance = 'plot recomputed from verified PCA coordinates; MiniLM not rerun'
assert len(pca)==len(documents)
pca_figure=plot_pca_coordinates(pca,pca_provenance); display(pca_figure); plt.close(pca_figure)
print({'rows':len(pca),'status':computation_status(MODE),'provenance':pca_provenance})"""),
        md("""## STEP 08 — Redis indexing і index audit

Full виконує Redis preflight, індексує `Document` із поточними embeddings і читає `FT.INFO`. Replay перевіряє historical index metadata, але не називає Redis доступним. FLAT/COSINE виправдані малим корпусом; fingerprint захищає від непомітної заміни документів."""),
        code("""if MODE == 'full':
    redis_client = redis_preflight(settings.redis_url)
    store = build_store(documents, embeddings, settings.redis_url)
    index_report = index_info(redis_client, settings.index_name, expected_count=len(documents))
    index_status = STATUS_LIVE
else:
    store = None
    index_report = json.loads((PORTFOLIO/'index_info.json').read_text(encoding='utf-8'))
    index_status = STATUS_HISTORICAL
display(pd.DataFrame([{'index':index_report['index'],'documents':index_report['document_count'],
                       'algorithm':index_report['vector']['algorithm'],'dimension':index_report['vector']['dim'],
                       'metric':index_report['vector']['distance_metric'],'status':index_status}]))
assert index_report['document_count']==len(documents) and index_report['vector']['dim']==384"""),
        md("""## STEP 09 — Semantic retrieval queries

Full надсилає всі запити до Redis vector store. Replay завантажує row-level ranked evidence, а не готовий metric summary, і перевіряє ranks, distances, IDs та provenance. Нижче q01 показаний як audit table; менша cosine distance означає ближчий embedding."""),
        code("""queries = json.loads((LAB_ROOT/'queries.json').read_text(encoding='utf-8'))['queries']
if MODE == 'full':
    retrieval, retrieval_summary, live_pairs = run_retrieval(store, queries, settings.output_dir, k=5)
else:
    retrieval = pd.read_csv(PORTFOLIO/'semantic_results.csv', dtype={'product_id':str})
assert retrieval.groupby('query_id')['rank'].apply(list).map(lambda x:x==list(range(1,len(x)+1))).all()
display(retrieval[retrieval.query_id=='q01'][['rank','cosine_distance','product_id','source_file','row_id','preview']])
display(retrieval.groupby('query_id').agg(results=('rank','size'), min_distance=('cosine_distance','min'),
                                          max_distance=('cosine_distance','max')))
print({'queries':retrieval.query_id.nunique(),'rows':len(retrieval),'status':computation_status(MODE)})"""),
        md("""## STEP 10 — Переобчислення Hit@5, MRR і Recall@5

Ground truth використовується тільки після завершення retrieval. $MRR=1/r_1$; $Recall@5=|R_5∩G|/|G|$. Для q03 дев'ять релевантних товарів означають теоретичну межу Recall@5 = 5/9. Жодна готова aggregate table не є входом цього кроку."""),
        code("""metrics, routing, structured_rows, hybrid_ids = evaluate_semantic_and_hybrid(catalog, queries, retrieval, k=5)
display(metrics[['query_id','configuration','expected_count','returned','hit_at_k','first_relevant_rank','mrr','recall_at_k']])
metric_summary = metrics.groupby('configuration').agg(hit_at_5=('hit_at_k','mean'),mrr=('mrr','mean'),
                                                       recall_at_5=('recall_at_k','mean'))
display(metric_summary)
assert metrics.query_id.nunique()==8
print({'status':computation_status(MODE),'ground_truth_boundary':'evaluation only'})"""),
        md("""## STEP 11 — Один повний grounded chain

Показано `question → parsed intent → retrieved rows → Documents → prompt`. q04 навмисно зберігається як відомий multi-record failure mode semantic baseline. У full prompt надсилається LM Studio; у replay він лише перебудовується, а historical answer не видається за новий inference."""),
        code("""chain_query = next(q for q in queries if q['id']=='q04')
semantic_ids = replay_rankings(retrieval)
chain_result = route_and_retrieve(catalog, chain_query['question'], semantic_ids['q04'], k=5)
chain_documents = dataframe_to_documents(chain_result.rows)
messages = build_prompt(chain_query['question'], chain_documents)
display(pd.DataFrame([{'question':chain_query['question'],'intent':chain_result.intent.strategy,
                       'product_ids':chain_result.product_ids,
                       'record_ids':[d.metadata['record_id'] for d in chain_documents]}]))
display(pd.DataFrame([{'role':m.type,'content_preview':m.content[:1200]} for m in messages]))
if MODE == 'full':
    available_models = list_models(settings)
    llm = create_llm(settings)
    chain_answer, _ = generate_answer(llm, chain_query['question'], chain_documents)
    chain_answer_status = STATUS_LIVE
else:
    available_models=[]; chain_answer='Not executed in replay mode'; chain_answer_status=STATUS_HISTORICAL
print({'answer_status':chain_answer_status,'models':available_models,'answer':chain_answer[:1000]})"""),
        md("""## STEP 12 — Correctness і citation audit

Historical LLM text є лише вхідним evidence. `answer_audit` заново перевіряє field presence, дозволені citations і невідомі citations проти фактично retrieved records. Lexical match не доводить entailment, тому ручна оцінка залишається окремою."""),
        code("""answer_path = PORTFOLIO/'historical_semantic_answers.csv' if MODE=='replay' else settings.output_dir/'rag/final_answers.csv'
answers = pd.read_csv(answer_path)
answer_checks = answer_audit(answers, queries, catalog, replay_rankings(retrieval))
display(answer_checks)
print({'answers':len(answer_checks),'status':STATUS_HISTORICAL if MODE=='replay' else STATUS_LIVE,
       'warning':'lexical checks require manual semantic review'})"""),
        md("""## STEP 13 — Failure categories

q01: конфлікт listed final price проти discount formula. q03: unit/citation error. q04: неповне multi-row retrieval. Інші категорії охоплюють missing-value semantics, global aggregation, arithmetic, unsupported claim та correct abstention. Їх не видалено після покращення router."""),
        code("""reviews = pd.read_csv(PORTFOLIO/'error_taxonomy.csv')
failure_analysis = reviews.merge(answer_checks, on='query_id', how='left')
display(failure_analysis[['query_id','failure_category','valid_citation','fields_present','fields_expected','review']])
assert {'q01','q03','q04'} <= set(failure_analysis.query_id)
print({'categories':failure_analysis.failure_category.nunique(),'status':computation_status(MODE)})"""),
        md("""## STEP 14 — Semantic vs hybrid через один harness

`route_and_retrieve` не приймає query ID, expected IDs або labels. Він бачить лише question, catalog і semantic candidates. Той самий `evaluate_id_lists` оцінює обидві конфігурації після inference, що усуває ground-truth leakage."""),
        code("""display(routing[['query_id','strategy','conditions','returned_ids']])
display(structured_rows[['query_id','product_id','title','final_price_numeric','source_file','row_id']].head(20))
comparison = metrics.pivot(index='query_id',columns='configuration',values='recall_at_k')
display(comparison)
print({'inference_signature':'question + catalog + semantic candidates','labels_used_by_router':False,
       'status':computation_status(MODE)})"""),
        md("""## STEP 15 — Dashboard із щойно переобчислених метрик

Dashboard отримує `metrics` цього kernel session. Він не читає `baseline_vs_hybrid.png`. Середні значення мають описовий характер: 8 curated queries не дають statistical confidence interval для production traffic."""),
        code("""dashboard_figure=plot_retrieval_dashboard(metrics); display(dashboard_figure); plt.close(dashboard_figure)
display(metric_summary)
display(failure_analysis.failure_category.value_counts().rename_axis('failure_category').to_frame('queries'))
print({'chart_source':'current metrics DataFrame','status':computation_status(MODE)})"""),
        md("""## STEP 16 — Висновки, підтверджені виконанням

Поточний запуск підтвердив 185 унікальних записів, 384-dimensional embedding contract, 8 evaluation queries і збережені q01/q03/q04 failure modes. Semantic search придатний для paraphrase similarity, але не гарантує exact ID, exhaustive filters чи arithmetic. Hybrid router покращує структуровані запити без labels під час inference. Обмеження: replay не перевиконує MiniLM/Redis/LM Studio; historical answers не є новою генерацією; PCA coordinates не відновлюють повну матрицю. Production потребує versioned corpus, repeated latency benchmarks, human relevance judgments, access control і monitoring."""),
    ]
    write(ROOT/'labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb', cells)


def lab4():
    cells = [
        md("""# Лабораторна робота №4: executable Stable Diffusion Text-to-Image

Notebook виконує або доказово відтворює 13 SD 1.5 generations. У `full` кожен `GenerationSpec` проходить через `generate_image`; у `replay` кожен PNG звіряється з параметрами manifest і SHA-256, після чого galleries та performance plots будуються заново. Replay не називається новою генерацією."""),
        md("""## STEP 01 — Generative AI і межі експерименту

VAE стискає та декодує, GAN оптимізує generator проти discriminator, diffusion навчає noise prediction. Тут досліджується inference Stable Diffusion 1.5, а не навчання моделі. Одиничні images демонструють sensitivity до controls, але не дають статистичного ranking якості."""),
        md("""## STEP 02 — Forward/reverse diffusion

$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$, $\\epsilon\\sim\\mathcal N(0,I)$. Training мінімізує $\\mathbb E\\|\\epsilon-\\epsilon_\\theta(x_t,t,c)\\|_2^2$. Inference починається з latent noise; scheduler задає послідовність reverse updates. Seed фіксує генератор шуму, але не гарантує bitwise equality між різними kernels/hardware."""),
        md("""## STEP 03 — Архітектура

Tokenizer → CLIP text encoder → conditioning; U-Net прогнозує noise residual; scheduler оновлює latent; VAE decoder створює RGB. CFG: $\\hat\\epsilon=\\epsilon_u+w(\\epsilon_c-\\epsilon_u)$. Більший $w$ посилює conditioning, але не гарантує кращу композицію."""),
        md("""## STEP 04 — Bootstrap, imports, environment і конфігурація

Режим і paths визначаються один раз. Поточне середовище показується окремо від historical GPU metadata. У replay потрібні pandas/Pillow/matplotlib; у full додатково потрібні torch/diffusers/model weights і достатня пам'ять."""),
        code("""from pathlib import Path
import json, os, platform, sys, time
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

cwd=Path.cwd().resolve()
REPO_ROOT=next((p for p in (cwd,*cwd.parents) if (p/'labs/lab04_stable_diffusion/src').is_dir()),None)
if REPO_ROOT is None: raise FileNotFoundError('Repository root not found')
sys.path.insert(0,str(REPO_ROOT))
LAB_ROOT=REPO_ROOT/'labs/lab04_stable_diffusion'; PORTFOLIO=LAB_ROOT/'artifacts/portfolio'
MODE=os.getenv('PORTFOLIO_MODE','replay').lower()
if MODE not in {'full','replay'}: raise ValueError('PORTFOLIO_MODE must be full or replay')
print('FULL EXPERIMENT' if MODE=='full' else 'REPLAYED FROM VERIFIED EVIDENCE')
from labs.lab04_stable_diffusion.src.config import ModelConfig
from labs.lab04_stable_diffusion.src.evaluation import load_generation_manifest, performance_summary, quality_rubric, scheduler_experiment_plan, validate_images
from labs.lab04_stable_diffusion.src.experiments import experiment_groups, unique_plan
from labs.lab04_stable_diffusion.src.notebook_helpers import *
from labs.lab04_stable_diffusion.src.pipeline import environment, load_pipeline

model_config=ModelConfig(); groups=experiment_groups(); plan=unique_plan()
current_env=pd.DataFrame([{'component':'Python','value':sys.version.split()[0]},
                          {'component':'OS','value':platform.platform()},
                          {'component':'mode','value':MODE},
                          {'component':'model_id','value':model_config.model_id},
                          {'component':'revision','value':model_config.revision}])
display(current_env); print({'specs':len(plan),'status':STATUS_LIVE if MODE=='full' else STATUS_REPLAY})"""),
        md("""## STEP 05 — Pipeline load або historical model audit

Full реально викликає `load_pipeline(ModelConfig)`, фіксує class, scheduler, dtype, attention processors і memory strategy. Replay не створює pipeline: він показує тільки historical metadata та перевіряє, що всі 13 records належать одному pinned model/revision."""),
        code("""manifest=load_generation_manifest(PORTFOLIO/'generations.csv')
if MODE=='full':
    pipe, pipeline_context=load_pipeline(model_config)
    pipeline_status=STATUS_LIVE
    pipeline_table=pd.DataFrame([{k:v for k,v in pipeline_context.items() if k!='scheduler_config'}])
else:
    pipe=None; pipeline_context=None; pipeline_status='HISTORICAL INFERENCE'
    pipeline_table=manifest[['model_id','revision','pipeline_class','scheduler','dtype','device_strategy','model_load_seconds']].drop_duplicates()
display(pipeline_table)
assert len(manifest)==13 and manifest.artifact_id.nunique()==13
print({'pipeline_loaded_now':pipe is not None,'status':pipeline_status})"""),
        md("""## STEP 06 — Baseline `GenerationSpec` і resolver

Baseline: seed 42, 30 steps, CFG 7.5, 512×512, PNDM. `resolve_generation` у full викликає production `generate_image`; у replay зіставляє всі controls із manifest, відкриває PNG і звіряє SHA-256. Output record містить provenance та runtime."""),
        code("""baseline_spec=groups['baseline'][0]
display(experiment_plan([baseline_spec]))
baseline_records=resolve_group([baseline_spec],mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
display(records_table(baseline_records))
baseline_figure=plot_generation_gallery(baseline_records,'Baseline generation',columns=1)
display(baseline_figure); plt.close(baseline_figure)
print({'case':'baseline','status':baseline_records[0]['status'],'sha256':baseline_records[0]['image_sha256']})"""),
        md("""## STEP 07 — Seed experiment group

Код формує чотири specs з незмінними prompt/steps/CFG і seeds 42, 123, 777, 2026. Галерея складається з resolved image objects у поточній комірці. Відмінність композиції можна пов'язувати із seed лише в межах цієї контрольованої серії."""),
        code("""seed_specs=groups['seed']; display(experiment_plan(seed_specs))
seed_records=resolve_group(seed_specs,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
display(records_table(seed_records))
seed_figure=plot_generation_gallery(seed_records,'Seed comparison',columns=2)
display(seed_figure); plt.close(seed_figure)
print({'resolved':len(seed_records),'statuses':sorted({r['status'] for r in seed_records})})"""),
        md("""## STEP 08 — Inference steps 20/30/50

Більше steps означає більше U-Net evaluations, але не монотонне покращення. Prompt, seed і CFG сталі. Runtime chart будується з metadata resolved records; кожна точка — один run, тому dispersion і confidence interval відсутні."""),
        code("""step_specs=groups['steps']; display(experiment_plan(step_specs))
step_records=resolve_group(step_specs,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
display(records_table(step_records))
step_figure=plot_generation_gallery(step_records,'Inference-step comparison'); display(step_figure); plt.close(step_figure)
runtime_figure=plot_runtime(step_records); display(runtime_figure); plt.close(runtime_figure)
print({'resolved':len(step_records),'warning':'single timing per control'})"""),
        md("""## STEP 09 — Classifier-Free Guidance 5/7.5/10

CFG масштабує різницю conditional/unconditional predictions. Низьке значення може послабити prompt adherence, високе — посилити saturation або geometry artifacts. Seed, prompt і steps фіксовані; результати не дозволяють стверджувати універсальний optimum."""),
        code("""cfg_specs=groups['cfg']; display(experiment_plan(cfg_specs))
cfg_records=resolve_group(cfg_specs,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
display(records_table(cfg_records))
cfg_figure=plot_generation_gallery(cfg_records,'CFG comparison'); display(cfg_figure); plt.close(cfg_figure)
print({'resolved':len(cfg_records),'cfg':[r['guidance_scale'] for r in cfg_records]})"""),
        md("""## STEP 10 — Negative-prompt control pairs

Для seeds 42 і 123 кожна пара відрізняється лише `negative_prompt`. Negative prompt впливає на всю denoising trajectory; це не post-processing filter і не гарантія видалення watermark/text/artifacts."""),
        code("""negative_records={}
for group_name in ('negative_prompt_seed42','negative_prompt_seed123'):
    specs=groups[group_name]; display(experiment_plan(specs))
    records=resolve_group(specs,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
    negative_records[group_name]=records; display(records_table(records))
    figure=plot_generation_gallery(records,group_name,columns=2); display(figure); plt.close(figure)
print({'pairs':len(negative_records),'images':sum(map(len,negative_records.values())),
       'status':sorted({r['status'] for rows in negative_records.values() for r in rows})})"""),
        md("""## STEP 11 — Watercolor, isometric, photorealistic

`experiment_groups()['styles']` програмно створює три specs із однаковими seed/steps/CFG та різним prompt suffix. Це prompt conditioning, не neural style transfer і не LoRA. Однаковий seed не фіксує geometry, коли conditioning змінено."""),
        code("""style_specs=groups['styles']; display(experiment_plan(style_specs))
style_records=resolve_group(style_specs,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context)
display(records_table(style_records))
style_figure=plot_generation_gallery(style_records,'Prompted styles'); display(style_figure); plt.close(style_figure)
print({'styles':[s.style for s in style_specs],'resolved':len(style_records)})"""),
        md("""## STEP 12 — Scheduler inspection

PNDM підтверджено всіма 13 records. Euler і DDIM позначено `NOT_EXECUTED`: однакова кількість steps не означає однакові sampling trajectories, а вигадані timings неприпустимі."""),
        code("""scheduler_plan=scheduler_experiment_plan()
executed_schedulers=manifest.scheduler.value_counts().rename_axis('scheduler').to_frame('verified_images')
display(scheduler_plan); display(executed_schedulers)
assert set(manifest.scheduler)=={'PNDMScheduler'}
print({'status':STATUS_REPLAY if MODE=='replay' else STATUS_LIVE,'unexecuted':['EulerDiscreteScheduler','DDIMScheduler']})"""),
        md("""## STEP 13 — Rubric-based evaluation

CLIPScore не обчислювався. `quality_rubric` відтворює historical human observations і явно позначає assessment як descriptive. Вони не є числовим ground truth. Повторна візуальна оцінка потребувала б незалежних raters і blind protocol."""),
        code("""rubric=quality_rubric(PORTFOLIO/'visual_observations.json')
display(rubric)
print({'rows':len(rubric),'origin':'historical manual observations','status':'HISTORICAL INFERENCE',
       'clipscore_computed':False})"""),
        md("""## STEP 14 — Runtime, memory і integrity audit

Статистики обчислюються з current manifest, а всі PNG повторно відкриваються. `mean/std` для experiment group не є benchmark: controls різні, baseline reused між групами, repeats відсутні. Peak memory — allocated CUDA memory під час historical run, не total GPU occupancy."""),
        code("""image_audit=validate_images(manifest.assign(image_path=manifest.public_image_path),LAB_ROOT)
perf=performance_summary(manifest)
display(image_audit); display(perf)
assert image_audit[['exists','valid_png']].all().all() and (image_audit[['width','height']]==512).all().all()
all_records=[]
for spec in plan:
    all_records.append(resolve_generation(spec,mode=MODE,manifest=manifest,pipeline=pipe,context=pipeline_context))
print({'valid_images':len(image_audit),'sha256_verified':sum(bool(r.get('sha256_verified',MODE=='full')) for r in all_records),
       'status':STATUS_REPLAY if MODE=='replay' else STATUS_LIVE})"""),
        md("""## STEP 15 — Dashboard із resolved records

Scatter plots створюються в цьому kernel із 13 records. Перший показує зв'язок steps/runtime; другий — runtime/peak allocated memory. Колір кодує experiment group. Correlation не інтерпретується причинно через confounding controls і один observation на spec."""),
        code("""dashboard=plot_performance_dashboard(all_records); display(dashboard); plt.close(dashboard)
display(records_table(all_records))
print({'chart_source':'records resolved in current session','records':len(all_records),
       'status':STATUS_REPLAY if MODE=='replay' else STATUS_LIVE})"""),
        md("""## STEP 16 — LoRA: що було б окремим експериментом

LoRA заморожує $W$ і навчає low-rank update $\\Delta W=BA$, де rank значно менший за dimensions $W$. Для чесного досліду потрібні dataset/license audit, train/validation split, adapter rank/alpha, learning-rate schedule, checkpoints і base-vs-adapter evaluation на фіксованих prompts/seeds. Тут LoRA не навчалася і не підключалася; жоден image не маркується як LoRA-generated."""),
        md("""## STEP 17 — Висновки

Поточний запуск сформував specs кодом, resolved 13 records, перевірив 13 SHA-256 у replay, заново побудував п'ять experiment galleries, runtime chart і performance dashboard. Seed змінює композицію; steps мають cost trade-off; CFG не має доведеного монотонного optimum; negative prompt не є гарантією; style prompting змінює style і geometry. Обмеження: SD 1.5, 512×512, PNDM-only evidence, один run на spec, historical GPU timings, відсутні CLIPScore та blind raters. Replay є перевіркою доказів і повторним аналізом, а не новим diffusion inference."""),
    ]
    write(ROOT/'labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb', cells)


if __name__ == '__main__':
    lab3(); lab4()
