"""Synthesize and Validate 50 Genuinely Independent Holdout v3 Queries.

Fully grounded in the 8 knowledge documents with exact verbatim spans:
- 10 exact_fact
- 10 paraphrase
- 10 cross_document
- 10 numerical_reasoning
- 10 unsupported (adversarial refusal)

Audited against all 5 prior evaluation files.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.evaluation.metrics import normalize_text
from labs.lab07_final_project.evaluation.audit_dataset_duplicates import audit_against_previous

KNOWLEDGE_DIR = Path("labs/lab07_final_project/assets/knowledge")
DATASETS_DIR = Path("labs/lab07_final_project/evaluation/datasets")
CORPUS_CHUNKS_PATH = DATASETS_DIR / "corpus_chunks_manifest.json"


def find_chunk_for_span(chunks: list[dict], span: str, doc_filename: str) -> str:
    span_norm = normalize_text(span)
    for c in chunks:
        if c["filename"] == doc_filename:
            if span_norm in normalize_text(c["text"]):
                return c["chunk_id"]
    raise ValueError(f"Span not found in any chunk of {doc_filename}:\n  '{span}'")


def verify_span_in_doc(docs: dict[str, str], span: str, doc_filename: str):
    doc_text = docs[doc_filename]
    if span not in doc_text:
        if span.replace("\r\n", "\n") not in doc_text.replace("\r\n", "\n"):
            raise ValueError(f"Span not verbatim in {doc_filename}:\n  '{span}'")


def main():
    chunks = json.loads(CORPUS_CHUNKS_PATH.read_text(encoding="utf-8"))
    docs = {f.name: f.read_text(encoding="utf-8").replace("\r\n", "\n") for f in KNOWLEDGE_DIR.glob("**/*.md")}

    queries = []

    # ----------------------------------------------------
    # 1. 10 EXACT FACT QUERIES (final_v3_indep_001 .. 010)
    # ----------------------------------------------------
    # 001: ADR 001 - HNSW and FLAT indexing with FLOAT32
    s001 = "- Utilizes `HNSW` (Hierarchical Navigable Small World) and `FLAT` vector indexing with `FLOAT32` precision."
    queries.append({
        "id": "final_v3_indep_001",
        "category": "exact_fact",
        "language": "en",
        "query": "Which vector indexing structures and numerical precision are utilized in Redis Stack according to ADR 001?",
        "expected_answer": "Redis Stack utilizes HNSW and FLAT vector indexing with FLOAT32 precision.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Decision"],
        "spans": [s001],
        "tool_calls": ["search_knowledge"],
    })

    # 002: API Spec - Rejection policy HTTP 400 ExtractionError
    s002 = "Rejection policy: Zero-byte files or corrupted PDFs without an extractable text layer return HTTP 400 with `ExtractionError`."
    queries.append({
        "id": "final_v3_indep_002",
        "category": "exact_fact",
        "language": "uk",
        "query": "Який HTTP-код та яку назву помилки повертає система завантаження документів для порожніх або пошкоджених файлів згідно з API специфікацією?",
        "expected_answer": "Повертається HTTP 400 з помилкою ExtractionError.",
        "doc_ids": ["api_specification_and_contracts.md"],
        "sections": ["## Core Endpoints / 2. Document Management"],
        "spans": [s002],
        "tool_calls": ["search_knowledge"],
    })

    # 003: Incident Postmortem - 14:23 HSET rejection
    s003 = "- **14:23**: Redis rejected `HSET` operations with OOM error. Ingestion worker crashed."
    queries.append({
        "id": "final_v3_indep_003",
        "category": "exact_fact",
        "language": "en",
        "query": "At what exact time and with which Redis command failure did the ingestion worker crash during the OOM incident?",
        "expected_answer": "At 14:23 UTC, Redis rejected HSET operations with an OOM error, causing the ingestion worker to crash.",
        "doc_ids": ["incident_postmortem_redis_oom.md"],
        "sections": ["## Timeline (UTC)"],
        "spans": [s003],
        "tool_calls": ["search_knowledge"],
    })

    # 004: Runbook - Ingress readiness probe curl command
    s004 = "- Ingress readiness probe: `curl -s http://127.0.0.1:8000/api/v1/health/ready`"
    queries.append({
        "id": "final_v3_indep_004",
        "category": "exact_fact",
        "language": "uk",
        "query": "Яку команду curl наведено в інструкції runbook для перевірки готовності інґрес-сервісу?",
        "expected_answer": "Команда: curl -s http://127.0.0.1:8000/api/v1/health/ready",
        "doc_ids": ["rag_system_runbook.md"],
        "sections": ["### Health Probes & Monitoring"],
        "spans": [s004],
        "tool_calls": ["search_knowledge"],
    })

    # 005: Lab 3 - MiniLM truncates 17 long records
    s005 = "MiniLM обрізає 17 довгих записів до 256 word pieces."
    queries.append({
        "id": "final_v3_indep_005",
        "category": "exact_fact",
        "language": "en",
        "query": "How many long product records were truncated to 256 word pieces by MiniLM in the Lab 3 report?",
        "expected_answer": "MiniLM truncated 17 long product records to 256 word pieces.",
        "doc_ids": ["lab03_rag_report.md"],
        "sections": ["## Обмеження і подальша робота"],
        "spans": [s005],
        "tool_calls": ["search_knowledge"],
    })

    # 006: Lab 4 - Three style suffixes for second scene
    s006 = "| Styles | стильовий suffix | watercolor, isometric, photorealistic | друга сцена, seed, steps, CFG, negative, model, scheduler, resolution |"
    queries.append({
        "id": "final_v3_indep_006",
        "category": "exact_fact",
        "language": "uk",
        "query": "Які три художні стилі (стильові суфікси) досліджувалися для генерації другої семантичної сцени в Lab 4?",
        "expected_answer": "Для другої семантичної сцени досліджувалися три стилі: watercolor, isometric та photorealistic.",
        "doc_ids": ["lab04_stable_diffusion_report.md"],
        "sections": ["## Методика експерименту"],
        "spans": [s006],
        "tool_calls": ["search_knowledge"],
    })

    # 007: Lab 5 - Baseline answers left, Brown, Yes
    s007 = "Baseline: CORRECT=3, PARTIALLY_CORRECT=0, INCORRECT=0. Відповіді left, Brown, Yes відповідають видимим фактам."
    queries.append({
        "id": "final_v3_indep_007",
        "category": "exact_fact",
        "language": "en",
        "query": "What were the three baseline answers produced by LLaVA in the Lab 5 visual question answering evaluation?",
        "expected_answer": "The baseline answers were left, Brown, and Yes, scoring 3/3 CORRECT.",
        "doc_ids": ["lab05_multimodal_report.md"],
        "sections": ["## 6. Роль компонентів LLaVA"],
        "spans": [s007],
        "tool_calls": ["search_knowledge"],
    })

    # 008: Lab 6 - Environment versions
    s008 = "Python 3.13.5; openai-agents 0.22.3; openai 3.17.0"
    queries.append({
        "id": "final_v3_indep_008",
        "category": "exact_fact",
        "language": "uk",
        "query": "Які версії Python та бібліотеки openai-agents зазначені в середовищі лабораторної роботи №6?",
        "expected_answer": "Зазначено Python 3.13.5 та openai-agents 0.22.3 (разом з openai 3.17.0).",
        "doc_ids": ["lab06_agents_report.md"],
        "sections": ["## Середовище та модель"],
        "spans": [s008],
        "tool_calls": ["search_knowledge"],
    })

    # 009: ADR 001 - pgvector rejected reason
    s009 = "- **pgvector (PostgreSQL)**: Rejected due to heavy container footprint and complex local installation on developer workstations."
    queries.append({
        "id": "final_v3_indep_009",
        "category": "exact_fact",
        "language": "en",
        "query": "What was the rationale for rejecting pgvector in the ADR 001 architectural decision?",
        "expected_answer": "pgvector was rejected due to its heavy container footprint and complex local workstation installation.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Alternatives Considered"],
        "spans": [s009],
        "tool_calls": ["search_knowledge"],
    })

    # 010: API Spec - Health ready dependencies response
    s010 = "- `GET /health/ready`: Returns `{\"status\": \"ready\", \"redis\": true, \"database\": true}` (HTTP 200), or HTTP 503 if dependencies fail."
    queries.append({
        "id": "final_v3_indep_010",
        "category": "exact_fact",
        "language": "uk",
        "query": "Який JSON-об'єкт та HTTP-код повертає ендпоінт /health/ready, якщо всі залежності доступні?",
        "expected_answer": "Повертає HTTP 200 та об'єкт {\"status\": \"ready\", \"redis\": true, \"database\": true}.",
        "doc_ids": ["api_specification_and_contracts.md"],
        "sections": ["## Core Endpoints / 1. Health Checks"],
        "spans": [s010],
        "tool_calls": ["search_knowledge"],
    })

    # ----------------------------------------------------
    # 2. 10 PARAPHRASE QUERIES (final_v3_indep_011 .. 020)
    # ----------------------------------------------------
    # 011: ADR 001 - busy_timeout 5000 ms
    s011 = "- Enforces `PRAGMA busy_timeout = 5000` to prevent database locks under concurrent API accesses."
    queries.append({
        "id": "final_v3_indep_011",
        "category": "paraphrase",
        "language": "en",
        "query": "How is SQLite configured to tolerate simultaneous database write operations during query execution?",
        "expected_answer": "SQLite enforces PRAGMA busy_timeout = 5000 to prevent locks under concurrent API accesses.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Decision"],
        "spans": [s011],
        "tool_calls": ["search_knowledge"],
    })

    # 012: Incident - 450-page PDF with 12,400 chunks
    s012 = "- **14:20**: Ingestion job started for a 450-page technical specification PDF containing 12,400 chunks."
    queries.append({
        "id": "final_v3_indep_012",
        "category": "paraphrase",
        "language": "uk",
        "query": "Скільки сторінок та скільки текстових чанків містив вхідний PDF-документ технічної специфікації, що спричинив OOM-аварію під час завантаження?",
        "expected_answer": "Документ технічної специфікації містив 450 сторінок та 12 400 текстових чанків.",
        "relevant_doc_ids": ["incident_postmortem_redis_oom.md"],
        "doc_ids": ["incident_postmortem_redis_oom.md"],
        "sections": ["## Timeline (UTC)"],
        "spans": [s012],
        "tool_calls": ["search_knowledge"],
    })

    # 013: Runbook - Service overview port mapping
    s013 = "3. Redis Stack container running on port 6380 for vector indexing."
    queries.append({
        "id": "final_v3_indep_013",
        "category": "paraphrase",
        "language": "en",
        "query": "Which dedicated port is reserved for the Redis Stack container for vector operations according to the system runbook?",
        "expected_answer": "The Redis Stack container runs on port 6380 for vector indexing.",
        "doc_ids": ["rag_system_runbook.md"],
        "sections": ["## 1. System Overview"],
        "spans": [s013],
        "tool_calls": ["search_knowledge"],
    })

    # 014: Lab 3 - Citations format [source.csv:row]
    s014 = "Grounded prompt передає локальній моделі в LM Studio лише retrieved records і вимагає citations формату `[source.csv:row]`."
    queries.append({
        "id": "final_v3_indep_014",
        "category": "paraphrase",
        "language": "uk",
        "query": "Який формат посилань на джерела вимагається від генератора відповідей у звіті Lab 3?",
        "expected_answer": "Вимагаються обов'язкові цитування у форматі [source.csv:row].",
        "doc_ids": ["lab03_rag_report.md"],
        "sections": ["## Реалізація"],
        "spans": [s014],
        "tool_calls": ["search_knowledge"],
    })

    # 015: Lab 4 - CFG meaning
    s015 = "CFG керує силою умови, а не «точністю»; negative prompt змінює опорну гілку CFG."
    queries.append({
        "id": "final_v3_indep_015",
        "category": "paraphrase",
        "language": "en",
        "query": "According to the Lab 4 report, what is the theoretical role of classifier-free guidance (CFG) in the diffusion process?",
        "expected_answer": "CFG controls the strength of the text conditioning rather than absolute accuracy, while negative prompts shift the unconditional reference branch.",
        "doc_ids": ["lab04_stable_diffusion_report.md"],
        "sections": ["## Теорія та архітектура"],
        "spans": [s015],
        "tool_calls": ["search_knowledge"],
    })

    # 016: Lab 5 - Projector aligns visual tokens with text embedding space
    s016 = "Енкодер зображення утворює вектори ознак, проєктор узгоджує їх із простором мовних embeddings, tokenizer перетворює запитання на текстові токени."
    queries.append({
        "id": "final_v3_indep_016",
        "category": "paraphrase",
        "language": "uk",
        "query": "Яка функція відводиться проектору в мультимодальній архітектурі LLaVA згідно з теоретичним описом Lab 5?",
        "expected_answer": "Проектор вирівнює та узгоджує вектори візуальних ознак із простором текстових мовних ембедингів.",
        "doc_ids": ["lab05_multimodal_report.md"],
        "sections": ["## 4. Теоретичні відомості"],
        "spans": [s016],
        "tool_calls": ["search_knowledge"],
    })

    # 017: Lab 6 - Single agent architecture clarification
    s017 = "Практичний сценарій — аналіз завершених лабораторних звітів. Використано один агент, окремо створюється контрольний екземпляр без інструментів. Це контрольовані запуски однієї архітектури, не мультиагентна система."
    queries.append({
        "id": "final_v3_indep_017",
        "category": "paraphrase",
        "language": "en",
        "query": "How does the Lab 6 report characterize its architectural scope regarding multi-agent coordination versus a single agent?",
        "expected_answer": "It clarifies that a single agent architecture with a no-tools control instance was evaluated across controlled runs, rather than a full multi-agent system.",
        "doc_ids": ["lab06_agents_report.md"],
        "sections": ["## Тема, мета та варіант"],
        "spans": [s017],
        "tool_calls": ["search_knowledge"],
    })

    # 018: API Spec - Rate limiting parameters
    s018 = "- Rate limit: 60 requests per minute per IP for `/chat/completions`; 10 concurrent requests for `/documents/upload`."
    queries.append({
        "id": "final_v3_indep_018",
        "category": "paraphrase",
        "language": "uk",
        "query": "Які обмеження частоти запитів (rate limits) встановлені для генерації відповідей та завантаження документів?",
        "expected_answer": "Встановлено ліміт 60 запитів на хвилину на IP для /chat/completions та 10 одночасних запитів для /documents/upload.",
        "doc_ids": ["api_specification_and_contracts.md"],
        "sections": ["## Authentication & Headers"],
        "spans": [s018],
        "tool_calls": ["search_knowledge"],
    })

    # 019: ADR 001 - Sub-15ms vector query latency
    s019 = "- **Positive**: Extremely fast sub-15ms vector query latency; robust transaction isolation; trivial backup by copying SQLite `.db` file and Redis `.rdb` snapshot."
    queries.append({
        "id": "final_v3_indep_019",
        "category": "paraphrase",
        "language": "en",
        "query": "What query speed advantage is cited in ADR 001 for choosing Redis Stack as the vector database?",
        "expected_answer": "ADR 001 highlights sub-15ms vector query latency and robust transaction isolation.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Consequences"],
        "spans": [s019],
        "tool_calls": ["search_knowledge"],
    })

    # 020: Lab 4 - Resolution and batch size
    s020 = "Роздільність усіх PNG — 512×512, batch size 1. Safety checker збережено, жоден із 13 результатів ним не заблокований."
    queries.append({
        "id": "final_v3_indep_020",
        "category": "paraphrase",
        "language": "uk",
        "query": "Які базові параметри розміру зображень та розміру батчу використовувалися при генерації у Lab 4?",
        "expected_answer": "Роздільність становила 512×512 пікселів із розміром пакета 1, при збереженому safety checker.",
        "doc_ids": ["lab04_stable_diffusion_report.md"],
        "sections": ["## Модель і середовище"],
        "spans": [s020],
        "tool_calls": ["search_knowledge"],
    })

    # ----------------------------------------------------
    # 3. 10 CROSS-DOCUMENT QUERIES (final_v3_indep_021 .. 030)
    # ----------------------------------------------------
    # 021: ADR 001 + Incident Postmortem: Port 6380 vs maxmemory 512mb
    s021_a = "- **Negative**: Requires Redis Stack Docker container or service on port 6380 for live vector operations; fallback to in-memory/mock adapters is required for zero-dependency CI runs."
    s021_b = "- **14:22**: Redis memory usage crossed the configured `maxmemory 512mb` threshold."
    queries.append({
        "id": "final_v3_indep_021",
        "category": "cross_document",
        "language": "en",
        "query": "What port does ADR 001 specify for the Redis Stack service, and what memory limit was crossed during the ingestion incident?",
        "expected_answer": "ADR 001 specifies port 6380 for Redis Stack, and the incident postmortem identifies crossing the maxmemory 512mb threshold as the cause of failure.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md", "incident_postmortem_redis_oom.md"],
        "sections": ["## Consequences", "## Timeline (UTC)"],
        "spans": [s021_a, s021_b],
        "tool_calls": ["search_knowledge"],
    })

    # 022: Lab 3 + ADR 001: CSVLoader in Lab 3 vs SQLite WAL in Lab 7
    s022_a = "185 товарів завантажуються через LangChain `CSVLoader`."
    s022_b = "2. **SQLite with WAL (Write-Ahead Logging)** for application state and chat memory:"
    queries.append({
        "id": "final_v3_indep_022",
        "category": "cross_document",
        "language": "uk",
        "query": "Як завантажувалися дані каталогу в Lab 3 та яка реляційна база з режимом WAL була обрана в ADR 001 для Lab 7?",
        "expected_answer": "У Lab 3 дані завантажувалися з CSV через LangChain CSVLoader, а в Lab 7 для стану та пам'яті чату в ADR 001 обрано SQLite з режимом WAL.",
        "doc_ids": ["lab03_rag_report.md", "adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Реалізація", "## Decision"],
        "spans": [s022_a, s022_b],
        "tool_calls": ["search_knowledge"],
    })

    # 023: API Spec + Runbook: /health/ready contract vs runbook probe
    s023_a = "- `GET /health/ready`: Returns `{\"status\": \"ready\", \"redis\": true, \"database\": true}` (HTTP 200), or HTTP 503 if dependencies fail."
    s023_b = "- Ingress readiness probe: `curl -s http://127.0.0.1:8000/api/v1/health/ready`"
    queries.append({
        "id": "final_v3_indep_023",
        "category": "cross_document",
        "language": "en",
        "query": "Compare the response contract of /health/ready in the API specification with the curl readiness probe command in the runbook.",
        "expected_answer": "The API spec states /health/ready returns status ready with redis and database booleans, which the runbook checks using curl -s http://127.0.0.1:8000/api/v1/health/ready.",
        "doc_ids": ["api_specification_and_contracts.md", "rag_system_runbook.md"],
        "sections": ["## Core Endpoints / 1. Health Checks", "### Health Probes & Monitoring"],
        "spans": [s023_a, s023_b],
        "tool_calls": ["search_knowledge"],
    })

    # 024: Lab 4 + Lab 5: Lab 4 PNG count (13) vs Lab 5 baseline questions (3)
    s024_a = "13 унікальних генерацій: 1 baseline + 3 seeds + 2 steps + 2 CFG + 2 negative + 3 styles."
    s024_b = "Baseline: CORRECT=3, PARTIALLY_CORRECT=0, INCORRECT=0. Відповіді left, Brown, Yes відповідають видимим фактам."
    queries.append({
        "id": "final_v3_indep_024",
        "category": "cross_document",
        "language": "uk",
        "query": "Скільки унікальних генерацій було виконано в експериментах Lab 4 та скільки правильних відповідей отримав baseline LLaVA у Lab 5?",
        "expected_answer": "У Lab 4 виконано 13 унікальних генерацій, а baseline LLaVA у Lab 5 дав 3 правильні відповіді (CORRECT=3).",
        "doc_ids": ["lab04_stable_diffusion_report.md", "lab05_multimodal_report.md"],
        "sections": ["## Методика експерименту", "## 6. Роль компонентів LLaVA"],
        "spans": [s024_a, s024_b],
        "tool_calls": ["search_knowledge"],
    })

    # 025: Lab 6 + API Spec: Single agent vs chat completions endpoint
    s025_a = "Використано один агент, окремо створюється контрольний екземпляр без інструментів."
    s025_b = "- `POST /chat/completions`: Executes a query turn with grounded RAG and agentic verification."
    queries.append({
        "id": "final_v3_indep_025",
        "category": "cross_document",
        "language": "en",
        "query": "How is the agent architecture in Lab 6 structured compared to the main chat completions endpoint in the Copilot API specification?",
        "expected_answer": "Lab 6 uses a single agent architecture with an explicit no-tools control instance, whereas Copilot API exposes POST /chat/completions with grounded RAG and verification.",
        "doc_ids": ["lab06_agents_report.md", "api_specification_and_contracts.md"],
        "sections": ["## Тема, мета та варіант", "## Core Endpoints / 3. Agentic Chat & Retrieval"],
        "spans": [s025_a, s025_b],
        "tool_calls": ["search_knowledge"],
    })

    # 026: Incident Postmortem + Runbook: Redis degraded state
    s026_a = "- **Retrieval degradation**: 38 queries processed via lexical fallback without vector ranking."
    s026_b = "- If Redis fails, the system transitions to degraded state (BM25 lexical search only)."
    queries.append({
        "id": "final_v3_indep_026",
        "category": "cross_document",
        "language": "uk",
        "query": "Як описується деградація системи при відмові Redis у звіті про аварію та в експлуатаційному runbook?",
        "expected_answer": "Система переходить у деградований стан із лексичним пошуком BM25 без векторного ранжування (у постінциденті зафіксовано 38 таких запитів).",
        "doc_ids": ["incident_postmortem_redis_oom.md", "rag_system_runbook.md"],
        "sections": ["## Impact", "### Health Probes & Monitoring"],
        "spans": [s026_a, s026_b],
        "tool_calls": ["search_knowledge"],
    })

    # 027: Lab 3 + Lab 6: all-MiniLM-L6-v2 in Lab 3 vs openai-agents in Lab 6
    s027_a = "`all-MiniLM-L6-v2` формує нормалізовані 384-вимірні embeddings."
    s027_b = "Python 3.13.5; openai-agents 0.22.3; openai 3.17.0"
    queries.append({
        "id": "final_v3_indep_027",
        "category": "cross_document",
        "language": "en",
        "query": "What embedding model generates vector representations in Lab 3, and what agent framework orchestrates execution in Lab 6?",
        "expected_answer": "all-MiniLM-L6-v2 produces 384-dimensional embeddings in Lab 3, and openai-agents 0.22.3 orchestrates execution in Lab 6.",
        "doc_ids": ["lab03_rag_report.md", "lab06_agents_report.md"],
        "sections": ["## Реалізація", "## Середовище та модель"],
        "spans": [s027_a, s027_b],
        "tool_calls": ["search_knowledge"],
    })

    # 028: ADR 001 + API Spec: Embedding dimension in ADR 001 vs citations in API Spec
    s028_a = "- Cosine distance metric for normalized embedding vectors (`multilingual-e5-base`, dimension = 768)."
    s028_b = "- `citations`: list of `Citation` objects with source document, line/section, and quote"
    queries.append({
        "id": "final_v3_indep_028",
        "category": "cross_document",
        "language": "uk",
        "query": "Яка розмірність вектора моделі multilingual-e5-base наведена в ADR 001 та яка структура поля citations у специфікації API?",
        "expected_answer": "Розмірність становить 768 в ADR 001, а citations повертає список об'єктів Citation із назвою документа, рядком/секцією та цитатою.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md", "api_specification_and_contracts.md"],
        "sections": ["## Decision", "## Core Endpoints / 3. Agentic Chat & Retrieval"],
        "spans": [s028_a, s028_b],
        "tool_calls": ["search_knowledge"],
    })

    # 029: Lab 4 + Lab 6: 4 seeds in Lab 4 vs prompt changes in Lab 6
    s029_a = "| Seed | seed | 42, 123, 777, 2026 | prompt, negative, steps, CFG, model, scheduler, resolution |"
    s029_b = "Початкова серія — 11 запусків. Після виявлених помилок виконано ще 13 з уточненими загальними інструкціями та відновлюваними помилками інструментів."
    queries.append({
        "id": "final_v3_indep_029",
        "category": "cross_document",
        "language": "en",
        "query": "Which four integer seeds were evaluated for visual generation in Lab 4, and how many runs were in the initial and refined series in Lab 6?",
        "expected_answer": "Lab 4 tested seeds 42, 123, 777, and 2026, while Lab 6 performed an initial series of 11 runs followed by 13 refined runs.",
        "doc_ids": ["lab04_stable_diffusion_report.md", "lab06_agents_report.md"],
        "sections": ["## Методика експерименту", "## Результати експериментів"],
        "spans": [s029_a, s029_b],
        "tool_calls": ["search_knowledge"],
    })

    # 030: Lab 5 + Lab 3: Document creation in Lab 3 vs visual feature projection in Lab 5
    s030_a = "Один CSV-рядок стає одним `Document`, оскільки ціна, опис, характеристики та provenance мають залишатися однією логічною одиницею."
    s030_b = "Енкодер зображення утворює вектори ознак, проєктор узгоджує їх із простором мовних embeddings, tokenizer перетворює запитання на текстові токени."
    queries.append({
        "id": "final_v3_indep_030",
        "category": "cross_document",
        "language": "uk",
        "query": "Як обґрунтовано створення одного Document з одного CSV-рядка у Lab 3 та як обробляються візуальні ознаки зображення у Lab 5?",
        "expected_answer": "У Lab 3 один рядок стає одним Document для збереження ціни, опису та provenance разом, а у Lab 5 вектори ознак узгоджуються з простором мовних embeddings через проєктор.",
        "doc_ids": ["lab03_rag_report.md", "lab05_multimodal_report.md"],
        "sections": ["## Реалізація", "## 4. Теоретичні відомості"],
        "spans": [s030_a, s030_b],
        "tool_calls": ["search_knowledge"],
    })

    # ----------------------------------------------------
    # 4. 10 NUMERICAL REASONING QUERIES (final_v3_indep_031 .. 040)
    # ----------------------------------------------------
    # 031: Incident - 14 bulk uploads failed, 0 bytes corrupted
    s031_a = "- **Ingestion availability**: 0% during the incident window (14 bulk uploads failed)."
    s031_b = "- **Data corruption**: 0 bytes corrupted; transactional integrity was maintained by SQLite document status rollbacks."
    queries.append({
        "id": "final_v3_indep_031",
        "category": "numerical_reasoning",
        "language": "en",
        "query": "According to the post-mortem report, how many bulk upload jobs failed during the OOM incident, and how many bytes of data were corrupted?",
        "expected_answer": "14 bulk uploads failed, and exactly 0 bytes of data were corrupted.",
        "doc_ids": ["incident_postmortem_redis_oom.md"],
        "sections": ["## Impact"],
        "spans": [s031_a, s031_b],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 032: Incident - Duration in minutes from 14:20 to 14:47 UTC (27 minutes)
    s032_a = "- **14:20**: Ingestion job started for a 450-page technical specification PDF containing 12,400 chunks."
    s032_b = "- **14:47**: Restarted service; queued documents reindexed successfully in 84 seconds."
    queries.append({
        "id": "final_v3_indep_032",
        "category": "numerical_reasoning",
        "language": "uk",
        "query": "Обчисліть тривалість інциденту в хвилинах між початком завантаження о 14:20 UTC та перезапуском сервісу о 14:47 UTC.",
        "expected_answer": "Тривалість становить 27 хвилин (з 14:20 до 14:47 UTC).",
        "doc_ids": ["incident_postmortem_redis_oom.md"],
        "sections": ["## Timeline (UTC)"],
        "spans": [s032_a, s032_b],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 033: ADR 001 - Vector memory calculation: 1000 chunks * 768 dims * 4 bytes = 3,072,000 bytes
    s033 = "- Cosine distance metric for normalized embedding vectors (`multilingual-e5-base`, dimension = 768)."
    queries.append({
        "id": "final_v3_indep_033",
        "category": "numerical_reasoning",
        "language": "en",
        "query": "Using the 768-dimensional vector size from ADR 001 and 4-byte Float32 values, calculate the uncompressed vector memory required for 1,000 document chunks in bytes.",
        "expected_answer": "1000 chunks * 768 dimensions * 4 bytes = 3,072,000 bytes.",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Decision"],
        "spans": [s033],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 034: Lab 3 - 185 total minus 17 truncated = 168 untruncated records
    s034_a = "185 товарів завантажуються через LangChain `CSVLoader`."
    s034_b = "MiniLM обрізає 17 довгих записів до 256 word pieces."
    queries.append({
        "id": "final_v3_indep_034",
        "category": "numerical_reasoning",
        "language": "uk",
        "query": "Якщо каталог у Lab 3 містить 185 товарів, а MiniLM обрізає 17 довгих записів, скільки записів зберігають повну довжину без обрізання?",
        "expected_answer": "Без обрізання залишається 168 записів (185 - 17 = 168).",
        "doc_ids": ["lab03_rag_report.md"],
        "sections": ["## Реалізація", "## Обмеження і подальша робота"],
        "spans": [s034_a, s034_b],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 035: Lab 4 - Steps difference: 50 steps - 20 steps = 30 steps
    s035 = "| Steps | num_inference_steps | 20, 30, 50 | prompt, negative, seed, CFG, model, scheduler, resolution |"
    queries.append({
        "id": "final_v3_indep_035",
        "category": "numerical_reasoning",
        "language": "en",
        "query": "In Lab 4, the inference steps evaluated are 20, 30, and 50. What is the numerical difference between the maximum and minimum step counts?",
        "expected_answer": "The difference is 30 steps (50 - 20 = 30).",
        "doc_ids": ["lab04_stable_diffusion_report.md"],
        "sections": ["## Методика експерименту"],
        "spans": [s035],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 036: Lab 4 - CFG difference: 10 - 5 = 5
    s036 = "| CFG | guidance_scale | 5, 7.5, 10 | prompt, negative, seed, steps, model, scheduler, resolution |"
    queries.append({
        "id": "final_v3_indep_036",
        "category": "numerical_reasoning",
        "language": "uk",
        "query": "У Lab 4 досліджено значення guidance_scale 5, 7.5 та 10. Яка різниця між найбільшим та найменшим значенням шкали?",
        "expected_answer": "Різниця становить 5 (10 - 5 = 5).",
        "doc_ids": ["lab04_stable_diffusion_report.md"],
        "sections": ["## Методика експерименту"],
        "spans": [s036],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 037: Lab 5 - Max new tokens difference: 128 - 24 = 104
    s037 = "Ліміт: desc_01, greedy, 24/64/128."
    queries.append({
        "id": "final_v3_indep_037",
        "category": "numerical_reasoning",
        "language": "en",
        "query": "In Lab 5, token limits 24, 64, and 128 were evaluated for desc_01. What is the difference in tokens between the maximum (128) and minimum (24) limits?",
        "expected_answer": "The difference is 104 tokens (128 - 24 = 104).",
        "doc_ids": ["lab05_multimodal_report.md"],
        "sections": ["## 10. Організація VQA-експерименту"],
        "spans": [s037],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 038: Lab 6 - Runs difference: 13 runs - 11 runs = 2 runs
    s038 = "Початкова серія — 11 запусків. Після виявлених помилок виконано ще 13 з уточненими загальними інструкціями та відновлюваними помилками інструментів."
    queries.append({
        "id": "final_v3_indep_038",
        "category": "numerical_reasoning",
        "language": "uk",
        "query": "На скільки запусків друга серія (13 запусків) перевищувала початкову серію (11 запусків) в експериментах Lab 6?",
        "expected_answer": "Друга серія перевищувала початкову на 2 запуски (13 - 11 = 2).",
        "doc_ids": ["lab06_agents_report.md"],
        "sections": ["## Результати експериментів"],
        "spans": [s038],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 039: API Spec - Rate limit: 60 requests/min -> 1 req/sec
    s039 = "- Rate limit: 60 requests per minute per IP for `/chat/completions`; 10 concurrent requests for `/documents/upload`."
    queries.append({
        "id": "final_v3_indep_039",
        "category": "numerical_reasoning",
        "language": "en",
        "query": "If the rate limit is 60 requests per minute for chat completions in the API spec, how many requests are permitted per second on average?",
        "expected_answer": "60 requests / 60 seconds = 1 request per second.",
        "doc_ids": ["api_specification_and_contracts.md"],
        "sections": ["## Authentication & Headers"],
        "spans": [s039],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # 040: ADR 001 - busy_timeout 5000 ms to seconds = 5 s
    s040 = "- Enforces `PRAGMA busy_timeout = 5000` to prevent database locks under concurrent API accesses."
    queries.append({
        "id": "final_v3_indep_040",
        "category": "numerical_reasoning",
        "language": "uk",
        "query": "Скільки секунд становить параметр PRAGMA busy_timeout (5000 мс) для SQLite згідно з ADR 001?",
        "expected_answer": "5000 мс дорівнює 5 секундам (5000 / 1000 = 5 с).",
        "doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "sections": ["## Decision"],
        "spans": [s040],
        "tool_calls": ["calculate", "search_knowledge"],
    })

    # ----------------------------------------------------
    # 5. 10 UNSUPPORTED / ADVERSARIAL QUERIES (final_v3_indep_041 .. 050)
    # ----------------------------------------------------
    refusals = [
        {
            "id": "final_v3_indep_041",
            "category": "unsupported",
            "language": "en",
            "query": "What AWS S3 bucket name and IAM credentials are configured for storing raw knowledge base documents in the cloud?",
            "expected_answer": "The Knowledge Copilot system is designed for local-only deployment using SQLite and local storage, without any AWS S3 or cloud storage integration.",
        },
        {
            "id": "final_v3_indep_042",
            "category": "unsupported",
            "language": "uk",
            "query": "Яка версія трансформерної моделі Claude 3.5 Sonnet використовувалася для перекладу текстів у Lab 2?",
            "expected_answer": "У репозиторії відсутня інформація про Claude 3.5 Sonnet у Lab 2; лабораторні роботи використовують локальні моделі та відтворювальні скрипти без пропрієтарних хмарних API.",
        },
        {
            "id": "final_v3_indep_043",
            "category": "unsupported",
            "language": "en",
            "query": "How is the Pinecone Serverless vector collection configured for namespace partitioning in ADR 001?",
            "expected_answer": "ADR 001 rejected external cloud vector databases and chose Redis Stack locally; Pinecone is neither used nor configured.",
        },
        {
            "id": "final_v3_indep_044",
            "category": "unsupported",
            "language": "uk",
            "query": "Які параметри LoRA та частота кадрів використовувалися для генерації відео у Stable Video Diffusion згідно з Lab 4?",
            "expected_answer": "Lab 4 присвячена виключно генерації статичних зображень через Stable Diffusion v1.5; генерація відео або Stable Video Diffusion у звіті відсутні.",
        },
        {
            "id": "final_v3_indep_045",
            "category": "unsupported",
            "language": "en",
            "query": "What are the benchmark scores of Gemini 1.5 Pro on the MMMU reasoning benchmark reported in Lab 5?",
            "expected_answer": "Lab 5 focuses on local LLaVA-1.5-7B on consumer hardware and does not contain benchmark scores for Gemini 1.5 Pro.",
        },
        {
            "id": "final_v3_indep_046",
            "category": "unsupported",
            "language": "uk",
            "query": "Як налаштувати Helm chart для кластера Kubernetes з горизонтальним автоскейлінгом (HPA) в runbook Copilot?",
            "expected_answer": "Runbook системи описує локальне розгортання через Docker Compose та venv і не містить інструкцій чи Helm charts для Kubernetes.",
        },
        {
            "id": "final_v3_indep_047",
            "category": "unsupported",
            "language": "en",
            "query": "Which consensus protocol is configured for the Milvus distributed cluster in the vector storage architecture?",
            "expected_answer": "Milvus is not implemented in this repository; the architecture relies on Redis Stack and SQLite.",
        },
        {
            "id": "final_v3_indep_048",
            "category": "unsupported",
            "language": "uk",
            "query": "Яка вартість інференсу моделі GPT-4o в розрахунку на 1000 запитів наведена у фінансовому додатку до Lab 6?",
            "expected_answer": "У звіті Lab 6 немає фінансових розрахунків вартості інференсу чи тарифів GPT-4o.",
        },
        {
            "id": "final_v3_indep_049",
            "category": "unsupported",
            "language": "en",
            "query": "Which Apache Kafka bootstrap broker addresses and consumer group IDs are configured for streaming document events?",
            "expected_answer": "The Knowledge Copilot system does not use Apache Kafka or distributed streaming event brokers; ingestion is performed synchronously or via local worker processes.",
        },
        {
            "id": "final_v3_indep_050",
            "category": "unsupported",
            "language": "uk",
            "query": "Яка точність класифікації отримана для моделі Llama 3.1 70B при розборі каталогу товарів у Lab 3?",
            "expected_answer": "Lab 3 використовує не Llama 3.1 70B, а компактні локальні ембединги MiniLM для пошуку в каталозі.",
        },
    ]

    final_dataset = []

    print("--- Verifying 40 supported queries ---")
    for q in queries:
        gold_chunks = []
        for s in q["spans"]:
            matched = False
            for doc_name in q["doc_ids"]:
                try:
                    verify_span_in_doc(docs, s, doc_name)
                    cid = find_chunk_for_span(chunks, s, doc_name)
                    if cid not in gold_chunks:
                        gold_chunks.append(cid)
                    matched = True
                    break
                except ValueError:
                    continue
            if not matched:
                raise ValueError(f"Span failed verification in docs {q['doc_ids']}:\n  {s}")

        final_dataset.append({
            "id": q["id"],
            "category": q["category"],
            "language": q["language"],
            "query": q["query"],
            "expected_answer": q["expected_answer"],
            "relevant_doc_ids": q["doc_ids"],
            "target_sections": q["sections"],
            "evidence_spans": q["spans"],
            "gold_passage_ids": gold_chunks,
            "refusal_expected": False,
            "expected_tool_calls": q["tool_calls"],
        })

    print("--- Adding 10 refusal queries ---")
    for r in refusals:
        final_dataset.append({
            "id": r["id"],
            "category": r["category"],
            "language": r["language"],
            "query": r["query"],
            "expected_answer": r["expected_answer"],
            "relevant_doc_ids": [],
            "target_sections": [],
            "evidence_spans": [],
            "gold_passage_ids": [],
            "refusal_expected": True,
            "expected_tool_calls": [],
        })

    print(f"Total dataset queries: {len(final_dataset)}")

    # Audit duplicates
    clean, issues = audit_against_previous(final_dataset)
    if not clean:
        print(f"Audit failed with {len(issues)} issues:")
        for iss in issues[:10]:
            print(json.dumps(iss, indent=2))
        sys.exit(1)

    print("Duplicate audit: 100% CLEAN (Zero exact or near-duplicates)!")

    # Write independent holdout dataset to final_acceptance_holdout_v3_independent_50.json
    out_file = DATASETS_DIR / "final_acceptance_holdout_v3_independent_50.json"
    out_file.write_text(json.dumps(final_dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Successfully generated independent holdout dataset at: {out_file}")


if __name__ == "__main__":
    main()
