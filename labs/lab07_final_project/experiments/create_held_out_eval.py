"""Generate and freeze the 30-query held-out retrieval evaluation set."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from labs.lab07_final_project.src.config import Settings
from labs.lab07_final_project.src.knowledge_base import KnowledgeBase

kb = KnowledgeBase(Settings())
chunks_by_src = {}
for c in kb.chunks:
    chunks_by_src.setdefault(c.metadata["source_id"], []).append(c.page_content)

queries = [
    # 1. Exact model IDs
    {
        "query_id": "ho_01",
        "query": "What exact model checkpoint was used for Stable Diffusion in Lab 4?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["runwayml/stable-diffusion-v1-5"],
        "split": "held_out",
        "question_type": "exact_model_id",
        "notes": "Exact model identifier in Lab 4 report"
    },
    {
        "query_id": "ho_02",
        "query": "Який чекпоїнт мультимодальної моделі досліджувався у лабораторній роботі 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["llava-hf/llava-interleave-qwen-0.5b-hf"],
        "split": "held_out",
        "question_type": "exact_model_id",
        "notes": "Exact multimodal model in Ukrainian"
    },
    {
        "query_id": "ho_03",
        "query": "Which model was connected via LM Studio in Lab 6?",
        "expected_source_ids": ["lab06"],
        "expected_evidence": ["mistralai/ministral-3-3b"],
        "split": "held_out",
        "question_type": "exact_model_id",
        "notes": "Exact agent model in Lab 6"
    },
    {
        "query_id": "ho_04",
        "query": "Яка модель ембеддингів використовувалася в Лабораторній роботі 3?",
        "expected_source_ids": ["lab03"],
        "expected_evidence": ["all-MiniLM-L6-v2"],
        "split": "held_out",
        "question_type": "exact_model_id",
        "notes": "Embedding model in Lab 3 in Ukrainian"
    },

    # 2. Versions and revisions
    {
        "query_id": "ho_05",
        "query": "What Git commit hash or revision was specified for Stable Diffusion v1.5 in Lab 4?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["451f4fe16113bff5a5d2269ed5ad43b0592e9a14"],
        "split": "held_out",
        "question_type": "version_revision",
        "notes": "Exact Git revision hash"
    },
    {
        "query_id": "ho_06",
        "query": "Який локальний порт та endpoint використовувалися для LM Studio в лабораторній роботі №6?",
        "expected_source_ids": ["lab06"],
        "expected_evidence": ["http://localhost:1234/v1"],
        "split": "held_out",
        "question_type": "version_revision",
        "notes": "Local server endpoint URI"
    },

    # 3. Numerical counts
    {
        "query_id": "ho_07",
        "query": "How many unique generated outputs were produced in the Lab 4 experiment?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["13 унікальних генерацій"],
        "split": "held_out",
        "question_type": "numerical_count",
        "notes": "Exact generation count in Lab 4"
    },
    {
        "query_id": "ho_08",
        "query": "Скільки унікальних інференсів VQA було виконано в лабораторній роботі №5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["17 унікальних inference"],
        "split": "held_out",
        "question_type": "numerical_count",
        "notes": "Unique VQA inferences in Lab 5"
    },
    {
        "query_id": "ho_09",
        "query": "How many long documents or records were truncated by MiniLM in Lab 3?",
        "expected_source_ids": ["lab03"],
        "expected_evidence": ["17 довгих записів"],
        "split": "held_out",
        "question_type": "numerical_count",
        "notes": "Truncated records count in Lab 3"
    },
    {
        "query_id": "ho_10",
        "query": "Скільки рядків таблиці результатів містив звіт лабораторної роботи 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["разом 19 рядків"],
        "split": "held_out",
        "question_type": "numerical_count",
        "notes": "Table rows count vs unique inferences"
    },

    # 4. Architectural details
    {
        "query_id": "ho_11",
        "query": "Which vision encoder component extracts visual features in LLaVA according to Lab 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["SigLIP Vision"],
        "split": "held_out",
        "question_type": "architectural_details",
        "notes": "Vision backbone component"
    },
    {
        "query_id": "ho_12",
        "query": "Який планувальник (scheduler) використовувався в пайплайні Stable Diffusion у Лабораторній 4?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["PNDMScheduler"],
        "split": "held_out",
        "question_type": "architectural_details",
        "notes": "Diffusion scheduler class"
    },
    {
        "query_id": "ho_13",
        "query": "What mechanism provides session state and persistence in Lab 6 agents?",
        "expected_source_ids": ["lab06"],
        "expected_evidence": ["SQLiteSession"],
        "split": "held_out",
        "question_type": "architectural_details",
        "notes": "SQLite persistence backend"
    },
    {
        "query_id": "ho_14",
        "query": "Яка векторна база даних використовувалася для індексації знань у лабораторній роботі 3?",
        "expected_source_ids": ["lab03"],
        "expected_evidence": ["Redis"],
        "split": "held_out",
        "question_type": "architectural_details",
        "notes": "Vector store in Lab 3"
    },

    # 5. Comparisons and parameters
    {
        "query_id": "ho_15",
        "query": "What guidance scale or CFG parameter was investigated in Lab 4?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["guidance_scale"],
        "split": "held_out",
        "question_type": "parameters",
        "notes": "Classifier-free guidance parameter"
    },
    {
        "query_id": "ho_16",
        "query": "Які значення температури генерації аналізувалися для LLaVA в Лабораторній роботі 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["temperature"],
        "split": "held_out",
        "question_type": "parameters",
        "notes": "Sampling temperature parameter"
    },
    {
        "query_id": "ho_17",
        "query": "What values of max_new_tokens were tested for generation length in Lab 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["max_new_tokens"],
        "split": "held_out",
        "question_type": "parameters",
        "notes": "Token limit parameter"
    },

    # 6. Hallucination cases
    {
        "query_id": "ho_18",
        "query": "What invented bounding box coordinates or hallucinated data were output by LLaVA in Lab 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["[0.542,0.544,0.856,0.731]"],
        "split": "held_out",
        "question_type": "hallucination",
        "notes": "Invented bounding box coordinates"
    },
    {
        "query_id": "ho_19",
        "query": "Чому LLaVA згенерувала неіснуючі координати в експерименті контролю галюцинацій у Lab 5?",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["модель не визнала відсутність інформації"],
        "split": "held_out",
        "question_type": "hallucination",
        "notes": "Failure rationale for hallucination"
    },

    # 7. Negative results and failure analysis
    {
        "query_id": "ho_20",
        "query": "What behavioral flaw or failure mode occurred with auto tool selection in Lab 6?",
        "expected_source_ids": ["lab06"],
        "expected_evidence": ["автономного вибору"],
        "split": "held_out",
        "question_type": "failure_analysis",
        "notes": "Agent autonomous tool selection limitation"
    },
    {
        "query_id": "ho_21",
        "query": "Яке обмеження роздільної здатності зображення було зафіксовано для Stable Diffusion у Lab 4?",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["512×512"],
        "split": "held_out",
        "question_type": "negative_result",
        "notes": "Fixed spatial resolution limit"
    },
    {
        "query_id": "ho_22",
        "query": "What token truncation limit occurred for long texts with MiniLM in Lab 3?",
        "expected_source_ids": ["lab03"],
        "expected_evidence": ["256"],
        "split": "held_out",
        "question_type": "negative_result",
        "notes": "Token truncation ceiling"
    },

    # 8. Cross-document comparisons
    {
        "query_id": "ho_23",
        "query": "Compare the text-to-image pipeline in Lab 4 with the multimodal VQA pipeline in Lab 5.",
        "expected_source_ids": ["lab04", "lab05"],
        "expected_evidence": ["StableDiffusionPipeline", "LLaVA"],
        "split": "held_out",
        "question_type": "cross_document",
        "notes": "Comparing generative vision vs VQA"
    },
    {
        "query_id": "ho_24",
        "query": "Які апаратні обмеження GPU RTX 3050 Laptop спостерігалися в лабораторних роботах 4 та 5?",
        "expected_source_ids": ["lab04", "lab05"],
        "expected_evidence": ["4 ГБ", "RTX 3050"],
        "split": "held_out",
        "question_type": "cross_document",
        "notes": "Hardware VRAM ceiling across labs"
    },
    {
        "query_id": "ho_25",
        "query": "How do document tools in Lab 6 differ from Redis vector retrieval in Lab 3?",
        "expected_source_ids": ["lab03", "lab06"],
        "expected_evidence": ["Redis", "read_workspace_file"],
        "split": "held_out",
        "question_type": "cross_document",
        "notes": "Direct file read vs vector search"
    },

    # 9. Ukrainian / English paraphrases
    {
        "query_id": "ho_26",
        "query": "Вплив відʼємних підказок на якість генерації зображень у Лабораторній 4",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["Negative prompt"],
        "split": "held_out",
        "question_type": "paraphrase",
        "notes": "Ukrainian paraphrase of negative prompt"
    },
    {
        "query_id": "ho_27",
        "query": "Consistency of answers to rephrased questions in visual question answering",
        "expected_source_ids": ["lab05"],
        "expected_evidence": ["Узгодженість перефразованих питань"],
        "split": "held_out",
        "question_type": "paraphrase",
        "notes": "English paraphrase of consistency test"
    },
    {
        "query_id": "ho_28",
        "query": "Дослідження фіксованого зерна випадковості seed у генерації Stable Diffusion",
        "expected_source_ids": ["lab04"],
        "expected_evidence": ["seed"],
        "split": "held_out",
        "question_type": "paraphrase",
        "notes": "Ukrainian paraphrase of seed experiment"
    },

    # 10. Unsupported requests (grounded refusal)
    {
        "query_id": "ho_29",
        "query": "What was the ambient room temperature during the Lab 4 execution run?",
        "expected_source_ids": [],
        "expected_evidence": [],
        "split": "held_out",
        "question_type": "unsupported",
        "notes": "Unsupported environmental question"
    },
    {
        "query_id": "ho_30",
        "query": "Хто є автором зовнішнього датасету COCO для тестування Лабораторної 5?",
        "expected_source_ids": [],
        "expected_evidence": [],
        "split": "held_out",
        "question_type": "unsupported",
        "notes": "Unsupported external author question in Ukrainian"
    }
]

print(f"Total queries: {len(queries)}")
missing = 0
for q in queries:
    for ev in q["expected_evidence"]:
        srcs = q["expected_source_ids"]
        found = False
        for s in srcs:
            if any(ev.casefold() in text.casefold() for text in chunks_by_src.get(s, [])):
                found = True
                break
        if not found:
            print(f"MISMATCH in {q['query_id']}: evidence '{ev}' not found in sources {srcs}")
            missing += 1

if missing == 0:
    out_path = Path("labs/lab07_final_project/experiments/held_out_retrieval_queries.json")
    out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully saved {len(queries)} verified held-out queries to {out_path}")
else:
    print(f"Failed verification with {missing} missing evidence passages.")
