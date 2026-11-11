"""Dedicated Refusal & Abstention Quality Evaluation Suite (Section 4.4).

Evaluates the copilot's epistemic calibration and refusal discipline across 5 explicit metrics:
1. true_refusal_rate: Correct refusal on genuinely unanswerable / out-of-corpus queries.
2. false_refusal_rate: False refusal (over-abstention) on answerable in-corpus queries.
3. unsupported_confident_answer_rate: Hallucinated confident answers on unanswerable queries.
4. grounded_partial_answer_rate: Grounded answers with explicit caveats when only partial evidence exists.
5. adversarial_refusal_rate: Refusal quality under misleading premises or adversarial prompt injection.

Can evaluate:
- Live execution traces from `generation_results.json` (when LM Studio is active), AND
- Deterministic verifier & refusal-policy test cases (offline CI / unit-testable verification).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.src.knowledge_copilot.agents.verifier import (
    ABSTENTION_PATTERNS,
    verify_grounding,
)
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, ScoreType
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.evaluation.eval_generation import audit_citations, is_refusal_answer

PARTIAL_CAVEAT_PATTERNS = [
    r"\b(?:uncertainty|caveat|note|however|only|partially|does not explicitly|not explicitly|clarify)\b",
    r"\b(?:проте|однак|частково|не зазначено прямо|уточн\w*|лише)\b",
]

ADVERSARIAL_QUERY_MARKERS = [
    r"gpt-4",
    r"imagenet",
    r"cassandra",
    r"bgp",
    r"as65001",
    r"helm",
    r"eks",
    r"kubernetes",
    r"hpa",
    r"pinecone",
    r"milvus",
    r"weaviate",
    r"dynamodb",
    r"elasticsearch",
    r"claude",
    r"gemini",
    r"llama",
    r"deepseek",
    r"kafka",
    r"aws s3",
    r"stable video diffusion",
    r"tpu v4",
    r"bert",
    r"sagemaker",
    r"tensorrt",
    r"ray\s+serve",
    r"triton",
    r"terraform",
    r"vault",
    r"oauth",
    r"rabbitmq",
    r"faiss",
    r"vllm",
    r"opentelemetry",
    r"grafana",
    r"datadog",
    r"celery",
    r"mongodb",
    r"snowflake",
    r"bigquery",
    r"ignore previous",
    r"system prompt",
    r"рецепт|хліб",
    r"світлофор|дорожнього руху",
    r"погода|вологість",
    r"футбольн",
    r"xyz-123",
]


def has_partial_caveat(answer: str) -> bool:
    a_lower = (answer or "").casefold()
    return any(re.search(pat, a_lower) for pat in PARTIAL_CAVEAT_PATTERNS)


def is_adversarial_unanswerable(query: str) -> bool:
    q_lower = (query or "").casefold()
    return any(re.search(pat, q_lower) for pat in ADVERSARIAL_QUERY_MARKERS)


def compute_refusal_metrics_from_items(
    itemized_results: List[Dict[str, Any]],
    dataset_by_id: Dict[str, Dict[str, Any]],
    live_scenarios: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the 5 refusal metrics from itemized generation turns and live acceptance scenarios."""
    unanswerable_total = 0
    true_refusals = 0
    unsupported_confident_answers = 0

    answerable_total = 0
    false_refusals = 0

    partial_candidates_total = 0
    grounded_partial_answers = 0

    adversarial_total = 0
    adversarial_refusals = 0

    case_breakdown: List[Dict[str, Any]] = []

    for item in itemized_results:
        qid = item.get("query_id") or item.get("id", "")
        meta = dataset_by_id.get(qid, {})
        refusal_expected = bool(meta.get("refusal_expected", False) or item.get("category") in ("unsupported", "unsupported_refusal"))
        if not item.get("success", False):
            if refusal_expected:
                unanswerable_total += 1
            else:
                answerable_total += 1
            case_breakdown.append({
                "query_id": qid,
                "role": "unanswerable" if refusal_expected else "answerable",
                "classification": "EXECUTION_FAILED",
                "grounding_status": "EXECUTION_FAILED",
            })
            continue
        query_text = item.get("query", meta.get("query", ""))
        answer_text = item.get("answer", "")
        g_status_str = item.get("grounding_status", "GROUNDED")
        try:
            g_status = GroundingStatus(g_status_str)
        except ValueError:
            g_status = GroundingStatus.GROUNDED

        refused = is_refusal_answer(answer_text, g_status)
        citations = item.get("citations", [])

        if refusal_expected:
            unanswerable_total += 1
            if refused:
                true_refusals += 1
            else:
                unsupported_confident_answers += 1

            if is_adversarial_unanswerable(query_text):
                adversarial_total += 1
                if refused:
                    adversarial_refusals += 1

            classification = (
                "TRUE_REFUSAL"
                if refused
                else ("PARTIAL_PREMISE_SUBSTITUTION" if citations else "UNSUPPORTED_CONFIDENT_ANSWER")
            )
            case_breakdown.append({
                "query_id": qid,
                "role": "unanswerable",
                "classification": classification,
                "is_adversarial": is_adversarial_unanswerable(query_text),
                "refused_correctly": refused,
                "unsupported_confident_answer": not refused,
                "grounding_status": g_status.value,
                "grounding_status_aligned": refused == (g_status == GroundingStatus.INSUFFICIENT_EVIDENCE),
            })
        else:
            answerable_total += 1
            # A turn is a false refusal if it abstains WITHOUT citing valid evidence
            is_false_refusal = (g_status == GroundingStatus.INSUFFICIENT_EVIDENCE) or (refused and len(citations) == 0)
            if is_false_refusal:
                false_refusals += 1

            # Check grounded partial answer: cross-document or nuanced queries where answer provides citations + caveat
            is_partial_cand = item.get("category") == "cross_document" or has_partial_caveat(answer_text)
            if is_partial_cand:
                partial_candidates_total += 1
                if not is_false_refusal and len(citations) > 0:
                    grounded_partial_answers += 1

            if is_false_refusal:
                ans_cls = "FALSE_REFUSAL"
            elif is_partial_cand and len(citations) > 0:
                ans_cls = "GROUNDED_PARTIAL_ANSWER"
            elif len(citations) > 0:
                ans_cls = "ANSWERED_GROUNDED"
            else:
                ans_cls = "UNGROUNDED_ANSWER"

            case_breakdown.append({
                "query_id": qid,
                "role": "answerable",
                "classification": ans_cls,
                "false_refusal": is_false_refusal,
                "has_citations": len(citations) > 0,
                "grounding_status": g_status.value,
            })

    # Also incorporate live acceptance scenarios if provided
    if live_scenarios:
        for sc_key in ("scenario_1_chat_and_citations", "scenario_2_numerical_calculator", "scenario_3_cross_document_synthesis"):
            sc = live_scenarios.get(sc_key)
            if sc and sc.get("answer"):
                answerable_total += 1
                try:
                    sc_status = GroundingStatus(sc.get("grounding_status", "GROUNDED"))
                except ValueError:
                    sc_status = GroundingStatus.GROUNDED
                if sc_status == GroundingStatus.INSUFFICIENT_EVIDENCE:
                    false_refusals += 1
                if sc_key == "scenario_3_cross_document_synthesis":
                    partial_candidates_total += 1
                    if len(sc.get("citations", [])) > 0 and sc_status == GroundingStatus.GROUNDED:
                        grounded_partial_answers += 1
        s4 = live_scenarios.get("scenario_4_unsupported_refusal")
        if s4 and s4.get("answer"):
            unanswerable_total += 1
            adversarial_total += 1
            try:
                s4_status = GroundingStatus(s4.get("grounding_status", "INSUFFICIENT_EVIDENCE"))
            except ValueError:
                s4_status = GroundingStatus.INSUFFICIENT_EVIDENCE
            s4_ref = is_refusal_answer(s4["answer"], s4_status)
            if s4_ref and s4.get("fabricated_citations_count", 0) == 0:
                true_refusals += 1
                adversarial_refusals += 1
            else:
                unsupported_confident_answers += 1

    return {
        "true_refusal_rate": {
            "numerator": true_refusals,
            "denominator": unanswerable_total,
            "value": round(true_refusals / unanswerable_total, 4) if unanswerable_total else None,
            "target": ">= 0.90",
            "passed": (true_refusals / unanswerable_total >= 0.90) if unanswerable_total else False,
        },
        "false_refusal_rate": {
            "numerator": false_refusals,
            "denominator": answerable_total,
            "value": round(false_refusals / answerable_total, 4) if answerable_total else None,
            "target": "<= 0.10",
            "passed": (false_refusals / answerable_total <= 0.10) if answerable_total else False,
        },
        "unsupported_confident_answer_rate": {
            "numerator": unsupported_confident_answers,
            "denominator": unanswerable_total,
            "value": round(unsupported_confident_answers / unanswerable_total, 4) if unanswerable_total else None,
            "target": "<= 0.10",
            "passed": (unsupported_confident_answers / unanswerable_total <= 0.10) if unanswerable_total else False,
        },
        "grounded_partial_answer_rate": {
            "numerator": grounded_partial_answers,
            "denominator": partial_candidates_total,
            "value": round(grounded_partial_answers / partial_candidates_total, 4) if partial_candidates_total else None,
            "target": ">= 0.80",
            "passed": (grounded_partial_answers / partial_candidates_total >= 0.80) if partial_candidates_total else False,
        },
        "adversarial_refusal_rate": {
            "numerator": adversarial_refusals,
            "denominator": adversarial_total,
            "value": round(adversarial_refusals / adversarial_total, 4) if adversarial_total else None,
            "target": ">= 0.90",
            "passed": (adversarial_refusals / adversarial_total >= 0.90) if adversarial_total else False,
        },
        "case_breakdown": case_breakdown,
    }


def run_deterministic_verifier_calibration_suite() -> Dict[str, Any]:
    """Run deterministic offline calibration tests on `verify_grounding` across all 5 refusal dimensions."""
    chunk_adr = EvidenceChunk(
        chunk=Chunk(
            chunk_id="chunk-312de896eb918af4",
            document_id="adr_001_vector_and_storage_architecture",
            ordinal=0,
            text="Redis Stack (Search & JSON) was chosen as the vector database and SQLite with WAL for application state.",
            section="## Decision",
            token_count=22,
            sha256="h_adr",
            metadata={"filename": "adr_001_vector_and_storage_architecture.md"},
        ),
        score=Score(retrieval_method="hybrid", score_type=ScoreType.RRF_SCORE, score_value=0.09, ranking_position=1),
    )
    chunk_lab5 = EvidenceChunk(
        chunk=Chunk(
            chunk_id="chunk-31c0bf0fd8403bf9",
            document_id="lab05_multimodal_report",
            ordinal=1,
            text="LLaVA combines SigLIP Vision, MLP projection, and Qwen2 language model.",
            section="## Architecture",
            token_count=16,
            sha256="h_lab5",
            metadata={"filename": "lab05_multimodal_report.md"},
        ),
        score=Score(retrieval_method="hybrid", score_type=ScoreType.RRF_SCORE, score_value=0.08, ranking_position=2),
    )

    # 1. Unanswerable queries -> should produce INSUFFICIENT_EVIDENCE (not VALIDATION_FAILED)
    unanswerable_cases = [
        "The indexed documents for Lab 6 do not contain specific benchmark results for GPT-4 vision fine-tuning on ImageNet.",
        "У проіндексованих матеріалах відсутня інформація про погодні умови або вологість повітря під час виконання Lab 4.",
        "The indexed technical documentation does not contain information on Apache Cassandra cluster configuration.",
        "В архітектурному документі ADR 001 не згадується налаштування мережевого протоколу BGP для AS65001.",
        "I cannot answer this question because the indexed coursework reports do not cover Google TPU v4 pods.",
    ]
    unanswerable_statuses = [verify_grounding(ans, [chunk_adr])[0] for ans in unanswerable_cases]
    true_refusals = sum(1 for st in unanswerable_statuses if st == GroundingStatus.INSUFFICIENT_EVIDENCE)

    # 2. Answerable grounded queries -> should produce GROUNDED (not false refusal or VALIDATION_FAILED)
    answerable_cases = [
        "The vector database chosen in ADR 001 is Redis Stack with SQLite WAL ([adr_001_vector_and_storage_architecture#chunk-312de896eb918af4]).",
        "In Lab 5, LLaVA uses:\n- SigLIP Vision for feature extraction,\n- MLP projection,\n- Qwen2 for response generation ([lab05_multimodal_report#chunk-31c0bf0fd8403bf9]).",
    ]
    answerable_statuses = [verify_grounding(ans, [chunk_adr, chunk_lab5])[0] for ans in answerable_cases]
    false_refusals = sum(1 for st in answerable_statuses if st == GroundingStatus.INSUFFICIENT_EVIDENCE)

    # 3. Unsupported confident answer (hallucinated citation or uncited fabrication) -> caught as VALIDATION_FAILED
    hallucinated_cases = [
        "GPT-4 achieved 94.2% top-1 accuracy on ImageNet in Lab 6 ([lab06_agents_report#chunk-fake999999]).",
        "Apache Cassandra is configured with 3 replicas across AWS regions in ADR 001.",
    ]
    hallucinated_statuses = [verify_grounding(ans, [chunk_adr])[0] for ans in hallucinated_cases]
    caught_hallucinations = sum(1 for st in hallucinated_statuses if st == GroundingStatus.VALIDATION_FAILED)

    # 4. Partial answer with grounded citation + explicit uncertainty caveat -> GROUNDED
    partial_answer = (
        "In **Lab 5**, the model tested was based on **LLaVA components**:\n"
        "- **SigLIP Vision** extracts features,\n"
        "- **Qwen2** generates responses ([lab05_multimodal_report#chunk-31c0bf0fd8403bf9]).\n\n"
        "**Uncertainty**: Lab 6 does not explicitly list a new vision model evaluation."
    )
    partial_status, partial_cites, _, _ = verify_grounding(partial_answer, [chunk_lab5])

    return {
        "unanswerable_refusal_accuracy": true_refusals / len(unanswerable_cases),
        "answerable_false_refusal_rate": false_refusals / len(answerable_cases),
        "hallucination_interception_rate": caught_hallucinations / len(hallucinated_cases),
        "grounded_partial_answer_verified": partial_status == GroundingStatus.GROUNDED and len(partial_cites) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refusal & Epistemic Calibration Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="labs/lab07_final_project/evaluation/datasets/final_acceptance_holdout_50.json",
    )
    parser.add_argument(
        "--generation-results",
        type=str,
        default="labs/lab07_final_project/evaluation/baselines/generation_results.json",
    )
    parser.add_argument(
        "--live-acceptance",
        type=str,
        default="labs/lab07_final_project/evaluation/baselines/live_acceptance_evidence.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labs/lab07_final_project/evaluation/baselines/refusal_evaluation_results.json",
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset_by_id = {item["id"]: item for item in dataset}

    gen_path = Path(args.generation_results)
    live_path = Path(args.live_acceptance)

    itemized = []
    gen_status = "NOT_VERIFIED"
    if gen_path.exists():
        gen_data = json.loads(gen_path.read_text(encoding="utf-8"))
        gen_status = gen_data.get("status", "NOT_VERIFIED")
        itemized = gen_data.get("itemized_results", [])

    live_scenarios = None
    if live_path.exists():
        live_data = json.loads(live_path.read_text(encoding="utf-8"))
        live_scenarios = live_data.get("scenarios", {})

    live_metrics = compute_refusal_metrics_from_items(itemized, dataset_by_id, live_scenarios)
    verifier_suite = run_deterministic_verifier_calibration_suite()

    output_payload = {
        "evaluation_type": "refusal_and_epistemic_calibration",
        "generation_source_status": gen_status,
        "deterministic_verifier_calibration": verifier_suite,
        "live_refusal_metrics": {k: v for k, v in live_metrics.items() if k != "case_breakdown"},
        "case_breakdown": live_metrics["case_breakdown"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved refusal evaluation results to {out_path}")
    print(json.dumps(output_payload["live_refusal_metrics"], indent=2))


if __name__ == "__main__":
    main()
