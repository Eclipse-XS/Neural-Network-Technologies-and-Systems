"""Unit tests for generation evaluation metrics and grounding faithfulness."""
import pytest
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, ScoreType
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.evaluation.eval_generation import (
    is_refusal_answer,
    check_numerical_match,
    compute_factual_correctness,
    verify_claim_evidence_support,
)


def make_evidence(doc_id: str, chunk_id: str, text: str) -> EvidenceChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        ordinal=0,
        text=text,
        section="Architecture",
        token_count=len(text.split()),
        sha256="test_sha256",
    )
    score = Score(
        retrieval_method="hybrid",
        score_type=ScoreType.RRF_SCORE,
        score_value=0.033,
        ranking_position=1,
    )
    return EvidenceChunk(chunk=chunk, score=score)


class TestRefusalDetection:
    """Validate comprehensive refusal detection across languages and grammatical inflections."""

    def test_ukrainian_morphology_and_genders(self):
        assert is_refusal_answer(
            "Інформація про порушення ПДР у документах не знайдена.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "У звіті не містяться відомості про погоду чи вологість повітря.",
            GroundingStatus.GROUNDED,
        )
        assert is_refusal_answer(
            "У даному документі не описано конкретного кулінарного рецепта для випікання хліба.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "В архітектурному документі відсутня інформація щодо BGP.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "У документах немає згадок про вологість повітря.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "Дані про умови навколишнього середовища в цьому документі не наведено.",
            GroundingStatus.VALIDATION_FAILED,
        )

    def test_english_abstention_patterns(self):
        assert is_refusal_answer(
            "The indexed documents do not provide specific guidance on the optimal batch size.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "The documentation does not cover Kubernetes Helm charts; it runs locally.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "No evidence was found in the indexed coursework reports.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "I cannot determine or answer this question based on the indexed documents.",
            GroundingStatus.VALIDATION_FAILED,
        )

    def test_affirmative_answers_not_refusals(self):
        assert not is_refusal_answer(
            "FastAPI backend works on port 8000 and Streamlit on port 8501.",
            GroundingStatus.GROUNDED,
        )
        assert not is_refusal_answer(
            "У лабораторній роботі №4 було згенеровано 17 унікальних зображень.",
            GroundingStatus.GROUNDED,
        )


class TestNumericalVerification:
    """Validate numeric derivation, formatting, scaling, and rounding."""

    def test_formatted_comma_numbers(self):
        assert check_numerical_match(
            "The difference is 364,031,264 parameters.",
            "The difference is 364,031,264 (864,031,264 - 500,000,000).",
        )
        assert check_numerical_match(
            "The total sum is 1,152.",
            "The sum of dimensions is 1152 (384 + 768).",
        )

    def test_unit_scales_and_conversions(self):
        # GiB to MiB (0.77 GiB vs 770 MiB)
        assert check_numerical_match(
            "The remaining free VRAM is 770 MiB (or 0.77 GiB).",
            "The used VRAM is 0.770 GiB (4.000 - 3.230).",
        )
        # GB to MB (1.5 GB vs 1536 MB)
        assert check_numerical_match(
            "Різниця становить 1536 МБ (2048 - 512).",
            "Різниця становить приблизно 1.5 GB (або 1536 MB при 2048 - 512).",
        )

    def test_rounding_tolerance(self):
        # 0.284 vs 0.2837
        assert check_numerical_match(
            "Різниця становить приблизно 0.284.",
            "The difference is 0.2837 (0.8571 - 0.5734).",
        )

    def test_derived_result_only(self):
        # Answer provides only the computed delta 501 without echoing 8501 and 8000
        assert check_numerical_match(
            "Різниця між портами становить 501.",
            "Різниця становить 501 (8501 - 8000).",
        )

    def test_adversarial_unit_mismatch(self):
        # Magnitude match with wrong unit category or wrong scaled unit MUST FAIL
        # 788.48 GiB is 1024 times larger than 0.77 GiB (expected 788.48 MiB)
        assert not check_numerical_match(
            "The remaining free VRAM is 788.48 GiB.",
            "The used VRAM is 0.770 GiB (4.000 - 3.230).",
        )
        # 5000 s is 1000 times larger than 5000 ms (expected 5 s)
        assert not check_numerical_match(
            "Latency is 5000 s.",
            "The latency threshold is 5000 ms.",
        )
        # Matching units with correct scale conversion MUST PASS
        assert check_numerical_match(
            "Latency is 5 s.",
            "The latency threshold is 5000 ms.",
        )
        assert check_numerical_match(
            "The remaining free VRAM is 788.48 MiB.",
            "The used VRAM is 0.770 GiB (4.000 - 3.230).",
        )


class TestFactualCorrectness:
    """Validate factual equivalence across languages and structured technical paraphrases."""

    def test_cross_lingual_equivalence(self):
        expected = "185 products were loaded via LangChain CSVLoader."
        evidence_spans = ["185 товарів завантажуються через LangChain `CSVLoader`."]
        answer = "В лабораторній роботі №3 через CSVLoader було завантажено 185 товарів (одне CSV-рядок стає одним документом)."
        assert compute_factual_correctness(answer, expected, evidence_spans)

    def test_structured_technical_paraphrase(self):
        expected = "Redis Stack (Search & JSON) was chosen for vector storage and SQLite with WAL for application state."
        evidence_spans = ["Redis Stack for vectors, SQLite with WAL for state."]
        answer = (
            "In **ADR 001**, the following databases were chosen:\n\n"
            "- **Vector database:** Redis Stack (with HNSW and FLAT vector indexing).\n"
            "- **Relational database:** SQLite with WAL (Write-Ahead Logging) for application state and chat memory."
        )
        assert compute_factual_correctness(answer, expected, evidence_spans)

    def test_detects_actual_contradiction(self):
        expected = "SQLite with WAL is used for chat memory persistence."
        answer = "SQLite with WAL is NOT used for chat memory persistence, it was rejected."
        assert not compute_factual_correctness(answer, expected, [])

    def test_intra_answer_contradiction_fails(self):
        expected = "FastAPI backend is used for serving HTTP endpoints."
        evidence_spans = ["FastAPI backend runs on port 8000 for serving HTTP endpoints."]
        # Answer first affirms and then negates the same proposition
        answer = (
            "FastAPI backend is used for serving HTTP endpoints on port 8000. "
            "However, FastAPI backend is not used for serving HTTP endpoints in this system."
        )
        assert not compute_factual_correctness(answer, expected, evidence_spans)

    def test_domain_technical_vocabulary_not_penalized(self):
        expected = "RRF combines BM25 ranking and dense cosine distance scores."
        evidence_spans = ["RRF combines reciprocal ranks from BM25 and vector search."]
        # Answer has rich Ukrainian domain explanation with terms not in any fixed vocabulary
        answer = (
            "Алгоритм реципрокного злиття рангів (Reciprocal Rank Fusion, RRF) обчислює "
            "гармонізовані скори для документів, комбінуючи розріджений пошук BM25 та векторний скоринг."
        )
        assert compute_factual_correctness(answer, expected, evidence_spans)

    def test_verbatim_span_with_wrong_numbers_fails(self):
        expected = "Used VRAM is 0.77 GiB."
        evidence_spans = ["Used VRAM is 0.77 GiB."]
        # Contains the span text partially but asserts an incorrect number (e.g. 50.0 GiB)
        answer = "Used VRAM is 50.0 GiB according to the test report."
        assert not compute_factual_correctness(answer, expected, evidence_spans)


class TestClaimEvidenceSupport:
    """Validate cumulative evidence grounding across multi-chunk contexts."""

    def test_multi_chunk_evidence_synthesis(self):
        ev1 = make_evidence(
            doc_id="adr_001",
            chunk_id="chunk-001",
            text="FastAPI backend runs on port 8000 for serving HTTP endpoints.",
        )
        ev2 = make_evidence(
            doc_id="adr_001",
            chunk_id="chunk-002",
            text="Streamlit user interface runs on port 8501 for user interaction.",
        )
        answer = "FastAPI backend runs on port 8000, while Streamlit interface operates on port 8501."
        assert verify_claim_evidence_support(answer, [ev1, ev2])

    def test_numerical_reasoning_claim_support(self):
        ev1 = make_evidence(
            doc_id="lab05",
            chunk_id="chunk-vram",
            text="RTX 3050 has total VRAM 4.000 GiB. Current allocation leaves 3.230 GiB available.",
        )
        answer = "The remaining free VRAM is 770 MiB (or 0.77 GiB), calculated as 4.000 minus 3.230 GiB."
        assert verify_claim_evidence_support(answer, [ev1])

    def test_adversarial_unsupported_citation(self):
        ev1 = make_evidence(
            doc_id="adr_001",
            chunk_id="chunk-001",
            text="FastAPI backend runs on port 8000 for serving HTTP endpoints.",
        )
        # Sentence cites chunk-001 for a claim that is completely unrelated to chunk-001
        answer = "Quantum computing algorithms accelerate cryptographic key exchange [adr_001#chunk-001]."
        is_supported, judgments = verify_claim_evidence_support(answer, [ev1], return_judgments=True)
        assert not is_supported
        assert any(j.get("verdict") == "UNSUPPORTED_CITATION" for j in judgments)


class TestHoldoutV3RegressionFixes:
    """Regression tests covering confirmed failures in v3 refusal, SI bytes, bilingual citations, and markdown/polarity."""

    def test_refusal_markdown_adverbs_and_hallucination_exclusion(self):
        # Confirmed valid refusals from v3_041, v3_045, v3_047, v3_048, v3_050
        assert is_refusal_answer(
            "The indexed documents do **not** explicitly mention an AWS S3 bucket name or IAM credentials.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "The indexed documents for **Lab 5** do not explicitly report benchmark scores for Gemini 1.5 Pro.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "The **Milvus distributed cluster** is **not explicitly configured with a consensus protocol**.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "Інформація про вартість GPT-4o не міститься у відомих документах.",
            GroundingStatus.VALIDATION_FAILED,
        )
        assert is_refusal_answer(
            "В звіті **Lab 3** для моделі Llama 3.1 70B не було прямо зазначено точності.",
            GroundingStatus.VALIDATION_FAILED,
        )
        # Confirmed premise substitutions / hallucinations (v3_042, v3_043) MUST remain False
        assert not is_refusal_answer(
            "В лабораторній роботі №5 для перекладу використовувалася модель Qwen/Qwen1.5-0.5B-Chat. "
            "Модель Claude 3.5 Sonnet не використовувалася.",
            GroundingStatus.PARTIALLY_SUPPORTED,
        )
        assert not is_refusal_answer(
            "The **Pinecone Serverless vector collection** in ADR 001 is configured with lab07:kb:* [adr_001#chunk-312].",
            GroundingStatus.GROUNDED,
        )

    def test_decimal_si_bytes_vs_wrong_clock_echo(self):
        # v3_033: 3.072 MB == 3,072,000 bytes (SI decimal prefix)
        assert check_numerical_match(
            "The uncompressed vector memory required for 1,000 chunks is 3.072 MB.",
            "1000 chunks * 768 dimensions * 4 bytes = 3,072,000 bytes.",
        )
        # v3_032: Echoing 14:20..14:47 with wrong duration 2h 42m MUST FAIL against 27 minutes
        assert not check_numerical_match(
            "Тривалість інциденту становить 2 години і 42 хвилини (Початок: 14:20 UTC, Перезапуск: 14:47 UTC).",
            "Тривалість становить 27 хвилин (з 14:20 до 14:47 UTC).",
        )

    def test_polarity_caveat_not_false_contradiction(self):
        # v3_023 & v3_026: 'which is not explicitly documented' and 'система не могла'
        exp23 = "The API spec states /health/ready returns status ready with redis and database booleans, which the runbook checks via curl."
        ans23 = "Both specify status, redis, and database flags. The runbook mentions a degraded state, which is not explicitly documented in the API spec."
        assert compute_factual_correctness(ans23, exp23, [])

        from labs.lab07_final_project.evaluation.eval_generation import _detect_polarity_contradiction
        exp26 = "Система переходить у деградований стан із лексичним пошуком BM25 без векторного ранжування."
        ans26 = "При перевищенні maxmemory система переходить на деградований режим пошуку BM25. Без належних настроєнь система не могла витримати навантаження."
        assert _detect_polarity_contradiction(ans26, exp26) is False
        assert compute_factual_correctness(
            ans26,
            exp26,
            ["lexical fallback without vector ranking", "degraded state (BM25 lexical search only)"],
        )

    def test_bilingual_citation_verbatim_error_and_markdown_headers(self):
        # v3_012 bilingual UK->EN + space thousands
        ev12 = make_evidence(
            "incident_postmortem_redis_oom",
            "chunk-ec91ddb2dece2958",
            "Ingestion job started for a 450-page technical specification PDF containing 12,400 chunks.",
        )
        ans12 = (
            "### **Розмір PDF-документа**\n"
            "Завантаження стосувалося великого технічного PDF-файлу обсягом 450 сторінок, "
            "який містив 12 400 чанків [incident_postmortem_redis_oom#chunk-ec91ddb2dece2958]."
        )
        ok12, jud12 = verify_claim_evidence_support(ans12, [ev12], return_judgments=True)
        assert ok12, f"Bilingual v3_012 failed: {jud12}"

        # v3_003 verbatim error quote with 'not allowed'
        ev03 = make_evidence(
            "incident_postmortem_redis_oom",
            "chunk-oom",
            "The pipeline failed with redis.exceptions.ResponseError: OOM command not allowed when used memory > 'maxmemory'.",
        )
        ans03 = "The ingestion pipeline raised ResponseError: OOM command not allowed when used memory > 'maxmemory' [incident_postmortem_redis_oom#chunk-oom]."
        ok03, jud03 = verify_claim_evidence_support(ans03, [ev03], return_judgments=True)
        assert ok03, f"Verbatim quote v3_003 failed: {jud03}"

        # v3_021 rejected alternative matching evidence
        ev21 = make_evidence(
            "adr_001_vector_and_storage_architecture",
            "chunk-rej",
            "Alternatives Considered: Milvus / Qdrant rejected due to high memory overhead (>=2 GB idle RAM).",
        )
        ans21 = "Milvus and Qdrant were rejected because of high memory overhead (>=2 GB idle RAM) [adr_001_vector_and_storage_architecture#chunk-rej]."
        ok21, jud21 = verify_claim_evidence_support(ans21, [ev21], return_judgments=True)
        assert ok21, f"Rejected alt v3_021 failed: {jud21}"

        # Premise hallucination in citation must still FAIL claim verification
        ev_redis = make_evidence("adr_001", "chunk-r", "Redis Stack uses namespace isolation lab07:kb:*.")
        ok_pine, jud_pine = verify_claim_evidence_support(
            "Pinecone Serverless uses namespace isolation lab07:kb:* [adr_001#chunk-r].",
            [ev_redis],
            return_judgments=True,
        )
        assert not ok_pine
        assert any(j.get("verdict") == "UNSUPPORTED_CITATION" for j in jud_pine)

    def test_recovered_calc_provenance_hhmm_and_ungrounded_rejection(self):
        ev_post = make_evidence(
            "incident_postmortem_redis_oom",
            "chunk-oom-ts",
            "Timeline: 14:20 UTC ingestion started; 14:47 UTC service restarted.",
        )
        ev_noise = make_evidence(
            "adr_001",
            "chunk-noise",
            "FastAPI runs on port 8000.",
        )
        ans_ok = (
            "Тривалість інциденту становить 27 хвилин.\n\n"
            "[Розраховано: 887 - 860 = 27] [incident_postmortem_redis_oom#chunk-oom-ts]"
        )
        ok, jud_ok = verify_claim_evidence_support(ans_ok, [ev_post, ev_noise], return_judgments=True)
        assert ok, f"Valid recovered HH:MM provenance failed: {jud_ok}"

        ans_bad_cite = "[Розраховано: 887 - 860 = 27] [adr_001#chunk-noise]"
        ok_bad, jud_bad = verify_claim_evidence_support(ans_bad_cite, [ev_post, ev_noise], return_judgments=True)
        assert not ok_bad
        assert any(j.get("verdict") == "UNSUPPORTED_CITATION" for j in jud_bad)

        ans_wrong_num = "Тривалість інциденту становить 99 хвилин."
        ok_wrong_num, jud_wrong_num = verify_claim_evidence_support(ans_wrong_num, [ev_post], return_judgments=True)
        assert not ok_wrong_num
        assert any(j.get("verdict") == "INSUFFICIENT_EVIDENCE" for j in jud_wrong_num)
