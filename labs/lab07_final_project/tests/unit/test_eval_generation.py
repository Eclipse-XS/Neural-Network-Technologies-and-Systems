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
