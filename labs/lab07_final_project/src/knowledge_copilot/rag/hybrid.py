"""Reciprocal Rank Fusion (RRF) for hybrid retrieval with section diversity."""
from typing import Dict, List, Optional, Tuple

from ..domain.enums import ScoreType
from ..domain.models import Chunk, EvidenceChunk, Score


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],  # (chunk, cosine_distance) sorted by distance asc
    lexical_results: List[Tuple[Chunk, float]],  # (chunk, bm25_score) sorted by score desc
    k: int = 5,
    rrf_k: int = 30,
    dense_weight: float = 1.5,
    lexical_weight: float = 1.0,
    max_per_section: int = 1,
    min_lexical_score: float = 2.0,
) -> List[EvidenceChunk]:
    """Combine dense and lexical rankings using RRF with section-level diversity."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    # Process dense ranking (1-based rank)
    for rank, (chunk, _) in enumerate(dense_results, 1):
        chunk_map[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (dense_weight / (rrf_k + rank))

    # Process lexical ranking (filter out noise below min_lexical_score if any matches exist)
    filtered_lexical = [
        (c, s) for c, s in lexical_results
        if s >= min_lexical_score or len(lexical_results) <= 1
    ]
    if not filtered_lexical and lexical_results:
        # If all lexical scores are below threshold, retain the top candidate
        filtered_lexical = lexical_results[:1]

    for rank, (chunk, _) in enumerate(filtered_lexical, 1):
        chunk_map[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (lexical_weight / (rrf_k + rank))

    # Sort descending by fused RRF score, breaking ties by chunk_id
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    evidence_chunks: List[EvidenceChunk] = []
    sec_counts: Dict[Tuple[str, str], int] = {}
    deferred: List[Tuple[str, float]] = []

    # First pass: enforce section diversity (prevent single section from crowding out others)
    for chunk_id, rrf_score in sorted_items:
        chunk = chunk_map[chunk_id]
        sec_key = (chunk.document_id, chunk.section)
        if sec_counts.get(sec_key, 0) < max_per_section:
            sec_counts[sec_key] = sec_counts.get(sec_key, 0) + 1
            rank = len(evidence_chunks) + 1
            score_obj = Score(
                retrieval_method="hybrid",
                score_type=ScoreType.RRF_SCORE,
                score_value=rrf_score,
                ranking_position=rank,
            )
            evidence_chunks.append(EvidenceChunk(chunk=chunk, score=score_obj))
            if len(evidence_chunks) >= k:
                break
        else:
            deferred.append((chunk_id, rrf_score))

    # Second pass: fill remainder from deferred items if fewer than k chunks available
    while len(evidence_chunks) < k and deferred:
        chunk_id, rrf_score = deferred.pop(0)
        chunk = chunk_map[chunk_id]
        rank = len(evidence_chunks) + 1
        score_obj = Score(
            retrieval_method="hybrid",
            score_type=ScoreType.RRF_SCORE,
            score_value=rrf_score,
            ranking_position=rank,
        )
        evidence_chunks.append(EvidenceChunk(chunk=chunk, score=score_obj))

    return evidence_chunks
