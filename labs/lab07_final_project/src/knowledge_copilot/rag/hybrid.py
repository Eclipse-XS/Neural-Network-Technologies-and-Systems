"""Reciprocal Rank Fusion (RRF) for hybrid retrieval."""
from typing import Dict, List, Tuple
from ..domain.enums import ScoreType
from ..domain.models import Chunk, EvidenceChunk, Score


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],  # (chunk, cosine_distance) sorted by distance asc
    lexical_results: List[Tuple[Chunk, float]],  # (chunk, bm25_score) sorted by score desc
    k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> List[EvidenceChunk]:
    """Combine dense and lexical rankings using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}
    
    # Process dense ranking (1-based rank)
    for rank, (chunk, _) in enumerate(dense_results, 1):
        chunk_map[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (dense_weight / (rrf_k + rank))
        
    # Process lexical ranking (1-based rank)
    for rank, (chunk, _) in enumerate(lexical_results, 1):
        chunk_map[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (lexical_weight / (rrf_k + rank))
        
    # Sort descending by RRF score
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    
    evidence_chunks: List[EvidenceChunk] = []
    for rank, (chunk_id, rrf_score) in enumerate(sorted_items[:k], 1):
        chunk = chunk_map[chunk_id]
        score_obj = Score(
            retrieval_method="hybrid",
            score_type=ScoreType.RRF_SCORE,
            score_value=rrf_score,
            ranking_position=rank,
        )
        evidence_chunks.append(EvidenceChunk(chunk=chunk, score=score_obj))
        
    return evidence_chunks
