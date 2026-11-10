"""Gate C4 / P0-5: Verification of embedding model integrity and fail-fast behavior."""
import pytest
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings


def test_missing_model_fails_fast_in_production_mode():
    """Embedding model must raise RuntimeError if model cannot be loaded when offline_fallback=False (default)."""
    with pytest.raises(RuntimeError, match="Silent synthetic fallback is disabled"):
        CopilotEmbeddings(
            model_name="nonexistent/completely-fake-embedding-model-999",
            offline_fallback=False,
        )


def test_offline_fallback_only_enabled_when_explicitly_configured():
    """Offline synthetic embeddings are permitted exclusively when explicitly set to True."""
    emb = CopilotEmbeddings(
        model_name="nonexistent/completely-fake-embedding-model-999",
        offline_fallback=True,
    )
    assert emb._offline is True
    vecs = emb.embed_documents(["Test passage text"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
