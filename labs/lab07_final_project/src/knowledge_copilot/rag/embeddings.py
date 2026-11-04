"""Local embedding model adapter with prefix handling and unit normalization."""
from typing import List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings


class CopilotEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.is_e5 = "e5" in model_name.casefold()
        
        from langchain_huggingface import HuggingFaceEmbeddings
        self._hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "revision": revision, "local_files_only": local_files_only},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        self.tokenizer = self._hf_embeddings._client.tokenizer
        self.max_seq_length = self._hf_embeddings._client.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, truncation=False))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts] if self.is_e5 else texts
        vectors = self._hf_embeddings.embed_documents(prefixed)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}" if self.is_e5 else text
        vector = self._hf_embeddings.embed_query(prefixed)
        return vector

    def probe_dimension(self) -> int:
        vec = np.asarray(self.embed_query("dimension probe"), dtype=np.float32)
        if vec.ndim != 1 or not np.isfinite(vec).all() or not np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3):
            raise ValueError("Probe embedding failed normalization or finite check")
        return len(vec)
