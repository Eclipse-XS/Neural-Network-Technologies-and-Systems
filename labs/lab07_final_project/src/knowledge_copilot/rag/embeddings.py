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
        offline_fallback: bool = False,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.is_e5 = "e5" in model_name.casefold()
        self.dim = 384
        self._offline = False
        self._hf_embeddings = None
        self.tokenizer = None
        self.max_seq_length = 512

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device, "revision": revision, "local_files_only": local_files_only},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            )
            self.tokenizer = self._hf_embeddings._client.tokenizer
            self.max_seq_length = self._hf_embeddings._client.max_seq_length
        except Exception as exc:
            if not offline_fallback:
                raise RuntimeError(
                    f"Configured embedding model '{model_name}' (rev {revision}) could not be loaded: {exc}. "
                    f"Silent synthetic fallback is disabled in production and benchmark modes."
                ) from exc
            self._offline = True

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, truncation=False))
        return max(1, int(len(text.split()) * 1.3))

    def _mock_vector(self, text: str) -> List[float]:
        import hashlib
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rng = np.random.RandomState(h % (2**32))
        vec = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._offline or self._hf_embeddings is None:
            return [self._mock_vector(t) for t in texts]
        prefixed = [f"passage: {t}" for t in texts] if self.is_e5 else texts
        return self._hf_embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        if self._offline or self._hf_embeddings is None:
            return self._mock_vector(text)
        prefixed = f"query: {text}" if self.is_e5 else text
        return self._hf_embeddings.embed_query(prefixed)

    def probe_dimension(self) -> int:
        if self._offline or self._hf_embeddings is None:
            return self.dim
        vec = np.asarray(self.embed_query("dimension probe"), dtype=np.float32)
        if vec.ndim != 1 or not np.isfinite(vec).all() or not np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3):
            raise ValueError("Probe embedding failed normalization or finite check")
        return len(vec)
