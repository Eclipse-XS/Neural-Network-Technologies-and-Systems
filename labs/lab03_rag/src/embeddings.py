"""CPU MiniLM with content-addressed local cache, independent of Redis."""
import hashlib
import json
from pathlib import Path
import numpy as np
from langchain_core.embeddings import Embeddings
from .config import MODEL_NAME, MODEL_REVISION


class CachedEmbeddings(Embeddings):
    def __init__(self, backend, directory):
        self.backend = backend
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def embed_documents(self, texts):
        paths = [self.directory / (hashlib.sha256(
            (MODEL_NAME + MODEL_REVISION + "cpu:normalized:v1" + text).encode()).hexdigest() + ".npy")
                 for text in texts]
        missing = {p: text for p, text in zip(paths, texts) if not p.exists()}
        if missing:
            vectors = self.backend.embed_documents(list(missing.values()))
            if len(vectors) != len(missing):
                raise ValueError("Embedding count mismatch")
            for path, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=np.float32)
                validate_vectors([vector])
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, vector, allow_pickle=False)
                temporary.replace(path)
        vectors = [np.load(path, allow_pickle=False).tolist() for path in paths]
        if vectors:
            validate_vectors(vectors)
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def validate_vectors(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 384 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite (n,384) embeddings; got {array.shape}")
    if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4):
        raise ValueError("Expected unit-normalized embeddings")
    return dict(shape=list(array.shape), finite=True, normalized=True)


def create_embeddings(cache_dir, *, local_files_only=True):
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        backend = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs=dict(device="cpu", revision=MODEL_REVISION,
                              local_files_only=local_files_only),
            encode_kwargs=dict(normalize_embeddings=True, batch_size=16),
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("MiniLM unavailable. In the embeddings cell use local_files_only=False "
                           "once to download the specified small embedding model; no LLM download.") from exc
    return CachedEmbeddings(backend, cache_dir)


def embedding_audit(embeddings, documents):
    """Measures real token lengths; never silently claims full embedding coverage."""
    client = embeddings.backend._client
    lengths = [len(client.tokenizer.encode(d.page_content, truncation=False)) for d in documents]
    limit = client.max_seq_length
    return dict(model=MODEL_NAME, revision=MODEL_REVISION, device="cpu", dimension=384,
                normalize_embeddings=True, max_seq_length=limit,
                token_lengths=lengths, truncated_rows=sum(n > limit for n in lengths),
                sanity=validate_vectors(embeddings.embed_documents([d.page_content for d in documents[:3]])))
