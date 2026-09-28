"""LangChain Redis VectorStore with a corpus-specific COSINE/FLAT index."""
import hashlib
import json
import time
from redis import Redis
from redis.exceptions import RedisError
from langchain_redis import RedisConfig, RedisVectorStore
from .config import MODEL_NAME, MODEL_REVISION

METADATA_SCHEMA = [dict(name=name, type="tag") for name in
                   ("source_file", "record_id", "product_id", "category", "price_conflict")]
METADATA_SCHEMA += [dict(name=name, type="numeric") for name in
                    ("row_id", "rating", "ratings_count", "initial_price", "final_price", "discount")]


def redis_preflight(url):
    client = Redis.from_url(url, socket_connect_timeout=3, socket_timeout=5)
    try:
        client.ping()
        indexes = client.execute_command("FT._LIST")
        return client, dict(status="ready", indexes=[x.decode() if isinstance(x, bytes) else x for x in indexes])
    except RedisError as exc:
        raise RuntimeError("Redis Search unavailable. Start Docker Desktop, then run: "
                           "docker compose -f labs/lab03_rag/docker-compose.yml up -d. "
                           f"Original error: {exc}") from exc


def corpus_fingerprint(documents):
    payload = dict(model=MODEL_NAME, revision=MODEL_REVISION, normalized=True,
                   schema=METADATA_SCHEMA, dimension=384, metric="COSINE", algorithm="FLAT",
                   docs=[dict(text=d.page_content, metadata=d.metadata) for d in documents])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, (list, tuple)):
        return [_decode(x) for x in value]
    if isinstance(value, dict):
        return {_decode(k): _decode(v) for k, v in value.items()}
    return value


def index_info(client, name, expected_count=None):
    raw = _decode(client.execute_command("FT.INFO", name))
    info = raw if isinstance(raw, dict) else dict(zip(raw[::2], raw[1::2]))
    attrs = [a if isinstance(a, dict) else dict(zip(a[::2], a[1::2])) for a in info["attributes"]]
    vector = next(a for a in attrs if a.get("type") == "VECTOR")
    if int(vector["dim"]) != 384 or vector["distance_metric"].upper() != "COSINE":
        raise ValueError("Existing Redis vector schema differs from COSINE/384")
    if vector["algorithm"].upper() != "FLAT":
        raise ValueError("Unexpected Redis indexing algorithm")
    for field in METADATA_SCHEMA:
        if not any(a.get("attribute") == field["name"] and a.get("type", "").lower() == field["type"] for a in attrs):
            raise ValueError(f"Missing/incompatible metadata field: {field['name']}")
    count = int(info["num_docs"])
    if int(info.get("hash_indexing_failures", 0)):
        raise ValueError("Redis reports hash indexing failures")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"Redis document count {count} != expected {expected_count}")
    return dict(index=name, document_count=count, vector=vector, metadata_schema=attrs)


def build_store(documents, embeddings, redis_url):
    if not documents:
        raise ValueError("Cannot index an empty corpus")
    client, _ = redis_preflight(redis_url)
    fingerprint = corpus_fingerprint(documents)
    name = "lab03_" + fingerprint[:20]
    config = RedisConfig(index_name=name, key_prefix=name, redis_client=client,
                         embedding_dimensions=384, distance_metric="COSINE",
                         indexing_algorithm="FLAT", vector_datatype="FLOAT32",
                         storage_type="hash", metadata_schema=METADATA_SCHEMA,
                         legacy_key_format=False)
    store = RedisVectorStore(embeddings, config=config)
    index_info(client, name)
    # Stable IDs and content-specific namespace make reruns idempotent, no FLUSHDB/drop.
    ids = [hashlib.sha256(d.metadata["record_id"].encode()).hexdigest() for d in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record IDs")
    missing = [(d, key) for d, key in zip(documents, ids) if not client.exists(f"{name}:{key}")]
    if missing:
        store.add_documents([d for d, _ in missing], ids=[key for _, key in missing])
    deadline = time.monotonic() + 10
    while True:
        info = index_info(client, name)
        if info["document_count"] == len(documents) or time.monotonic() >= deadline:
            break
        time.sleep(.1)
    info = index_info(client, name, len(documents))
    info.update(corpus_sha256=fingerprint, inserted=len(missing))
    return store, info
