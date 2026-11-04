"""Hash-verified Markdown corpus, CPU embeddings and real Redis vector search."""
import hashlib
import json
import statistics
import time
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .config import KNOWLEDGE, OUTPUTS, Settings

SCHEMA = [{"name": n, "type": "tag"} for n in
          ("source_id", "filename", "lab_number", "section", "chunk_id", "source_sha256", "evidence_quality")]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_corpus(directory: Path = KNOWLEDGE) -> list[dict]:
    """Validate the copied corpus independently of the original labs (DEFECT-05)."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        filename = item["workspace_filename"]
        path = (directory / filename).resolve()
        if Path(filename).name != filename or not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid corpus path")
        data = path.read_bytes()
        if not data.strip() or len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Corpus integrity failure: {filename}")
    return manifest


def load_chunks(settings: Settings, directory: Path = KNOWLEDGE, *, token_counter=None,
                max_tokens: int | None = None) -> list[Document]:
    """Preserve headings/provenance and enforce an embedding-token budget.

    Character splitting is only a first pass. When a tokenizer counter is supplied,
    oversized pieces are reduced deterministically until every embedding input fits.
    """
    token_budget = settings.embedding_max_tokens if max_tokens is None else max_tokens
    if token_budget < 32:
        raise ValueError("Effective embedding token budget must be at least 32")
    headers = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    manifest_items = validate_corpus(directory)
    for item in manifest_items:
        sections = headers.split_text((directory / item["workspace_filename"]).read_text(encoding="utf-8"))
        candidates = splitter.split_documents(sections)
        bounded = []
        for doc in candidates:
            heading = " / ".join(doc.metadata[x] for x in ("h2", "h3") if x in doc.metadata)
            if heading and heading.casefold() not in doc.page_content[:200].casefold():
                doc = Document(page_content=f"## {heading}\n{doc.page_content}", metadata=doc.metadata)
            pending = [doc]
            while pending:
                current = pending.pop(0)
                if token_counter is None or token_counter(current.page_content) <= token_budget:
                    bounded.append(current); continue
                text = current.page_content
                midpoint = len(text) // 2
                cut = max(text.rfind("\n", 0, midpoint), text.rfind(". ", 0, midpoint), text.rfind(" ", 0, midpoint))
                if cut < 1:
                    cut = midpoint
                overlap = min(settings.chunk_overlap, max(0, cut // 4))
                left, right = text[:cut].strip(), text[max(0, cut-overlap):].strip()
                if not left or not right or left == text or right == text:
                    raise ValueError("Unable to split text within embedding token budget")
                pending[:0] = [Document(page_content=left, metadata=current.metadata),
                               Document(page_content=right, metadata=current.metadata)]
        for ordinal, doc in enumerate(bounded):
            section = " / ".join(doc.metadata[x] for x in ("h2", "h3") if x in doc.metadata) or doc.metadata.get("h1", "Вступ")
            digest = hashlib.sha256(json.dumps([item["sha256"], section, " ".join(doc.page_content.split()), ordinal], ensure_ascii=False).encode()).hexdigest()[:16]
            transcript_sections = (
                "system instructions", "дизайн експерименту", "підсумкова таблиця",
                "багатокрокове виконання", "додаткова арифметична", "керована діагностика",
                "без інструментів", "невідома інформація",
                "ефективність рішень", "фактичні помилки", "зауваження до кожної",
                "результати", "comparison",
            )
            raw_output_markers = (
                "session a,", "session b,", "to confirm, i will check", "otherwise, based on",
                "final answer:", "the sum of the inferred", "lab_4_results.csv",
            )
            quality = "quoted_model_output" if item.get("source_id") == "lab06" and (
                any(marker in section.casefold() for marker in transcript_sections) or
                any(marker in doc.page_content.casefold() for marker in raw_output_markers)
            ) else "canonical_report"
            doc.metadata = dict(source_id=item["source_id"], filename=item["workspace_filename"], lab_number=str(item.get("lab_number", "")),
                section=section, chunk_id=f"chunk-{digest}", source_sha256=item["sha256"], evidence_quality=quality)
            chunks.append(doc)
    return chunks


def chunk_audit(chunks: list[Document]) -> dict:
    lengths = [len(d.page_content) for d in chunks]
    sources = sorted({d.metadata['source_id'] for d in chunks})
    return dict(document_count=len(sources), chunk_count=len(chunks),
        chunks_per_source={s: sum(d.metadata['source_id'] == s for d in chunks) for s in sources},
        min_length=min(lengths) if lengths else 0, median_length=statistics.median(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        tiny_under_100=sum(n < 100 for n in lengths), samples=[dict(text=d.page_content, **d.metadata) for d in chunks[::max(1, len(chunks)//8)]])


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    filename: str
    lab_number: str
    section: str
    chunk_id: str
    evidence_quality: str
    text: str
    cosine_distance: float
    retrieval_method: str = "dense"
    score_type: str = "cosine_distance"
    score_value: float = 0.0
    ranking_position: int = 1

    def to_dict(self) -> dict:
        d = dict(asdict(self), citation=f"[{self.source_id}#{self.chunk_id}]")
        d["score_type"] = self.score_type
        d["score_value"] = round(self.score_value, 4)
        d["retrieval_method"] = self.retrieval_method
        d["ranking_position"] = self.ranking_position
        return d


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, (list, tuple)):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        return {decode(k): decode(v) for k, v in value.items()}
    return value


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manifest = validate_corpus()
        self.chunks = load_chunks(settings)
        self.store = None
        self.info = None
        self._lexical = None

    def initialize(self) -> dict:
        """Reuse only a verified index; rebuilding is restricted to Lab 7 keys."""
        from redis import Redis
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_redis import RedisConfig, RedisVectorStore
        s = self.settings
        client = Redis.from_url(s.redis_url, socket_connect_timeout=3, socket_timeout=10)
        client.ping()
        existing = decode(client.execute_command("FT._LIST"))
        base_embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model,
            model_kwargs=dict(device="cpu", revision=s.embedding_revision, local_files_only=True),
            encode_kwargs=dict(normalize_embeddings=True, batch_size=16))
        use_e5_prefixes = "e5" in s.embedding_model.casefold()

        class RetrievalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                values = [f"passage: {text}" for text in texts] if use_e5_prefixes else texts
                return base_embeddings.embed_documents(values)

            def embed_query(self, text):
                return base_embeddings.embed_query(f"query: {text}" if use_e5_prefixes else text)

        embeddings = RetrievalEmbeddings()
        tokenizer = base_embeddings._client.tokenizer
        effective_token_budget = min(s.embedding_max_tokens, base_embeddings._client.max_seq_length - 8)
        self.chunks = load_chunks(s, token_counter=lambda text: len(tokenizer.encode(text, truncation=False)),
                                  max_tokens=effective_token_budget)
        self._build_lexical_index()
        vector = np.asarray(embeddings.embed_query("embedding dimension probe"), dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError("Invalid normalized embedding")
        dim = len(vector)
        token_lengths = [len(tokenizer.encode(d.page_content, truncation=False)) for d in self.chunks]
        save_json(OUTPUTS / "metadata/embedding.json", dict(model=s.embedding_model, revision=s.embedding_revision,
            dimension=dim, device="cpu", normalization=True, max_seq_length=base_embeddings._client.max_seq_length,
            query_passage_prefixes=use_e5_prefixes,
            truncated_chunks=sum(n > base_embeddings._client.max_seq_length for n in token_lengths), token_lengths=token_lengths))
        payload = dict(manifest=self.manifest, model=s.embedding_model, revision=s.embedding_revision, dimension=dim,
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, embedding_max_tokens=effective_token_budget,
            schema=SCHEMA, version=2, normalized=True,
            chunks=[dict(text=d.page_content, metadata=d.metadata) for d in self.chunks])
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        marker = s.key_prefix + ":fingerprint"
        reuse = s.index_name in existing and client.get(marker) == fingerprint.encode()
        if not reuse:
            if s.index_name in existing:
                client.execute_command("FT.DROPINDEX", s.index_name)
            keys = list(client.scan_iter(match=s.key_prefix + ":*"))
            if keys:
                client.delete(*keys)
        config = RedisConfig(index_name=s.index_name, key_prefix=s.key_prefix, redis_client=client,
            embedding_dimensions=dim, distance_metric="COSINE", indexing_algorithm="FLAT", vector_datatype="FLOAT32",
            storage_type="hash", metadata_schema=SCHEMA, legacy_key_format=False)
        self.store = RedisVectorStore(embeddings, config=config)
        missing = [d for d in self.chunks if not client.exists(f"{s.key_prefix}:{d.metadata['chunk_id']}")]
        if missing:
            self.store.add_documents(missing, ids=[d.metadata['chunk_id'] for d in missing])
            client.set(marker, fingerprint)
        probe = self.store.similarity_search_with_score("test probe", k=1)
        actual = bool(probe and probe[0][0].metadata.get("chunk_id"))
        self.info = dict(redis_url=s.redis_url, index_name=s.index_name, total_keys=client.dbsize(),
            corpus_fingerprint=fingerprint, embedding_model=s.embedding_model, embedding_dimensions=dim,
            chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, embedding_max_tokens=effective_token_budget,
            status='reused' if reuse else 'created', inserted=len(missing),
            redis_version=client.info()['redis_version'], verified_vector=actual)
        save_json(OUTPUTS / 'metadata/index.json', self.info)
        save_json(OUTPUTS / 'metadata/chunks.json', chunk_audit(self.chunks))
        return self.info

    def sources(self) -> list[dict]:
        return [{k: item.get(k, "") for k in ('source_id', 'lab_number', 'title', 'workspace_filename')} for item in self.manifest]

    @staticmethod
    def _terms(text: str) -> list[str]:
        return re.findall(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ_./+-]+", text.casefold())

    def _build_lexical_index(self) -> None:
        docs = [self._terms(d.page_content) for d in self.chunks]
        df = Counter(term for terms in docs for term in set(terms))
        self._lexical = (docs, df, sum(map(len, docs)) / max(1, len(docs)))

    def _lexical_search(self, query: str, k: int, source_id: str | None) -> list[tuple[Document, float]]:
        if self._lexical is None:
            self._build_lexical_index()
        docs, df, avgdl = self._lexical
        qterms = self._terms(query)
        scored = []
        n = len(docs)
        for doc, terms in zip(self.chunks, docs):
            if source_id and doc.metadata['source_id'] != source_id:
                continue
            if doc.metadata['evidence_quality'] != 'canonical_report':
                continue
            tf = Counter(terms); score = 0.0
            for term in qterms:
                freq = tf[term]
                if not freq: continue
                idf = math.log(1 + (n - df[term] + .5) / (df[term] + .5))
                score += idf * freq * 2.2 / (freq + 1.2 * (.25 + .75 * len(terms) / max(avgdl, 1)))
            if score > 0:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].metadata['chunk_id']))[:k]

    def search(self, query: str, k: int | None = None, source_id: str | None = None,
               mode: str | None = None) -> list[RetrievedChunk]:
        """Return raw evidence with explicit score metadata (DEFECT-06)."""
        from redisvl.query.filter import Tag
        k = self.settings.top_k if k is None else k
        if not isinstance(query, str) or not query.strip() or len(query) > 600 or type(k) is not int or not 1 <= k <= 8:
            raise ValueError('Use a nonempty query up to 600 characters and k from 1 to 8')
        if source_id is not None and source_id not in {x['source_id'] for x in self.manifest}:
            raise ValueError('Unknown source_id; use list_knowledge_sources')
        if source_id is None:
            mentioned = set(re.findall(r"(?:lab|лабораторн(?:а|ої)\s+робот(?:а|и))\s*№?\s*([3-6])\b",
                                       query.casefold()))
            if len(mentioned) == 1:
                source_id = f"lab{mentioned.pop().zfill(2)}"
        retrieval_query = query
        if re.search(r"\b(?:how many|count|total|number|кільк(?:ість|и)|скільки)\b|унікальн|inference|інферен", query.casefold()):
            retrieval_query += " загальна кількість разом унікальних генерацій inference виконано total count"
        if self.store is None:
            raise RuntimeError('Initialize knowledge index before searching')
        mode = mode or self.settings.retrieval_mode
        if mode not in {'dense','lexical','hybrid'}:
            raise ValueError('mode must be dense, lexical or hybrid')
        
        dense = self.store.similarity_search_with_score(retrieval_query, k=min(len(self.chunks), max(k * 5, k)),
            filter=(Tag('source_id') == source_id) if source_id else None) if mode != 'lexical' else []
        dense = [(doc, score) for doc, score in dense if doc.metadata.get('evidence_quality') == 'canonical_report']
        lexical = self._lexical_search(retrieval_query, min(len(self.chunks), max(k * 3, k)), source_id) if mode != 'dense' else []
        
        retrieved: list[RetrievedChunk] = []
        if mode == 'dense':
            for rank, (doc, dist) in enumerate(dense[:k], 1):
                retrieved.append(RetrievedChunk(
                    **{n: str(doc.metadata[n]) for n in ('source_id','filename','lab_number','section','chunk_id','evidence_quality')},
                    text=doc.page_content, cosine_distance=float(dist),
                    retrieval_method="dense", score_type="cosine_distance", score_value=float(dist), ranking_position=rank
                ))
        elif mode == 'lexical':
            for rank, (doc, bm25_score) in enumerate(lexical[:k], 1):
                retrieved.append(RetrievedChunk(
                    **{n: str(doc.metadata[n]) for n in ('source_id','filename','lab_number','section','chunk_id','evidence_quality')},
                    text=doc.page_content, cosine_distance=float(bm25_score),
                    retrieval_method="lexical", score_type="bm25_score", score_value=float(bm25_score), ranking_position=rank
                ))
        else:
            # Hybrid Reciprocal-rank fusion with explicit score metadata
            fused = {}
            by_id = {}
            distances = {}
            for ranking, items in [('dense', dense), ('lexical', lexical)]:
                for rank, (doc, score) in enumerate(items, 1):
                    cid = doc.metadata['chunk_id']
                    by_id[cid] = doc
                    weight = 2.0 if ranking == 'dense' else 1.0
                    fused[cid] = fused.get(cid, 0.0) + weight / (60 + rank)
                    if ranking == 'dense': distances[cid] = float(score)
            order = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
            for rank, cid in enumerate(order, 1):
                doc = by_id[cid]
                raw_dist = distances.get(cid, 0.0)
                rrf_val = fused[cid]
                retrieved.append(RetrievedChunk(
                    **{n: str(doc.metadata[n]) for n in ('source_id','filename','lab_number','section','chunk_id','evidence_quality')},
                    text=doc.page_content, cosine_distance=raw_dist,
                    retrieval_method="hybrid", score_type="rrf_score", score_value=rrf_val, ranking_position=rank
                ))
        return retrieved
