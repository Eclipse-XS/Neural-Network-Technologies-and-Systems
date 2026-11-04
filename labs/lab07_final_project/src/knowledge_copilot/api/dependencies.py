"""FastAPI dependency injection container."""
import os
from pathlib import Path
from typing import Optional

from ..application.ingestion_service import IngestionService
from ..application.query_service import QueryService
from ..application.session_service import SessionService
from ..domain.enums import RetrievalMode
from ..infrastructure.llm_client import LocalLLMClient
from ..infrastructure.redis_store import RedisVectorAdapter
from ..infrastructure.sqlite_repo import SQLiteRepository
from ..rag.embeddings import CopilotEmbeddings
from ..agents.orchestrator import AgentOrchestrator

_INGESTION_SERVICE: Optional[IngestionService] = None
_QUERY_SERVICE: Optional[QueryService] = None
_SESSION_SERVICE: Optional[SessionService] = None
_SQLITE_REPO: Optional[SQLiteRepository] = None
_REDIS_ADAPTER: Optional[RedisVectorAdapter] = None
_EMBEDDINGS: Optional[CopilotEmbeddings] = None
_LLM_CLIENT: Optional[LocalLLMClient] = None
_ORCHESTRATOR: Optional[AgentOrchestrator] = None


def get_sqlite_repo() -> SQLiteRepository:
    global _SQLITE_REPO
    if _SQLITE_REPO is None:
        db_path = Path(os.getenv("SESSION_DB", "outputs/sessions/assistant_memory.sqlite"))
        if not db_path.is_absolute():
            lab_root = Path(__file__).resolve().parents[3]
            db_path = lab_root / db_path
        _SQLITE_REPO = SQLiteRepository(db_path)
    return _SQLITE_REPO


def get_redis_adapter() -> RedisVectorAdapter:
    global _REDIS_ADAPTER
    if _REDIS_ADAPTER is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6380")
        index_name = os.getenv("REDIS_INDEX_NAME", "lab07_course_kb_idx")
        _REDIS_ADAPTER = RedisVectorAdapter(redis_url=redis_url, index_name=index_name)
    return _REDIS_ADAPTER


def get_embeddings() -> CopilotEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        revision = os.getenv("EMBEDDING_REVISION", "614241f622f53c4eeff9890bdc4f31cfecc418b3")
        _EMBEDDINGS = CopilotEmbeddings(model_name=model_name, revision=revision)
    return _EMBEDDINGS


def get_ingestion_service() -> IngestionService:
    global _INGESTION_SERVICE
    if _INGESTION_SERVICE is None:
        repo = get_sqlite_repo()
        redis = get_redis_adapter()
        emb = get_embeddings()
        _INGESTION_SERVICE = IngestionService(
            sqlite_repo=repo,
            redis_adapter=redis,
            embeddings_model=emb,
            max_tokens=int(os.getenv("EMBEDDING_MAX_TOKENS", "240")),
        )
    return _INGESTION_SERVICE


def get_session_service() -> SessionService:
    global _SESSION_SERVICE
    if _SESSION_SERVICE is None:
        repo = get_sqlite_repo()
        _SESSION_SERVICE = SessionService(sqlite_repo=repo)
    return _SESSION_SERVICE


def get_llm_client() -> LocalLLMClient:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        model_id = os.getenv("LM_STUDIO_MODEL", "mistralai/ministral-3-3b")
        _LLM_CLIENT = LocalLLMClient(base_url=base_url, model_id=model_id)
    return _LLM_CLIENT


def get_query_service() -> QueryService:
    global _QUERY_SERVICE, _ORCHESTRATOR
    if _QUERY_SERVICE is None:
        ingestion = get_ingestion_service()
        redis = get_redis_adapter()
        repo = get_sqlite_repo()
        llm = get_llm_client()
        
        mode_str = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
        mode = RetrievalMode(mode_str) if mode_str in {"dense", "lexical", "hybrid"} else RetrievalMode.HYBRID
        
        _QUERY_SERVICE = QueryService(
            ingestion_service=ingestion,
            redis_adapter=redis,
            sqlite_repo=repo,
            default_retrieval_mode=mode,
            top_k=int(os.getenv("TOP_K", "5")),
        )
        
        _ORCHESTRATOR = AgentOrchestrator(
            llm_client=llm,
            search_fn=_QUERY_SERVICE.search,
            list_sources_fn=_QUERY_SERVICE.list_sources,
            max_turns=int(os.getenv("AGENT_MAX_TURNS", "10")),
        )
        _QUERY_SERVICE.orchestrator = _ORCHESTRATOR

    return _QUERY_SERVICE
