"""FastAPI application factory and middleware configuration."""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..observability.logging import get_logger
from .dependencies import get_embeddings, get_ingestion_service, get_redis_adapter
from .routers import chat, documents, health

logger = get_logger("knowledge_copilot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure Redis vector index is initialized and default coursework docs are loaded
    logger.info("Initializing Knowledge Copilot API...")
    try:
        redis_adapter = get_redis_adapter()
        embeddings = get_embeddings()
        ingestion = get_ingestion_service()
        
        if redis_adapter.ping():
            redis_adapter.initialize_store(embeddings)
            
            # Load default coursework documents if available
            lab_root = Path(__file__).resolve().parents[3]
            knowledge_dir = lab_root / "assets/knowledge"
            manifest_file = knowledge_dir / "manifest.json"
            
            if manifest_file.exists():
                import json
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                for item in manifest:
                    fpath = knowledge_dir / item["workspace_filename"]
                    if fpath.exists():
                        try:
                            ingestion.ingest_file(
                                filename=item["workspace_filename"],
                                content=fpath.read_bytes(),
                                collection="coursework",
                                document_id=item["source_id"],
                            )
                        except Exception as e:
                            logger.warning(f"Could not ingest initial doc {item['source_id']}: {e}")
        else:
            logger.warning("Redis is not currently reachable on startup.")
    except Exception as exc:
        logger.error(f"Error during startup initialization: {exc}")
        
    yield
    logger.info("Shutting down Knowledge Copilot API...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local Engineering Knowledge Copilot API",
        version="2.0.0",
        description="Privacy-oriented local agentic RAG and document ingestion API.",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and tracing middleware
    @app.middleware("http")
    async def request_id_and_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)",
            extra={"request_id": req_id, "duration_ms": duration_ms},
        )
        return response

    # Include routers
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(health.router)

    return app


app = create_app()
