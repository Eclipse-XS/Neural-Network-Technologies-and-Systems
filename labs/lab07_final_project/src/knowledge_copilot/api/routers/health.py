"""Health check and telemetry endpoints."""
from fastapi import APIRouter, Depends, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..dependencies import get_llm_client, get_redis_adapter, get_sqlite_repo
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health/live", response_model=HealthResponse)
async def liveness(
    sqlite_repo=Depends(get_sqlite_repo),
    redis_adapter=Depends(get_redis_adapter),
    llm_client=Depends(get_llm_client),
):
    """Liveness probe: verifies process is alive and basic services respond."""
    redis_ok = redis_adapter.ping()
    try:
        with sqlite_repo._get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        sqlite_ok = True
    except Exception:
        sqlite_ok = False
    llm_ok = await llm_client.is_available()
    
    return HealthResponse(
        status="live",
        version="2.0.0",
        redis_connected=redis_ok,
        sqlite_connected=sqlite_ok,
        llm_available=llm_ok,
    )


@router.get("/api/v1/health/ready", response_model=HealthResponse)
async def readiness(
    response: Response,
    sqlite_repo=Depends(get_sqlite_repo),
    redis_adapter=Depends(get_redis_adapter),
    llm_client=Depends(get_llm_client),
):
    """Readiness probe: verifies all infrastructure dependencies are ready."""
    redis_ok = redis_adapter.ping()
    try:
        with sqlite_repo._get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        sqlite_ok = True
    except Exception:
        sqlite_ok = False
    llm_ok = await llm_client.is_available()
    
    ready = redis_ok and sqlite_ok
    status_str = "ready" if ready else "degraded"
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return HealthResponse(
        status=status_str,
        version="2.0.0",
        redis_connected=redis_ok,
        sqlite_connected=sqlite_ok,
        llm_available=llm_ok,
    )


@router.get("/metrics")
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
