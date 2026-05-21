"""
Nexus — Enterprise Engineering Copilot
FastAPI application with full middleware stack.
"""
from contextlib import asynccontextmanager

import structlog
from fastapi                     import FastAPI, Request, status
from fastapi.middleware.cors     import CORSMiddleware
from fastapi.responses           import JSONResponse

from app.core.config             import get_settings
from app.core.database           import init_db
from app.api.v1.routes           import router
from app.ingestion.scheduler     import start_scheduler, stop_scheduler

settings = get_settings()
log      = structlog.get_logger()

# ── Structured logging setup ───────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("nexus.startup", version=settings.app_version, env=settings.environment)

    # Init database tables
    await init_db()

    # Warm up cross-encoder model
    from app.graph.nodes import _reranker
    _reranker()
    log.info("nexus.reranker_loaded")

    # Start scheduled re-indexing
    start_scheduler()

    yield

    # Shutdown
    stop_scheduler()
    log.info("nexus.shutdown")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.app_name,
    version     = settings.app_version,
    description = """
## Nexus — Enterprise Engineering Copilot

Multi-tenant SaaS RAG platform for engineering teams.

### Features
- **Modular RAG** with LangGraph — adaptive retrieval, agentic loop
- **Multi-tenant** — complete data isolation per tenant
- **Code-aware chunking** — function/class boundaries, runbook sections, incident formats
- **Hybrid search** — BM25 + vector + RRF fusion
- **Streaming** — Server-Sent Events for real-time progress
- **RBAC** — admin / member / viewer roles
- **Webhooks** — HMAC-signed event delivery
- **Slack/Teams** — native bot integration
- **Analytics** — query volume, latency, RAGAS scores
- **Scheduled re-indexing** — delta sync every 6 hours

### Authentication
All endpoints (except /health and /auth/login) require a Bearer JWT.
""",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── Middleware ─────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Log every request with method, path, status, and latency."""
    import time
    start = time.perf_counter()
    response = await call_next(request)
    latency  = round((time.perf_counter() - start) * 1000, 1)
    log.info(
        "http.request",
        method  = request.method,
        path    = request.url.path,
        status  = response.status_code,
        ms      = latency,
    )
    return response


# ── Global error handler ───────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Routes ────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")
