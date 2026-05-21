"""
Nexus API v1 — all routes.

Endpoint groups:
  /auth          → login, token
  /tenants       → CRUD (admin only)
  /users         → user management
  /documents     → index + list + delete
  /query         → REST query + SSE streaming
  /webhooks      → register + list + delete
  /analytics     → dashboard metrics
  /integrations  → Slack + Teams event endpoints
  /health        → system health
"""
import uuid, time
from typing import AsyncGenerator

from fastapi            import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses  import StreamingResponse
from sqlalchemy         import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse  import EventSourceResponse

from app.core.database  import get_db, Tenant, User, Document, WebhookEndpoint
from app.core.auth      import (
    get_current_user, CurrentUser, require_role,
    hash_password, verify_password, create_access_token,
)
from app.core.schemas   import (
    LoginRequest, TokenResponse, TenantCreate, UserCreate,
    IndexRequest, IndexResponse, DocumentOut,
    QueryRequest, QueryResponse,
    WebhookCreate, WebhookOut,
    AnalyticsSummary, WebhookEvent,
)
from app.core.config     import get_settings
from app.core.vectorstore import vector_count
from app.ingestion.indexer import index_document
from app.graph.runner    import run as rag_run, stream_run
from app.integrations.webhooks import fire_event
from app.integrations.slack    import handle_slash_command, handle_app_mention, verify_slack_signature
from app.analytics.service     import log_query, get_analytics

settings = get_settings()
router   = APIRouter()


# ── Health ─────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health():
    return {
        "status":  "ok",
        "version": settings.app_version,
        "env":     settings.environment,
    }


# ── Auth ───────────────────────────────────────────────────────

@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(
        select(User).where(User.email == req.email, User.is_active == True)
    )).scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_pw):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    token = create_access_token(user.id, user.tenant_id, user.role)
    return TokenResponse(
        access_token = token,
        expires_in   = settings.jwt_expire_minutes * 60,
    )


# ── Tenants ────────────────────────────────────────────────────

@router.post("/tenants", status_code=201, tags=["tenants"])
async def create_tenant(
    req:  TenantCreate,
    db:   AsyncSession = Depends(get_db),
    # In production: restrict to platform admins via API key
):
    existing = await db.get(Tenant, req.id)
    if existing:
        raise HTTPException(409, f"Tenant '{req.id}' already exists")
    tenant = Tenant(id=req.id, name=req.name, plan=req.plan)
    db.add(tenant)
    await db.commit()
    return {"id": tenant.id, "name": tenant.name, "plan": tenant.plan}


@router.post("/tenants/{tenant_id}/users", status_code=201, tags=["tenants"])
async def create_user(
    tenant_id: str,
    req:       UserCreate,
    db:        AsyncSession = Depends(get_db),
    _user:     CurrentUser  = Depends(require_role("admin")),
):
    if _user.tenant_id != tenant_id:
        raise HTTPException(403, "Cannot create users in another tenant")

    user = User(
        id         = str(uuid.uuid4()),
        tenant_id  = tenant_id,
        email      = req.email,
        hashed_pw  = hash_password(req.password),
        role       = req.role.value,
    )
    db.add(user)
    await db.commit()
    return {"id": user.id, "email": user.email, "role": user.role}


# ── Documents ─────────────────────────────────────────────────

@router.post(
    "/documents",
    response_model = IndexResponse,
    status_code    = 201,
    tags           = ["documents"],
    summary        = "Index a document into the knowledge base",
)
async def index_doc(
    req:  IndexRequest,
    bg:   BackgroundTasks,
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(require_role("admin", "member")),
):
    """
    Index a document. Supported source types:
      - upload: local file path
      - github: repository file URL
      - confluence: page content (pass text as source_ref)
      - pagerduty: incident JSON path
    """
    result = await index_document(
        tenant_id   = user.tenant_id,
        source_type = req.source_type.value,
        source_ref  = req.source_ref,
        doc_type    = req.doc_type.value,
        title       = req.title,
        db          = db,
        metadata    = req.metadata,
    )

    # Fire webhook in background
    bg.add_task(
        fire_event, db, user.tenant_id,
        WebhookEvent.INDEX_DONE,
        {"source_ref": req.source_ref, "chunks": result["chunks_added"]},
    )

    return IndexResponse(**result)


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
async def list_documents(
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(get_current_user),
):
    docs = (await db.execute(
        select(Document).where(Document.tenant_id == user.tenant_id)
        .order_by(Document.created_at.desc())
    )).scalars().all()
    return [DocumentOut.model_validate(d) for d in docs]


@router.get("/documents/stats", tags=["documents"])
async def document_stats(user: CurrentUser = Depends(get_current_user)):
    return {
        "tenant_id":    user.tenant_id,
        "vector_chunks": vector_count(user.tenant_id),
    }


@router.delete("/documents/{doc_id}", tags=["documents"])
async def delete_document(
    doc_id: str,
    db:     AsyncSession = Depends(get_db),
    user:   CurrentUser  = Depends(require_role("admin")),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.tenant_id != user.tenant_id:
        raise HTTPException(404, "Document not found")
    await db.delete(doc)
    await db.commit()
    return {"deleted": doc_id}


# ── Query ──────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model = QueryResponse,
    tags           = ["query"],
    summary        = "Ask a question (full Modular RAG + LangGraph pipeline)",
)
async def query(
    req:  QueryRequest,
    bg:   BackgroundTasks,
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(get_current_user),
):
    if vector_count(user.tenant_id) == 0:
        raise HTTPException(400, "No documents indexed. POST to /documents first.")

    start = time.perf_counter()
    response = await rag_run(
        tenant_id       = user.tenant_id,
        user_id         = user.user_id,
        question        = req.question,
        doc_type_filter = req.doc_type_filter.value if req.doc_type_filter else None,
    )

    # Log query asynchronously
    bg.add_task(log_query, db, user.tenant_id, user.user_id, req.question, response)

    # Fire webhook
    bg.add_task(
        fire_event, db, user.tenant_id,
        WebhookEvent.QUERY_COMPLETED,
        {"question": req.question, "confidence": response.confidence},
    )

    return response


@router.get(
    "/query/stream",
    tags    = ["query"],
    summary = "Ask a question with Server-Sent Events streaming",
    description = (
        "Returns SSE stream. Events: node_start, node_done, done, error.\n"
        "Clients see live progress as the graph executes."
    ),
)
async def query_stream(
    question:        str,
    doc_type_filter: str | None = None,
    user:            CurrentUser = Depends(get_current_user),
):
    if vector_count(user.tenant_id) == 0:
        raise HTTPException(400, "No documents indexed.")

    async def generator():
        async for chunk in stream_run(
            tenant_id       = user.tenant_id,
            user_id         = user.user_id,
            question        = question,
            doc_type_filter = doc_type_filter,
        ):
            yield chunk

    return EventSourceResponse(generator())


# ── Webhooks ───────────────────────────────────────────────────

@router.post("/webhooks", response_model=WebhookOut, status_code=201, tags=["webhooks"])
async def register_webhook(
    req:  WebhookCreate,
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(require_role("admin")),
):
    secret = req.secret or str(uuid.uuid4()).replace("-", "")
    wh     = WebhookEndpoint(
        id        = str(uuid.uuid4()),
        tenant_id = user.tenant_id,
        url       = req.url,
        secret    = secret,
        events    = [e.value for e in req.events],
    )
    db.add(wh)
    await db.commit()
    return WebhookOut.model_validate(wh)


@router.get("/webhooks", response_model=list[WebhookOut], tags=["webhooks"])
async def list_webhooks(
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(require_role("admin")),
):
    whs = (await db.execute(
        select(WebhookEndpoint).where(WebhookEndpoint.tenant_id == user.tenant_id)
    )).scalars().all()
    return [WebhookOut.model_validate(w) for w in whs]


@router.delete("/webhooks/{webhook_id}", tags=["webhooks"])
async def delete_webhook(
    webhook_id: str,
    db:         AsyncSession = Depends(get_db),
    user:       CurrentUser  = Depends(require_role("admin")),
):
    wh = await db.get(WebhookEndpoint, webhook_id)
    if not wh or wh.tenant_id != user.tenant_id:
        raise HTTPException(404, "Webhook not found")
    await db.delete(wh)
    await db.commit()
    return {"deleted": webhook_id}


# ── Analytics ──────────────────────────────────────────────────

@router.get(
    "/analytics",
    response_model = AnalyticsSummary,
    tags           = ["analytics"],
    summary        = "Query volume, latency, and RAGAS eval scores",
)
async def analytics(
    days: int        = 30,
    db:   AsyncSession = Depends(get_db),
    user: CurrentUser  = Depends(require_role("admin", "member")),
):
    return await get_analytics(db, user.tenant_id, days)


# ── Slack integration ──────────────────────────────────────────

@router.post("/integrations/slack/events", tags=["integrations"])
async def slack_events(
    request: "Request",    # type: ignore
    bg:      BackgroundTasks,
    db:      AsyncSession = Depends(get_db),
):
    """Slack Events API endpoint. Handles app_mention and message events."""
    from fastapi import Request
    body      = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(body, timestamp, signature):
        raise HTTPException(403, "Invalid Slack signature")

    import json
    payload = json.loads(body)

    # URL verification challenge
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event     = payload.get("event", {})
    event_type = event.get("type")

    # Resolve tenant from Slack team_id
    team_id = payload.get("team_id", "")
    tenant  = (await db.execute(
        select(Tenant).where(Tenant.settings["slack_team_id"].as_string() == team_id)
    )).scalar_one_or_none()

    if not tenant:
        return {"ok": True}   # unregistered team

    if event_type == "app_mention":
        bg.add_task(handle_app_mention, event, tenant.id, "slack_user")

    return {"ok": True}


@router.post("/integrations/slack/commands", tags=["integrations"])
async def slack_slash_command(
    request: "Request",    # type: ignore
    db:      AsyncSession = Depends(get_db),
):
    """Handle /nexus slash command from Slack."""
    from fastapi import Request
    body      = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(body, timestamp, signature):
        raise HTTPException(403, "Invalid Slack signature")

    from urllib.parse import parse_qs
    payload   = {k: v[0] for k, v in parse_qs(body.decode()).items()}
    team_id   = payload.get("team_id", "")

    tenant = (await db.execute(
        select(Tenant).where(Tenant.settings["slack_team_id"].as_string() == team_id)
    )).scalar_one_or_none()

    if not tenant:
        return {"text": "Nexus is not configured for this workspace."}

    return await handle_slash_command(payload, tenant.id, "slack_user")
