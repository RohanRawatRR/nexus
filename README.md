# Nexus — Enterprise Engineering Copilot

> Multi-tenant SaaS RAG platform for engineering teams. Ask questions across your codebase, runbooks, incident history, and API docs — and get grounded, cited answers in seconds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│   REST API · SSE Stream · Slack Bot · Teams Bot · Webhooks  │
└──────────────────────────┬──────────────────────────────────┘
                           │  JWT + RBAC
┌──────────────────────────▼──────────────────────────────────┐
│              FastAPI Gateway (tenant_id on every request)    │
│              Rate limiting · Structured logging · CORS       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              LangGraph — Modular RAG Engine                  │
│                                                              │
│  analyze_query → hybrid_retrieve → adequacy_check → generate│
│                         ↑__________________|                 │
│                    (agentic loop, max 3 iterations)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     DATA LAYER                               │
│  ChromaDB (vector, per-tenant) · BM25 (lexical, per-tenant) │
│  SQLite (tenants, users, docs, query logs, webhooks)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### LangGraph for Modular RAG
The RAG pipeline is implemented as a LangGraph `StateGraph`. This gives us:
- **Native cycles** — the adequacy loop retries retrieval without manual recursion
- **State reducers** — `Annotated[list, operator.add]` accumulates chunks across iterations automatically
- **Checkpointing** — full execution replay for debugging
- **Conditional routing** — query type drives which retrieval strategy activates

### Multi-tenant isolation
Each tenant gets physically isolated storage:
- `data/chroma/{tenant_id}/` — separate ChromaDB PersistentClient
- `data/bm25/{tenant_id}/index.pkl` — separate BM25 index

No query can ever cross tenant boundaries. `tenant_id` is injected from the validated JWT on every request — it is never a user-supplied parameter.

### Code-aware chunking
Different document types split at different boundaries:
- **Code** → function/class definitions (`def`, `class`, `func`)
- **Runbooks/SOPs** → H2/H3 markdown headings
- **Incidents** → standard sections (Timeline, Root Cause, Action Items)
- **API docs** → endpoint definitions (GET/POST/PUT...)
- **Prose** → token sliding window with overlap

### Hybrid search + RRF
Vector search (semantic) and BM25 (lexical) run independently per query. Reciprocal Rank Fusion merges the result lists without needing score normalization. Incident and API doc queries double-weight BM25 to boost exact entity matching.

### Streaming with SSE
The `/query/stream` endpoint uses `EventSourceResponse` (sse-starlette). LangGraph's `astream_events` emits node lifecycle events, giving clients real-time progress:
```
event: node_start  → {"node": "analyze"}
event: node_done   → {"node": "analyze", "query_type": "code", "queries": [...]}
event: node_start  → {"node": "retrieve"}
event: node_done   → {"node": "retrieve", "chunks": 6, "iteration": 1}
event: node_done   → {"node": "adequacy", "sufficient": true, "confidence": 0.87}
event: done        → {"answer": "...", "sources": [...], "latency_ms": 1240}
```

---

## Project Structure

```
nexus/
├── app/
│   ├── api/v1/routes.py        # All FastAPI endpoints
│   ├── core/
│   │   ├── auth.py             # JWT + RBAC
│   │   ├── config.py           # Pydantic settings
│   │   ├── database.py         # SQLAlchemy async models
│   │   ├── schemas.py          # Pydantic request/response
│   │   └── vectorstore.py      # Multi-tenant ChromaDB + BM25
│   ├── graph/
│   │   ├── nodes.py            # All LangGraph nodes + state
│   │   ├── graph.py            # Graph compilation
│   │   └── runner.py           # run() + stream_run() (SSE)
│   ├── ingestion/
│   │   ├── processor.py        # Code-aware chunking
│   │   ├── indexer.py          # Full index pipeline
│   │   └── scheduler.py        # APScheduler delta sync
│   ├── integrations/
│   │   ├── slack.py            # Slack bot (slash cmd + mention)
│   │   └── webhooks.py         # HMAC-signed webhook delivery
│   └── analytics/
│       └── service.py          # Query metrics + RAGAS aggregation
├── tests/
│   └── unit/test_nexus.py      # 20+ unit tests (no API key needed)
├── main.py                     # FastAPI app + lifespan
├── requirements.txt
└── .env.example
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/nexus
cd nexus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set OPENAI_API_KEY in .env

mkdir -p data
python main.py
# → http://localhost:8000/docs
```

### 1. Create a tenant

```bash
curl -X POST http://localhost:8000/api/v1/tenants \
  -H "Content-Type: application/json" \
  -d '{"id": "acme-corp", "name": "Acme Corp", "plan": "growth"}'
```

### 2. Create an admin user

```bash
curl -X POST http://localhost:8000/api/v1/tenants/acme-corp/users \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@acme.com", "password": "secure123", "role": "admin"}'
```

### 3. Login + get JWT

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@acme.com", "password": "secure123"}' \
  | jq -r .access_token)
```

### 4. Index engineering docs

```bash
# Index a runbook
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "upload",
    "source_ref":  "docs/deploy-runbook.md",
    "doc_type":    "runbook",
    "title":       "Production Deployment Runbook"
  }'

# Index Python source
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "upload",
    "source_ref":  "src/auth/service.py",
    "doc_type":    "code",
    "title":       "Auth Service"
  }'
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I roll back a failed deployment?"}'
```

### 6. Stream a question (SSE)

```bash
curl -N "http://localhost:8000/api/v1/query/stream?question=What+caused+the+DB+outage+last+month" \
  -H "Authorization: Bearer $TOKEN"
```

### 7. Register a webhook

```bash
curl -X POST http://localhost:8000/api/v1/webhooks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url":    "https://your-server.com/nexus-events",
    "events": ["query.completed", "index.done"]
  }'
```

### 8. View analytics

```bash
curl http://localhost:8000/api/v1/analytics?days=30 \
  -H "Authorization: Bearer $TOKEN"
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET  | `/api/v1/health` | None | System health |
| POST | `/api/v1/auth/login` | None | Get JWT token |
| POST | `/api/v1/tenants` | None | Create tenant |
| POST | `/api/v1/tenants/{id}/users` | Admin | Create user |
| POST | `/api/v1/documents` | Member+ | Index a document |
| GET  | `/api/v1/documents` | Any | List documents |
| GET  | `/api/v1/documents/stats` | Any | Chunk count |
| DELETE | `/api/v1/documents/{id}` | Admin | Delete document |
| POST | `/api/v1/query` | Any | Ask a question |
| GET  | `/api/v1/query/stream` | Any | SSE streaming query |
| POST | `/api/v1/webhooks` | Admin | Register webhook |
| GET  | `/api/v1/webhooks` | Admin | List webhooks |
| DELETE | `/api/v1/webhooks/{id}` | Admin | Delete webhook |
| GET  | `/api/v1/analytics` | Member+ | Dashboard metrics |
| POST | `/api/v1/integrations/slack/events` | Slack | Slack Events API |
| POST | `/api/v1/integrations/slack/commands` | Slack | Slash commands |

---

## Running Tests

```bash
# Unit tests — no API key or server needed
pytest tests/ -v -m "not integration"

# Full suite — requires running server
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI + uvicorn |
| RAG orchestration | LangGraph (StateGraph) |
| LLM + embeddings | OpenAI (gpt-4o, text-embedding-3-large) |
| Reranker | sentence-transformers (ms-marco-MiniLM-L-6-v2) |
| Vector store | ChromaDB (persistent, per-tenant) |
| Lexical search | rank-bm25 |
| Database | SQLAlchemy async + SQLite |
| Auth | JWT (python-jose) + bcrypt |
| Streaming | sse-starlette |
| Scheduling | APScheduler |
| Webhooks | httpx + HMAC-SHA256 |
| Slack | slack-sdk |
| Logging | structlog (JSON) |
| Evaluation | RAGAS |
