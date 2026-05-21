from pydantic import BaseModel, Field, EmailStr
from typing   import Optional, Any
from enum     import Enum
from datetime import datetime


# ── Enums ──────────────────────────────────────────────────────

class DocType(str, Enum):
    CODE      = "code"
    RUNBOOK   = "runbook"
    INCIDENT  = "incident"
    API_DOC   = "api_doc"
    SOP       = "sop"
    OTHER     = "other"

class SourceType(str, Enum):
    GITHUB      = "github"
    CONFLUENCE  = "confluence"
    PAGERDUTY   = "pagerduty"
    NOTION      = "notion"
    UPLOAD      = "upload"

class QueryType(str, Enum):
    CODE        = "code"       # implementation, "how does X work"
    RUNBOOK     = "runbook"    # operational procedures
    INCIDENT    = "incident"   # past incidents, post-mortems
    API         = "api"        # API docs, endpoints, schemas
    GENERAL     = "general"    # cross-domain

class UserRole(str, Enum):
    ADMIN   = "admin"
    MEMBER  = "member"
    VIEWER  = "viewer"

class WebhookEvent(str, Enum):
    QUERY_COMPLETED = "query.completed"
    INDEX_DONE      = "index.done"
    INDEX_FAILED    = "index.failed"


# ── Auth ───────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int

class TenantCreate(BaseModel):
    id:   str  = Field(..., pattern=r"^[a-z0-9-]+$", description="Slug: acme-corp")
    name: str
    plan: str  = "starter"

class UserCreate(BaseModel):
    email:     str
    password:  str  = Field(..., min_length=8)
    role:      UserRole = UserRole.MEMBER


# ── Documents ─────────────────────────────────────────────────

class IndexRequest(BaseModel):
    source_type:  SourceType
    source_ref:   str   = Field(..., description="URL, repo path, page ID, or file path")
    title:        Optional[str]    = None
    doc_type:     DocType          = DocType.OTHER
    metadata:     dict             = Field(default_factory=dict)

class IndexResponse(BaseModel):
    document_id:  str
    chunks_added: int
    source_ref:   str
    message:      str

class DocumentOut(BaseModel):
    id:           str
    source_type:  str
    source_ref:   str
    title:        Optional[str]
    doc_type:     str
    chunk_count:  int
    last_indexed: Optional[datetime]

    class Config:
        from_attributes = True


# ── Query ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:       str   = Field(..., min_length=3)
    doc_type_filter: Optional[DocType]   = None
    stream:         bool  = False

class ChunkOut(BaseModel):
    text:      str
    score:     float
    source:    str
    doc_type:  Optional[str] = None
    title:     Optional[str] = None

class QueryResponse(BaseModel):
    query_id:          str
    answer:            str
    query_type:        QueryType
    rewritten_queries: list[str]
    chunks_used:       list[ChunkOut]
    sources:           list[str]
    confidence:        float
    iterations:        int
    latency_ms:        float
    faithfulness:      Optional[float] = None
    answer_relevancy:  Optional[float] = None


# ── Webhooks ───────────────────────────────────────────────────

class WebhookCreate(BaseModel):
    url:    str
    events: list[WebhookEvent]
    secret: Optional[str] = None   # auto-generated if not provided

class WebhookOut(BaseModel):
    id:       str
    url:      str
    events:   list[str]
    is_active: bool

    class Config:
        from_attributes = True


# ── Analytics ─────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    period:              str
    total_queries:       int
    avg_latency_ms:      float
    avg_confidence:      float
    avg_faithfulness:    Optional[float]
    avg_answer_relevancy: Optional[float]
    top_query_types:     dict[str, int]
    queries_per_day:     list[dict]
