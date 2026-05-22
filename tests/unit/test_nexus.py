"""
Nexus test suite.
pytest tests/ -v -m "not integration"
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


# ── Unit: chunking ─────────────────────────────────────────────

class TestCodeChunking:
    def test_python_functions_split_at_def(self):
        from app.ingestion.processor import _chunk_code
        code = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        chunks = _chunk_code(code)
        assert len(chunks) >= 2
        assert any("def foo" in c for c in chunks)
        assert any("def bar" in c for c in chunks)

    def test_large_function_falls_back_to_window(self):
        from app.ingestion.processor import _chunk_code
        big_func = "def huge():\n" + "    x = 1\n" * 500
        chunks   = _chunk_code(big_func)
        assert len(chunks) >= 2

    def test_markdown_splits_on_headings(self):
        from app.ingestion.processor import _chunk_by_headings
        md = "## Deploy\nStep 1\n\n## Rollback\nStep 2\n"
        chunks = _chunk_by_headings(md)
        assert any("Deploy" in c for c in chunks)
        assert any("Rollback" in c for c in chunks)

    def test_incident_splits_on_sections(self):
        from app.ingestion.processor import _chunk_incident
        report = "## Summary\nOutage.\n\n## Timeline\n10:00 Alert\n\n## Root Cause\nDB issue\n"
        chunks = _chunk_incident(report)
        assert any("Summary" in c for c in chunks)
        assert any("Root Cause" in c for c in chunks)

    def test_parent_child_structure(self):
        from app.ingestion.processor import _build_parent_child
        raws   = ["chunk A", "chunk B", "chunk C", "chunk D"]
        result = _build_parent_child(raws)
        assert all("child_text"   in c for c in result)
        assert all("parent_text"  in c for c in result)
        assert all("parent_index" in c for c in result)
        # Parent must contain the child
        for c in result:
            assert c["child_text"] in c["parent_text"]

    def test_token_count(self):
        from app.ingestion.processor import count_tokens
        assert count_tokens("hello world") > 0
        assert count_tokens("") == 0


# ── Unit: RRF ──────────────────────────────────────────────────

class TestRRF:
    def test_merges_two_lists(self):
        from app.graph.nodes import _rrf
        list1 = [{"text": "a", "metadata": {"source_ref": "f", "parent_index": 0, "chunk_index": 0}}]
        list2 = [{"text": "a", "metadata": {"source_ref": "f", "parent_index": 0, "chunk_index": 0}}]
        result = _rrf([list1, list2])
        assert len(result) == 1
        assert result[0]["rrf_score"] > 0

    def test_higher_rank_gets_higher_score(self):
        from app.graph.nodes import _rrf
        def c(ref):
            return {"text": ref, "metadata": {"source_ref": ref, "parent_index": 0, "chunk_index": 0}}
        list1 = [c("best"), c("second"), c("third")]
        list2 = [c("best"), c("third"),  c("second")]
        result = _rrf([list1, list2])
        scores = {r["text"]: r["rrf_score"] for r in result}
        assert scores["best"] > scores["second"]
        assert scores["best"] > scores["third"]

    def test_empty_lists_handled(self):
        from app.graph.nodes import _rrf
        assert _rrf([]) == []
        assert _rrf([[]]) == []


# ── Unit: auth ─────────────────────────────────────────────────

class TestAuth:
    def test_password_hash_and_verify(self):
        from app.core.auth import hash_password, verify_password
        hashed = hash_password("secret123")
        assert verify_password("secret123", hashed)
        assert not verify_password("wrong", hashed)

    def test_create_and_decode_token(self):
        from app.core.auth import create_access_token, decode_token
        token   = create_access_token("user_1", "acme-corp", "admin")
        payload = decode_token(token)
        assert payload.sub       == "user_1"
        assert payload.tenant_id == "acme-corp"
        assert payload.role      == "admin"

    def test_invalid_token_raises(self):
        from app.core.auth import decode_token
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            decode_token("invalid.token.here")
        assert exc.value.status_code == 401


# ── Unit: webhook signing ──────────────────────────────────────

class TestWebhooks:
    def test_sign_and_verify(self):
        from app.integrations.webhooks import _sign, verify_signature
        secret  = "test-secret"
        payload = b'{"event": "query.completed"}'
        sig     = _sign(secret, payload)
        assert verify_signature(secret, payload, sig)

    def test_wrong_secret_fails(self):
        from app.integrations.webhooks import _sign, verify_signature
        payload = b'{"event": "test"}'
        sig     = _sign("secret-a", payload)
        assert not verify_signature("secret-b", payload, sig)

    def test_tampered_payload_fails(self):
        from app.integrations.webhooks import _sign, verify_signature
        sig = _sign("secret", b"original")
        assert not verify_signature("secret", b"tampered", sig)


# ── Unit: schemas ──────────────────────────────────────────────

class TestSchemas:
    def test_query_type_enum(self):
        from app.core.schemas import QueryType
        assert QueryType("code")     == QueryType.CODE
        assert QueryType("incident") == QueryType.INCIDENT

    def test_query_request_min_length(self):
        from app.core.schemas import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="hi")   # too short

    def test_index_request_valid(self):
        from app.core.schemas import IndexRequest, SourceType, DocType
        req = IndexRequest(
            source_type = SourceType.UPLOAD,
            source_ref  = "/tmp/runbook.md",
            doc_type    = DocType.RUNBOOK,
        )
        assert req.doc_type == DocType.RUNBOOK


# ── Unit: analytics ────────────────────────────────────────────

class TestAnalytics:
    @pytest.mark.asyncio
    async def test_empty_logs_returns_zeros(self):
        from app.analytics.service import get_analytics
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: [])))
        result = await get_analytics(mock_db, "test-tenant", days=7)
        assert result.total_queries == 0
        assert result.avg_latency_ms == 0.0


# ── Integration: API ───────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
class TestAPI:
    async def test_health(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000") as c:
            r = await c.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_login_invalid(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000") as c:
            r = await c.post("/api/v1/auth/login", json={
                "email": "nobody@example.com", "password": "wrong"
            })
        assert r.status_code == 401

    async def test_query_requires_auth(self):
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000") as c:
            r = await c.post("/api/v1/query", json={"question": "How does auth work?"})
        assert r.status_code == 403
