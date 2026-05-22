"""
Graph runner — two modes:
  1. run()        → await full response (REST API)
  2. stream_run() → async generator of SSE events (streaming endpoint)
"""
from __future__ import annotations
import json, time, uuid
from typing import AsyncGenerator

from app.graph.graph  import nexus_graph
from app.graph.nodes  import NexusState
from app.core.schemas import QueryResponse, ChunkOut


async def run(
    tenant_id:       str,
    user_id:         str,
    question:        str,
    doc_type_filter: str | None = None,
) -> QueryResponse:
    """Run the full Modular RAG graph and return a complete response."""
    query_id = str(uuid.uuid4())
    start_ts = time.perf_counter()

    initial_state: NexusState = {
        "tenant_id":      tenant_id,
        "user_id":        user_id,
        "question":       question,
        "doc_type_filter": doc_type_filter,
        "query_id":       query_id,
        "start_ts":       start_ts,
        # Defaults (nodes will fill these)
        "query_type":     "general",
        "sub_questions":  [],
        "entities":       [],
        "search_queries": [],
        "all_chunks":     [],
        "current_chunks": [],
        "iteration":      0,
        "max_iterations": 3,
        "is_sufficient":  False,
        "follow_up_query": None,
        "confidence":     0.0,
        "answer":         "",
        "sources":        [],
        "faithfulness":   None,
        "answer_relevancy": None,
    }

    config = {"configurable": {"thread_id": query_id}}
    final  = await nexus_graph.ainvoke(initial_state, config=config)

    latency_ms = round((time.perf_counter() - start_ts) * 1000, 1)

    chunks_out = [
        ChunkOut(
            text    = c["text"],
            score   = c.get("rerank_score", c.get("rrf_score", 0.0)),
            source  = c.get("metadata", {}).get("source_type", ""),
            doc_type= c.get("metadata", {}).get("doc_type"),
            title   = c.get("metadata", {}).get("title"),
        )
        for c in (final.get("all_chunks") or [])[:6]
    ]

    return QueryResponse(
        query_id          = query_id,
        answer            = final.get("answer", ""),
        query_type        = final.get("query_type", "general"),
        rewritten_queries = final.get("search_queries", [question]),
        chunks_used       = chunks_out,
        sources           = final.get("sources", []),
        confidence        = final.get("confidence", 0.0),
        iterations        = final.get("iteration", 1),
        latency_ms        = latency_ms,
    )


async def stream_run(
    tenant_id:       str,
    user_id:         str,
    question:        str,
    doc_type_filter: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Yield Server-Sent Events as the graph executes.
    Each node completion sends an event so the client sees progress live.

    Event types:
      node_start      → which node just started
      node_done       → node completed with key outputs
      token           → generation token (streamed from LLM)
      done            → final complete response
      error           → pipeline error
    """
    query_id = str(uuid.uuid4())
    start_ts = time.perf_counter()

    initial_state: NexusState = {
        "tenant_id":      tenant_id,
        "user_id":        user_id,
        "question":       question,
        "doc_type_filter": doc_type_filter,
        "query_id":       query_id,
        "start_ts":       start_ts,
        "query_type":     "general",
        "sub_questions":  [],
        "entities":       [],
        "search_queries": [],
        "all_chunks":     [],
        "current_chunks": [],
        "iteration":      0,
        "max_iterations": 3,
        "is_sufficient":  False,
        "follow_up_query": None,
        "confidence":     0.0,
        "answer":         "",
        "sources":        [],
        "faithfulness":   None,
        "answer_relevancy": None,
    }

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        config = {"configurable": {"thread_id": query_id}}

        async for event in nexus_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name",  "")

            if kind == "on_chain_start" and name in ("analyze", "retrieve", "adequacy", "generate"):
                yield _sse("node_start", {"node": name})

            elif kind == "on_chain_end" and name == "analyze":
                out = event.get("data", {}).get("output", {})
                yield _sse("node_done", {
                    "node":       "analyze",
                    "query_type": out.get("query_type"),
                    "queries":    out.get("search_queries", []),
                })

            elif kind == "on_chain_end" and name == "retrieve":
                out = event.get("data", {}).get("output", {})
                yield _sse("node_done", {
                    "node":      "retrieve",
                    "chunks":    len(out.get("current_chunks") or []),
                    "iteration": out.get("iteration", 1),
                })

            elif kind == "on_chain_end" and name == "adequacy":
                out = event.get("data", {}).get("output", {})
                yield _sse("node_done", {
                    "node":         "adequacy",
                    "sufficient":   out.get("is_sufficient"),
                    "confidence":   out.get("confidence"),
                })

            elif kind == "on_chain_end" and name == "generate":
                out        = event.get("data", {}).get("output", {})
                latency_ms = round((time.perf_counter() - start_ts) * 1000, 1)
                yield _sse("done", {
                    "query_id":   query_id,
                    "answer":     out.get("answer", ""),
                    "sources":    out.get("sources", []),
                    "confidence": out.get("confidence", 0.0),
                    "latency_ms": latency_ms,
                })

    except Exception as e:
        yield _sse("error", {"message": str(e), "query_id": query_id})
