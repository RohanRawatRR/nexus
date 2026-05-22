"""
Nexus LangGraph — graph compilation.

The graph is compiled once at startup and reused for all requests.
Tenant isolation is enforced via state fields (tenant_id), not graph instances.
"""
from langgraph.graph             import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.nodes import (
    NexusState,
    analyze_query_node,
    hybrid_retrieve_node,
    adequacy_check_node,
    generate_answer_node,
    should_continue_retrieving,
)


def build_graph() -> StateGraph:
    builder = StateGraph(NexusState)

    # ── Register nodes ────────────────────────────────────────
    builder.add_node("analyze",   analyze_query_node)
    builder.add_node("retrieve",  hybrid_retrieve_node)
    builder.add_node("adequacy",  adequacy_check_node)
    builder.add_node("generate",  generate_answer_node)

    # ── Fixed edges ───────────────────────────────────────────
    builder.add_edge(START,      "analyze")
    builder.add_edge("analyze",  "retrieve")
    builder.add_edge("retrieve", "adequacy")

    # ── Conditional edge: agentic loop or generate ─────────────
    builder.add_conditional_edges(
        "adequacy",
        should_continue_retrieving,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    builder.add_edge("generate", END)

    return builder


# Compile with in-memory checkpointer (swap for Redis/Postgres in production)
_checkpointer = MemorySaver()
nexus_graph   = build_graph().compile(checkpointer=_checkpointer)
