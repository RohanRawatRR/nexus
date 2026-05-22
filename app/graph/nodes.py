"""
Nexus LangGraph — Modular RAG graph nodes and state.

Graph flow:
  START
    → analyze_query
    → route_query          (conditional: sets active modules)
    → hybrid_retrieve      (vector + BM25 + RRF)
    → adequacy_check       (conditional: loop or continue)
    → generate_answer      (streams tokens)
    → self_evaluate        (async RAGAS)
  END

State accumulates chunks across iterations via reducer.
"""
from __future__ import annotations
import json, time, uuid
from typing import TypedDict, Annotated, Optional
import operator

from openai                  import AsyncOpenAI
from sentence_transformers   import CrossEncoder
from functools               import lru_cache

from app.core.config         import get_settings
from app.core.vectorstore    import vector_search, BM25Store
from app.core.schemas        import QueryType, DocType

settings = get_settings()
_oai     = AsyncOpenAI(api_key=settings.openai_api_key)


# ── State ──────────────────────────────────────────────────────

class NexusState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    tenant_id:      str
    user_id:        str
    question:       str
    doc_type_filter: Optional[str]

    # ── Analysis ─────────────────────────────────────────────
    query_type:     str
    sub_questions:  list[str]
    entities:       list[str]
    search_queries: list[str]          # after rewriting / HyDE

    # ── Retrieval (accumulates across iterations) ────────────
    all_chunks:     Annotated[list, operator.add]
    current_chunks: list
    iteration:      int
    max_iterations: int

    # ── Adequacy ─────────────────────────────────────────────
    is_sufficient:  bool
    follow_up_query: Optional[str]
    confidence:     float

    # ── Generation ───────────────────────────────────────────
    answer:         str
    sources:        list[str]

    # ── Eval (async, written after response sent) ─────────────
    faithfulness:       Optional[float]
    answer_relevancy:   Optional[float]

    # ── Metadata ─────────────────────────────────────────────
    query_id:       str
    start_ts:       float


# ── LLM helpers ───────────────────────────────────────────────

async def _embed(texts: list[str]) -> list[list[float]]:
    res = await _oai.embeddings.create(model=settings.embedding_model, input=texts)
    return [r.embedding for r in res.data]


async def _chat(prompt: str, system: str = "", as_json: bool = False) -> str:
    kwargs: dict = dict(
        model       = settings.chat_model,
        temperature = 0.0,
        messages    = [
            {"role": "system", "content": system or "You are a precise engineering assistant."},
            {"role": "user",   "content": prompt},
        ],
    )
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}
    res = await _oai.chat.completions.create(**kwargs)
    return res.choices[0].message.content


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


# ── Node 1: analyze_query ──────────────────────────────────────

_ANALYZE_PROMPT = """You are analyzing an engineering question to optimize retrieval.

Question: {question}

Classify and decompose. Output JSON only:
{{
  "query_type": "code|runbook|incident|api|general",
  "sub_questions": ["list if complex, else just the original"],
  "entities": ["service names, repo names, API names, error codes mentioned"],
  "needs_hyde": true/false
}}

query_type guide:
  code     → implementation questions, "how does X work", debugging
  runbook  → deployment steps, operational procedures, "how to"
  incident → past outages, post-mortems, "what caused", "when did X fail"
  api      → endpoints, request/response schemas, authentication
  general  → spans multiple types"""


async def analyze_query_node(state: NexusState) -> dict:
    raw    = await _chat(
        _ANALYZE_PROMPT.format(question=state["question"]),
        as_json=True,
    )
    data   = json.loads(raw)

    # Generate search queries
    q_type = data.get("query_type", "general")
    subs   = data.get("sub_questions", [state["question"]])

    if data.get("needs_hyde"):
        queries = [await _hyde(state["question"])]
    else:
        queries = subs if subs else [state["question"]]

    return {
        "query_type":    q_type,
        "sub_questions": subs,
        "entities":      data.get("entities", []),
        "search_queries": queries,
        "iteration":     0,
        "max_iterations": settings.max_rag_iterations,
        "is_sufficient": False,
        "confidence":    0.0,
        "all_chunks":    [],
    }


async def _hyde(question: str) -> str:
    """Generate a hypothetical engineering doc to improve embedding alignment."""
    return await _chat(
        f"""Write a short technical documentation passage (3-4 sentences) that would
directly answer this engineering question. Use technical language, code terms,
and specific details as if from real internal docs.

Question: {question}
Passage:""",
        temperature=0.1,
    )


# ── Node 2: hybrid_retrieve ────────────────────────────────────

def _rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists."""
    from collections import defaultdict
    scores: dict = defaultdict(float)
    chunks: dict = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked):
            uid = f"{chunk['metadata'].get('source_ref','')}_{chunk['metadata'].get('parent_index','')}_{chunk['metadata'].get('chunk_index','')}"
            scores[uid] += 1.0 / (k + rank + 1)
            chunks[uid]  = chunk
    top = sorted(scores, key=scores.get, reverse=True)
    return [{**chunks[uid], "rrf_score": round(scores[uid], 5)} for uid in top]


async def hybrid_retrieve_node(state: NexusState) -> dict:
    iteration = state["iteration"]

    # On re-retrieval use the follow-up query from adequacy check
    if iteration > 0 and state.get("follow_up_query"):
        queries = [state["follow_up_query"]]
    else:
        queries = state["search_queries"]

    # Build metadata filter
    where = None
    if state.get("doc_type_filter"):
        where = {"doc_type": {"$eq": state["doc_type_filter"]}}
    if state["entities"]:
        # Prioritize docs mentioning key entities via BM25
        pass

    all_lists = []
    bm25_store = BM25Store(state["tenant_id"])

    for q in queries:
        # Vector retrieval
        q_emb    = (await _embed([q]))[0]
        vec_res  = vector_search(state["tenant_id"], q_emb, k=settings.top_k_retrieval, where=where)
        if vec_res:
            all_lists.append(vec_res)

        # BM25 retrieval
        bm25_res = bm25_store.search(q, k=settings.top_k_retrieval)
        if bm25_res:
            all_lists.append(bm25_res)

        # Double BM25 weight for incident/api (entity-heavy)
        if state["query_type"] in ("incident", "api") and bm25_res:
            all_lists.append(bm25_res)

    if not all_lists:
        return {"current_chunks": [], "iteration": iteration + 1}

    # Fuse
    fused = _rrf(all_lists)

    # Swap child → parent text
    seen, unique = set(), []
    for chunk in fused:
        meta  = chunk.get("metadata", {})
        p_key = f"{meta.get('source_ref')}_{meta.get('parent_index')}"
        if p_key not in seen:
            seen.add(p_key)
            c = chunk.copy()
            c["text"] = meta.get("parent_text", chunk["text"])
            unique.append(c)

    # Cross-encoder reranking
    if unique:
        q_main = queries[0]
        pairs  = [(q_main, c["text"]) for c in unique]
        import numpy as np
        scores = _reranker().predict(pairs)
        norm   = (1 / (1 + np.exp(-np.array(scores)))).tolist()
        ranked = sorted(zip(norm, unique), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, chunk in ranked[:settings.top_k_rerank]:
            c = chunk.copy()
            c["rerank_score"] = round(float(score), 4)
            reranked.append(c)
    else:
        reranked = unique[:settings.top_k_rerank]

    return {
        "current_chunks": reranked,
        "all_chunks":     reranked,     # reducer appends across iterations
        "iteration":      iteration + 1,
    }


# ── Node 3: adequacy_check ─────────────────────────────────────

_ADEQUACY_PROMPT = """You are evaluating whether retrieved engineering documentation
is sufficient to answer the question completely.

Question: {question}
Retrieved context (first 3 chunks):
{context}

Respond JSON only:
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "gaps": ["what is still missing"],
  "follow_up_query": "targeted search query if needed, else null"
}}"""


async def adequacy_check_node(state: NexusState) -> dict:
    chunks  = state["current_chunks"]
    if not chunks:
        return {
            "is_sufficient":   False,
            "confidence":      0.0,
            "follow_up_query": state["question"],
        }

    context = "\n\n---\n\n".join(c["text"] for c in chunks[:3])
    raw     = await _chat(
        _ADEQUACY_PROMPT.format(question=state["question"], context=context),
        as_json=True,
    )
    data = json.loads(raw)
    return {
        "is_sufficient":   data.get("is_sufficient", False),
        "confidence":      round(float(data.get("confidence", 0.5)), 3),
        "follow_up_query": data.get("follow_up_query"),
    }


# ── Node 4: generate_answer ────────────────────────────────────

_GEN_SYSTEM = """You are Nexus, an enterprise engineering copilot.
Answer questions using only the provided documentation context.
Be precise, cite sources, and include code examples where relevant.
If the context is insufficient, say so explicitly — never hallucinate."""

_GEN_PROMPT = """Answer the engineering question using ONLY the context below.

Rules:
- Cite the source (title/filename) for each key claim
- Include relevant code snippets exactly as they appear in the context
- For runbooks/SOPs: preserve step numbering
- Do NOT add information not present in the context

<context>
{context}
</context>

<question>{question}</question>"""


async def generate_answer_node(state: NexusState) -> dict:
    # Deduplicate all chunks across iterations
    seen, unique = set(), []
    for chunk in state["all_chunks"]:
        meta  = chunk.get("metadata", {})
        key   = f"{meta.get('source_ref')}_{meta.get('parent_index')}"
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('title') or c['metadata'].get('source_ref','unknown')}]\n{c['text']}"
        for c in unique[:8]
    )
    answer  = await _chat(
        _GEN_PROMPT.format(context=context, question=state["question"]),
        system=_GEN_SYSTEM,
    )
    sources = list({
        c["metadata"].get("title") or c["metadata"].get("source_ref", "")
        for c in unique[:8]
    })
    scores  = [c.get("rerank_score", 0.5) for c in unique[:8]]
    import numpy as np
    confidence = round(float(np.mean(scores)) if scores else 0.5, 3)

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": confidence,
    }


# ── Routing functions ──────────────────────────────────────────

def should_continue_retrieving(state: NexusState) -> str:
    if (
        not state.get("is_sufficient")
        and state["iteration"] < state["max_iterations"]
        and state.get("follow_up_query")
    ):
        return "retrieve"
    return "generate"
