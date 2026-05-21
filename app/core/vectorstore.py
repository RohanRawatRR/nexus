"""
Multi-tenant vector store.
Each tenant gets its own ChromaDB PersistentClient at:
  {chroma_base_dir}/{tenant_id}/

This ensures:
  - Complete data isolation at the storage level
  - No cross-tenant leakage is possible
  - Each tenant's index can be independently backed up or deleted
"""
import os
import pickle
from pathlib import Path
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.core.config import get_settings

settings = get_settings()


# ── ChromaDB (vector) ──────────────────────────────────────────

def _chroma_path(tenant_id: str) -> str:
    path = Path(settings.chroma_base_dir) / tenant_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_chroma_collection(tenant_id: str) -> chromadb.Collection:
    """Get or create the ChromaDB collection for a tenant."""
    client = chromadb.PersistentClient(
        path=_chroma_path(tenant_id),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=f"nexus_{tenant_id}",
        metadata={"hnsw:space": "cosine"},
    )


def vector_add(
    tenant_id:  str,
    ids:        list[str],
    documents:  list[str],
    embeddings: list[list[float]],
    metadatas:  list[dict],
) -> None:
    """Add chunks to tenant's vector store. Skips existing IDs."""
    col      = get_chroma_collection(tenant_id)
    existing = set(col.get(ids=ids)["ids"])
    mask     = [i for i, id_ in enumerate(ids) if id_ not in existing]
    if not mask:
        return
    col.add(
        ids        = [ids[i]        for i in mask],
        documents  = [documents[i]  for i in mask],
        embeddings = [embeddings[i] for i in mask],
        metadatas  = [metadatas[i]  for i in mask],
    )


def vector_search(
    tenant_id:       str,
    query_embedding: list[float],
    k:               int,
    where:           Optional[dict] = None,
) -> list[dict]:
    """ANN search in tenant's collection."""
    col  = get_chroma_collection(tenant_id)
    n    = min(k, col.count())
    if n == 0:
        return []

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)
    return [
        {
            "text":     res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "score":    round(1.0 - res["distances"][0][i], 4),
            "source":   "vector",
        }
        for i in range(len(res["documents"][0]))
    ]


def vector_count(tenant_id: str) -> int:
    return get_chroma_collection(tenant_id).count()


def vector_delete_tenant(tenant_id: str) -> None:
    """Wipe all vectors for a tenant (on offboarding)."""
    import shutil
    shutil.rmtree(_chroma_path(tenant_id), ignore_errors=True)


# ── BM25 (lexical) ─────────────────────────────────────────────

def _bm25_path(tenant_id: str) -> Path:
    path = Path(settings.bm25_base_dir) / tenant_id
    path.mkdir(parents=True, exist_ok=True)
    return path / "index.pkl"


class BM25Store:
    """
    Persistent BM25 index per tenant.
    Serialised with pickle between requests.
    For large tenants, swap this for Elasticsearch/OpenSearch.
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._path     = _bm25_path(tenant_id)
        self.corpus_texts: list[str]  = []
        self.corpus_meta:  list[dict] = []
        self._bm25:        Optional[BM25Okapi] = None
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self.corpus_texts = data["texts"]
            self.corpus_meta  = data["meta"]
            self._rebuild()

    def _save(self):
        with open(self._path, "wb") as f:
            pickle.dump({"texts": self.corpus_texts, "meta": self.corpus_meta}, f)

    def _rebuild(self):
        if self.corpus_texts:
            tokenized   = [t.lower().split() for t in self.corpus_texts]
            self._bm25  = BM25Okapi(tokenized)

    def add(self, texts: list[str], metadatas: list[dict]):
        self.corpus_texts.extend(texts)
        self.corpus_meta.extend(metadatas)
        self._rebuild()
        self._save()

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not self._bm25 or not self.corpus_texts:
            return []
        import numpy as np
        scores  = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append({
                    "text":     self.corpus_texts[idx],
                    "metadata": self.corpus_meta[idx],
                    "score":    round(float(scores[idx]), 4),
                    "source":   "bm25",
                })
        return results
