"""
Document indexer.
Orchestrates: extract → chunk → embed → store (vector + BM25)
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from openai                 import AsyncOpenAI

from app.core.config     import get_settings
from app.core.database   import Document
from app.core.vectorstore import vector_add, BM25Store
from app.core.schemas    import DocType, SourceType
from app.ingestion.processor import extract_text, chunk_document

settings = get_settings()
_oai     = AsyncOpenAI(api_key=settings.openai_api_key)


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed in batches of 100 to respect API limits."""
    all_embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        res   = await _oai.embeddings.create(model=settings.embedding_model, input=batch)
        all_embs.extend(r.embedding for r in res.data)
    return all_embs


async def index_document(
    tenant_id:   str,
    source_type: str,
    source_ref:  str,
    doc_type:    str,
    title:       str | None,
    db:          AsyncSession,
    metadata:    dict | None = None,
) -> dict:
    """
    Full indexing pipeline for one document.
    Returns summary dict.
    """
    doc_id    = str(uuid.uuid4())
    meta      = metadata or {}

    # 1. Extract text
    pages     = extract_text(source_ref, source_type)
    if not pages:
        raise ValueError(f"No text extracted from {source_ref}")

    # 2. Chunk (code-aware)
    chunks    = chunk_document(pages, DocType(doc_type))

    # 3. Build IDs + metadata for each child chunk
    ids, child_texts, metadatas = [], [], []
    parent_texts, parent_metas  = [], []

    for chunk in chunks:
        chunk_id = f"{tenant_id}_{doc_id}_p{chunk['parent_index']}_c{chunk['chunk_index']}"
        ids.append(chunk_id)
        child_texts.append(chunk["child_text"])
        parent_texts.append(chunk["parent_text"])
        metadatas.append({
            "doc_id":       doc_id,
            "source_type":  source_type,
            "source_ref":   source_ref,
            "doc_type":     doc_type,
            "title":        title or source_ref,
            "parent_index": chunk["parent_index"],
            "chunk_index":  chunk["chunk_index"],
            "parent_text":  chunk["parent_text"],
            **meta,
        })
        parent_metas.append({
            "doc_id":      doc_id,
            "source_ref":  source_ref,
            "doc_type":    doc_type,
            "title":       title or source_ref,
        })

    # 4. Embed child texts
    embeddings = await _embed_batch(child_texts)

    # 5. Store in vector DB (children)
    vector_add(tenant_id, ids, child_texts, embeddings, metadatas)

    # 6. Store in BM25 index (parents — better for keyword search over full context)
    bm25 = BM25Store(tenant_id)
    bm25.add(parent_texts, parent_metas)

    # 7. Upsert document record in SQLite
    doc_record = Document(
        id           = doc_id,
        tenant_id    = tenant_id,
        source_type  = source_type,
        source_ref   = source_ref,
        title        = title,
        doc_type     = doc_type,
        chunk_count  = len(chunks),
        last_indexed = datetime.now(timezone.utc),
        metadata_    = meta,
    )
    db.add(doc_record)
    await db.commit()

    return {
        "document_id":  doc_id,
        "chunks_added": len(chunks),
        "source_ref":   source_ref,
        "message":      f"Indexed {len(pages)} pages → {len(chunks)} chunks",
    }
