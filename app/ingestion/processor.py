"""
Code-aware document processor.
Different content types need different chunking strategies:

  CODE:     Function/class boundaries > fixed-size. Preserve indentation.
  RUNBOOK:  Split on H2/H3 headings. Keep steps together.
  INCIDENT: Split on sections (Timeline, Root Cause, Action Items).
  API_DOC:  Split on endpoint definitions. Keep request/response together.
  PROSE:    Standard token overlap sliding window.
"""
import re
import tiktoken
from pathlib import Path
from pypdf   import PdfReader
from app.core.schemas import DocType
from app.core.config  import get_settings

settings = get_settings()
_enc     = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def decode_tokens(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ── Extractors ─────────────────────────────────────────────────

def extract_text(source_ref: str, source_type: str) -> list[dict]:
    """Route to correct extractor. Returns [{text, page, raw_type}]."""
    path = Path(source_ref)
    ext  = path.suffix.lower() if path.exists() else ""

    if ext == ".pdf":
        return _extract_pdf(source_ref)
    elif ext in (".py", ".js", ".ts", ".go", ".java", ".rb", ".rs", ".cpp"):
        return [{"text": path.read_text(errors="replace"), "page": 1, "raw_type": "code"}]
    elif ext in (".md", ".mdx"):
        return [{"text": path.read_text(errors="replace"), "page": 1, "raw_type": "markdown"}]
    elif ext in (".txt", ".rst"):
        return [{"text": path.read_text(errors="replace"), "page": 1, "raw_type": "text"}]
    else:
        # Treat as plain text
        try:
            return [{"text": path.read_text(errors="replace"), "page": 1, "raw_type": "text"}]
        except Exception:
            return []


def _extract_pdf(path: str) -> list[dict]:
    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"text": text, "page": i + 1, "raw_type": "text"})
    return pages


# ── Chunkers ───────────────────────────────────────────────────

def chunk_document(
    pages:    list[dict],
    doc_type: DocType,
) -> list[dict]:
    """
    Dispatch to the right chunker based on doc_type.
    Returns list of {child_text, parent_text, page, chunk_index, parent_index}.
    """
    all_text = "\n\n".join(p["text"] for p in pages)
    page_map = {p["page"]: p for p in pages}

    if doc_type == DocType.CODE:
        raw_chunks = _chunk_code(all_text)
    elif doc_type in (DocType.RUNBOOK, DocType.SOP):
        raw_chunks = _chunk_by_headings(all_text)
    elif doc_type == DocType.INCIDENT:
        raw_chunks = _chunk_incident(all_text)
    elif doc_type == DocType.API_DOC:
        raw_chunks = _chunk_api_doc(all_text)
    else:
        raw_chunks = _chunk_sliding_window(
            all_text,
            max_tokens=settings.chunk_size_docs,
            overlap=settings.chunk_overlap,
        )

    return _build_parent_child(raw_chunks)


def _chunk_sliding_window(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(decode_tokens(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def _chunk_code(text: str) -> list[str]:
    """
    Split code at function/class definitions.
    Falls back to sliding window if no boundaries found.
    """
    # Match Python defs, JS functions, Go funcs, etc.
    pattern = re.compile(
        r"(?=^(?:def |class |func |function |async def |export function |const \w+ = ))",
        re.MULTILINE,
    )
    parts = pattern.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    for part in parts:
        if count_tokens(part) <= settings.chunk_size_code * 2:
            chunks.append(part)
        else:
            # Function too large — sliding window within it
            chunks.extend(_chunk_sliding_window(
                part,
                max_tokens=settings.chunk_size_code,
                overlap=settings.chunk_overlap,
            ))
    return chunks or _chunk_sliding_window(
        text,
        max_tokens=settings.chunk_size_code,
        overlap=settings.chunk_overlap,
    )


def _chunk_by_headings(text: str) -> list[str]:
    """Split markdown on H1/H2/H3. Merge small sections, split large ones."""
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks   = []
    buffer   = ""
    for section in sections:
        if not section.strip():
            continue
        combined = (buffer + "\n\n" + section).strip() if buffer else section
        if count_tokens(combined) <= settings.chunk_size_docs:
            buffer = combined
        else:
            if buffer:
                chunks.append(buffer)
            # Section itself too large — sliding window
            if count_tokens(section) > settings.chunk_size_docs:
                chunks.extend(_chunk_sliding_window(
                    section,
                    max_tokens=settings.chunk_size_docs,
                    overlap=settings.chunk_overlap,
                ))
                buffer = ""
            else:
                buffer = section
    if buffer:
        chunks.append(buffer)
    return chunks


def _chunk_incident(text: str) -> list[str]:
    """
    Split incident reports on standard section headers.
    Keeps Timeline, Root Cause, and Action Items as atomic chunks.
    """
    section_pattern = re.compile(
        r"(?=^(?:## |### |Timeline|Root Cause|Impact|Mitigation|Action Items|Summary|Detection))",
        re.MULTILINE | re.IGNORECASE,
    )
    parts  = section_pattern.split(text)
    chunks = []
    for part in (p.strip() for p in parts if p.strip()):
        if count_tokens(part) <= settings.parent_chunk_size:
            chunks.append(part)
        else:
            chunks.extend(_chunk_sliding_window(
                part,
                max_tokens=settings.chunk_size_docs,
                overlap=settings.chunk_overlap,
            ))
    return chunks


def _chunk_api_doc(text: str) -> list[str]:
    """
    Split API docs on endpoint definitions.
    Keeps method + path + description + request/response together.
    """
    endpoint_pattern = re.compile(
        r"(?=^(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) |\n#{2,3} (?:GET|POST|PUT|PATCH|DELETE))",
        re.MULTILINE,
    )
    parts  = endpoint_pattern.split(text)
    chunks = []
    for part in (p.strip() for p in parts if p.strip()):
        if count_tokens(part) <= settings.parent_chunk_size:
            chunks.append(part)
        else:
            chunks.extend(_chunk_sliding_window(
                part,
                max_tokens=settings.chunk_size_docs,
                overlap=settings.chunk_overlap,
            ))
    return chunks or _chunk_sliding_window(
        text,
        max_tokens=settings.chunk_size_docs,
        overlap=settings.chunk_overlap,
    )


def _build_parent_child(raw_chunks: list[str]) -> list[dict]:
    """
    Build parent/child chunk pairs from raw chunks.
    Parent = up to 2 consecutive raw chunks merged (broader context).
    Child  = single raw chunk (precise matching).
    """
    result = []
    for p_idx in range(0, len(raw_chunks), 2):
        child_a = raw_chunks[p_idx]
        child_b = raw_chunks[p_idx + 1] if p_idx + 1 < len(raw_chunks) else ""
        parent  = (child_a + "\n\n" + child_b).strip() if child_b else child_a

        for c_idx, child in enumerate([child_a, child_b] if child_b else [child_a]):
            result.append({
                "child_text":   child,
                "parent_text":  parent,
                "parent_index": p_idx // 2,
                "chunk_index":  c_idx,
            })
    return result
