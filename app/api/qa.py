import re
from typing import Optional, Dict

from fastapi import APIRouter, UploadFile, Form, File
from groq import Groq

from app.services import extractor, embedder, indexer
from app.core.config import settings
from app.utils.prompts import load_prompt

router = APIRouter()

# -----------------------
# Groq client (single)
# -----------------------
client = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = load_prompt("llm_prompt.txt")


# -----------------------
# Helpers
# -----------------------
def serialize_candidates(candidates):
    out = []
    for c in candidates:
        out.append({
            "id": c.get("id"),
            "score": float(c.get("score", 0.0)),
            "metadata": {
                "type": c.get("metadata", {}).get("type"),
                "caption": c.get("metadata", {}).get("caption", ""),
                "content": c.get("metadata", {}).get("content", ""),
            }
        })
    return out


def extract_fig(question: str) -> Optional[int]:
    q = question.lower().replace("\u00a0", " ")
    m = re.search(r"\b(fig|figure)\b[^\d]{0,5}(\d+)", q)
    return int(m.group(2)) if m else None


def is_structural_query(question: str) -> bool:
    """Detect queries asking about document structure (chapters, TOC, outline, etc.)"""
    structural_keywords = [
        "chapter", "chapters", "contents", "table of contents", "toc",
        "outline", "sections", "parts", "structure", "overview",
        "topics", "index", "summary", "summarize", "all about"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in structural_keywords)


def expand_structural_query(question: str) -> list[str]:
    """Use LLM to generate multiple search queries for structural questions."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": f"""Generate 5 different search queries to find the answer to this question about a book/document.
Return ONLY the queries, one per line, no numbering or explanation.

Question: {question}

Example output for "What are the chapters?":
table of contents
chapter 1
chapter 2
list of chapters
contents page"""
        }],
        max_completion_tokens=100,
    )
    queries = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()][:5]


def truncate_context(context: str, max_chars: int = 16000) -> str:
    """Truncate context to fit within token limits (~4 chars per token)."""
    if len(context) <= max_chars:
        return context
    # Truncate and add indicator
    return context[:max_chars] + "\n\n[Context truncated due to length...]"


def run_llm(context: str, question: str) -> str:
    """
    Direct Groq call.
    No LangChain.
    """
    # Truncate context to stay within Groq's token limits
    truncated_context = truncate_context(context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"DOCUMENT CONTEXT:\n{truncated_context}\n\nQUESTION:\n{question}"
            },
        ],
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content.strip()


# -----------------------
# API
# -----------------------
@router.post("/qa/")
async def qa(
    question: str = Form(...),
    doc_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> Dict:

    # -------- Upload / ingest --------
    if file is not None:
        raw = await file.read()
        doc_id, bundles = extractor.extract_and_prepare(
            raw, file.filename, doc_id
        )
        embedder.embed_and_store(bundles, doc_id)

    if not doc_id:
        return {"answer": "The document does not contain this information."}

    # -------- Query type --------
    fig_num = extract_fig(question)
    is_figure_query = fig_num is not None
    structural_query = is_structural_query(question)

    # Determine how many chunks to retrieve
    if is_figure_query:
        top_k = 10
    elif structural_query:
        # Structural queries need more context (chapters, TOC, etc.)
        top_k = 40
    else:
        top_k = 20

    # -------- Vector search --------
    if structural_query:
        # For structural queries, expand and search with multiple queries
        expanded_queries = expand_structural_query(question)
        print(f"[DEBUG] Expanded queries: {expanded_queries}")

        all_matches = {}
        for eq in [question] + expanded_queries:
            q_emb = embedder.get_embeddings([eq])[0]
            matches = indexer.query(doc_id, q_emb, top_k=15)
            for m in matches:
                mid = m.get("id")
                if mid not in all_matches or m.get("score", 0) > all_matches[mid].get("score", 0):
                    all_matches[mid] = m

        # Sort by score and take top results
        matches = sorted(all_matches.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    else:
        q_emb = embedder.get_embeddings([question])[0]
        matches = indexer.query(doc_id, q_emb, top_k=top_k)

    if is_figure_query:
        # For figure queries, filter strictly to matching figures
        candidates = [
            m for m in matches
            if m["metadata"].get("type") == "figure"
            and f"figure {fig_num}" in m["metadata"].get("caption", "").lower()
        ]
    else:
        # For general queries, keep ALL retrieved content (text + figures)
        # Let the LLM decide what's relevant
        candidates = matches

    # -------- Build context --------
    # Log retrieved chunks for debugging
    print(f"\n[DEBUG] Retrieved {len(candidates)} candidates for: '{question}'")
    for i, m in enumerate(candidates[:10]):  # Log first 10
        content_preview = m["metadata"].get("content", "")[:100].replace("\n", " ")
        print(f"  [{i}] score={m.get('score', 0):.3f} | {content_preview}...")

    context = "\n\n".join(
        (
            m["metadata"].get("caption", "")
            + "\n"
            + m["metadata"].get("content", "")
        ).strip()
        for m in candidates
        if m["metadata"].get("caption") or m["metadata"].get("content")
    )

    if not context.strip():
        return {
            "answer": "The document does not contain this information.",
            "doc_id": doc_id,
            "candidates": [],
        }

    answer = run_llm(context, question)

    return {
        "answer": answer,
        "doc_id": doc_id,
        "candidates": serialize_candidates(candidates),
    }