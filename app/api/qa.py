import re
from typing import Optional, Dict

from fastapi import APIRouter, UploadFile, Form, File
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services import extractor, embedder, indexer
from app.core.config import settings
from app.utils.prompts import load_prompt


router = APIRouter()



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


SYSTEM_PROMPT = load_prompt("llm_prompt.txt")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}")
])

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=settings.GROQ_API_KEY,
)

chain = prompt | llm | StrOutputParser()


@router.post("/qa/")
async def qa(
    question: str = Form(...),
    doc_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> Dict:

    if file is not None:
        raw = await file.read()
        doc_id, bundles = extractor.extract_and_prepare(
            raw, file.filename, doc_id
        )
        embedder.embed_and_store(bundles, doc_id)

    if not doc_id:
        return {"answer": "The document does not contain this information."}

    fig_num = extract_fig(question)
    is_figure_query = fig_num is not None

    allowed_type = "figure" if is_figure_query else "text"
    top_k = 6 if is_figure_query else 8

    q_emb = embedder.get_embeddings([question])[0]
    matches = indexer.query(doc_id, q_emb, top_k=top_k)

    candidates = [
        m for m in matches
        if m["metadata"].get("type") == allowed_type
        and (
            not is_figure_query
            or f"figure {fig_num}" in m["metadata"].get("caption", "").lower()
        )
    ]

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

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "doc_id": doc_id,
        "candidates": serialize_candidates(candidates),
    }