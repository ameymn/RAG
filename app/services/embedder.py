import uuid
from typing import List
import requests

from app.core.config import settings
from app.services import indexer

JINA_URL = "https://api.jina.ai/v1/embeddings"


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Jina CLIP v2.
    """
    resp = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-clip-v2",
            "input": texts,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def embed_and_store(bundles: list, doc_id: str):
    """
    Embeds and stores:
    - Text chunks
    - Figure captions + vision summaries 
    """

    embed_inputs: List[str] = []
    valid_bundles: List[dict] = []

    for i, b in enumerate(bundles):
        if "bundle_id" not in b:
            b["bundle_id"] = f"auto-{i}-{uuid.uuid4().hex[:6]}"

        btype = b.get("type")
        caption = b.get("caption", "")
        content = b.get("content", "")

        if btype == "figure":
            # Combine caption + vision description
            text = f"{caption}\n{content}".strip()
            if text:
                embed_inputs.append(text)
                valid_bundles.append(b)

        elif btype == "text" and content.strip():
            embed_inputs.append(content.strip())
            valid_bundles.append(b)

    if not embed_inputs:
        return

    # Generate embeddings
    embeddings = get_embeddings(embed_inputs)

    # Ensure index exists
    indexer.create_index_if_needed(dimension=len(embeddings[0]))

    # Store in Pinecone
    indexer.upsert_bundles(
        doc_id=doc_id,
        bundles=valid_bundles,
        embeddings=embeddings,
    )