import uuid
import time
from typing import List
import requests

from app.core.config import settings
from app.services import indexer

JINA_URL = "https://api.jina.ai/v1/embeddings"
EMBEDDING_BATCH_SIZE = 32  # Number of texts per API call
MAX_RETRIES = 5
BASE_DELAY = 1.0  # Base delay for exponential backoff


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a single batch with retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
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

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - exponential backoff
                delay = BASE_DELAY * (2 ** attempt)
                print(f"Rate limited, waiting {delay}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(delay)
            else:
                raise

    raise Exception(f"Failed to get embeddings after {MAX_RETRIES} retries")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Jina CLIP v2 with batching.
    """
    if not texts:
        return []

    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        embeddings = get_embeddings_batch(batch)
        all_embeddings.extend(embeddings)

        # Small delay between batches to avoid rate limiting
        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(0.2)

    return all_embeddings


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