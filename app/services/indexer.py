from typing import List
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

INDEX_NAME = settings.PINECONE_INDEX_NAME
NAMESPACE = "vision-rag"


def create_index_if_needed(dimension: int):
    """Create Pinecone index if it does not exist."""
    if pc.has_index(INDEX_NAME):
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=settings.PINECONE_ENVIRONMENT
        )
    )


def upsert_bundles(
    doc_id: str,
    bundles: List[dict],
    embeddings: List[List[float]]
):
    """Upsert bundles and their embeddings into Pinecone."""
    index = pc.Index(INDEX_NAME)

    vectors = []
    for bundle, vector in zip(bundles, embeddings):
        vectors.append({
            "id": f"{doc_id}:{bundle['bundle_id']}",
            "values": vector,
            "metadata": {
                "doc_id": doc_id,
                "bundle_id": bundle["bundle_id"],
                "type": bundle["type"],
                "content": bundle["content"],
                "caption": bundle.get("caption", "")
            }
        })

    index.upsert(vectors=vectors, namespace=NAMESPACE)


def query(
    doc_id: str,
    query_vec: List[float],
    top_k: int = 10
) -> List[dict]:
    """Query Pinecone index."""
    index = pc.Index(INDEX_NAME)

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=NAMESPACE,
        filter={"doc_id": doc_id},
        include_metadata=True
    )

    return res.get("matches", [])