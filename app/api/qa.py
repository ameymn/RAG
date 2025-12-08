from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from typing import Optional, Dict
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings

from app.services import extractor, embedder, indexer, llm

router = APIRouter()

_s3_client = boto3.client(
    's3',
    endpoint_url=settings.S3_ENDPOINT or None,
    aws_access_key_id=settings.S3_ACCESS_KEY or None,
    aws_secret_access_key=settings.S3_SECRET_KEY or None,
    region_name=settings.S3_REGION or None
)

indexer.create_index_if_needed(dimension=embedder.DEFAULT_DIM)


@router.post("/")
async def qa(
    question: str = Form(...),
    doc_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> Dict:
    """
    Minimal QA endpoint with clean flow:
    1. Upload filet to S3
    2. Extract bundles (text chunks + figure images)
    3. Embed bundles
    4. Upsert into Pinecone
    5. Embed question
    6. Query Pinecone
    7. Generate LLM answer
    """

    saved_info = None
    created_doc_id = doc_id

    if file is not None:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        created_doc_id = str(uuid.uuid4())
        filename = file.filename or f"upload-{created_doc_id}"
        key = f"{created_doc_id}/{filename}"

        try:
            _s3_client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=raw)
        except (BotoCoreError, ClientError) as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

        saved_info = {
            "doc_id": created_doc_id,
            "path": f"s3://{settings.S3_BUCKET}/{key}",
            "file_name": filename
        }

        created_doc_id, bundles = extractor.extract_and_prepare(raw, filename, doc_id=created_doc_id)

        embed_inputs = [b["content"] for b in bundles]
        bundle_embeds = embedder.get_embeddings(embed_inputs, dimensions=embedder.DEFAULT_DIM)


        indexer.upsert_bundles(doc_id=created_doc_id, bundles=bundles, embeddings=bundle_embeds)


    q_vec = embedder.get_embeddings([question], dimensions=embedder.DEFAULT_DIM)[0]


    candidates = indexer.query(doc_id=created_doc_id, query_vec=q_vec, top_k=8)


    answer = llm.generate_answer(question, candidates)

    return {
        "answer": answer,
        "doc_id": created_doc_id,
        "file_saved": bool(saved_info),
        "saved_file": saved_info,
        "candidates": candidates
    }