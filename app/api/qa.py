from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from typing import Optional, Dict
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings

router = APIRouter()


_s3_client = boto3.client(
    's3',
    endpoint_url=settings.S3_ENDPOINT or None,
    aws_access_key_id=settings.S3_ACCESS_KEY or None,
    aws_secret_access_key=settings.S3_SECRET_KEY or None,
    region_name=settings.S3_REGION or None
)



@router.post("/")
async def qa(
    question: str = Form(...),
    doc_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),            
) -> Dict :
    """
    Minimal QA endpoint
    if the file is uploaded, save it to S3
    """

    saved_info = None
    if file is not None:
        raw = await file.read()

        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        new_doc_id = str(uuid.uuid4())
        filename = file.filename or f"upload-{new_doc_id}"
        key = f"{new_doc_id}/{filename}"


        try:
            _s3_client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=raw)
        except (BotoCoreError, ClientError) as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
        s3_uri = f"s3://{settings.S3_BUCKET}/{key}"

        saved_info = {"doc_id": new_doc_id, "path": s3_uri, "file_name": filename}

        returned_doc_id = doc_id or (saved_info["doc_id"] if saved_info else None)

        return {
            "answer": f"Received question: '{question}'",
            "doc_id": returned_doc_id,
            "file_saved": bool(saved_info),
            "saved_file": saved_info,
        }