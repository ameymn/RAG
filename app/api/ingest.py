from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from typing import Optional, Dict
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings

router = APIRouter()

s3_client = boto3.client(
    's3',
    endpoint_url=settings.S3_ENDPOINT or None,
    aws_access_key_id=settings.S3_ACCESS_KEY or None,
    aws_secret_access_key=settings.S3_SECRET_KEY or None,
    region_name=settings.S3_REGION or None
)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF or image file upload and saves it to the disk
    Return a generated doc_id and saved path.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    raw= await file.read()
    doc_id = str(uuid.uuid4())
    filename = file.filename or f"upload-{doc_id}"
    key = f"{doc_id}/{filename}"
    
    try:
        s3_client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=raw)
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {e}")
    s3_uri = f"s3://{settings.S3_BUCKET}/{key}"
    return {"status": "ok", "doc_id": doc_id, "path": s3_uri}