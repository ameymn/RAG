import io
import uuid
from typing import List, Tuple, Optional
import fitz

import boto3

from app.core.config import settings

s3_client = boto3.client(
    's3',
    endpoint_url=settings.S3_ENDPOINT or None,
    aws_access_key_id=settings.S3_ACCESS_KEY or None,
    aws_secret_access_key=settings.S3_SECRET_KEY or None,
    region_name=settings.S3_REGION or None
)


def upload_bytes_to_s3(b: bytes, key: str) -> str:
    """Upload bytes to S3 and return s3:// URI."""
    s3_client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=b)

    return f"s3://{settings.S3_BUCKET}/{key}"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into smaller pieces with overlap."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def find_caption(page_text: str, image_index_hint: Optional[int]) -> Optional[str]:
    """Find caption for the image based on its index in the page text."""
    lines = page_text.split('\n')
    image_count = 0

    for line in lines:
        if line.strip() == "":
            continue
        if "Figure" in line or "Fig." in line:
            if image_count == image_index_hint:
                return line.strip()
            image_count += 1

    return None


def extract_and_prepare(file_bytes: bytes, filename: str, doc_id: Optional[str] = None) -> Tuple[str, List[dict]]:
    """Extract text and images from PDF bytes, chunk text, and upload to S3."""
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    bundles: List[dict] = []
    filename = filename.lower()

    if filename.endswith(('.png', ".jpg", ".jpeg")):
        bundle_id = str(uuid.uuid4())
        key = f"{doc_id}/images/{bundle_id}.png"
        s3_uri = upload_bytes_to_s3(file_bytes, key)
        bundles.append({
           "bundle_id": bundle_id,
            "type": "image",
            "content": s3_uri,
            "caption": None,
            "snippet": None,
            "page": None,
            "bbox": None,
            "metadata": {}
        })
        return doc_id, bundles
    

    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        page_text = page.get_text("text") or ""
        text_chunks = chunk_text(page_text)

        for i, chunk in enumerate(text_chunks):
            bundles.append({
                "bundle_id": str(uuid.uuid4()),
                "type": "text",
                "content": chunk,
                "caption": None,
                "snippet": chunk[:200],
                "page": page_idx + 1,
                "bbox": None,
                "metadata": {
                    "chunk_index:": i,
                }
            })

        images = page.get_images(full=True)
        if not images:
            continue

        for img_index, img in enumerate(images):
            xref = img[0]
            pix =  fitz.Pixmap(doc, xref)

            if pix.n < 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_bytes = pix.tobytes("png")
            pix = None

            bundle_id = str(uuid.uuid4())
            key = f"{doc_id}/images/{bundle_id}.png"
            s3_uri = upload_bytes_to_s3(img_bytes, key)
            caption = find_caption(page_text, img_index)

            bundles.append({
                "bundle_id": bundle_id,
                "type": "figure",
                "content": s3_uri,
                "caption": caption,
                "snippet": caption,
                "page": page_idx + 1,
                "bbox": None,
                "metadata": {"image_index": img_index}
            }) 
    doc.close()
    return doc_id, bundles
        