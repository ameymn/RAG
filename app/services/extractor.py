import re
import uuid
from typing import List, Tuple, Dict, Any

import fitz
import boto3

from app.core.config import settings
from app.services.vision_summarizer import summarize_figure

s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.S3_ACCESS_KEY,
    aws_secret_access_key=settings.S3_SECRET_KEY,
    region_name=settings.S3_REGION,
)

FIGURE_REGEX = re.compile(
    r"(Figure\s+(\d+)\s*:\s*)(.*?)(?=\nFigure\s+\d+\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

def extract_and_prepare(
    raw_pdf: bytes,
    filename: str,
    doc_id: str | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:

    if doc_id is None:
        doc_id = str(uuid.uuid4())

    pdf = fitz.open(stream=raw_pdf, filetype="pdf")
    full_text = "\n".join(p.get_text("text") for p in pdf)

    bundles: List[Dict[str, Any]] = []
    figure_spans = []
    figure_numbers = set()

    for m in FIGURE_REGEX.finditer(full_text):
        fig_num = int(m.group(2))
        caption = m.group(3).strip()

        bundles.append({
            "bundle_id": f"fig-{fig_num}-{uuid.uuid4().hex[:6]}",
            "type": "figure",
            "content": "",               
            "caption": f"Figure {fig_num}: {caption}",
            "metadata": {
                "type": "figure",
                "figure_number": fig_num,
                "doc_id": doc_id,
            },
        })

        figure_spans.append(m.span())
        figure_numbers.add(fig_num)

    for page_idx, page in enumerate(pdf):
        page_text = page.get_text("text").lower()
        m = re.search(r"figure\s+(\d+)", page_text)
        if not m:
            continue

        fig_num = int(m.group(1))
        if fig_num not in figure_numbers:
            continue

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = pdf.extract_image(xref)

            image_bytes = base["image"]
            ext = base.get("ext", "png")

            s3_key = f"{doc_id}/figures/figure_{fig_num}_{img_idx}.{ext}"

            s3.put_object(
                Bucket=settings.S3_BUCKET,
                Key=s3_key,
                Body=image_bytes,
                ContentType=f"image/{ext}",
            )



            vision_text = summarize_figure(s3_key)

            for b in bundles:
                if b["metadata"]["figure_number"] == fig_num:
                    b["content"] += "\n" + vision_text

    clean_text = full_text
    for s, e in reversed(figure_spans):
        clean_text = clean_text[:s] + clean_text[e:]

    for i, chunk in enumerate(_chunk_text(clean_text)):
        if chunk.strip():
            bundles.append({
                "bundle_id": f"text-{i}-{uuid.uuid4().hex[:6]}",
                "type": "text",
                "content": chunk.strip(),
                "caption": "",
                "metadata": {"type": "text", "doc_id": doc_id},
            })

    return doc_id, bundles


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    chunks, buf = [], ""
    for line in text.splitlines():
        if len(buf) + len(line) > max_chars:
            chunks.append(buf.strip())
            buf = line
        else:
            buf += "\n" + line
    if buf.strip():
        chunks.append(buf.strip())
    return chunks