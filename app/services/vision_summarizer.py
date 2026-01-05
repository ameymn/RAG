from groq import Groq
from app.core.config import settings
import boto3
from botocore.client import Config
from app.utils.prompts import load_prompt


client = Groq(api_key=settings.GROQ_API_KEY)
VISION_PROMPT = load_prompt("vlm_prompt.txt")


s3 = boto3.client(
    "s3",
    region_name=settings.S3_REGION,
    aws_access_key_id=settings.S3_ACCESS_KEY,
    aws_secret_access_key=settings.S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

def presigned_image_url(bucket: str, key: str, expires=900) -> str:
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

    return url


def summarize_figure(s3_key: str) -> str:
    image_url = presigned_image_url(settings.S3_BUCKET, s3_key)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        max_completion_tokens=400,
    )

    return response.choices[0].message.content.strip()