from groq import Groq
from app.core.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)

VISION_SYSTEM_PROMPT = """
You are a scientific vision assistant.

Describe the figure in detail:
- Identify trends (increase/decrease/comparison)
- Mention axes, bars, curves if present
- Explain what the figure conveys scientifically
- Do NOT guess values
- Be factual and concise
"""

def summarize_figure(image_url: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this figure."},
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()