# Multimodal RAG — Document Question Answering

Multimodal RAG is a **multimodal Retrieval Augmented Generation (RAG)** system for asking **grounded questions over PDF documents**, including **figure-specific questions that require visual understanding**.

The system answers **strictly from the document (text + figures)** and explicitly refuses when the information is not present.

---

## What You Can Ask

- “What is this paper about?”
- “What methodology is proposed?”
- “What does Figure 3 show?”
- “What does the ablation study indicate?”

If the answer is not supported by the document, the system responds exactly:

> **The document does not contain this information.**

---

## Key Features

- PDF ingestion and parsing  
- Robust figure caption extraction  
- Multimodal figure understanding using a Vision-Language Model (VLM)  
- Intent-aware retrieval (text vs figure queries)  
- CLIP-based embeddings (Jina CLIP v2)  
- Pinecone vector database  
- Groq-powered LLM and VLM inference  
- Confidence-gated responses to prevent hallucinations  
- Streamlit user interface  

---

## Project Structure

```text
RAG/
├── app/
│   ├── api/
│   │   └── qa.py
│   ├── services/
│   │   ├── extractor.py
│   │   ├── vision_summarizer.py
│   │   ├── embedder.py
│   │   └── indexer.py
│   └── core/
│       └── config.py
├── app/utils/prompts/
│   └── llm_prompt.txt
├── ui/
│   └── streamlit_app.py
├── pyproject.toml
├── uv.lock
├── .env
└── README.md
```

---

## Requirements

- Python 3.10+
- `uv` package manager
- Pinecone account
- Jina AI API key
- Groq API key
- AWS S3 credentials

---

## Setup

### Clone the Repository

```bash
git clone git@github.com:ameymn/RAG.git
cd RAG
```

---

### Install Dependencies

```bash
uv sync
```

This will create a local virtual environment and install dependencies from `pyproject.toml`.

---

### Environment Configuration

This project uses a **single `.env` file**.

Create it manually if it does not exist:

```bash
touch .env
```

Example `.env`:

```env
GROQ_API_KEY="gsk_..."

PINECONE_API_KEY="..."

JINA_API_KEY="..."

S3_ACCESS_KEY="..."
S3_SECRET_KEY="..."
```



---

## Running the Application

### Streamlit UI (Recommended)

```bash
uv run streamlit run ui/streamlit_app.py
```

Access at:

```
http://localhost:8501
```

---

### FastAPI Backend (Optional)

```bash
uv run uvicorn app.main:app --reload
```

Access at:

```
http://localhost:8000
```

---

## How It Works

1. User uploads a PDF  
2. The document is parsed into:
   - Text chunks  
   - Figure captions  
   - Extracted figure images  
3. Figure images are uploaded to S3 and summarized using a Vision-Language Model  
4. Text and figure summaries are embedded using CLIP (Jina CLIP v2)  
5. Embeddings are stored in Pinecone  
6. User asks a question  
7. The system:
   - Detects intent (text vs figure)
   - Retrieves relevant context
   - Applies confidence gating  
8. The LLM answers **only from retrieved context**  
9. If context is insufficient → explicit refusal

---

