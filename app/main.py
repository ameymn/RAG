from fastapi import FastAPI
from app.api.qa import router as qa_router


app = FastAPI(title="Vision RAG API")

app.include_router(qa_router, prefix="/api", tags=["QA"])