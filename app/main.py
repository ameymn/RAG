from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.qa import router as qa_router

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router, prefix="/api", tags=["QA"])

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse(UI_DIR / "index.html")