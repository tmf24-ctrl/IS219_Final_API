from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes.analyze import router as analyze_router
from app.routes.improve_resume import router as improve_resume_router
from app.services.llm_service import LLMService

_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="AI Job Application Assistant API",
    version="1.0.0",
    description=(
        "Analyze resume and job descriptions, identify skill gaps, and suggest "
        "resume improvements using an LLM-backed service."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mode", tags=["health"])
def mode_check() -> dict:
    service = LLMService()
    return {"mock": service.using_mock, "model": service.model}


app.include_router(analyze_router)
app.include_router(improve_resume_router)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
