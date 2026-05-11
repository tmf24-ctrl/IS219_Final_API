from fastapi import FastAPI

from app.routes.analyze import router as analyze_router
from app.routes.improve_resume import router as improve_resume_router

app = FastAPI(
    title="AI Job Application Assistant API",
    version="1.0.0",
    description=(
        "Analyze resume and job descriptions, identify skill gaps, and suggest "
        "resume improvements using an LLM-backed service."
    ),
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(analyze_router)
app.include_router(improve_resume_router)
