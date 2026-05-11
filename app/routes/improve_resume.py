from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import AnalysisRequest, ImproveResumeResponse
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service

router = APIRouter(tags=["resume-improvement"])


@router.post("/improve-resume", response_model=ImproveResumeResponse)
def improve_resume(
    request: AnalysisRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> ImproveResumeResponse:
    try:
        result = llm_service.improve_resume(
            resume_text=request.resume_text,
            job_description_text=request.job_description_text,
            api_key=request.api_key,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {exc}",
        ) from exc

    return ImproveResumeResponse(
        improved_bullet_points=result.improved_bullet_points,
        suggestions=result.suggestions,
    )
