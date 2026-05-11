from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import AnalysisRequest, AnalyzeResponse
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(
    request: AnalysisRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> AnalyzeResponse:
    try:
        result = llm_service.analyze_resume(
            resume_text=request.resume_text,
            job_description_text=request.job_description_text,
            api_key=request.api_key,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM service error: {exc}",
        ) from exc

    return AnalyzeResponse(
        matching_skills=result.matching_skills,
        missing_skills=result.missing_skills,
        summary=result.summary,
    )
