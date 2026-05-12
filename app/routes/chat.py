from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat_endpoint(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    try:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        reply = llm_service.chat(message=request.message, history=history)
        return ChatResponse(reply=reply)
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
