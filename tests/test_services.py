import pytest

from app.services.llm_service import LLMService, LLMServiceError


def test_llm_service_raises_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMServiceError, match="GEMINI_API_KEY"):
        LLMService()
