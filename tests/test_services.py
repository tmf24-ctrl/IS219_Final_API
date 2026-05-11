from app.services.llm_service import LLMService


def test_llm_service_uses_mock_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = LLMService()

    assert service.using_mock is True


def test_mock_analyze_returns_expected_shape(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = LLMService()

    resume = "Experienced in Python, FastAPI, SQL, and pytest."
    job_description = "Need Python, FastAPI, Docker, AWS, and SQL."

    result = service.analyze_resume(resume, job_description)

    assert "python" in result.matching_skills
    assert "fastapi" in result.matching_skills
    assert "docker" in result.missing_skills
    assert isinstance(result.summary, str)
    assert result.summary


def test_mock_improve_generates_bullets(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = LLMService()

    resume = """
    - Built backend APIs for analytics dashboards
    - Automated ETL jobs for reporting
    """
    job_description = "Need Python and Docker experience."

    result = service.improve_resume(resume, job_description)

    assert len(result.improved_bullet_points) >= 1
    assert len(result.suggestions) >= 1
