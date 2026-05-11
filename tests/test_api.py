from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import LLMAnalysisResult, LLMImproveResult
from app.services.llm_service import LLMServiceError, get_llm_service


class FakeLLMService:
    def analyze_resume(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        return LLMAnalysisResult(
            matching_skills=["python", "fastapi"],
            missing_skills=["docker"],
            summary="Strong backend fit with one key infrastructure gap.",
        )

    def improve_resume(self, resume_text: str, job_description_text: str) -> LLMImproveResult:
        return LLMImproveResult(
            improved_bullet_points=[
                "Built FastAPI services that reduced response latency by 35%.",
                "Added CI pipelines that cut release failures by 20%.",
            ],
            suggestions=[
                "Quantify outcomes in each bullet.",
                "Highlight cloud and deployment experience.",
            ],
        )


class FailingLLMService:
    def analyze_resume(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        raise LLMServiceError("upstream failure")

    def improve_resume(self, resume_text: str, job_description_text: str) -> LLMImproveResult:
        raise LLMServiceError("upstream failure")


def test_analyze_endpoint_success() -> None:
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)

    payload = {
        "resume_text": "Python FastAPI backend experience with pytest.",
        "job_description_text": "Need Python, FastAPI, Docker.",
    }
    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["matching_skills"] == ["python", "fastapi"]
    assert body["missing_skills"] == ["docker"]
    assert "summary" in body

    app.dependency_overrides.clear()


def test_improve_resume_success() -> None:
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)

    payload = {
        "resume_text": "- Built APIs\n- Wrote tests",
        "job_description_text": "Looking for cloud backend engineer.",
    }
    response = client.post("/improve-resume", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["improved_bullet_points"]) == 2
    assert len(body["suggestions"]) == 2

    app.dependency_overrides.clear()


def test_invalid_input_returns_422() -> None:
    client = TestClient(app)
    payload = {"resume_text": "too short", "job_description_text": "short"}

    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_llm_error_returns_502() -> None:
    app.dependency_overrides[get_llm_service] = lambda: FailingLLMService()
    client = TestClient(app)

    payload = {
        "resume_text": "Python FastAPI backend experience with pytest and SQL.",
        "job_description_text": "Need Python, FastAPI, Docker and AWS experience.",
    }
    response = client.post("/analyze", json=payload)

    assert response.status_code == 502
    assert "LLM service error" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_root_serves_html() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_mode_endpoint_returns_mock_flag() -> None:
    client = TestClient(app)
    response = client.get("/mode")
    assert response.status_code == 200
    body = response.json()
    assert "mock" in body
    assert isinstance(body["mock"], bool)
    assert "model" in body
