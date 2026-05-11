import json
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from app.models.schemas import LLMAnalysisResult, LLMImproveResult
from app.utils.text_processing import extract_bullet_points, extract_skills

load_dotenv()


class LLMServiceError(Exception):
    """Raised when the LLM service fails to produce a valid response."""


class LLMService:
    def __init__(self, model: str = "gemini-2.0-flash-lite") -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=(
                    "You are an expert resume reviewer. "
                    "Always return valid JSON with no markdown or code fences."
                ),
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
        else:
            self._client = None

    @property
    def using_mock(self) -> bool:
        return self._client is None

    def analyze_resume(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        if self.using_mock:
            return self._mock_analyze(resume_text, job_description_text)

        prompt = self._analysis_prompt(resume_text, job_description_text)
        data = self._call_llm_json(prompt)
        try:
            return LLMAnalysisResult(**data)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Invalid analysis response shape: {exc}") from exc

    def improve_resume(self, resume_text: str, job_description_text: str) -> LLMImproveResult:
        if self.using_mock:
            return self._mock_improve(resume_text, job_description_text)

        prompt = self._improve_prompt(resume_text, job_description_text)
        data = self._call_llm_json(prompt)
        try:
            return LLMImproveResult(**data)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Invalid improve response shape: {exc}") from exc

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        if not self._client:
            raise LLMServiceError("Gemini client not initialized")

        try:
            response = self._client.generate_content(prompt)
            raw_text = response.text
        except Exception as exc:
            msg = str(exc)
            if "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg or "401" in msg:
                raise LLMServiceError(
                    "Gemini authentication failed. Verify GEMINI_API_KEY in your .env file."
                ) from exc
            raise LLMServiceError("LLM request failed due to an upstream provider error.") from exc

        # Strip accidental markdown fences Gemini may still add
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text.strip(), flags=re.IGNORECASE)
        raw_text = re.sub(r"```$", "", raw_text.strip())

        if not raw_text:
            raise LLMServiceError("LLM returned an empty response")

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(f"LLM returned non-JSON output: {exc}") from exc

    def _analysis_prompt(self, resume_text: str, job_description_text: str) -> str:
        return (
            "Analyze the candidate resume against the job description. "
            "Return strict JSON with this exact shape: "
            '{"matching_skills": ["..."], "missing_skills": ["..."], "summary": "..."}. '
            "Include concise, practical findings.\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description_text}"
        )

    def _improve_prompt(self, resume_text: str, job_description_text: str) -> str:
        return (
            "Rewrite resume bullet points to better align with the job description. "
            "Return strict JSON with this exact shape: "
            '{"improved_bullet_points": ["..."], "suggestions": ["..."]}. '
            "Keep bullets concise and quantified when possible.\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description_text}"
        )

    def _mock_analyze(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description_text)

        matching = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)

        summary = (
            f"Found {len(matching)} matching skills and {len(missing)} missing skills "
            "based on keyword analysis."
        )
        return LLMAnalysisResult(
            matching_skills=matching,
            missing_skills=missing,
            summary=summary,
        )

    def _mock_improve(self, resume_text: str, job_description_text: str) -> LLMImproveResult:
        bullets = extract_bullet_points(resume_text)
        jd_skills = sorted(extract_skills(job_description_text))

        improved: list[str] = []
        for bullet in bullets[:6]:
            tailored = bullet.rstrip(".")
            if jd_skills:
                tailored += f"; aligned with {jd_skills[0]} requirements"
            improved.append(tailored + ".")

        if not improved:
            improved = [
                "Delivered backend features with measurable impact and strong cross-team collaboration.",
                "Improved reliability and performance through testing, monitoring, and iterative optimization.",
            ]

        suggestions = [
            "Quantify outcomes in each bullet, such as latency reduction or revenue impact.",
            "Prioritize bullets that match required job skills and technologies.",
            "Group skills by category (languages, frameworks, cloud, ML/AI) for readability.",
        ]
        return LLMImproveResult(improved_bullet_points=improved, suggestions=suggestions)


def get_llm_service() -> LLMService:
    return LLMService()
