import json
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from app.models.schemas import LLMAnalysisResult, LLMImproveResult
from app.utils.text_processing import extract_bullet_points, extract_skills

load_dotenv()

_SYSTEM_INSTRUCTION = (
    "You are a senior technical recruiter and professional resume coach. "
    "Always return valid JSON only — no markdown, no code fences, no extra text."
)


class LLMServiceError(Exception):
    """Raised when the LLM service fails to produce a valid response."""


class LLMService:
    def __init__(self, model: str = "gemini-2.0-flash-lite") -> None:
        self._env_api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", model)
        self._server_client = self._build_client(self._env_api_key)

    @property
    def using_mock(self) -> bool:
        return self._server_client is None and not self._env_api_key

    def _build_client(self, api_key: str | None) -> Any | None:
        if not api_key or not api_key.strip():
            return None
        genai.configure(api_key=api_key.strip())
        return genai.GenerativeModel(
            model_name=self.model,
            system_instruction=_SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

    def _get_client(self, api_key: str | None) -> Any | None:
        """Return a per-request client when a request-level key is given, else fall back to server client."""
        if api_key and api_key.strip():
            return self._build_client(api_key.strip())
        return self._server_client

    # ── Public API ──────────────────────────────────────────────────────────────

    def analyze_resume(
        self,
        resume_text: str,
        job_description_text: str,
        api_key: str | None = None,
    ) -> LLMAnalysisResult:
        client = self._get_client(api_key)
        if client is None:
            return self._mock_analyze(resume_text, job_description_text)

        prompt = self._analysis_prompt(resume_text, job_description_text)
        data = self._call_llm_json(client, prompt)
        try:
            return LLMAnalysisResult(**data)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Invalid analysis response shape: {exc}") from exc

    def improve_resume(
        self,
        resume_text: str,
        job_description_text: str,
        api_key: str | None = None,
    ) -> LLMImproveResult:
        client = self._get_client(api_key)
        if client is None:
            return self._mock_improve(resume_text, job_description_text)

        prompt = self._improve_prompt(resume_text, job_description_text)
        data = self._call_llm_json(client, prompt)
        try:
            return LLMImproveResult(**data)
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Invalid improve response shape: {exc}") from exc

    # ── Core LLM call ───────────────────────────────────────────────────────────

    def _call_llm_json(self, client: Any, prompt: str) -> dict[str, Any]:
        try:
            response = client.generate_content(prompt)
            raw_text = response.text
        except Exception as exc:
            msg = str(exc)
            if any(k in msg for k in ("API_KEY_INVALID", "PERMISSION_DENIED", "401", "403")):
                raise LLMServiceError(
                    "Gemini authentication failed. Check that your API key is valid."
                ) from exc
            raise LLMServiceError("LLM request failed due to an upstream provider error.") from exc

        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text.strip(), flags=re.IGNORECASE)
        raw_text = re.sub(r"```$", "", raw_text.strip())

        if not raw_text:
            raise LLMServiceError("LLM returned an empty response")

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(f"LLM returned non-JSON output: {exc}") from exc

    # ── Prompts ─────────────────────────────────────────────────────────────────

    def _analysis_prompt(self, resume_text: str, job_description_text: str) -> str:
        return f"""You are a senior technical recruiter reviewing a candidate for a specific role.

Carefully read both the resume and the job description, then return a JSON object with this exact shape:
{{
  "matching_skills": ["..."],
  "missing_skills": ["..."],
  "summary": "..."
}}

Rules:
- matching_skills: specific skills, tools, and technologies present in BOTH the resume and the job description. Use exact names (e.g. "Python", "Docker", "REST APIs", "CI/CD").
- missing_skills: skills, tools, or qualifications the job description asks for that are NOT on the resume. Be specific and prioritize the most important gaps.
- summary: 3-4 sentences. Assess the overall fit honestly. Highlight the candidate's strongest relevant experience, name the most critical gap, and give one concrete recommendation for what they should do to become a stronger candidate.

Be specific and direct. Do not be vague. Do not list every technology — focus on what matters most for this role.

Resume:
{resume_text}

Job Description:
{job_description_text}"""

    def _improve_prompt(self, resume_text: str, job_description_text: str) -> str:
        return f"""You are an expert resume writer who specializes in tech and engineering roles.

Your task: rewrite the resume bullet points so they better match the job description. Then provide targeted career advice.

Return a JSON object with this exact shape:
{{
  "improved_bullet_points": ["..."],
  "suggestions": ["..."]
}}

Rules for improved_bullet_points:
- Rewrite every bullet point from the resume. Keep the same experiences but strengthen the language.
- Start each bullet with a strong action verb (Engineered, Reduced, Automated, Designed, Led, etc.)
- Quantify results wherever possible — add realistic estimates if none exist (e.g. "reduced deployment time by ~40%")
- Weave in keywords from the job description naturally — do NOT keyword-stuff
- Keep each bullet to one line, maximum 20 words
- Return at least 4 improved bullets

Rules for suggestions (return exactly 4):
1. One suggestion about skills to add or certifications to pursue based on what the job description emphasizes
2. One suggestion about how to reframe or highlight their existing experience better
3. One suggestion about the resume format or structure
4. One honest assessment of their biggest weakness for this specific role and how to address it

Be specific. Reference actual technologies and skills from both the resume and the job description.

Resume:
{resume_text}

Job Description:
{job_description_text}"""

    # ── Mock fallbacks ──────────────────────────────────────────────────────────

    def _mock_analyze(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description_text)

        matching = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)

        summary = (
            f"Keyword scan found {len(matching)} matching skill(s) and {len(missing)} missing skill(s). "
            "Add a Gemini API key for a detailed, AI-powered analysis with real recommendations."
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
                tailored += f"; relevant to {jd_skills[0]} requirements"
            improved.append(tailored + ".")

        if not improved:
            improved = [
                "Delivered backend features with measurable impact and cross-team collaboration.",
                "Improved system reliability through automated testing and iterative optimization.",
            ]

        suggestions = [
            "Add a Gemini API key to get real AI-powered bullet rewrites tailored to this specific role.",
            "Quantify outcomes in each bullet — numbers (%, $, time saved) make bullets 3x more compelling.",
            "Prioritize bullets that use keywords from the job description's requirements section.",
            "Group your skills section by category: Languages, Frameworks, Cloud, Tools, AI/ML.",
        ]
        return LLMImproveResult(improved_bullet_points=improved, suggestions=suggestions)


def get_llm_service() -> LLMService:
    return LLMService()

