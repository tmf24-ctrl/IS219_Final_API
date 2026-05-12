import json
import os
import re
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from app.models.schemas import LLMAnalysisResult, LLMImproveResult

load_dotenv()

_SYSTEM_INSTRUCTION = (
    "You are a senior technical recruiter and professional resume coach. "
    "Always return valid JSON only -- no markdown, no code fences, no extra text."
)


class LLMServiceError(Exception):
    pass


class LLMService:
    def __init__(self, model: str = "gemini-2.0-flash-lite") -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", model)
        if not api_key:
            raise LLMServiceError(
                "GEMINI_API_KEY is not set. Add it to your .env file and restart the server."
            )
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=_SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

    def analyze_resume(self, resume_text: str, job_description_text: str) -> LLMAnalysisResult:
        prompt = self._analysis_prompt(resume_text, job_description_text)
        data = self._call_llm_json(prompt)
        try:
            return LLMAnalysisResult(**data)
        except Exception as exc:
            raise LLMServiceError(f"Invalid analysis response shape: {exc}") from exc

    def improve_resume(self, resume_text: str, job_description_text: str) -> LLMImproveResult:
        prompt = self._improve_prompt(resume_text, job_description_text)
        data = self._call_llm_json(prompt)
        try:
            return LLMImproveResult(**data)
        except Exception as exc:
            raise LLMServiceError(f"Invalid improve response shape: {exc}") from exc

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        try:
            response = self._client.generate_content(prompt)
            raw_text = response.text
        except Exception as exc:
            msg = str(exc)
            if any(k in msg for k in ("API_KEY_INVALID", "PERMISSION_DENIED", "401", "403")):
                raise LLMServiceError("Gemini authentication failed. Check that your API key is valid.") from exc
            raise LLMServiceError(f"Gemini request failed: {type(exc).__name__}: {msg}") from exc

        raw_text = re.sub(r"^``[a-z]*\n?", "", raw_text.strip(), flags=re.IGNORECASE)
        raw_text = re.sub(r"``$", "", raw_text.strip())

        if not raw_text:
            raise LLMServiceError("LLM returned an empty response")

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(f"LLM returned non-JSON output: {exc}") from exc

    def _analysis_prompt(self, resume_text: str, job_description_text: str) -> str:
        return (
            "You are a senior technical recruiter reviewing a candidate for a specific role.\n\n"
            "Carefully read both the resume and the job description, then return a JSON object with this exact shape:\n"
            '{\n  "matching_skills": ["..."],\n  "missing_skills": ["..."],\n  "summary": "..."\n}\n\n'
            "Rules:\n"
            "- matching_skills: specific skills, tools, and technologies present in BOTH the resume and the job description.\n"
            "- missing_skills: skills the job description asks for that are NOT on the resume.\n"
            "- summary: 3-4 sentences assessing overall fit, strongest relevant experience, critical gap, and a concrete recommendation.\n\n"
            f"Resume:\n{resume_text}\n\nJob Description:\n{job_description_text}"
        )

    def _improve_prompt(self, resume_text: str, job_description_text: str) -> str:
        return (
            "You are an expert resume writer who specializes in tech and engineering roles.\n\n"
            "Rewrite the resume bullet points to better match the job description, then provide targeted advice.\n\n"
            "Return a JSON object with this exact shape:\n"
            '{\n  "improved_bullet_points": ["..."],\n  "suggestions": ["..."]\n}\n\n'
            "Rules for improved_bullet_points:\n"
            "- Start each bullet with a strong action verb. Quantify results. Weave in job description keywords.\n"
            "- Keep each bullet to one line, max 20 words. Return at least 4 bullets.\n\n"
            "Rules for suggestions (return exactly 4):\n"
            "1. Skills to add or certifications to pursue.\n"
            "2. How to reframe existing experience.\n"
            "3. A resume format or structure improvement.\n"
            "4. Biggest weakness for this role and how to address it.\n\n"
            f"Resume:\n{resume_text}\n\nJob Description:\n{job_description_text}"
        )


def get_llm_service() -> LLMService:
    return LLMService()
