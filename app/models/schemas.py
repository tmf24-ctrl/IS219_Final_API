from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Raw resume text")
    job_description_text: str = Field(
        ..., min_length=10, description="Target job description text"
    )

    @field_validator("resume_text", "job_description_text")
    @classmethod
    def no_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Input text cannot be empty.")
        return value


class AnalyzeResponse(BaseModel):
    matching_skills: list[str]
    missing_skills: list[str]
    summary: str


class ImproveResumeResponse(BaseModel):
    improved_bullet_points: list[str]
    suggestions: list[str]


class LLMAnalysisResult(BaseModel):
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    summary: str = ""


class LLMImproveResult(BaseModel):
    improved_bullet_points: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class LLMResponseEnvelope(BaseModel):
    payload: dict
