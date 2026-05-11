# AI Job Application Assistant API

A professional FastAPI backend that compares a resume against a job description, identifies skill gaps, and rewrites resume bullet points to better match a target role — powered by an LLM (Google Gemini) with a fully functional mock mode for offline use.

---

## Why I Built This

I researched Applied AI Engineer and AI Product Engineer roles on LinkedIn and Indeed. Across multiple job descriptions, three requirements kept appearing that were underrepresented on my resume:

1. **LLM API integration** — calling structured AI completions from application code
2. **Production-style REST API design** — versioned, documented, validated endpoints
3. **AI-augmented developer tooling** — building tools that use AI to help real workflows

I used ChatGPT to analyze the gap between my resume and those job descriptions. The output pointed to the same conclusion: I had Python experience but had not yet shipped a project that combined AI services with a clean backend API.

I then discussed project ideas with AI, and we landed on this: a resume analysis API that a developer could actually use — something small enough to build well but credible enough to show employers.

---

## What This Project Demonstrates

| Skill | How it is demonstrated |
|---|---|
| LLM API integration | `LLMService` calls Google Gemini with structured prompts and parses JSON responses |
| FastAPI REST design | Two validated endpoints with Pydantic schemas, dependency injection, and OpenAPI docs |
| Graceful degradation | Mock mode activates automatically when no API key is present — no crashes, no config required |
| Backend testing | 7 pytest tests covering endpoint success, validation errors, and service-level LLM failures |
| Clean code structure | Separation of routes, services, models, and utilities |

---

## Features

- `POST /analyze` — compares resume text against a job description; returns matching skills, missing skills, and a summary
- `POST /improve-resume` — rewrites resume bullet points to better align with the target role; returns improved bullets and actionable suggestions
- **Mock mode** — works fully offline with no API key using local keyword extraction
- **LLM mode** — plug in a free Gemini API key for real AI-powered analysis
- Input validation with Pydantic (returns HTTP 422 on bad input)
- LLM errors surfaced as HTTP 502 with clear messages

---

## Tech Stack

- Python 3.10+
- FastAPI
- Google Gemini API (`google-generativeai`)
- Pydantic v2
- Pytest + HTTPX

---

## Project Structure

```text
is219-final/
├── app/
│   ├── main.py                  # FastAPI app and router registration
│   ├── routes/
│   │   ├── analyze.py           # POST /analyze endpoint
│   │   └── improve_resume.py    # POST /improve-resume endpoint
│   ├── services/
│   │   └── llm_service.py       # Gemini integration + mock fallback
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       └── text_processing.py   # Skill extraction and bullet parsing
├── tests/
│   ├── test_api.py              # Endpoint integration tests
│   └── test_services.py        # LLM service unit tests
├── conftest.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd is219-final
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and add your Gemini API key if you want real AI responses.  
**No key? No problem.** The app runs in mock mode automatically.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Run the Tests

```bash
pytest -v
```

All 7 tests pass with no API key required.

```
tests/test_api.py::test_analyze_endpoint_success        PASSED
tests/test_api.py::test_improve_resume_success          PASSED
tests/test_api.py::test_invalid_input_returns_422       PASSED
tests/test_api.py::test_llm_error_returns_502           PASSED
tests/test_services.py::test_llm_service_uses_mock_...  PASSED
tests/test_services.py::test_mock_analyze_returns_...   PASSED
tests/test_services.py::test_mock_improve_generates...  PASSED
```

---

## API Reference

### `POST /analyze`

Compares a resume against a job description.

**Request body**

```json
{
  "resume_text": "Python backend engineer with FastAPI, SQL, and pytest experience.",
  "job_description_text": "Seeking Python, FastAPI, Docker, AWS, and CI/CD experience."
}
```

**Response**

```json
{
  "matching_skills": ["fastapi", "python", "sql"],
  "missing_skills": ["aws", "ci/cd", "docker"],
  "summary": "Found 3 matching skills and 3 missing skills based on keyword analysis."
}
```

---

### `POST /improve-resume`

Rewrites resume bullet points to better target a role.

**Request body**

```json
{
  "resume_text": "- Built APIs for internal tools\n- Improved test coverage",
  "job_description_text": "Need Python and cloud-native backend experience with measurable impact."
}
```

**Response**

```json
{
  "improved_bullet_points": [
    "Built APIs for internal tools; aligned with python requirements.",
    "Improved test coverage; aligned with python requirements."
  ],
  "suggestions": [
    "Quantify outcomes in each bullet, such as latency reduction or revenue impact.",
    "Prioritize bullets that match required job skills and technologies.",
    "Group skills by category (languages, frameworks, cloud, ML/AI) for readability."
  ]
}
```

---

## How Mock Mode Works

When `GEMINI_API_KEY` is not set, `LLMService` activates mock mode:

- `/analyze` uses regex-based keyword matching against a curated skill list to find matching and missing skills
- `/improve-resume` extracts bullet points from the resume and appends job-aligned suggestions
- All tests run against this mode — results are deterministic and no network call is made

To switch to live AI responses, add a valid `GEMINI_API_KEY` to your `.env` file. Free keys are available at [aistudio.google.com](https://aistudio.google.com/apikey).

---

## Connection to Target Role

Applied AI Engineer and AI Product Engineer roles require developers who can:

- integrate LLM APIs into real application code
- build APIs that are reliable enough for production use
- handle failure modes gracefully (bad input, upstream LLM errors, missing credentials)

This project practices all three in a self-contained backend that solves a real problem. It is small enough to understand fully and professional enough to discuss in an interview.

---

## License

Educational and portfolio use.

