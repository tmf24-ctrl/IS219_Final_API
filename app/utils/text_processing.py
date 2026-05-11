import re

COMMON_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "nosql",
    "fastapi",
    "flask",
    "django",
    "node.js",
    "react",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "openai",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "git",
    "ci/cd",
    "pytest",
    "redis",
    "postgresql",
    "mongodb",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text: str) -> set[str]:
    normalized = normalize_text(text)
    found: set[str] = set()
    for skill in COMMON_SKILLS:
        if any(ch.isalnum() for ch in skill):
            pattern = re.escape(skill)
            if re.search(rf"\b{pattern}\b", normalized):
                found.add(skill)
        elif skill in normalized:
            found.add(skill)
    return found


def extract_bullet_points(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped):
            bullets.append(re.sub(r"^[-*]\s+", "", stripped))
    return bullets
