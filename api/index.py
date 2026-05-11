import os
import sys

# Make project root importable so `from app.*` works inside Vercel's runtime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: F401  — Vercel picks up the `app` ASGI variable
