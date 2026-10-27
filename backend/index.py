"""Entry point for Vercel's Python runtime (it looks for `app` in index.py).
Locally and on Render you can keep using: uvicorn app.main:app"""
from app.main import app  # noqa: F401
