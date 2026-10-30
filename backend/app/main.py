"""FastAPI backend for Speech Emotion Recognition.

Run locally:   uvicorn app.main:app --reload --port 8000
Docs:          http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import Registry, run_prediction

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
registry = Registry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load()
    print(f"[startup] loaded models: {list(registry.models)} (default: {registry.default})")
    yield


app = FastAPI(title="Speech Emotion Recognition API", version="1.0.0", lifespan=lifespan)

# Comma-separated list, e.g. "https://ser.vercel.app,http://localhost:5173"
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX"),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    n = len(registry.models)
    return {
        "status": "ok" if n else "degraded",
        "models_loaded": n,
        "default_model": registry.default,
        "model_ids": list(registry.models),
    }


@app.get("/api/models")
def models():
    return {"default": registry.default, "models": [m.meta for m in registry.models.values()]}


@app.get("/api/results")
def results():
    lb = registry.leaderboard()
    if lb is None:
        raise HTTPException(404, "No leaderboard.json in the model folder yet. Train and export first.")
    return lb


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str | None = Query(None)):
    if not registry.models:
        raise HTTPException(503, "No model loaded on the server.")
    model_id = model or registry.default
    if model_id not in registry.models:
        raise HTTPException(404, f"Unknown model '{model_id}'. See /api/models.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large.")
    try:
        return run_prediction(registry.models[model_id], data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
