"""FastAPI service for uploads, queued video processing, and job status."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from redis.asyncio import Redis
from starlette.exceptions import HTTPException as StarletteHTTPException

from logging_config import configure_logging, get_logger

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "static" / "uploads"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", BASE_DIR / "static" / "processed"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
QUEUE_KEY = "stampedeshield:queue"
JOB_TTL_SECONDS = 7 * 24 * 60 * 60
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

configure_logging()
logger = get_logger(__name__)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app.state.redis = Redis.from_url(REDIS_URL, decode_responses=False,
                                    socket_connect_timeout=3, socket_timeout=5)
    yield
    await app.state.redis.aclose()


app = FastAPI(title="StampedeShield API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/processed", StaticFiles(directory=str(PROCESSED_DIR), check_dir=False), name="processed")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        logger.error("http_error", path=request.url.path, status_code=exc.status_code)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    issues = [{"location": error["loc"], "message": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    logger.warning("request_validation_failed", path=request.url.path, issue_count=len(issues))
    return JSONResponse(status_code=422, content={"error": "Request validation failed", "issues": issues})


async def redis_client(request: Request) -> Redis:
    return request.app.state.redis


async def save_upload(file: UploadFile, job_id: str) -> tuple[Path, int]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported video format")
    destination = UPLOAD_DIR / f"{job_id}{suffix}"
    total = 0
    try:
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds upload size limit")
                await output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    return destination, total


async def enqueue(redis: Redis, job_id: str, input_path: Path, size_bytes: int) -> None:
    data = {"job_id": job_id, "input_path": str(input_path), "output_path": str(PROCESSED_DIR / f"{job_id}.mp4")}
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(f"stampedeshield:job:{job_id}", mapping={
            "status": "queued", "progress": "0", "risk_score": "0", "count": "0",
            "created_at": datetime.now(timezone.utc).isoformat(), "size_bytes": str(size_bytes),
            "input_name": input_path.name,
        })
        pipe.expire(f"stampedeshield:job:{job_id}", JOB_TTL_SECONDS)
        pipe.rpush(QUEUE_KEY, json.dumps(data))
        await pipe.execute()
    await redis.set("stampedeshield:latest_job", job_id, ex=JOB_TTL_SECONDS)


async def read_job(redis: Redis, job_id: str) -> dict | None:
    raw = await redis.hgetall(f"stampedeshield:job:{job_id}")
    if not raw:
        return None
    return {k.decode(): v.decode() for k, v in raw.items()}


async def resolve_job_id(redis: Redis, job_id: str | None) -> str:
    resolved = job_id or await redis.get("stampedeshield:latest_job")
    if isinstance(resolved, bytes):
        resolved = resolved.decode()
    if not resolved:
        raise HTTPException(status_code=404, detail="No video job is available")
    return str(resolved)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload", status_code=202)
async def upload(request: Request, video: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    input_path, size = await save_upload(video, job_id)
    redis = await redis_client(request)
    try:
        await enqueue(redis, job_id, input_path, size)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    logger.info("job_queued", job_id=job_id, bytes=size)
    return {"message": "Upload accepted", "job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(request: Request, job_id: str):
    job = await read_job(await redis_client(request), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job, "result_url": f"/processed/{job_id}.mp4" if job.get("status") == "completed" else None}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness(request: Request):
    try:
        redis = await redis_client(request)
        await redis.ping()
        if await redis.exists("stampedeshield:worker:heartbeat") == 0:
            raise HTTPException(status_code=503, detail="Inference worker unavailable")
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.warning("readiness_check_failed", dependency="redis", error=str(exc))
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc
    return {"status": "ready", "dependencies": {"redis": "ok", "worker": "ok"}}


@app.get("/live_preview", response_class=HTMLResponse)
async def live_preview(request: Request, job_id: str | None = None):
    job_id = await resolve_job_id(await redis_client(request), job_id)
    return templates.TemplateResponse(request=request, name="live_preview.html", context={"job_id": job_id})


@app.get("/video_feed")
async def video_feed(request: Request, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    async def frames():
        last_frame = None
        while True:
            frame = await redis.get(f"stampedeshield:frame:{job_id}")
            if frame and frame != last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/risk_score")
async def risk_score(request: Request, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    job = await read_job(redis, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"score": int(job.get("risk_score", 0)), "count": int(job.get("count", 0)), "status": job["status"]}


@app.get("/set_zoom")
async def set_zoom(request: Request, row: int = -1, col: int = -1, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    if await redis.exists(f"stampedeshield:job:{job_id}") == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    await redis.set(f"stampedeshield:zoom:{job_id}", json.dumps({"row": row, "col": col}), ex=JOB_TTL_SECONDS)
    return {"status": "OK", "row": row, "col": col}


@app.get("/zoom_feed")
async def zoom_feed(request: Request, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    async def frames():
        last_frame = None
        while True:
            frame = await redis.get(f"stampedeshield:zoom_frame:{job_id}")
            if frame and frame != last_frame:
                last_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/jobs/{job_id}/heatmap")
async def set_heatmap(request: Request, job_id: str, enabled: bool):
    redis = await redis_client(request)
    if await redis.exists(f"stampedeshield:job:{job_id}") == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    await redis.set(f"stampedeshield:heatmap:{job_id}", "1" if enabled else "0", ex=JOB_TTL_SECONDS)
    return {"status": "ok", "enabled": enabled}


@app.get("/toggle_heatmap")
async def legacy_toggle_heatmap(request: Request, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    key = f"stampedeshield:heatmap:{job_id}"
    enabled = await redis.get(key) == b"1"
    await redis.set(key, "0" if enabled else "1", ex=JOB_TTL_SECONDS)
    return RedirectResponse(url=f"/live_preview?job_id={job_id}", status_code=303)


@app.post("/toggle_heatmap")
async def toggle_heatmap(request: Request, job_id: str | None = None):
    redis = await redis_client(request)
    job_id = await resolve_job_id(redis, job_id)
    key = f"stampedeshield:heatmap:{job_id}"
    enabled = await redis.get(key) == b"1"
    await redis.set(key, "0" if enabled else "1", ex=JOB_TTL_SECONDS)
    return {"status": "ok", "enabled": not enabled}


@app.get("/process_video")
async def process_video_route(request: Request):
    legacy_input = UPLOAD_DIR / "input.mp4"
    if not legacy_input.is_file():
        raise HTTPException(status_code=404, detail="static/uploads/input.mp4 was not found")
    job_id = uuid.uuid4().hex
    await enqueue(await redis_client(request), job_id, legacy_input, legacy_input.stat().st_size)
    return RedirectResponse(url=f"/live_preview?job_id={job_id}", status_code=303)
