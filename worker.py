"""Redis-backed, single-process video inference worker."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))

import cv2
import httpx
import redis

from logging_config import configure_logging, get_logger
from yolo_inference import YOLOInference

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "stampedeshield:queue"
PROCESSING_KEY = "stampedeshield:processing"
HEARTBEAT_KEY = "stampedeshield:worker:heartbeat"
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "75"))
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))
logger = get_logger(__name__)
stopping = False


def _stop(_signum, _frame):
    global stopping
    stopping = True


def send_discord_alert(job_id: str, score: int, count: int, store: redis.Redis) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook or score < ALERT_THRESHOLD:
        return
    # One alert per configured cooldown across worker restarts.
    cooldown_key = "stampedeshield:alert-cooldown:global"
    if not store.set(cooldown_key, "1", nx=True, ex=ALERT_COOLDOWN):
        return
    try:
        response = httpx.post(webhook, json={"content": f"StampedeShield high crowd risk: {score}/100, {count} people (job {job_id})."}, timeout=8)
        response.raise_for_status()
        logger.info("discord_alert_sent", job_id=job_id, score=score)
    except Exception as exc:
        # Avoid retrying every single video frame during a Discord outage.
        store.expire(cooldown_key, min(60, ALERT_COOLDOWN))
        logger.error("discord_alert_failed", job_id=job_id, error=str(exc))


def process_job(store: redis.Redis, inference: YOLOInference, job: dict) -> None:
    job_id = job["job_id"]
    job_key = f"stampedeshield:job:{job_id}"
    frame_key = f"stampedeshield:frame:{job_id}"
    heatmap_key = f"stampedeshield:heatmap:{job_id}"
    output_path = Path(job["output_path"])
    store.hset(job_key, mapping={"status": "processing", "started_at": str(time.time()), "error": ""})
    logger.info("job_started", job_id=job_id, model=MODEL_PATH)

    def publish(frame, risk_score, count):
        store.set(HEARTBEAT_KEY, str(time.time()), ex=60)
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            store.set(frame_key, encoded.tobytes(), ex=3600)
        zoom = store.get(f"stampedeshield:zoom:{job_id}")
        if zoom:
            setting = json.loads(zoom)
            inference.set_zoom_cell(int(setting.get("row", -1)), int(setting.get("col", -1)))
            subimage = inference.get_zoomed_subimage()
            if subimage is not None:
                zoom_ok, zoom_encoded = cv2.imencode(".jpg", subimage)
                if zoom_ok:
                    store.set(f"stampedeshield:zoom_frame:{job_id}", zoom_encoded.tobytes(), ex=3600)
        store.hset(job_key, mapping={"risk_score": str(risk_score), "count": str(count)})
        send_discord_alert(job_id, risk_score, count, store)

    try:
        inference.set_heatmap_enabled(store.get(heatmap_key) == b"1")
        inference.process_video(job["input_path"], str(output_path), on_frame=publish,
                                heatmap_enabled=lambda: store.get(heatmap_key) == b"1")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Video could not be opened or produced no output")
        store.hset(job_key, mapping={"status": "completed", "progress": "100", "completed_at": str(time.time()),
                                     "output_bytes": str(output_path.stat().st_size)})
        logger.info("job_completed", job_id=job_id, output_bytes=output_path.stat().st_size)
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        store.hset(job_key, mapping={"status": "failed", "error": str(exc)[:1000], "completed_at": str(time.time())})
        logger.exception("job_failed", job_id=job_id, error=str(exc))
    finally:
        Path(job["input_path"]).unlink(missing_ok=True)


def main() -> None:
    global stopping
    configure_logging()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    store = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=5)
    while not stopping:
        try:
            store.ping()
            break
        except redis.RedisError as exc:
            logger.warning("redis_unavailable_retrying", error=str(exc))
            time.sleep(2)
    if stopping:
        return
    # There is one worker service by default; recover items left in-flight by an abrupt stop.
    for pending in store.lrange(PROCESSING_KEY, 0, -1):
        store.lpush(QUEUE_KEY, pending)
    store.delete(PROCESSING_KEY)
    inference = YOLOInference(model_path=MODEL_PATH)
    logger.info("worker_ready", model=MODEL_PATH)
    while not stopping:
        try:
            store.set(HEARTBEAT_KEY, str(time.time()), ex=60)
            item = store.blmove(QUEUE_KEY, PROCESSING_KEY, timeout=2, src="LEFT", dest="RIGHT")
            if item:
                try:
                    process_job(store, inference, json.loads(item))
                finally:
                    state = store.hget(f"stampedeshield:job:{json.loads(item)['job_id']}", "status")
                    if state in (b"completed", b"failed"):
                        store.lrem(PROCESSING_KEY, 1, item)
        except redis.RedisError as exc:
            logger.error("worker_redis_error", error=str(exc))
            time.sleep(2)
    store.delete(HEARTBEAT_KEY)


if __name__ == "__main__":
    main()
