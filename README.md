# StampedeShield

Crowd monitoring for uploaded video with person detection/tracking, a smoothed risk score, heatmap output, and optional Discord notifications.

## Architecture

```text
Browser --video upload/status--> FastAPI --Redis list--> inference worker
Browser <--job status/risk---- FastAPI <--Redis hashes/frame cache-- worker
                                                     | YOLO + DeepSort + OpenCV
                                                     + processed MP4 in shared storage
```

FastAPI only validates and streams uploads to disk, stores job metadata, enqueues work, serves status, and streams cached preview frames. It does not run inference in its request handlers. One worker process loads the model and consumes jobs sequentially. Redis stores the queue, job states and metrics, current preview frames, heatmap settings, and alert cooldown keys. The processing list retains an in-flight job so that a worker restart can requeue it. Keep one worker service instance unless in-flight recovery is upgraded to use worker leases; jobs can be processed concurrently by multiple worker instances, but startup recovery assumes one instance.

Outputs and uploads are stored in shared filesystem paths. The worker removes source uploads after processing. Completed MP4s are available at `/processed/{job_id}.mp4`.

## Local development (Python 3.11 recommended)

Install Redis separately or start the Compose Redis service. Install PyTorch's CPU build on machines without CUDA, then the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements-dev.txt
docker compose up -d redis
```

In two terminals with the virtual environment active, run:

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
python -m uvicorn app:app --reload
```

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
python worker.py
```

The worker's default `MODEL_PATH` is `yolov8n.pt`; Ultralytics may download that model on first use if it is not present. DeepSort also needs its MobileNet embedder weights. Set `MODEL_PATH` to an existing model file to use a different checkpoint. Upload a supported video at the UI. `/health` is liveness and `/ready` requires Redis and a model-loaded worker heartbeat.

Run the API/queue unit tests with:

```powershell
python -m pytest -q
```

## Docker Compose

Place the selected YOLO checkpoint in `./models` (or set `MODEL_DIR` to its folder). The model is deliberately not bundled in the image. Configure `MODEL_PATH` to the model's container path if it is not `/models/yolov8n.pt`. `.dockerignore` excludes dataset images, training runs, model weights, and video output from the build context.

```powershell
Copy-Item .env.example .env
# Edit .env and set MODEL_PATH / MODEL_DIR as needed.
docker compose up --build
```

Open `http://localhost:8000`. View service logs with `docker compose logs -f api worker redis`; stop with `docker compose down`. Persistent video and Redis data use named volumes. To remove them as well, use `docker compose down -v`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `MODEL_PATH` | `yolov8n.pt` | YOLO checkpoint path, resolved by Ultralytics |
| `UPLOAD_DIR` | `static/uploads` | Upload storage |
| `PROCESSED_DIR` | `static/processed` | Processed MP4 storage |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum upload size (500 MiB) |
| `DISCORD_WEBHOOK_URL` | unset | Optional Discord incoming webhook |
| `ALERT_THRESHOLD` | `75` | Minimum risk score that triggers an alert |
| `ALERT_COOLDOWN_SECONDS` | `300` | Global repeat-alert cooldown |
| `LOG_LEVEL` | `INFO` | Structured JSON log level |

Never commit webhook URLs or model weights. Discord alerting is disabled when the webhook is unset. Webhook delivery failures are logged and do not fail video processing.

## API

- `POST /upload` — multipart field `video`; returns HTTP 202 and a `job_id`.
- `GET /jobs/{job_id}` — job state and result URL when complete.
- `GET /api/risk_score?job_id=...` — current score, person count, and processing state.
- `POST /toggle_heatmap?job_id=...` — toggles the job's heatmap setting.
- `GET /live_preview?job_id=...` — dashboard; `/video_feed?job_id=...` streams its frames.
- `GET /health`, `GET /ready` — process liveness and Redis/worker readiness.

## Training

Dataset paths in `dataset/data.yaml` are relative to that directory (`images/train`, `images/val`). Run `python train_yolo.py`; set `TRAIN_MODEL` or `TRAIN_DEVICE` to override the checkpoint or device. Training data and resulting weights are not included.

## Deployment notes

The Compose stack is suitable as an EC2 starting point: expose port 8000 only behind a TLS-terminating reverse proxy/load balancer, persist the Redis and video volumes, provide a model file, and inject Discord credentials through the host's secret mechanism. The default image uses CPU-only PyTorch and is intended for CPU instances. GPU EC2 requires a compatible NVIDIA driver/container runtime and a CUDA-enabled PyTorch image; that variant is not configured or verified here. No AWS resources are created by this project.

The current implementation does not include authentication, rate limiting, malware scanning, automatic video retention, or multi-instance worker leases. Configure trusted access controls before exposing the API publicly.
