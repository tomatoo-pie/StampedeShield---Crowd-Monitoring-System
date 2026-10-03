import json

from fastapi.testclient import TestClient

import app as api


class MemoryPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def hset(self, key, mapping):
        self.commands.append(("hset", key, mapping))

    def rpush(self, key, value):
        self.commands.append(("rpush", key, value))

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))

    async def execute(self):
        for command, key, value in self.commands:
            if command == "hset":
                self.redis.hashes.setdefault(key, {}).update(value)
            elif command == "rpush":
                self.redis.queues.setdefault(key, []).append(value)


class MemoryRedis:
    def __init__(self):
        self.hashes = {}
        self.queues = {}
        self.values = {}

    def pipeline(self, transaction=True):
        return MemoryPipeline(self)

    async def hgetall(self, key):
        return {k.encode(): str(v).encode() for k, v in self.hashes.get(key, {}).items()}

    async def exists(self, key):
        return int(key in self.hashes or key in self.values)

    async def set(self, key, value, **kwargs):
        self.values[key] = value.encode() if isinstance(value, str) else value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def ping(self):
        return True

    async def aclose(self):
        pass


def test_upload_queues_job_and_status(monkeypatch, tmp_path):
    fake_redis = MemoryRedis()
    monkeypatch.setattr(api.Redis, "from_url", lambda *args, **kwargs: fake_redis)
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path / "processed")

    with TestClient(api.app) as client:
        response = client.post("/upload", files={"video": ("crowd.mp4", b"sample video", "video/mp4")})
        assert response.status_code == 202
        result = response.json()
        assert result["status"] == "queued"
        job_id = result["job_id"]
        assert (tmp_path / "uploads" / f"{job_id}.mp4").read_bytes() == b"sample video"
        assert len(fake_redis.queues[api.QUEUE_KEY]) == 1
        queued = json.loads(fake_redis.queues[api.QUEUE_KEY][0])
        assert queued["job_id"] == job_id
        status = client.get(f"/jobs/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "queued"
        assert status.json()["result_url"] is None
        preview = client.get(f"/live_preview?job_id={job_id}")
        assert preview.status_code == 200
        assert f"/video_feed?job_id={job_id}" in preview.text


def test_upload_rejects_unsupported_extension():
    with TestClient(api.app) as client:
        response = client.post("/upload", files={"video": ("notes.txt", b"not a video", "text/plain")})
    assert response.status_code == 415


def test_health_and_readiness(monkeypatch):
    fake_redis = MemoryRedis()
    fake_redis.values["stampedeshield:worker:heartbeat"] = b"alive"
    monkeypatch.setattr(api.Redis, "from_url", lambda *args, **kwargs: fake_redis)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["dependencies"] == {"redis": "ok", "worker": "ok"}


def test_readiness_fails_when_worker_has_no_heartbeat(monkeypatch):
    fake_redis = MemoryRedis()
    monkeypatch.setattr(api.Redis, "from_url", lambda *args, **kwargs: fake_redis)
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 503


def test_missing_job_is_404(monkeypatch):
    fake_redis = MemoryRedis()
    monkeypatch.setattr(api.Redis, "from_url", lambda *args, **kwargs: fake_redis)
    with TestClient(api.app) as client:
        assert client.get("/jobs/no-such-job").status_code == 404
