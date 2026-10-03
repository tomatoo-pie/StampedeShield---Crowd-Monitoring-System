import numpy as np

from worker import process_job, send_discord_alert


class MemoryStore:
    def __init__(self):
        self.hashes = {}
        self.values = {}
        self.ttls = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key):
        self.values.pop(key, None)

    def expire(self, key, seconds):
        self.ttls[key] = seconds
        return True


class FakeInference:
    def set_heatmap_enabled(self, enabled):
        self.heatmap = enabled

    def process_video(self, input_path, output_path, on_frame, heatmap_enabled):
        with open(output_path, "wb") as output:
            output.write(b"processed-video")
        on_frame(np.zeros((24, 24, 3), dtype=np.uint8), 82, 17)

    def set_zoom_cell(self, row, col):
        pass

    def get_zoomed_subimage(self):
        return None


def test_worker_publishes_preview_metrics_and_result(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"source")
    job = {"job_id": "test-job", "input_path": str(input_path), "output_path": str(output_path)}
    store = MemoryStore()

    process_job(store, FakeInference(), job)

    state = store.hashes["stampedeshield:job:test-job"]
    assert state["status"] == "completed"
    assert state["risk_score"] == "82"
    assert state["count"] == "17"
    assert store.values["stampedeshield:frame:test-job"].startswith(b"\xff\xd8")
    assert output_path.read_bytes() == b"processed-video"
    assert not input_path.exists()


def test_worker_marks_inference_failure_and_cleans_input(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"source")

    class FailingInference(FakeInference):
        def process_video(self, *args, **kwargs):
            raise ValueError("bad test video")

    store = MemoryStore()
    process_job(store, FailingInference(), {
        "job_id": "failed-job", "input_path": str(input_path), "output_path": str(output_path)
    })

    state = store.hashes["stampedeshield:job:failed-job"]
    assert state["status"] == "failed"
    assert state["error"] == "bad test video"
    assert not input_path.exists()


def test_discord_alert_is_sent_once_during_cooldown(monkeypatch):
    import worker

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(worker.httpx, "post", lambda *args, **kwargs: calls.append((args, kwargs)) or Response())
    store = MemoryStore()
    send_discord_alert("crowd-job", 85, 23, store)
    send_discord_alert("crowd-job-2", 95, 40, store)

    assert len(calls) == 1
    assert calls[0][1]["json"]["content"].startswith("StampedeShield high crowd risk: 85/100")
    assert "stampedeshield:alert-cooldown:global" in store.values
