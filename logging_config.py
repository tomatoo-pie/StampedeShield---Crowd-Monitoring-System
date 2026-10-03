"""JSON structured logging shared by the API and inference worker."""
import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StructuredLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _write(self, level: int, message: str, **fields):
        self.logger.log(level, message, extra={"fields": fields})

    def info(self, message: str, **fields):
        self._write(logging.INFO, message, **fields)

    def debug(self, message: str, **fields):
        self._write(logging.DEBUG, message, **fields)

    def warning(self, message: str, **fields):
        self._write(logging.WARNING, message, **fields)

    def error(self, message: str, **fields):
        self._write(logging.ERROR, message, **fields)

    def exception(self, message: str, **fields):
        self.logger.exception(message, extra={"fields": fields})


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))
