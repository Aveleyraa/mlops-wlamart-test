import logging
import os
import json
import time
from typing import Any, Dict, Callable
from functools import wraps

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        # Attach extras if provided via logger.extra / LoggerAdapter
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)

    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL)
            logger.propagate = False
        return logger

    def log_timing(logger: logging.Logger, label: str) -> Callable:
        """Decorator to log function execution time."""
        def _decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def _wrapper(*args, **kwargs):
                t0 = time.time()
                try:
                    return fn(*args, **kwargs)
                finally:
                    dt = (time.time() - t0) * 1000
                    logger.info(f"{label}: {dt:.1f}ms")
            return _wrapper
        return _decorator