from __future__ import annotations

import json
import logging as std_logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, TextIO

from eager.config import is_secret_field

SENSITIVE_RUNTIME_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:authorization|credential|cookie|session)(?:$|[_-])"
)
VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._/-]{8,}"),
)


def redact_text(value: str) -> str:
    output = value
    for pattern in VALUE_PATTERNS:
        output = pattern.sub("<redacted>", output)
    return output


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            output[key] = (
                "<redacted>"
                if is_secret_field(key) or SENSITIVE_RUNTIME_KEY.search(key)
                else redact(child)
            )
        return output
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, tuple):
        return [redact(child) for child in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class RedactionFilter(std_logging.Filter):
    def filter(self, record: std_logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        else:
            record.msg = redact(record.msg)
        if record.args:
            record.args = redact(record.args)
        for key, value in list(record.__dict__.items()):
            if key not in std_logging.LogRecord(None, 0, "", 0, "", (), None).__dict__:
                record.__dict__[key] = (
                    "<redacted>"
                    if is_secret_field(key) or SENSITIVE_RUNTIME_KEY.search(key)
                    else redact(value)
                )
        return True


class JsonFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload: MutableMapping[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        standard = std_logging.LogRecord(None, 0, "", 0, "", (), None).__dict__
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard and key not in {"message", "asctime"}
        }
        if extras:
            payload["fields"] = extras
        return json.dumps(redact(payload), ensure_ascii=False, sort_keys=True)


def configure_logging(
    *,
    level: str = "INFO",
    stream: TextIO | None = None,
    log_path: Path | None = None,
    logger_name: str = "eager",
) -> std_logging.Logger:
    logger = std_logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    redaction_filter = RedactionFilter()

    stream_handler = std_logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(JsonFormatter())
    stream_handler.addFilter(redaction_filter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = std_logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(redaction_filter)
        logger.addHandler(file_handler)
    return logger
