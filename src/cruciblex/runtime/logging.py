from __future__ import annotations

import logging
from typing import Any

LOGGER_NAME = "cruciblex"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def bind_event(event: str, **fields: Any) -> str:
    parts = [event]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    return " ".join(parts)
