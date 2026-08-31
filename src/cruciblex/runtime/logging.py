from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER_NAME = "cruciblex"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def configure_run_logging(output_root: Path, log_name: str = "run.log") -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / log_name
    logger = get_logger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        if getattr(handler, "_cx_run_log_path", None) is not None:
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler._cx_run_log_path = str(log_path)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return log_path


def append_run_log(output_root: Path, text: str, log_name: str = "run.log") -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / log_name
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
    return log_path


def bind_event(event: str, **fields: Any) -> str:
    parts = [event]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    return " ".join(parts)
