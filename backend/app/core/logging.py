"""Structured logging configuration and request correlation."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

import structlog

from app.core.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _inject_request_id(_logger, _name, event_dict):  # type: ignore[no-untyped-def]
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging() -> None:
    """Install structlog processors and route stdlib logging through them."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_request_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Third-party libraries are noisy at INFO; keep their warnings only.
    for noisy in ("rasterio", "rasterio._env", "botocore", "urllib3", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
