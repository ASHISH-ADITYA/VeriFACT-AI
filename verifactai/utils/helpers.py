"""
Shared utilities: logging, timing, hashing, retry logic.
"""

from __future__ import annotations

import hashlib
import time
import functools
from typing import Callable, TypeVar, Any

from loguru import logger

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Structured logger configuration
# ---------------------------------------------------------------------------
logger.remove()  # remove default stderr handler
logger.add(
    "verifactai.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),   # also print INFO+ to console
    level="INFO",
    format="{time:HH:mm:ss} | {level:<8} | {message}",
)


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def md5_hash(text: str) -> str:
    """Deterministic cache key for a text input."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Call *func* with exponential backoff on failure."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"{func.__qualname__} attempt {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {delay:.1f}s …"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{func.__qualname__} failed after {attempts} attempts"
    ) from last_exc
