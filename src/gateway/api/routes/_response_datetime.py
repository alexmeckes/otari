"""Datetime formatting helpers for API response models."""

from datetime import datetime
from typing import Any


def optional_datetime_isoformat(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None
