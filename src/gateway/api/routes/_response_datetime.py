"""Datetime formatting helpers for API response models."""

from datetime import datetime
from typing import Any


def datetime_isoformat(value: datetime) -> str:
    return value.isoformat()


def optional_datetime_isoformat(value: Any) -> str | None:
    if isinstance(value, datetime):
        return datetime_isoformat(value)
    return None
