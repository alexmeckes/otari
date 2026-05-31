"""Shared helpers for typed server-sent events."""

from typing import Protocol


class TypedStreamEvent(Protocol):
    type: str

    def model_dump_json(self, *, exclude_none: bool = False) -> str: ...


def format_typed_stream_event(event: TypedStreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
