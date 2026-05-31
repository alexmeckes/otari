from typing import Any


def ensure_stream_usage_options(completion_kwargs: dict[str, Any]) -> None:
    if completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}
