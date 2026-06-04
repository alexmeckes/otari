from typing import Any

from pydantic import BaseModel, Field


class BatchRequestItem(BaseModel):
    custom_id: str
    body: dict[str, Any]


class CreateBatchRequest(BaseModel):
    model: str
    requests: list[BatchRequestItem] = Field(min_length=1, max_length=10_000)
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None
