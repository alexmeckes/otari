from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from gateway.api.routes._response_datetime import datetime_isoformat, optional_datetime_isoformat
from gateway.models.entities import APIKey


class CreateKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    key_name: str | None = Field(default=None, description="Optional name for the key")
    user_id: str | None = Field(default=None, description="Optional user ID to associate with this key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class CreateKeyResponse(BaseModel):
    """Response model for creating a new API key."""

    id: str
    key: str
    key_name: str | None
    user_id: str | None
    created_at: str
    expires_at: str | None
    is_active: bool
    metadata: dict[str, Any]


class KeyInfo(BaseModel):
    """Response model for key information."""

    id: str
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, key: APIKey) -> "KeyInfo":
        return cls(
            id=str(key.id),
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=datetime_isoformat(key.created_at),
            last_used_at=optional_datetime_isoformat(key.last_used_at),
            expires_at=optional_datetime_isoformat(key.expires_at),
            is_active=bool(key.is_active),
            metadata=key.metadata_dict(),
        )


class UpdateKeyRequest(BaseModel):
    """Request model for updating a key."""

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None
