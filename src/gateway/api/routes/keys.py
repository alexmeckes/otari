import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.api.routes._database import commit_or_database_error, get_api_key_or_404
from gateway.api.routes._key_models import CreateKeyRequest, CreateKeyResponse, KeyInfo, UpdateKeyRequest
from gateway.auth.models import generate_api_key, hash_key
from gateway.models.entities import APIKey, User

router = APIRouter(prefix="/v1/keys", tags=["keys"])


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_key(
    request: CreateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CreateKeyResponse:
    """Create a new API key.

    Requires master key authentication.

    If user_id is provided, the key will be associated with that user (creates user if it doesn't exist).
    If user_id is not provided, a new user will be created automatically and the key will be associated with it.
    """
    api_key = generate_api_key()
    key_hash = hash_key(api_key)
    key_id = uuid.uuid4()

    if request.user_id:
        result = await db.execute(select(User).where(User.user_id == request.user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                user_id=request.user_id,
                alias=f"User {request.user_id}",
            )
            db.add(user)
        elif user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{request.user_id}' has been deleted. Recreate via POST /v1/users first.",
            )
        user_id = request.user_id
    else:
        user_id = f"apikey-{key_id}"
        user = User(
            user_id=user_id,
            alias=f"Virtual user for API key: {request.key_name or 'unnamed'}",
        )
        db.add(user)

    db_key = APIKey(
        id=str(key_id),
        key_hash=key_hash,
        key_name=request.key_name,
        user_id=user_id,
        expires_at=request.expires_at,
        metadata_=request.metadata,
    )

    db.add(db_key)
    await commit_or_database_error(db)
    await db.refresh(db_key)

    key_info = KeyInfo.from_model(db_key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=api_key,
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[KeyInfo]:
    """List all API keys.

    Requires master key authentication.
    """
    result = await db.execute(select(APIKey).offset(skip).limit(limit))
    keys = result.scalars().all()

    return [KeyInfo.from_model(key) for key in keys]


@router.get("/{key_id}", dependencies=[Depends(verify_master_key)])
async def get_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeyInfo:
    """Get details of a specific API key.

    Requires master key authentication.
    """
    key = await get_api_key_or_404(db, key_id)
    return KeyInfo.from_model(key)


@router.patch("/{key_id}", dependencies=[Depends(verify_master_key)])
async def update_key(
    key_id: str,
    request: UpdateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeyInfo:
    """Update an API key.

    Requires master key authentication.
    """
    key = await get_api_key_or_404(db, key_id)
    if request.key_name is not None:
        key.key_name = request.key_name
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.expires_at is not None:
        key.expires_at = request.expires_at
    if request.metadata is not None:
        key.metadata_ = request.metadata

    await commit_or_database_error(db)
    await db.refresh(key)

    return KeyInfo.from_model(key)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete (revoke) an API key.

    Requires master key authentication.
    """
    key = await get_api_key_or_404(db, key_id)
    await db.delete(key)
    await commit_or_database_error(db)
