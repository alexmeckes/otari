"""OpenAI-compatible audio transcription and speech endpoints."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM, aspeech, atranscription
from any_llm.types.audio import Transcription
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._audio_models import AudioSpeechRequest
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._usage import rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import get_provider_kwargs

router = APIRouter(prefix="/v1", tags=["audio"])

# Maximum upload size for audio files (25 MB, matching OpenAI's limit)
_MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024

# Mapping from response_format to MIME type for speech endpoint
_SPEECH_CONTENT_TYPES: dict[str | None, str] = {
    None: "audio/mpeg",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",
}


@dataclass(frozen=True)
class _AudioRequestContext:
    api_key_id: str | None
    user_id: str
    rate_limit_info: RateLimitInfo | None


def _resolve_audio_request_context(
    *,
    raw_request: Request,
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
) -> _AudioRequestContext:
    api_key, is_master_key = auth_result
    user_id = resolve_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
    )
    return _AudioRequestContext(
        api_key_id=api_key.id if api_key else None,
        user_id=user_id,
        rate_limit_info=check_rate_limit(raw_request, user_id),
    )


async def _log_audio_usage(
    *,
    log_writer: LogWriter,
    context: _AudioRequestContext,
    model: str,
    provider: Any,
    endpoint: str,
    error: str | None = None,
) -> None:
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=context.api_key_id,
        user_id=context.user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )
    if error is None:
        # Audio endpoints do not expose measurable usage units yet, so cost is
        # left unset until a dedicated pricing metric is available.
        usage_log.prompt_tokens = 0
        usage_log.completion_tokens = 0
        usage_log.total_tokens = 0
    await log_writer.put(usage_log)


@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str | None = Form(None),
    temperature: float | None = Form(None),
    user: str | None = Form(None),
) -> dict[str, Any]:
    """OpenAI-compatible audio transcription endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    context = _resolve_audio_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=user,
    )

    _ = await validate_user_budget(db, context.user_id, model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model_name = AnyLLM.split_model_provider(model)

    provider_kwargs = get_provider_kwargs(config, provider)

    file_bytes = await file.read()
    if len(file_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    transcription_kwargs: dict[str, Any] = {
        "model": model_name,
        "file": file_bytes,
        "provider": provider,
        **provider_kwargs,
    }
    if language is not None:
        transcription_kwargs["language"] = language
    if prompt is not None:
        transcription_kwargs["prompt"] = prompt
    if response_format is not None:
        transcription_kwargs["response_format"] = response_format
    if temperature is not None:
        transcription_kwargs["temperature"] = temperature

    try:
        result: Transcription = await atranscription(**transcription_kwargs)
        await _log_audio_usage(
            log_writer=log_writer,
            context=context,
            model=model_name,
            provider=provider,
            endpoint="/v1/audio/transcriptions",
        )

    except HTTPException:
        raise
    except Exception as e:
        await _log_audio_usage(
            log_writer=log_writer,
            context=context,
            model=model_name,
            provider=provider,
            endpoint="/v1/audio/transcriptions",
            error=str(e),
        )

        logger.error("Provider call failed for %s:%s: %s", provider, model_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if context.rate_limit_info:
        for key, value in rate_limit_headers(context.rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump()


@router.post(
    "/audio/speech",
    response_model=None,
    responses={
        200: {
            "description": "Audio bytes in the requested format",
            "content": {
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                "audio/opus": {"schema": {"type": "string", "format": "binary"}},
                "audio/aac": {"schema": {"type": "string", "format": "binary"}},
                "audio/flac": {"schema": {"type": "string", "format": "binary"}},
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/L16": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
)
async def create_speech(
    raw_request: Request,
    response: Response,
    request: AudioSpeechRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> StreamingResponse:
    """OpenAI-compatible audio speech (TTS) endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    context = _resolve_audio_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=request.user,
    )

    _ = await validate_user_budget(db, context.user_id, request.model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model_name = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    speech_kwargs: dict[str, Any] = {
        "model": model_name,
        "input": request.input,
        "voice": request.voice,
        "provider": provider,
        **provider_kwargs,
    }
    if request.instructions is not None:
        speech_kwargs["instructions"] = request.instructions
    if request.response_format is not None:
        speech_kwargs["response_format"] = request.response_format
    if request.speed is not None:
        speech_kwargs["speed"] = request.speed

    try:
        audio_bytes: bytes = await aspeech(**speech_kwargs)
        await _log_audio_usage(
            log_writer=log_writer,
            context=context,
            model=model_name,
            provider=provider,
            endpoint="/v1/audio/speech",
        )

    except HTTPException:
        raise
    except Exception as e:
        await _log_audio_usage(
            log_writer=log_writer,
            context=context,
            model=model_name,
            provider=provider,
            endpoint="/v1/audio/speech",
            error=str(e),
        )

        logger.error("Provider call failed for %s:%s: %s", provider, model_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    headers: dict[str, str] = {}
    if context.rate_limit_info:
        headers.update(rate_limit_headers(context.rate_limit_info))

    return StreamingResponse(
        content=iter([audio_bytes]),
        media_type=content_type,
        headers=headers,
    )
