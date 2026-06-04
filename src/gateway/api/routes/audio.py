"""OpenAI-compatible audio transcription and speech endpoints."""

from typing import Annotated, Any

from any_llm import aspeech, atranscription
from any_llm.types.audio import Transcription
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._audio_models import AudioSpeechRequest
from gateway.api.routes._provider_context import resolve_openai_provider_request_context
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter

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
    context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=user,
        db=db,
        config=config,
        model=model,
    )

    file_bytes = await file.read()
    if len(file_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    transcription_kwargs = context.call_kwargs(
        file=file_bytes,
        optional={
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
        },
    )

    try:
        result: Transcription = await atranscription(**transcription_kwargs)
        await context.log_zero_token_usage(log_writer, endpoint="/v1/audio/transcriptions")

    except HTTPException:
        raise
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer=log_writer,
            endpoint="/v1/audio/transcriptions",
            error=e,
        )

    context.apply_rate_limit_headers(response)

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
    context = await resolve_openai_provider_request_context(
        raw_request=raw_request,
        auth_result=auth_result,
        user=request.user,
        db=db,
        config=config,
        model=request.model,
    )

    speech_kwargs = context.call_kwargs(
        input=request.input,
        voice=request.voice,
        optional={
            "instructions": request.instructions,
            "response_format": request.response_format,
            "speed": request.speed,
        },
    )

    try:
        audio_bytes: bytes = await aspeech(**speech_kwargs)
        await context.log_zero_token_usage(log_writer, endpoint="/v1/audio/speech")

    except HTTPException:
        raise
    except Exception as e:
        await context.log_and_raise_provider_error(
            log_writer=log_writer,
            endpoint="/v1/audio/speech",
            error=e,
        )

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    headers = context.rate_limit_headers()

    return StreamingResponse(
        content=iter([audio_bytes]),
        media_type=content_type,
        headers=headers,
    )
