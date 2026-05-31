from gateway.api.routes._provider_context import OpenAIProviderRequestContext
from gateway.services.log_writer import LogWriter


async def log_audio_usage(
    *,
    log_writer: LogWriter,
    context: OpenAIProviderRequestContext,
    endpoint: str,
    error: str | None = None,
) -> None:
    usage_log = context.usage_log(
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
        prompt_tokens=0 if error is None else None,
        completion_tokens=0 if error is None else None,
        total_tokens=0 if error is None else None,
    )
    await log_writer.put(usage_log)
