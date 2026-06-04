from pydantic import BaseModel


class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible audio speech (TTS) request."""

    model: str
    input: str
    voice: str
    instructions: str | None = None
    response_format: str | None = None
    speed: float | None = None
    user: str | None = None
