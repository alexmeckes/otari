from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    response_format: str | None = None
    user: str | None = None
