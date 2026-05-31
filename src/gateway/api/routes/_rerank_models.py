from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Rerank request."""

    model: str = Field(description="Provider-prefixed model ID, e.g. 'cohere:rerank-v3.5'")
    query: str = Field(description="The search query to rerank documents against")
    documents: list[str] = Field(description="List of document strings to rerank", min_length=1)
    top_n: int | None = Field(default=None, description="Maximum number of results to return", gt=0)
    max_tokens_per_doc: int | None = Field(default=None, description="Per-document truncation limit", gt=0)
    user: str | None = Field(default=None, description="User ID for usage attribution")
