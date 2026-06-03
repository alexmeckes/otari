"""Shared helper for building provider kwargs from gateway configuration."""

from typing import Any

from any_llm import LLMProvider

from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import GatewayConfig


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider kwargs from config for acompletion calls.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.)

    """
    provider_config = config.providers.get(provider.value)
    if provider_config is None:
        return {}

    if provider == LLMProvider.VERTEXAI:
        kwargs = setup_vertex_environment(
            credentials=provider_config.get("credentials"),
            project=provider_config.get("project"),
            location=provider_config.get("location"),
        )
        if "client_args" in provider_config:
            kwargs["client_args"] = provider_config["client_args"]
        return kwargs

    return dict(provider_config)
