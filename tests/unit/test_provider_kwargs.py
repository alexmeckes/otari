from typing import Any

import pytest
from any_llm import LLMProvider

from gateway.core.config import GatewayConfig
from gateway.services.provider_kwargs import get_provider_kwargs


def test_get_provider_kwargs_returns_empty_for_unconfigured_provider() -> None:
    config = GatewayConfig(
        database_url="sqlite:///test.db",
        master_key="test",
        providers={"openai": {"api_key": "sk-test"}},
    )

    assert get_provider_kwargs(config, LLMProvider.ANTHROPIC) == {}


def test_get_provider_kwargs_returns_copy_for_non_vertex_provider() -> None:
    provider_config: dict[str, Any] = {
        "api_key": "sk-test",
        "client_args": {"timeout": 30},
        "model": "provider-default-model",
    }
    config = GatewayConfig(
        database_url="sqlite:///test.db",
        master_key="test",
        providers={"openai": provider_config},
    )

    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI)

    assert kwargs == provider_config
    assert kwargs is not provider_config
    kwargs["api_key"] = "changed"
    assert provider_config["api_key"] == "sk-test"


def test_get_provider_kwargs_uses_vertex_setup_and_preserves_client_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_setup_vertex_environment(
        *,
        credentials: str | dict[str, Any] | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        calls.append(
            {
                "credentials": credentials,
                "project": project,
                "location": location,
            }
        )
        return {"project": "resolved-project", "location": "resolved-location"}

    monkeypatch.setattr(
        "gateway.services.provider_kwargs.setup_vertex_environment",
        fake_setup_vertex_environment,
    )
    config = GatewayConfig(
        database_url="sqlite:///test.db",
        master_key="test",
        providers={
            LLMProvider.VERTEXAI.value: {
                "credentials": {"project_id": "config-project"},
                "project": "configured-project",
                "location": "us-central1",
                "client_args": {"timeout": 30},
            }
        },
    )

    kwargs = get_provider_kwargs(config, LLMProvider.VERTEXAI)

    assert calls == [
        {
            "credentials": {"project_id": "config-project"},
            "project": "configured-project",
            "location": "us-central1",
        }
    ]
    assert kwargs == {
        "project": "resolved-project",
        "location": "resolved-location",
        "client_args": {"timeout": 30},
    }
