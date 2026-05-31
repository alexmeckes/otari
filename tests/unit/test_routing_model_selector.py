import pytest
from pydantic import ValidationError

from gateway.api.routes._chat_request import ChatCompletionRequest
from gateway.api.routes._responses_transform import ResponsesRequest
from gateway.api.routes._routing_models import ResolveRoutingRequest
from gateway.services.routing_policy_service import DEFAULT_ROUTING_MODEL, require_routing_model_selector


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_ROUTING_MODEL),
        ("default_routing", DEFAULT_ROUTING_MODEL),
        (" DEFAULT_ROUTING ", DEFAULT_ROUTING_MODEL),
        (" openai/gpt-4o ", "openai/gpt-4o"),
        (123, "123"),
    ],
)
def test_require_routing_model_selector_normalizes_supported_values(value: object, expected: str) -> None:
    assert require_routing_model_selector(value) == expected


def test_require_routing_model_selector_rejects_blank_value() -> None:
    with pytest.raises(ValueError, match="model must not be blank"):
        require_routing_model_selector("   ")


def test_request_models_use_required_routing_model_selector() -> None:
    messages = [{"role": "user", "content": "hello"}]

    chat_request = ChatCompletionRequest(model=" DEFAULT_ROUTING ", messages=messages)
    responses_request = ResponsesRequest(model=None, input="hello")
    resolve_request = ResolveRoutingRequest(model=" openai/gpt-4o ", messages=messages)

    assert chat_request.model == DEFAULT_ROUTING_MODEL
    assert responses_request.model == DEFAULT_ROUTING_MODEL
    assert resolve_request.model == "openai/gpt-4o"


def test_request_models_reject_blank_model() -> None:
    with pytest.raises(ValidationError, match="model must not be blank"):
        ChatCompletionRequest(model=" ", messages=[{"role": "user", "content": "hello"}])
