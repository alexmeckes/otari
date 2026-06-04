from any_llm.types.completion import ChatCompletion

from gateway.api.routes._responses_transform import chat_completion_to_response_payload


def test_chat_completion_to_response_payload_handles_empty_choices() -> None:
    completion = ChatCompletion(
        id="chatcmpl-empty",
        object="chat.completion",
        created=1_700_000_000,
        model="openai:gpt-4o-mini",
        choices=[],
        usage=None,
    )

    payload = chat_completion_to_response_payload(completion)

    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["vendor"] == "openai"
    assert payload["output_text"] == ""
    assert payload["output"][0]["content"][0]["text"] == ""
