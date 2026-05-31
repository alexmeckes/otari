from dataclasses import dataclass

from gateway.api.routes._stream_events import format_typed_stream_event


@dataclass
class FakeStreamEvent:
    type: str
    payload: str = "{}"

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        assert exclude_none is True
        return self.payload


def test_format_typed_stream_event_formats_event_and_data_lines() -> None:
    assert format_typed_stream_event(FakeStreamEvent(type="message_delta", payload='{"type":"message_delta"}')) == (
        'event: message_delta\ndata: {"type":"message_delta"}\n\n'
    )
