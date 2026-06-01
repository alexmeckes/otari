from typing import Any

import httpx
import pytest

from gateway.services import routing_guardrail_external


@pytest.mark.asyncio
async def test_external_classifier_trims_config_strings_and_violation_rules() -> None:
    captured: list[dict[str, Any]] = []

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        captured.append(
            {
                "url": url,
                "request_text": request_text,
                "timeout_seconds": timeout_seconds,
                "headers": headers,
            }
        )
        return 200, {"violations": [{"rule": " customer_pii "}]}, None

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {
                    "name": " dlp ",
                    "url": " https://classifier.example.test/check ",
                }
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert captured == [
        {
            "url": "https://classifier.example.test/check",
            "request_text": "hello",
            "timeout_seconds": 2.0,
            "headers": None,
        }
    ]
    assert violations == [{"type": "external_classifier", "rule": "customer_pii"}]
    assert results[0]["name"] == "dlp"
    assert results[0]["status"] == "flagged"


@pytest.mark.asyncio
async def test_external_classifier_headers_skip_blank_keys_without_trimming_kept_keys() -> None:
    captured: list[dict[str, Any]] = []

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        captured.append({"headers": headers})
        return 200, {"violations": []}, None

    await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {
                    "url": "https://classifier.example.test/check",
                    "headers": {" Authorization ": "Bearer test", " ": "skip", 0: 42},
                }
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert captured == [{"headers": {" Authorization ": "Bearer test", "0": "42"}}]


@pytest.mark.asyncio
async def test_external_classifier_threshold_uses_shared_score_parsing() -> None:
    async def post_classifier(
        **_kwargs: Any,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        return 200, {"score": 0.82, "label": " prompt_injection "}, None

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {
                    "name": "prompt-shield",
                    "url": "https://classifier.example.test/check",
                    "threshold": 0.8,
                }
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert violations == [{"type": "external_classifier", "rule": "prompt_injection"}]
    assert results[0]["status"] == "flagged"
    assert results[0]["score"] == 0.82
    assert results[0]["threshold"] == 0.8


@pytest.mark.asyncio
async def test_external_classifier_float_config_preserves_timeout_and_threshold_fallbacks() -> None:
    captured: list[dict[str, Any]] = []

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        captured.append(
            {
                "url": url,
                "request_text": request_text,
                "timeout_seconds": timeout_seconds,
                "headers": headers,
            }
        )
        return 200, {"score": 0.9}, None

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {
                    "url": "https://classifier.example.test/check",
                    "timeout_seconds": 0,
                    "threshold": -1,
                }
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert captured == [
        {
            "url": "https://classifier.example.test/check",
            "request_text": "hello",
            "timeout_seconds": 2.0,
            "headers": None,
        }
    ]
    assert violations == []
    assert results[0]["score"] == 0.9
    assert results[0]["threshold"] is None
    assert results[0]["status"] == "passed"


@pytest.mark.asyncio
async def test_external_classifier_blank_name_and_url_fall_back_to_skipped() -> None:
    async def post_classifier(
        **kwargs: Any,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        raise AssertionError(f"unexpected classifier post: {kwargs}")

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={"external_classifiers": [{"name": " ", "url": " "}]},
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert violations == []
    assert results == [{"name": "classifier_1", "status": "skipped", "reason": "missing_url"}]


@pytest.mark.asyncio
async def test_external_classifier_http_error_uses_trimmed_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, str],
            headers: dict[str, str] | None,
        ) -> httpx.Response:
            assert self.timeout == 3.0
            assert url == "https://classifier.example.test/check"
            assert json == {"text": "hello"}
            assert headers == {"Authorization": "Bearer test"}
            return httpx.Response(400, json={"detail": " blocked by classifier "})

    monkeypatch.setattr(routing_guardrail_external.httpx, "AsyncClient", FakeAsyncClient)

    status_code, payload, error = await routing_guardrail_external.post_external_guardrail_classifier(
        url="https://classifier.example.test/check",
        request_text="hello",
        timeout_seconds=3.0,
        headers={"Authorization": "Bearer test"},
    )

    assert status_code == 400
    assert payload == {"detail": " blocked by classifier "}
    assert error == "HTTP 400: blocked by classifier"
