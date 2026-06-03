from typing import Any

import httpx
import pytest

from gateway.services import routing_guardrail_external


def test_classifier_response_payload_accepts_object_json_only() -> None:
    assert routing_guardrail_external._classifier_response_payload(
        httpx.Response(200, json={"score": 0.5})
    ) == {"score": 0.5}
    assert routing_guardrail_external._classifier_response_payload(httpx.Response(200, json=["not-object"])) is None
    assert routing_guardrail_external._classifier_response_payload(httpx.Response(200, content=b"not json")) is None


def test_classifier_http_error_text_prefers_payload_detail() -> None:
    response = httpx.Response(429, text="body fallback")

    assert (
        routing_guardrail_external._classifier_http_error_text(response, {"detail": " blocked by classifier "})
        == "HTTP 429: blocked by classifier"
    )


def test_classifier_http_error_text_uses_payload_error_then_body() -> None:
    response = httpx.Response(503, text="body fallback")

    assert (
        routing_guardrail_external._classifier_http_error_text(response, {"detail": " ", "error": " upstream down "})
        == "HTTP 503: upstream down"
    )
    assert (
        routing_guardrail_external._classifier_http_error_text(response, {"detail": " "})
        == "HTTP 503: body fallback"
    )
    assert routing_guardrail_external._classifier_http_error_text(response, None) == "HTTP 503: body fallback"


def test_classifier_headers_skip_blank_keys_without_trimming_kept_keys() -> None:
    assert routing_guardrail_external._classifier_headers(
        {" Authorization ": "Bearer test", " ": "skip", 0: 42}
    ) == {
        " Authorization ": "Bearer test",
        "0": "42",
    }
    assert routing_guardrail_external._classifier_headers({" ": "skip"}) is None
    assert routing_guardrail_external._classifier_headers(["Authorization"]) is None


def test_classifier_configs_accept_single_dict_and_list_dicts() -> None:
    single = {"url": "https://classifier.example.test/check"}
    first = {"name": "first"}
    second = {"url": "https://classifier.example.test/other"}

    assert routing_guardrail_external._classifier_configs(single) == [single]
    assert routing_guardrail_external._classifier_configs([first, "skip", None, second]) == [first, second]
    assert routing_guardrail_external._classifier_configs("disabled") == []
    assert routing_guardrail_external._classifier_configs(None) == []


def test_classifier_flagged_preserves_boolean_and_threshold_rules() -> None:
    assert routing_guardrail_external._classifier_flagged({"blocked": True}, score=None, threshold=None)
    assert routing_guardrail_external._classifier_flagged({"flagged": True}, score=None, threshold=None)
    assert routing_guardrail_external._classifier_flagged({"score": 0.8}, score=0.8, threshold=0.8)
    assert not routing_guardrail_external._classifier_flagged({"score": 0.7}, score=0.7, threshold=0.8)
    assert not routing_guardrail_external._classifier_flagged({"score": 0.9}, score=None, threshold=0.8)
    assert not routing_guardrail_external._classifier_flagged({"score": 0.9}, score=0.9, threshold=None)


def test_classifier_rule_uses_configured_field_precedence() -> None:
    assert (
        routing_guardrail_external._classifier_rule(
            {
                "name": "classifier_name",
                "category": "category_name",
                "label": "label_name",
                "type": "type_name",
                "rule": " rule_name ",
            },
            fallback="fallback",
        )
        == "rule_name"
    )
    assert (
        routing_guardrail_external._classifier_rule(
            {
                "name": " classifier_name ",
                "category": " category_name ",
            },
            fallback="fallback",
        )
        == "category_name"
    )


def test_explicit_classifier_violations_preserve_order_and_fallbacks() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._explicit_classifier_violations(
        settings,
        {
            "violations": [
                {"category": " customer_pii "},
                " prompt_injection ",
                {"rule": " "},
            ]
        },
    ) == [
        {"type": "external_classifier", "rule": "customer_pii"},
        {"type": "external_classifier", "rule": "prompt_injection"},
        {"type": "external_classifier", "rule": "dlp"},
    ]


def test_explicit_classifier_violations_default_empty_for_unsupported_payloads() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._explicit_classifier_violations(settings, {"violations": {"rule": "pii"}}) == []
    assert routing_guardrail_external._explicit_classifier_violations(settings, {}) == []


def test_fallback_classifier_violation_preserves_label_fallback_rules() -> None:
    prompt_shield = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )
    dlp = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._fallback_classifier_violation(
        prompt_shield,
        {"label": " prompt_injection "},
    ) == {"type": "external_classifier", "rule": "prompt_injection"}
    assert routing_guardrail_external._fallback_classifier_violation(
        dlp,
        {"label": {"category": " customer_pii "}},
    ) == {"type": "external_classifier", "rule": "customer_pii"}
    assert routing_guardrail_external._fallback_classifier_violation(dlp, {"label": " "}) == {
        "type": "external_classifier",
        "rule": "dlp",
    }


def test_classifier_violations_preserve_explicit_rules_and_score() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )

    violations, score = routing_guardrail_external._classifier_violations(
        settings,
        {
            "violations": [
                {"category": " customer_pii "},
                " prompt_injection ",
            ],
            "flagged": True,
            "label": "fallback",
            "score": 0.91,
        },
    )

    assert violations == [
        {"type": "external_classifier", "rule": "customer_pii"},
        {"type": "external_classifier", "rule": "prompt_injection"},
    ]
    assert score == 0.91


def test_classifier_violations_use_threshold_label_fallback() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )

    violations, score = routing_guardrail_external._classifier_violations(
        settings,
        {"score": 0.82, "label": " prompt_injection "},
    )

    assert violations == [{"type": "external_classifier", "rule": "prompt_injection"}]
    assert score == 0.82


def test_classifier_violations_default_empty_when_not_flagged() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )

    violations, score = routing_guardrail_external._classifier_violations(
        settings,
        {"score": 0.7, "label": " prompt_injection "},
    )

    assert violations == []
    assert score == 0.7


def test_classifier_result_label_preserves_string_values_only() -> None:
    assert routing_guardrail_external._classifier_result_label({"label": " prompt_injection "}) == " prompt_injection "
    assert routing_guardrail_external._classifier_result_label({"label": {"rule": "ignored"}}) is None
    assert routing_guardrail_external._classifier_result_label({}) is None


def test_classifier_success_result_preserves_flagged_shape_and_string_label() -> None:
    violations = [{"type": "external_classifier", "rule": "prompt_injection"}]
    settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._classifier_success_result(
        settings,
        status_code=200,
        payload={"label": " prompt_injection "},
        score=0.82,
        violations=violations,
    ) == {
        "name": "prompt-shield",
        "status": "flagged",
        "status_code": 200,
        "score": 0.82,
        "threshold": 0.8,
        "label": " prompt_injection ",
        "violations": violations,
    }


def test_classifier_success_result_passed_omits_non_string_label() -> None:
    settings = routing_guardrail_external._ClassifierSettings(
        name="classifier_1",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._classifier_success_result(
        settings,
        status_code=204,
        payload={"label": {"rule": "ignored"}},
        score=None,
        violations=[],
    ) == {
        "name": "classifier_1",
        "status": "passed",
        "status_code": 204,
        "score": None,
        "threshold": None,
        "label": None,
        "violations": [],
    }


def test_classifier_error_text_truncates_long_errors_only() -> None:
    limit = routing_guardrail_external._CLASSIFIER_ERROR_TEXT_LIMIT

    assert routing_guardrail_external._classifier_error_text("x" * (limit + 5)) == "x" * limit
    assert routing_guardrail_external._classifier_error_text("short error") == "short error"


def test_classifier_error_result_truncates_error_and_preserves_fail_closed() -> None:
    limit = routing_guardrail_external._CLASSIFIER_ERROR_TEXT_LIMIT
    settings = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=True,
    )

    assert routing_guardrail_external._classifier_error_result(
        settings,
        status_code=503,
        error="x" * (limit + 5),
    ) == {
        "name": "dlp",
        "status": "error",
        "status_code": 503,
        "error": "x" * limit,
        "fail_closed": True,
    }


def test_classifier_error_violations_preserve_fail_closed_behavior() -> None:
    fail_closed = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=True,
    )
    fail_open = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    assert routing_guardrail_external._classifier_error_violations(fail_closed) == [
        {"type": "external_classifier_error", "rule": "dlp"}
    ]
    assert routing_guardrail_external._classifier_error_violations(fail_open) == []


def test_classifier_skipped_result_uses_missing_url_reason() -> None:
    assert routing_guardrail_external._classifier_skipped_result("classifier_1") == {
        "name": "classifier_1",
        "status": "skipped",
        "reason": routing_guardrail_external._CLASSIFIER_MISSING_URL_REASON,
    }


def test_classifier_settings_normalizes_request_fields() -> None:
    default_timeout = routing_guardrail_external._CLASSIFIER_DEFAULT_TIMEOUT_SECONDS

    assert routing_guardrail_external._classifier_settings(
        {
            "name": " dlp ",
            "url": " https://classifier.example.test/check ",
            "timeout_seconds": 0,
            "threshold": -1,
            "headers": {" Authorization ": "Bearer test", " ": "skip"},
            "fail_closed": True,
        },
        index=2,
    ) == routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=default_timeout,
        threshold=None,
        headers={" Authorization ": "Bearer test"},
        fail_closed=True,
    )


def test_classifier_settings_uses_index_name_fallback() -> None:
    assert routing_guardrail_external._classifier_settings({"name": " ", "url": " "}, index=3).name == "classifier_3"


@pytest.mark.asyncio
async def test_external_classifier_trims_config_strings_and_violation_rules() -> None:
    default_timeout = routing_guardrail_external._CLASSIFIER_DEFAULT_TIMEOUT_SECONDS
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
            "timeout_seconds": default_timeout,
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
    default_timeout = routing_guardrail_external._CLASSIFIER_DEFAULT_TIMEOUT_SECONDS
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
            "timeout_seconds": default_timeout,
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
    assert results == [
        {
            "name": "classifier_1",
            "status": "skipped",
            "reason": routing_guardrail_external._CLASSIFIER_MISSING_URL_REASON,
        }
    ]


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


@pytest.mark.asyncio
async def test_external_classifier_redirect_response_uses_http_error_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            assert url == "https://classifier.example.test/check"
            assert json == {"text": "hello"}
            assert headers is None
            return httpx.Response(302, json={"detail": " classifier moved "})

    monkeypatch.setattr(routing_guardrail_external.httpx, "AsyncClient", FakeAsyncClient)

    status_code, payload, error = await routing_guardrail_external.post_external_guardrail_classifier(
        url="https://classifier.example.test/check",
        request_text="hello",
        timeout_seconds=3.0,
        headers=None,
    )

    assert status_code == 302
    assert payload == {"detail": " classifier moved "}
    assert error == "HTTP 302: classifier moved"


@pytest.mark.asyncio
async def test_external_classifier_non_object_json_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
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
            assert headers is None
            return httpx.Response(200, text="not-json")

    monkeypatch.setattr(routing_guardrail_external.httpx, "AsyncClient", FakeAsyncClient)

    status_code, payload, error = await routing_guardrail_external.post_external_guardrail_classifier(
        url="https://classifier.example.test/check",
        request_text="hello",
        timeout_seconds=3.0,
        headers=None,
    )

    assert status_code == 200
    assert payload is None
    assert error == routing_guardrail_external._CLASSIFIER_NON_OBJECT_JSON_ERROR
