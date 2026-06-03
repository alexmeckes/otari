from typing import Any

import httpx
import pytest

from gateway.services import routing_guardrail_external


async def _post_classifier_response(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    *,
    request_headers: dict[str, str] | None = None,
) -> routing_guardrail_external.ExternalClassifierPostResult:
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
            assert headers == request_headers
            return response

    monkeypatch.setattr(routing_guardrail_external.httpx, "AsyncClient", FakeAsyncClient)

    return await routing_guardrail_external.post_external_guardrail_classifier(
        url="https://classifier.example.test/check",
        request_text="hello",
        timeout_seconds=3.0,
        headers=request_headers,
    )


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


def test_classifier_violations_preserve_label_fallback_rules() -> None:
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

    violations, score = routing_guardrail_external._classifier_violations(
        prompt_shield,
        {"blocked": True, "label": " prompt_injection "},
    )
    assert violations == [{"type": "external_classifier", "rule": "prompt_injection"}]
    assert score is None

    violations, score = routing_guardrail_external._classifier_violations(
        dlp,
        {"flagged": True, "label": {"category": " customer_pii "}},
    )
    assert violations == [{"type": "external_classifier", "rule": "customer_pii"}]
    assert score is None

    violations, score = routing_guardrail_external._classifier_violations(dlp, {"flagged": True, "label": " "})
    assert violations == [{"type": "external_classifier", "rule": "dlp"}]
    assert score is None


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
                {"rule": " "},
            ],
            "flagged": True,
            "label": "fallback",
            "score": 0.91,
        },
    )

    assert violations == [
        {"type": "external_classifier", "rule": "customer_pii"},
        {"type": "external_classifier", "rule": "prompt_injection"},
        {"type": "external_classifier", "rule": "dlp"},
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


def test_classifier_violations_preserve_unflagged_score_rules() -> None:
    threshold_settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )
    no_threshold_settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    violations, score = routing_guardrail_external._classifier_violations(
        threshold_settings,
        {"score": 0.7, "label": " prompt_injection "},
    )
    assert violations == []
    assert score == 0.7

    violations, score = routing_guardrail_external._classifier_violations(
        threshold_settings,
        {"label": " prompt_injection "},
    )
    assert violations == []
    assert score is None

    violations, score = routing_guardrail_external._classifier_violations(
        no_threshold_settings,
        {"score": 0.9, "label": " prompt_injection "},
    )
    assert violations == []
    assert score == 0.9

    violations, score = routing_guardrail_external._classifier_violations(
        no_threshold_settings,
        {"violations": {"rule": "prompt_injection"}},
    )
    assert violations == []
    assert score is None


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
    assert routing_guardrail_external._classifier_settings({"headers": {" ": "skip"}}, index=1).headers is None
    assert routing_guardrail_external._classifier_settings({"headers": ["Authorization"]}, index=1).headers is None


def test_classifier_settings_uses_index_name_fallback() -> None:
    assert routing_guardrail_external._classifier_settings({"name": " ", "url": " "}, index=3).name == "classifier_3"


@pytest.mark.asyncio
async def test_evaluate_classifier_from_settings_preserves_skipped_error_and_success_paths() -> None:
    limit = routing_guardrail_external._CLASSIFIER_ERROR_TEXT_LIMIT
    skipped_settings = routing_guardrail_external._ClassifierSettings(
        name="classifier_1",
        url=None,
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )
    error_settings = routing_guardrail_external._ClassifierSettings(
        name="dlp",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=True,
    )
    fail_open_error_settings = routing_guardrail_external._ClassifierSettings(
        name="fail-open",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )
    success_settings = routing_guardrail_external._ClassifierSettings(
        name="prompt-shield",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=0.8,
        headers=None,
        fail_closed=False,
    )
    passed_success_settings = routing_guardrail_external._ClassifierSettings(
        name="classifier_1",
        url="https://classifier.example.test/check",
        timeout_seconds=2.0,
        threshold=None,
        headers=None,
        fail_closed=False,
    )

    async def unexpected_post_classifier(**kwargs: Any) -> routing_guardrail_external.ExternalClassifierPostResult:
        raise AssertionError(f"unexpected classifier post: {kwargs}")

    assert await routing_guardrail_external._evaluate_classifier_from_settings(
        skipped_settings,
        request_text="hello",
        post_classifier=unexpected_post_classifier,
    ) == (
        [],
        {
            "name": "classifier_1",
            "status": "skipped",
            "reason": routing_guardrail_external._CLASSIFIER_MISSING_URL_REASON,
        },
    )

    async def error_post_classifier(**_kwargs: Any) -> routing_guardrail_external.ExternalClassifierPostResult:
        return 503, None, "x" * (limit + 5)

    violations, result = await routing_guardrail_external._evaluate_classifier_from_settings(
        error_settings,
        request_text="hello",
        post_classifier=error_post_classifier,
    )
    assert violations == [{"type": "external_classifier_error", "rule": "dlp"}]
    assert result == {
        "name": "dlp",
        "status": "error",
        "status_code": 503,
        "error": "x" * limit,
        "fail_closed": True,
    }

    async def fail_open_error_post_classifier(
        **_kwargs: Any,
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
        return None, None, "timeout"

    violations, result = await routing_guardrail_external._evaluate_classifier_from_settings(
        fail_open_error_settings,
        request_text="hello",
        post_classifier=fail_open_error_post_classifier,
    )
    assert violations == []
    assert result == {
        "name": "fail-open",
        "status": "error",
        "status_code": None,
        "error": "timeout",
        "fail_closed": False,
    }

    async def success_post_classifier(**_kwargs: Any) -> routing_guardrail_external.ExternalClassifierPostResult:
        return 200, {"score": 0.9, "label": "prompt_injection"}, None

    violations, result = await routing_guardrail_external._evaluate_classifier_from_settings(
        success_settings,
        request_text="hello",
        post_classifier=success_post_classifier,
    )
    assert violations == [{"type": "external_classifier", "rule": "prompt_injection"}]
    assert result == {
        "name": "prompt-shield",
        "status": "flagged",
        "status_code": 200,
        "score": 0.9,
        "threshold": 0.8,
        "label": "prompt_injection",
        "violations": violations,
    }

    async def passed_success_post_classifier(
        **_kwargs: Any,
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
        return 204, {"label": {"rule": "ignored"}}, None

    violations, result = await routing_guardrail_external._evaluate_classifier_from_settings(
        passed_success_settings,
        request_text="hello",
        post_classifier=passed_success_post_classifier,
    )
    assert violations == []
    assert result == {
        "name": "classifier_1",
        "status": "passed",
        "status_code": 204,
        "score": None,
        "threshold": None,
        "label": None,
        "violations": [],
    }


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
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
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
async def test_external_classifier_configs_accept_single_dict_and_list_dicts() -> None:
    captured_urls: list[str] = []

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
        captured_urls.append(url)
        return 200, {"violations": []}, None

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={"external_classifiers": {"name": "single", "url": "https://classifier.example.test/single"}},
        request_text="hello",
        post_classifier=post_classifier,
    )
    assert captured_urls == ["https://classifier.example.test/single"]
    assert violations == []
    assert [result["name"] for result in results] == ["single"]

    captured_urls.clear()
    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {"name": "first", "url": "https://classifier.example.test/first"},
                "skip",
                None,
                {"name": "second", "url": "https://classifier.example.test/second"},
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )
    assert captured_urls == [
        "https://classifier.example.test/first",
        "https://classifier.example.test/second",
    ]
    assert violations == []
    assert [result["name"] for result in results] == ["first", "second"]

    async def unexpected_post_classifier(**kwargs: Any) -> routing_guardrail_external.ExternalClassifierPostResult:
        raise AssertionError(f"unexpected classifier post: {kwargs}")

    assert await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={"external_classifiers": "disabled"},
        request_text="hello",
        post_classifier=unexpected_post_classifier,
    ) == ([], [])
    assert await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={},
        request_text="hello",
        post_classifier=unexpected_post_classifier,
    ) == ([], [])


@pytest.mark.asyncio
async def test_external_classifiers_preserve_call_result_and_violation_order() -> None:
    captured_urls: list[str] = []
    payloads: dict[str, dict[str, Any]] = {
        "https://classifier.example.test/first": {
            "violations": [{"rule": "first_rule"}, {"rule": "first_extra"}],
        },
        "https://classifier.example.test/second": {
            "score": 0.9,
            "label": "second_label",
        },
        "https://classifier.example.test/third": {
            "violations": [],
        },
    }

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
        captured_urls.append(url)
        return 200, payloads[url], None

    violations, results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails={
            "external_classifiers": [
                {"name": "first", "url": "https://classifier.example.test/first"},
                {"name": "second", "url": "https://classifier.example.test/second", "threshold": 0.8},
                {"name": "third", "url": "https://classifier.example.test/third"},
            ]
        },
        request_text="hello",
        post_classifier=post_classifier,
    )

    assert captured_urls == [
        "https://classifier.example.test/first",
        "https://classifier.example.test/second",
        "https://classifier.example.test/third",
    ]
    assert violations == [
        {"type": "external_classifier", "rule": "first_rule"},
        {"type": "external_classifier", "rule": "first_extra"},
        {"type": "external_classifier", "rule": "second_label"},
    ]
    assert [result["name"] for result in results] == ["first", "second", "third"]
    assert [result["status"] for result in results] == ["flagged", "flagged", "passed"]


@pytest.mark.asyncio
async def test_external_classifier_headers_skip_blank_keys_without_trimming_kept_keys() -> None:
    captured: list[dict[str, Any]] = []

    async def post_classifier(
        *,
        url: str,
        request_text: str,
        timeout_seconds: float,
        headers: dict[str, str] | None,
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
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
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
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
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
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
    ) -> routing_guardrail_external.ExternalClassifierPostResult:
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
async def test_external_classifier_object_json_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    status_code, payload, error = await _post_classifier_response(
        monkeypatch,
        httpx.Response(200, json={"score": 0.5}),
    )

    assert status_code == 200
    assert payload == {"score": 0.5}
    assert error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "request_headers", "expected_payload", "expected_error"),
    [
        (
            httpx.Response(400, json={"detail": " blocked by classifier "}),
            {"Authorization": "Bearer test"},
            {"detail": " blocked by classifier "},
            "HTTP 400: blocked by classifier",
        ),
        (
            httpx.Response(503, json={"detail": " ", "error": " upstream down "}),
            None,
            {"detail": " ", "error": " upstream down "},
            "HTTP 503: upstream down",
        ),
        (
            httpx.Response(503, text="body fallback"),
            None,
            None,
            "HTTP 503: body fallback",
        ),
        (
            httpx.Response(302, json={"detail": " classifier moved "}),
            None,
            {"detail": " classifier moved "},
            "HTTP 302: classifier moved",
        ),
    ],
)
async def test_external_classifier_http_error_prefers_payload_detail_error_then_body(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    request_headers: dict[str, str] | None,
    expected_payload: dict[str, str] | None,
    expected_error: str,
) -> None:
    status_code, payload, error = await _post_classifier_response(
        monkeypatch,
        response,
        request_headers=request_headers,
    )

    assert status_code == response.status_code
    assert payload == expected_payload
    assert error == expected_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=["not-object"]),
        httpx.Response(200, content=b"not-json"),
    ],
)
async def test_external_classifier_non_object_json_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
) -> None:
    status_code, payload, error = await _post_classifier_response(
        monkeypatch,
        response,
    )

    assert status_code == 200
    assert payload is None
    assert error == routing_guardrail_external._CLASSIFIER_NON_OBJECT_JSON_ERROR
