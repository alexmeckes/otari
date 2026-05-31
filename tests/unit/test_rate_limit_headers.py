from fastapi import Response

from gateway.api.routes._usage import apply_rate_limit_headers, optional_rate_limit_headers, rate_limit_headers
from gateway.rate_limit import RateLimitInfo


def test_rate_limit_headers_serializes_info() -> None:
    info = RateLimitInfo(limit=10, remaining=7, reset=123.9)

    assert rate_limit_headers(info) == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "7",
        "X-RateLimit-Reset": "123",
    }


def test_apply_rate_limit_headers_updates_response() -> None:
    response = Response()

    apply_rate_limit_headers(response, RateLimitInfo(limit=10, remaining=7, reset=123.9))

    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "7"
    assert response.headers["X-RateLimit-Reset"] == "123"


def test_optional_rate_limit_headers_serializes_info() -> None:
    info = RateLimitInfo(limit=10, remaining=7, reset=123.9)

    assert optional_rate_limit_headers(info) == rate_limit_headers(info)


def test_optional_rate_limit_headers_returns_empty_dict_without_info() -> None:
    assert optional_rate_limit_headers(None) == {}


def test_apply_rate_limit_headers_ignores_missing_info() -> None:
    response = Response()

    apply_rate_limit_headers(response, None)

    assert "X-RateLimit-Limit" not in response.headers
