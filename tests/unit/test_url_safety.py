"""Unit tests for `gateway.services.url_safety`."""

import ipaddress

import pytest

from gateway.services.url_safety import UnsafeURLError, validate_mcp_url, validate_outbound_fetch_url


def test_public_https_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.services import url_safety

    monkeypatch.setattr(url_safety, "_resolve_all", lambda _host: [ipaddress.ip_address("93.184.216.34")])
    validate_mcp_url("https://example.com/mcp", has_authorization_token=True)


def test_public_http_accepted_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.services import url_safety

    monkeypatch.setattr(url_safety, "_resolve_all", lambda _host: [ipaddress.ip_address("93.184.216.34")])
    validate_mcp_url("http://example.com/mcp", has_authorization_token=False)


def test_public_http_rejected_with_token() -> None:
    with pytest.raises(UnsafeURLError, match="https"):
        validate_mcp_url("http://example.com/mcp", has_authorization_token=True)


def test_loopback_allowed_by_default() -> None:
    validate_mcp_url("http://127.0.0.1:9201/mcp", has_authorization_token=False)


@pytest.mark.parametrize("value", ["0", "false", "no"])
def test_loopback_false_like_env_values_disable_loopback(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_LOOPBACK", value)
    with pytest.raises(UnsafeURLError, match="loopback"):
        validate_mcp_url("http://127.0.0.1/mcp", has_authorization_token=False)


@pytest.mark.parametrize("value", ["", "off", "maybe", "1", "true", "yes"])
def test_loopback_unknown_or_true_like_env_values_keep_loopback_allowed(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_LOOPBACK", value)
    validate_mcp_url("http://127.0.0.1:9201/mcp", has_authorization_token=False)


def test_rfc1918_rejected() -> None:
    for ip in ("10.0.0.5", "172.16.5.5", "192.168.1.1"):
        with pytest.raises(UnsafeURLError, match="private"):
            validate_mcp_url(f"https://{ip}/mcp", has_authorization_token=False)


def test_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        validate_mcp_url("https://169.254.169.254/latest/", has_authorization_token=False)


def test_ipv6_link_local_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="link-local"):
        validate_mcp_url("https://[fe80::1]/mcp", has_authorization_token=False)


def test_non_http_scheme_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="http or https"):
        validate_mcp_url("ftp://example.com/mcp", has_authorization_token=False)


def test_no_host_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="hostname"):
        validate_mcp_url("https:///mcp", has_authorization_token=False)


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_private_override_true_like_env_values_allow_internal(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", value)
    validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_private_override_false_or_unknown_env_values_reject_internal(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", value)
    with pytest.raises(UnsafeURLError, match="private"):
        validate_mcp_url("https://10.0.0.5/mcp", has_authorization_token=False)


def test_unresolvable_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hostnames that fail to resolve are rejected (DNS-rebinding TOCTOU).

    A name that doesn't resolve at validation time could resolve to an internal
    address at fetch time. Operators that genuinely want this behaviour opt in
    via GATEWAY_MCP_ALLOW_PRIVATE_HOSTS.
    """
    from gateway.services import url_safety

    monkeypatch.setattr(url_safety, "_resolve_all", lambda _host: [])
    with pytest.raises(UnsafeURLError, match="could not be resolved"):
        validate_mcp_url("https://does-not-exist.invalid/mcp", has_authorization_token=False)


def test_unresolvable_host_allowed_with_private_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private-hosts opt-out also covers unresolvable hostnames."""
    from gateway.services import url_safety

    monkeypatch.setenv("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setattr(url_safety, "_resolve_all", lambda _host: [])
    validate_mcp_url("https://does-not-exist.invalid/mcp", has_authorization_token=False)


@pytest.mark.parametrize("value", ["1", "true", "yes"])
@pytest.mark.asyncio
async def test_web_search_private_override_true_like_env_values_allow_internal(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", value)
    await validate_outbound_fetch_url("http://10.0.0.5/page")


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
@pytest.mark.asyncio
async def test_web_search_private_override_false_or_unknown_env_values_reject_internal(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", value)
    with pytest.raises(UnsafeURLError, match="private"):
        await validate_outbound_fetch_url("http://10.0.0.5/page")
