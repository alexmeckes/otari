from gateway.core.config import GatewayConfig
from gateway.services.platform_config import platform_int_setting, platform_timeout_seconds, platform_url


def test_platform_url_joins_base_and_path() -> None:
    assert platform_url("https://platform.local/api/v1/", "/gateway/usage") == "https://platform.local/api/v1/gateway/usage"


def test_platform_int_setting_uses_default_when_missing() -> None:
    assert platform_int_setting(GatewayConfig(platform={}), "usage_max_retries", 3) == 3


def test_platform_int_setting_uses_configured_integer() -> None:
    assert platform_int_setting(GatewayConfig(platform={"usage_max_retries": 7}), "usage_max_retries", 3) == 7


def test_platform_int_setting_converts_string_values() -> None:
    assert platform_int_setting(GatewayConfig(platform={"usage_max_retries": "6"}), "usage_max_retries", 3) == 6


def test_platform_timeout_seconds_uses_default_when_missing() -> None:
    assert platform_timeout_seconds(GatewayConfig(platform={}), "usage_timeout_ms") == 5


def test_platform_timeout_seconds_uses_configured_milliseconds() -> None:
    assert platform_timeout_seconds(GatewayConfig(platform={"usage_timeout_ms": 2500}), "usage_timeout_ms") == 2.5


def test_platform_timeout_seconds_converts_string_values() -> None:
    assert platform_timeout_seconds(GatewayConfig(platform={"resolve_timeout_ms": "1500"}), "resolve_timeout_ms") == 1.5
