"""Shared helpers for platform integration configuration."""

from gateway.core.config import GatewayConfig


def platform_base_url(config: GatewayConfig) -> str | None:
    value = config.platform.get("base_url")
    return str(value) if value else None


def platform_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def platform_int_setting(config: GatewayConfig, key: str, default: int) -> int:
    return int(config.platform.get(key, default))


def platform_timeout_seconds(config: GatewayConfig, key: str, default_ms: int = 5000) -> float:
    return platform_int_setting(config, key, default_ms) / 1000
