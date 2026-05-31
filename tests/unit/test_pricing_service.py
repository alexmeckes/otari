from unittest.mock import patch

from gateway.services.pricing_service import log_missing_pricing, pricing_model_ref


def test_pricing_model_ref_uses_provider_prefix() -> None:
    assert pricing_model_ref("openai", "gpt-4o") == "openai:gpt-4o"


def test_pricing_model_ref_uses_model_without_provider() -> None:
    assert pricing_model_ref(None, "gpt-4o") == "gpt-4o"


def test_log_missing_pricing_uses_standard_warning() -> None:
    with patch("gateway.services.pricing_service.logger.warning") as warning:
        log_missing_pricing("openai", "gpt-4o")

    warning.assert_called_once_with(
        "No pricing configured for '%s'. Usage will be tracked without cost.",
        "openai:gpt-4o",
    )
