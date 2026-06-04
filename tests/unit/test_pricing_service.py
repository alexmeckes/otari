from contextlib import AbstractContextManager, nullcontext
from unittest.mock import patch

import pytest

from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import (
    candidate_pricing_model_refs,
    input_metered_cost,
    legacy_pricing_model_ref,
    log_missing_pricing,
    normalized_pricing_model_ref,
    pricing_model_ref,
    split_pricing_model_ref,
    token_usage_cost,
)


def _pricing(input_price: float = 2.0, output_price: float = 4.0) -> ModelPricing:
    return ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )


def test_pricing_model_ref_uses_provider_prefix() -> None:
    assert pricing_model_ref("openai", "gpt-4o") == "openai:gpt-4o"


def test_pricing_model_ref_uses_model_without_provider() -> None:
    assert pricing_model_ref(None, "gpt-4o") == "gpt-4o"


def test_legacy_pricing_model_ref_uses_slash_separator() -> None:
    assert legacy_pricing_model_ref("openai", "gpt-4o") == "openai/gpt-4o"


def test_split_pricing_model_ref_returns_provider_value_and_model() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert split_pricing_model_ref("openai/gpt-4o") == ("openai", "gpt-4o")


def test_normalized_pricing_model_ref_uses_colon_separator() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert normalized_pricing_model_ref("openai/gpt-4o") == "openai:gpt-4o"


@pytest.mark.parametrize(
    ("model_ref", "expected", "warning_context"),
    [
        ("openai:gpt-4o", ["openai:gpt-4o", "openai/gpt-4o"], nullcontext()),
        ("openai/gpt-4o", ["openai/gpt-4o", "openai:gpt-4o"], pytest.warns(DeprecationWarning, match="provider/model")),
        ("gpt-4o", ["gpt-4o"], nullcontext()),
    ],
)
def test_candidate_pricing_model_refs(
    model_ref: str,
    expected: list[str],
    warning_context: AbstractContextManager[object],
) -> None:
    with warning_context:
        candidates = candidate_pricing_model_refs(model_ref)

    assert candidates == expected


def test_log_missing_pricing_uses_standard_warning() -> None:
    with patch("gateway.services.pricing_service.logger.warning") as warning:
        log_missing_pricing("openai", "gpt-4o")

    warning.assert_called_once_with(
        "No pricing configured for '%s'. Usage will be tracked without cost.",
        "openai:gpt-4o",
    )


def test_input_metered_cost_defaults_to_price_per_million_units() -> None:
    assert input_metered_cost(_pricing(input_price=2.0), units=250_000) == 0.5


def test_input_metered_cost_supports_custom_divisor() -> None:
    assert input_metered_cost(_pricing(input_price=0.04), units=2, price_divisor=1) == 0.08


def test_token_usage_cost_combines_input_and_output_rates() -> None:
    assert token_usage_cost(_pricing(input_price=2.0, output_price=4.0), prompt_tokens=1000, completion_tokens=500) == (
        1000 / 1_000_000
    ) * 2.0 + (500 / 1_000_000) * 4.0
