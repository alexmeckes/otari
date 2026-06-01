from types import SimpleNamespace

import pytest

from gateway.services.routing_constraints import (
    _candidate_regions,
    _normalize_model_key_for_constraint,
    _region_presence_failure,
    _request_region,
    apply_constraints,
)


def test_normalize_model_key_for_constraint_returns_canonical_selector() -> None:
    assert _normalize_model_key_for_constraint("openai:gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_normalizes_legacy_slash_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert _normalize_model_key_for_constraint("openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_preserves_invalid_selector() -> None:
    assert _normalize_model_key_for_constraint("gpt-4o") == "gpt-4o"


def test_candidate_regions_normalize_region_metadata() -> None:
    candidate = SimpleNamespace(metadata={"regions": [" EU ", "us"], "region": " Apac "})

    assert _candidate_regions(candidate) == {"eu", "us", "apac"}


def test_request_region_uses_trimmed_region_tag_and_value() -> None:
    assert _request_region({"region_tag": " request_region "}, {"request_region": " EU "}) == "eu"


def test_request_region_ignores_blank_region_tag_and_value() -> None:
    assert _request_region({"region_tag": " "}, {"region": "eu"}) is None
    assert _request_region({"region_tag": "region"}, {"region": " "}) is None


def test_region_presence_failure_reuses_unknown_and_mismatch_reasons() -> None:
    assert _region_presence_failure(set(), matches=False, mismatch_reason="region_not_allowed") == "region_unknown"
    assert (
        _region_presence_failure({"eu"}, matches=False, mismatch_reason="region_not_supported")
        == "region_not_supported"
    )
    assert _region_presence_failure({"eu"}, matches=True, mismatch_reason="region_not_allowed") is None


def test_apply_constraints_uses_configured_constraint_values() -> None:
    allowed, rejected = apply_constraints(
        [
            SimpleNamespace(
                model="openai:gpt-4o",
                provider="openai",
                estimated_cost=None,
                metadata={"region": "eu"},
            ),
            SimpleNamespace(
                model="anthropic:claude-3-5-haiku-latest",
                provider="anthropic",
                estimated_cost=0.001,
                metadata={"region": "eu"},
            ),
            SimpleNamespace(
                model="openai:gpt-4o-mini",
                provider="openai",
                estimated_cost=0.001,
                metadata={"region": "eu"},
            ),
            SimpleNamespace(
                model="openai:gpt-4o-large",
                provider="openai",
                estimated_cost=0.02,
                metadata={"regions": ["eu"]},
            ),
        ],
        config={
            "constraints": {
                "allowed_providers": ["openai"],
                "blocked_models": ["openai:gpt-4o-mini"],
                "region_tag": "request_region",
                "require_region_match": "true",
                "max_estimated_cost": 0.01,
                "allow_unknown_cost": "true",
            }
        },
        tags={"request_region": " EU "},
    )

    assert [candidate.model for candidate in allowed] == ["openai:gpt-4o"]
    assert [(item["model"], item["reason"]) for item in rejected] == [
        ("anthropic:claude-3-5-haiku-latest", "provider_not_allowed"),
        ("openai:gpt-4o-mini", "model_blocked"),
        ("openai:gpt-4o-large", "estimated_cost_exceeds_max"),
    ]
