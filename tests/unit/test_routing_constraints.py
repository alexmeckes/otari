from types import SimpleNamespace

import pytest

from gateway.services.routing_constraints import (
    _candidate_regions,
    _constraint_failure,
    _constraint_sets,
    _estimated_cost_failure,
    _membership_failure,
    _normalize_model_key_for_constraint,
    _provider_model_failure,
    _region_failure,
    _region_presence_failure,
    _rejected_candidate_payload,
    _request_region,
    _required_request_region,
    apply_constraints,
)


def test_normalize_model_key_for_constraint_returns_canonical_selector() -> None:
    assert _normalize_model_key_for_constraint("openai:gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_normalizes_legacy_slash_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert _normalize_model_key_for_constraint("openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_preserves_invalid_selector() -> None:
    assert _normalize_model_key_for_constraint("gpt-4o") == "gpt-4o"


def test_constraint_sets_normalize_provider_model_and_region_values() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        constraint_sets = _constraint_sets(
            {
                "allowed_providers": [" openai "],
                "blocked_providers": ["anthropic"],
                "allowed_models": ["openai/gpt-4o"],
                "blocked_models": ["openai:gpt-4o-mini"],
                "allowed_regions": [" EU "],
                "blocked_regions": ["us"],
            }
        )

    assert constraint_sets.allowed_providers == {"openai"}
    assert constraint_sets.blocked_providers == {"anthropic"}
    assert constraint_sets.allowed_models == {"openai:gpt-4o"}
    assert constraint_sets.blocked_models == {"openai:gpt-4o-mini"}
    assert constraint_sets.allowed_regions == {"eu"}
    assert constraint_sets.blocked_regions == {"us"}


def test_candidate_regions_normalize_region_metadata() -> None:
    candidate = SimpleNamespace(metadata={"regions": [" EU ", "us"], "region": " Apac "})

    assert _candidate_regions(candidate) == {"eu", "us", "apac"}


def test_request_region_uses_trimmed_region_tag_and_value() -> None:
    assert _request_region({"region_tag": " request_region "}, {"request_region": " EU "}) == "eu"


def test_request_region_ignores_blank_region_tag_and_value() -> None:
    assert _request_region({"region_tag": " "}, {"region": "eu"}) is None
    assert _request_region({"region_tag": "region"}, {"region": " "}) is None


def test_required_request_region_only_returns_region_when_match_is_required() -> None:
    assert _required_request_region({"region_tag": "request_region"}, {"request_region": "EU"}) is None
    assert (
        _required_request_region(
            {"require_region_match": "true", "region_tag": "request_region"},
            {"request_region": "EU"},
        )
        == "eu"
    )


def test_region_presence_failure_reuses_unknown_and_mismatch_reasons() -> None:
    assert _region_presence_failure(set(), matches=False, mismatch_reason="region_not_allowed") == "region_unknown"
    assert (
        _region_presence_failure({"eu"}, matches=False, mismatch_reason="region_not_supported")
        == "region_not_supported"
    )
    assert _region_presence_failure({"eu"}, matches=True, mismatch_reason="region_not_allowed") is None


def test_membership_failure_reuses_allowed_and_blocked_reasons() -> None:
    assert (
        _membership_failure(
            "anthropic",
            allowed_values={"openai"},
            blocked_values=set(),
            not_allowed_reason="provider_not_allowed",
            blocked_reason="provider_blocked",
        )
        == "provider_not_allowed"
    )
    assert (
        _membership_failure(
            "openai:gpt-4o-mini",
            allowed_values=set(),
            blocked_values={"openai:gpt-4o-mini"},
            not_allowed_reason="model_not_allowed",
            blocked_reason="model_blocked",
        )
        == "model_blocked"
    )
    assert (
        _membership_failure(
            "openai",
            allowed_values={"openai"},
            blocked_values=set(),
            not_allowed_reason="provider_not_allowed",
            blocked_reason="provider_blocked",
        )
        is None
    )


def test_provider_model_failure_preserves_provider_before_model_order() -> None:
    candidate = SimpleNamespace(provider="anthropic", model="anthropic:claude-3-5-haiku-latest")

    assert (
        _provider_model_failure(
            candidate,
            _constraint_sets({"allowed_providers": ["openai"], "blocked_models": [candidate.model]}),
        )
        == "provider_not_allowed"
    )
    assert (
        _provider_model_failure(candidate, _constraint_sets({"blocked_providers": ["anthropic"]}))
        == "provider_blocked"
    )
    assert (
        _provider_model_failure(candidate, _constraint_sets({"allowed_models": ["openai:gpt-4o"]}))
        == "model_not_allowed"
    )
    assert (
        _provider_model_failure(candidate, _constraint_sets({"blocked_models": [candidate.model]}))
        == "model_blocked"
    )
    assert (
        _provider_model_failure(
            candidate,
            _constraint_sets({"allowed_providers": ["anthropic"], "allowed_models": [candidate.model]}),
        )
        is None
    )


def test_estimated_cost_failure_preserves_absent_unknown_and_exceeded_behavior() -> None:
    unknown_cost = SimpleNamespace(estimated_cost=None)
    expensive = SimpleNamespace(estimated_cost=0.02)
    cheap = SimpleNamespace(estimated_cost=0.001)

    assert _estimated_cost_failure(expensive, {}) is None
    assert _estimated_cost_failure(unknown_cost, {"max_estimated_cost": 0.01}) == "estimated_cost_unknown"
    assert _estimated_cost_failure(unknown_cost, {"max_estimated_cost": 0.01, "allow_unknown_cost": "true"}) is None
    assert _estimated_cost_failure(expensive, {"max_estimated_cost": 0.01}) == "estimated_cost_exceeds_max"
    assert _estimated_cost_failure(cheap, {"max_estimated_cost": 0.01}) is None


def test_region_failure_preserves_allowed_blocked_and_request_region_order() -> None:
    assert _region_failure(set(), _constraint_sets({"allowed_regions": ["eu"]}), None) == "region_unknown"
    assert (
        _region_failure({"us"}, _constraint_sets({"allowed_regions": ["eu"]}), None)
        == "region_not_allowed"
    )
    assert (
        _region_failure({"us"}, _constraint_sets({"blocked_regions": ["us"]}), None)
        == "region_blocked"
    )
    assert (
        _region_failure(
            {"us"},
            _constraint_sets({}),
            "eu",
        )
        == "region_not_supported"
    )
    assert (
        _region_failure(
            {"eu"},
            _constraint_sets({"allowed_regions": ["eu"]}),
            "eu",
        )
        is None
    )


def test_rejected_candidate_payload_preserves_public_shape_and_sorted_regions() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=0.02,
        metadata={"regions": ["us", "eu"]},
    )

    assert _rejected_candidate_payload(candidate, "estimated_cost_exceeds_max") == {
        "model": "openai:gpt-4o",
        "provider": "openai",
        "reason": "estimated_cost_exceeds_max",
        "estimated_cost": 0.02,
        "regions": ["eu", "us"],
    }


def test_constraint_failure_reuses_precomputed_constraint_sets() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=0.02,
        metadata={"region": "eu"},
    )
    constraints = {
        "allowed_providers": ["openai"],
        "allowed_models": ["openai:gpt-4o"],
        "allowed_regions": ["eu"],
        "max_estimated_cost": 0.01,
    }

    assert (
        _constraint_failure(candidate, constraints, _constraint_sets(constraints), None)
        == "estimated_cost_exceeds_max"
    )


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
