from types import SimpleNamespace

import pytest

from gateway.services.routing_constraints import (
    _constraint_sets,
    _estimated_cost_failure,
    _normalize_model_key_for_constraint,
    _provider_model_failure,
    _region_failure,
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


def test_apply_constraints_normalizes_region_metadata_in_rejections() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={"regions": [" EU ", "us"], "region": " Apac "},
    )

    allowed, rejected = apply_constraints(
        [candidate],
        config={"constraints": {"allowed_regions": ["latam"]}},
        tags={},
    )

    assert allowed == []
    assert rejected == [
        {
            "model": "openai:gpt-4o",
            "provider": "openai",
            "reason": "region_not_allowed",
            "estimated_cost": None,
            "regions": ["apac", "eu", "us"],
        }
    ]


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

    assert (
        _estimated_cost_failure(expensive, max_estimated_cost=None, allow_unknown_cost=False)
        is None
    )
    assert (
        _estimated_cost_failure(
            unknown_cost,
            max_estimated_cost=0.01,
            allow_unknown_cost=False,
        )
        == "estimated_cost_unknown"
    )
    assert (
        _estimated_cost_failure(
            unknown_cost,
            max_estimated_cost=0.01,
            allow_unknown_cost=True,
        )
        is None
    )
    assert (
        _estimated_cost_failure(
            expensive,
            max_estimated_cost=0.01,
            allow_unknown_cost=False,
        )
        == "estimated_cost_exceeds_max"
    )
    assert (
        _estimated_cost_failure(
            cheap,
            max_estimated_cost=0.01,
            allow_unknown_cost=False,
        )
        is None
    )


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


def test_apply_constraints_ignores_missing_or_non_dict_constraints() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={},
    )

    for config in ({}, {"constraints": "disabled"}):
        allowed, rejected = apply_constraints([candidate], config=config, tags={})

        assert allowed == [candidate]
        assert rejected == []


def test_apply_constraints_normalizes_legacy_slash_model_constraints() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={},
    )

    with pytest.warns(DeprecationWarning, match="provider/model"):
        allowed, rejected = apply_constraints(
            [candidate],
            config={"constraints": {"blocked_models": ["openai/gpt-4o"]}},
            tags={},
        )

    assert allowed == []
    assert rejected == [
        {
            "model": "openai:gpt-4o",
            "provider": "openai",
            "reason": "model_blocked",
            "estimated_cost": None,
            "regions": [],
        }
    ]


def test_apply_constraints_ignores_request_region_unless_match_is_required() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={"region": "us"},
    )

    allowed, rejected = apply_constraints(
        [candidate],
        config={"constraints": {"region_tag": "request_region"}},
        tags={"request_region": "EU"},
    )

    assert allowed == [candidate]
    assert rejected == []


def test_apply_constraints_ignores_blank_request_region_inputs() -> None:
    candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={"region": "us"},
    )

    for constraints, tags in (
        ({"require_region_match": "true", "region_tag": " "}, {"region": "EU"}),
        ({"require_region_match": "true", "region_tag": "request_region"}, {"request_region": " "}),
    ):
        allowed, rejected = apply_constraints([candidate], config={"constraints": constraints}, tags=tags)

        assert allowed == [candidate]
        assert rejected == []


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
                model="openai:gpt-4o-regional",
                provider="openai",
                estimated_cost=0.001,
                metadata={"region": "us"},
            ),
            SimpleNamespace(
                model="openai:gpt-4o-large",
                provider="openai",
                estimated_cost=0.02,
                metadata={"regions": ["us", "eu"]},
            ),
        ],
        config={
            "constraints": {
                "allowed_providers": ["openai"],
                "blocked_models": ["openai:gpt-4o-mini"],
                "region_tag": " request_region ",
                "require_region_match": "true",
                "max_estimated_cost": 0.01,
                "allow_unknown_cost": "true",
            }
        },
        tags={"request_region": " EU "},
    )

    assert [candidate.model for candidate in allowed] == ["openai:gpt-4o"]
    assert rejected == [
        {
            "model": "anthropic:claude-3-5-haiku-latest",
            "provider": "anthropic",
            "reason": "provider_not_allowed",
            "estimated_cost": 0.001,
            "regions": ["eu"],
        },
        {
            "model": "openai:gpt-4o-mini",
            "provider": "openai",
            "reason": "model_blocked",
            "estimated_cost": 0.001,
            "regions": ["eu"],
        },
        {
            "model": "openai:gpt-4o-regional",
            "provider": "openai",
            "reason": "region_not_supported",
            "estimated_cost": 0.001,
            "regions": ["us"],
        },
        {
            "model": "openai:gpt-4o-large",
            "provider": "openai",
            "reason": "estimated_cost_exceeds_max",
            "estimated_cost": 0.02,
            "regions": ["eu", "us"],
        },
    ]
