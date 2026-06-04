from types import SimpleNamespace

import pytest

from gateway.services.routing_constraints import (
    _candidate_regions,
    _constraint_failure,
    _constraint_rejection,
    _constraint_rejection_for_candidate,
    _estimated_cost_constraint_failure,
    _lower_string_set,
    _model_constraint_set,
    _normalize_model_key_for_constraint,
    _prepared_constraints,
    _PreparedConstraints,
    _provider_model_constraint_failure,
    _region_constraint_failure,
    _requested_region,
    _string_set,
    apply_constraints,
)


def test_normalize_model_key_for_constraint_returns_canonical_selector() -> None:
    assert _normalize_model_key_for_constraint("openai:gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_normalizes_legacy_slash_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert _normalize_model_key_for_constraint("openai/gpt-4o") == "openai:gpt-4o"


def test_normalize_model_key_for_constraint_preserves_invalid_selector() -> None:
    assert _normalize_model_key_for_constraint("gpt-4o") == "gpt-4o"


def test_model_constraint_set_normalizes_model_values() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert _model_constraint_set(
            ["openai/gpt-4o", " anthropic:claude-3-5-haiku-latest ", "gpt-4o", " "]
        ) == {
            "anthropic:claude-3-5-haiku-latest",
            "gpt-4o",
            "openai:gpt-4o",
        }
    assert _model_constraint_set(None) == set()


def test_string_set_normalizes_string_values() -> None:
    assert _string_set([" openai ", "anthropic", None, " "]) == {"None", "anthropic", "openai"}
    assert _string_set(" openai ") == {"openai"}
    assert _string_set(None) == set()


def test_lower_string_set_normalizes_string_values() -> None:
    assert _lower_string_set([" EU ", "us", None, " "]) == {"eu", "none", "us"}
    assert _lower_string_set(" Apac ") == {"apac"}
    assert _lower_string_set(None) == set()


def test_apply_constraints_normalizes_provider_model_and_region_constraint_values() -> None:
    allowed_candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={"region": "eu"},
    )

    with pytest.warns(DeprecationWarning, match="provider/model"):
        allowed, rejected = apply_constraints(
            [allowed_candidate],
            config={
                "constraints": {
                    "allowed_providers": [" openai "],
                    "allowed_models": ["openai/gpt-4o"],
                    "allowed_regions": [" EU "],
                }
            },
            tags={},
        )

    assert allowed == [allowed_candidate]
    assert rejected == []

    blocked_candidates = [
        SimpleNamespace(
            model="anthropic:claude-3-5-haiku-latest",
            provider="anthropic",
            estimated_cost=None,
            metadata={"region": "eu"},
        ),
        SimpleNamespace(
            model="openai:gpt-4o-mini",
            provider="openai",
            estimated_cost=None,
            metadata={"region": "eu"},
        ),
        SimpleNamespace(
            model="openai:gpt-4o",
            provider="openai",
            estimated_cost=None,
            metadata={"region": "us"},
        ),
    ]

    with pytest.warns(DeprecationWarning, match="provider/model"):
        allowed, rejected = apply_constraints(
            blocked_candidates,
            config={
                "constraints": {
                    "blocked_providers": [" anthropic "],
                    "blocked_models": ["openai/gpt-4o-mini"],
                    "blocked_regions": [" US "],
                }
            },
            tags={},
        )

    assert allowed == []
    assert [item["reason"] for item in rejected] == ["provider_blocked", "model_blocked", "region_blocked"]


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


def test_constraint_rejection_preserves_candidate_fields_and_sorted_regions() -> None:
    candidate = SimpleNamespace(model="openai:gpt-4o", provider="openai", estimated_cost=0.01)

    assert _constraint_rejection(candidate, reason="region_not_allowed", candidate_regions={"us", "eu"}) == {
        "model": "openai:gpt-4o",
        "provider": "openai",
        "reason": "region_not_allowed",
        "estimated_cost": 0.01,
        "regions": ["eu", "us"],
    }


def test_constraint_rejection_for_candidate_returns_payload_or_none() -> None:
    constraints = _PreparedConstraints(
        allowed_providers=set(),
        blocked_providers=set(),
        allowed_models=set(),
        blocked_models=set(),
        allowed_regions={"eu"},
        blocked_regions=set(),
        requested_region=None,
        max_estimated_cost=None,
        allow_unknown_cost=False,
    )
    allowed_candidate = SimpleNamespace(
        model="openai:gpt-4o",
        provider="openai",
        estimated_cost=None,
        metadata={"region": " EU "},
    )
    rejected_candidate = SimpleNamespace(
        model="openai:gpt-4o-mini",
        provider="openai",
        estimated_cost=0.02,
        metadata={"region": " US "},
    )

    assert _constraint_rejection_for_candidate(allowed_candidate, constraints=constraints) is None
    assert _constraint_rejection_for_candidate(rejected_candidate, constraints=constraints) == {
        "model": "openai:gpt-4o-mini",
        "provider": "openai",
        "reason": "region_not_allowed",
        "estimated_cost": 0.02,
        "regions": ["us"],
    }


def test_constraint_failure_preserves_provider_region_and_cost_order() -> None:
    candidate = SimpleNamespace(model="openai:gpt-4o", provider="openai", estimated_cost=0.02)

    def reason_for(
        *,
        allowed_providers: set[str] | None = None,
        allowed_regions: set[str] | None = None,
        max_estimated_cost: float | None = 0.01,
    ) -> str | None:
        constraints = _PreparedConstraints(
            allowed_providers=allowed_providers or set(),
            blocked_providers=set(),
            allowed_models=set(),
            blocked_models=set(),
            allowed_regions=allowed_regions or set(),
            blocked_regions=set(),
            requested_region=None,
            max_estimated_cost=max_estimated_cost,
            allow_unknown_cost=False,
        )
        return _constraint_failure(
            candidate,
            candidate_regions={"us"},
            constraints=constraints,
        )

    assert reason_for(allowed_providers={"anthropic"}, allowed_regions={"eu"}) == "provider_not_allowed"
    assert reason_for(allowed_regions={"eu"}) == "region_not_allowed"
    assert reason_for() == "estimated_cost_exceeds_max"
    assert reason_for(max_estimated_cost=None) is None


def test_candidate_regions_normalizes_list_and_single_region_metadata() -> None:
    assert _candidate_regions({"regions": [" EU ", "us", None], "region": " Apac "}) == {
        "apac",
        "eu",
        "none",
        "us",
    }
    assert _candidate_regions({"regions": " EU ", "region": " "}) == {"eu"}
    assert _candidate_regions({}) == set()


def test_provider_model_constraint_failure_preserves_rejection_order() -> None:
    assert (
        _provider_model_constraint_failure(
            provider="anthropic",
            model="anthropic:claude-3-5-haiku-latest",
            allowed_providers={"openai"},
            blocked_providers={"anthropic"},
            allowed_models=set(),
            blocked_models=set(),
        )
        == "provider_not_allowed"
    )
    assert (
        _provider_model_constraint_failure(
            provider="anthropic",
            model="anthropic:claude-3-5-haiku-latest",
            allowed_providers=set(),
            blocked_providers={"anthropic"},
            allowed_models={"openai:gpt-4o"},
            blocked_models=set(),
        )
        == "provider_blocked"
    )
    assert (
        _provider_model_constraint_failure(
            provider="anthropic",
            model="anthropic:claude-3-5-haiku-latest",
            allowed_providers=set(),
            blocked_providers=set(),
            allowed_models={"openai:gpt-4o"},
            blocked_models={"anthropic:claude-3-5-haiku-latest"},
        )
        == "model_not_allowed"
    )
    assert (
        _provider_model_constraint_failure(
            provider="anthropic",
            model="anthropic:claude-3-5-haiku-latest",
            allowed_providers=set(),
            blocked_providers=set(),
            allowed_models=set(),
            blocked_models={"anthropic:claude-3-5-haiku-latest"},
        )
        == "model_blocked"
    )
    assert (
        _provider_model_constraint_failure(
            provider="anthropic",
            model="anthropic:claude-3-5-haiku-latest",
            allowed_providers={"anthropic"},
            blocked_providers=set(),
            allowed_models={"anthropic:claude-3-5-haiku-latest"},
            blocked_models=set(),
        )
        is None
    )


def test_apply_constraints_preserves_provider_model_rejection_order() -> None:
    candidate = SimpleNamespace(provider="anthropic", model="anthropic:claude-3-5-haiku-latest")

    def reason_for(constraints: dict[str, list[str]]) -> str | None:
        allowed, rejected = apply_constraints(
            [SimpleNamespace(**vars(candidate), estimated_cost=None, metadata={})],
            config={"constraints": constraints},
            tags={},
        )
        if rejected:
            return rejected[0]["reason"]
        assert len(allowed) == 1
        return None

    assert reason_for({"allowed_providers": ["openai"], "blocked_models": [candidate.model]}) == "provider_not_allowed"
    assert reason_for({"blocked_providers": ["anthropic"]}) == "provider_blocked"
    assert reason_for({"allowed_models": ["openai:gpt-4o"]}) == "model_not_allowed"
    assert reason_for({"blocked_models": [candidate.model]}) == "model_blocked"
    assert reason_for({"allowed_providers": ["anthropic"], "allowed_models": [candidate.model]}) is None


def test_region_constraint_failure_preserves_rejection_order() -> None:
    assert (
        _region_constraint_failure(
            set(),
            allowed_regions={"eu"},
            blocked_regions=set(),
            requested_region=None,
        )
        == "region_unknown"
    )
    assert (
        _region_constraint_failure(
            {"us"},
            allowed_regions={"eu"},
            blocked_regions={"us"},
            requested_region=None,
        )
        == "region_not_allowed"
    )
    assert (
        _region_constraint_failure(
            {"us"},
            allowed_regions=set(),
            blocked_regions={"us"},
            requested_region="eu",
        )
        == "region_blocked"
    )
    assert (
        _region_constraint_failure(
            {"us"},
            allowed_regions=set(),
            blocked_regions=set(),
            requested_region="eu",
        )
        == "region_not_supported"
    )
    assert (
        _region_constraint_failure(
            {"eu"},
            allowed_regions={"eu"},
            blocked_regions={"us"},
            requested_region="eu",
        )
        is None
    )


def test_apply_constraints_preserves_region_rejection_order() -> None:
    def reason_for(
        metadata: dict[str, str],
        constraints: dict[str, list[str] | bool | str],
        tags: dict[str, str] | None = None,
    ) -> str | None:
        candidate = SimpleNamespace(
            model="openai:gpt-4o",
            provider="openai",
            estimated_cost=None,
            metadata=metadata,
        )
        allowed, rejected = apply_constraints([candidate], config={"constraints": constraints}, tags=tags or {})
        if rejected:
            return rejected[0]["reason"]
        assert allowed == [candidate]
        return None

    assert reason_for({}, {"allowed_regions": ["eu"]}) == "region_unknown"
    assert reason_for({"region": "us"}, {"allowed_regions": ["eu"], "blocked_regions": ["us"]}) == "region_not_allowed"
    assert reason_for({"region": "us"}, {"blocked_regions": ["us"]}) == "region_blocked"
    assert (
        reason_for(
            {"region": "us"},
            {"require_region_match": True, "region_tag": "request_region"},
            {"request_region": "eu"},
        )
        == "region_not_supported"
    )
    assert (
        reason_for(
            {"region": "eu"},
            {"allowed_regions": ["eu"], "require_region_match": True},
            {"region": "eu"},
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


def test_requested_region_respects_match_gate_tag_and_blank_inputs() -> None:
    assert _requested_region({"region_tag": "request_region"}, {"request_region": "EU"}) is None
    assert (
        _requested_region({"require_region_match": "true", "region_tag": "request_region"}, {"request_region": " EU "})
        == "eu"
    )
    assert _requested_region({"require_region_match": "true", "region_tag": " "}, {"region": "EU"}) is None
    assert (
        _requested_region({"require_region_match": "true", "region_tag": "request_region"}, {"request_region": " "})
        is None
    )


def test_prepared_constraints_normalizes_config_values() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        prepared = _prepared_constraints(
            {
                "allowed_providers": [" openai "],
                "blocked_providers": [" anthropic "],
                "allowed_models": ["openai/gpt-4o"],
                "blocked_models": ["openai:gpt-4o-mini"],
                "allowed_regions": [" EU "],
                "blocked_regions": [" US "],
                "require_region_match": "true",
                "region_tag": " request_region ",
                "max_estimated_cost": 0.01,
                "allow_unknown_cost": "true",
            },
            {"request_region": " EU "},
        )

    assert prepared == _PreparedConstraints(
        allowed_providers={"openai"},
        blocked_providers={"anthropic"},
        allowed_models={"openai:gpt-4o"},
        blocked_models={"openai:gpt-4o-mini"},
        allowed_regions={"eu"},
        blocked_regions={"us"},
        requested_region="eu",
        max_estimated_cost=0.01,
        allow_unknown_cost=True,
    )


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


def test_estimated_cost_constraint_failure_preserves_unknown_and_limit_behavior() -> None:
    assert (
        _estimated_cost_constraint_failure(
            None,
            max_estimated_cost=0.01,
            allow_unknown_cost=False,
        )
        == "estimated_cost_unknown"
    )
    assert (
        _estimated_cost_constraint_failure(
            None,
            max_estimated_cost=0.01,
            allow_unknown_cost=True,
        )
        is None
    )
    assert (
        _estimated_cost_constraint_failure(
            0.02,
            max_estimated_cost=0.01,
            allow_unknown_cost=True,
        )
        == "estimated_cost_exceeds_max"
    )
    assert (
        _estimated_cost_constraint_failure(
            0.01,
            max_estimated_cost=0.01,
            allow_unknown_cost=False,
        )
        is None
    )
    assert (
        _estimated_cost_constraint_failure(
            None,
            max_estimated_cost=None,
            allow_unknown_cost=False,
        )
        is None
    )


def test_apply_constraints_preserves_estimated_cost_behavior() -> None:
    candidates = [
        SimpleNamespace(model="openai:unknown-cost", provider="openai", estimated_cost=None, metadata={}),
        SimpleNamespace(model="openai:cheap", provider="openai", estimated_cost=0.001, metadata={}),
        SimpleNamespace(model="openai:expensive", provider="openai", estimated_cost=0.02, metadata={}),
    ]

    allowed, rejected = apply_constraints(
        candidates,
        config={"constraints": {"max_estimated_cost": 0.01}},
        tags={},
    )
    assert [candidate.model for candidate in allowed] == ["openai:cheap"]
    assert rejected == [
        {
            "model": "openai:unknown-cost",
            "provider": "openai",
            "reason": "estimated_cost_unknown",
            "estimated_cost": None,
            "regions": [],
        },
        {
            "model": "openai:expensive",
            "provider": "openai",
            "reason": "estimated_cost_exceeds_max",
            "estimated_cost": 0.02,
            "regions": [],
        },
    ]

    allowed, rejected = apply_constraints(
        candidates,
        config={"constraints": {"max_estimated_cost": 0.01, "allow_unknown_cost": True}},
        tags={},
    )
    assert [candidate.model for candidate in allowed] == ["openai:unknown-cost", "openai:cheap"]
    assert [item["reason"] for item in rejected] == ["estimated_cost_exceeds_max"]

    allowed, rejected = apply_constraints(
        candidates,
        config={"constraints": {"allow_unknown_cost": False}},
        tags={},
    )
    assert allowed == candidates
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
