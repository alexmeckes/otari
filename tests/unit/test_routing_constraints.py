from types import SimpleNamespace

import pytest

from gateway.services.routing_constraints import (
    _candidate_regions,
    _normalize_model_key_for_constraint,
    _request_region,
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
