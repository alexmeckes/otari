import pytest

from gateway.services.routing_candidate_specs import (
    configured_candidate_specs,
    infer_tier_from_output_price,
    split_model_selector,
)


def test_split_model_selector_returns_normalized_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert split_model_selector("openai/gpt-4o") == ("openai", "gpt-4o", "openai:gpt-4o")


@pytest.mark.parametrize(
    ("output_price", "expected_tier"),
    [
        (None, None),
        (0.05, "simple"),
        (1.49, "simple"),
        (1.50, "medium"),
        (2.50, "complex"),
        (5.00, "reasoning"),
    ],
)
def test_infer_tier_from_output_price_preserves_boundaries(
    output_price: float | None,
    expected_tier: str | None,
) -> None:
    assert infer_tier_from_output_price(output_price) == expected_tier


def test_configured_candidate_specs_dedupes_with_normalized_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        specs = configured_candidate_specs(
            {
                "candidates": [
                    {
                        "model": "openai/gpt-4o",
                        "metadata": {"source": "first"},
                        "quality_score": 0.9,
                    },
                    {
                        "model": "openai:gpt-4o",
                        "metadata": {"source": "duplicate"},
                        "quality_score": 0.1,
                    },
                ]
            }
        )

    assert len(specs) == 1
    assert specs[0].model == "openai:gpt-4o"
    assert specs[0].metadata == {"source": "first"}
    assert specs[0].quality_score == 0.9


def test_configured_candidate_specs_trims_models_skips_blanks_and_normalizes_tiers() -> None:
    specs = configured_candidate_specs(
        {
            "candidates": [
                " openai:gpt-4o-mini ",
                " ",
                {
                    "model": " anthropic:claude-3-5-haiku-latest ",
                    "tier": " Complex ",
                },
                {
                    "model": " ",
                    "tier": "reasoning",
                },
            ]
        }
    )

    assert [(spec.model, spec.tier) for spec in specs] == [
        ("openai:gpt-4o-mini", None),
        ("anthropic:claude-3-5-haiku-latest", "complex"),
    ]


def test_configured_candidate_specs_collects_tiered_candidates() -> None:
    specs = configured_candidate_specs(
        {
            "tiers": {
                " Simple ": [
                    " openai:gpt-4o-mini ",
                    " ",
                    {"model": "anthropic:claude-3-5-sonnet-latest", "tier": "Reasoning"},
                ],
                "unknown": ["openai:gpt-4o"],
                "complex": {"model": "anthropic:claude-3-opus-latest"},
            }
        }
    )

    assert [(spec.model, spec.tier) for spec in specs] == [
        ("openai:gpt-4o-mini", "simple"),
        ("anthropic:claude-3-5-sonnet-latest", "reasoning"),
    ]


def test_configured_candidate_specs_reads_quality_score_aliases_from_metadata() -> None:
    specs = configured_candidate_specs(
        {
            "candidates": [
                {
                    "model": "openai:gpt-4o-mini",
                    "metadata": {"benchmark_score": 0.72},
                }
            ]
        }
    )

    assert specs[0].quality_score == 0.72


def test_configured_candidate_specs_merges_top_level_region_metadata_defaults() -> None:
    metadata = {"region": "metadata-region", "source": "catalog"}

    specs = configured_candidate_specs(
        {
            "candidates": [
                {
                    "model": "openai:gpt-4o-mini",
                    "metadata": metadata,
                    "region": "top-level-region",
                    "regions": ["eu", "us"],
                }
            ]
        }
    )

    assert specs[0].metadata == {
        "region": "metadata-region",
        "regions": ["eu", "us"],
        "source": "catalog",
    }
    assert metadata == {"region": "metadata-region", "source": "catalog"}


def test_configured_candidate_specs_prefers_top_level_quality_score_aliases() -> None:
    specs = configured_candidate_specs(
        {
            "candidates": [
                {
                    "model": "openai:gpt-4o-mini",
                    "score": 0.0,
                    "metadata": {"quality_score": 0.95},
                }
            ]
        }
    )

    assert specs[0].quality_score == 0.0
