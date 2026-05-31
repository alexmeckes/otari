import pytest

from gateway.services.routing_candidate_specs import configured_candidate_specs, split_model_selector


def test_split_model_selector_returns_normalized_selector() -> None:
    with pytest.warns(DeprecationWarning, match="provider/model"):
        assert split_model_selector("openai/gpt-4o") == ("openai", "gpt-4o", "openai:gpt-4o")


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
