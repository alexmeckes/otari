from gateway.services.routing_policy_eval_scores import EvalScoreInput, apply_eval_scores_to_policy_config


def test_apply_eval_scores_reports_previous_quality_score_alias() -> None:
    result = apply_eval_scores_to_policy_config(
        {
            "candidates": [
                {
                    "model": "openai:gpt-4o-mini",
                    "score": 0.0,
                    "metadata": {"quality_score": 0.9},
                }
            ]
        },
        [EvalScoreInput(model="openai:gpt-4o-mini", quality_score=0.7)],
        updated_at="2026-06-01T00:00:00Z",
    )

    assert result.applied_scores[0].previous_quality_score == 0.0
    assert result.applied_scores[0].quality_score == 0.7
    assert result.config["candidates"][0]["quality_score"] == 0.7


def test_apply_eval_scores_ignores_blank_dict_candidate_model() -> None:
    blank_candidate = {"model": "   ", "quality_score": 0.1}

    result = apply_eval_scores_to_policy_config(
        {
            "candidates": [
                blank_candidate,
                "openai:gpt-4o-mini",
            ]
        },
        [
            EvalScoreInput(model="openai:gpt-4o-mini", quality_score=0.7),
            EvalScoreInput(model="anthropic:claude-3-haiku", quality_score=0.8),
        ],
        updated_at="2026-06-01T00:00:00Z",
    )

    assert result.config["candidates"][0] == blank_candidate
    assert result.config["candidates"][1]["quality_score"] == 0.7
    assert [score.model for score in result.applied_scores] == ["openai:gpt-4o-mini"]
    assert result.unmatched_models == ["anthropic:claude-3-haiku"]


def test_apply_eval_scores_ignores_unsupported_candidate_items() -> None:
    result = apply_eval_scores_to_policy_config(
        {
            "candidates": [
                123,
            ]
        },
        [EvalScoreInput(model="openai:gpt-4o-mini", quality_score=0.7)],
        updated_at="2026-06-01T00:00:00Z",
    )

    assert result.config["candidates"] == [123]
    assert result.applied_scores == []
    assert result.unmatched_models == ["openai:gpt-4o-mini"]
