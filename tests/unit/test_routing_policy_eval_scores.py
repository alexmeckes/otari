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
