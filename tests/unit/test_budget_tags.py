from gateway.models.entities import Budget
from gateway.services.budget_tags import TAG_BUDGET_SCOPE, budget_matches_tags


def _tag_budget(match_tags: dict[str, str] | None = None) -> Budget:
    return Budget(
        scope_type=TAG_BUDGET_SCOPE,
        is_active=True,
        match_tags={"team": "platform"} if match_tags is None else match_tags,
    )


def test_budget_matches_tags_uses_request_tags() -> None:
    assert budget_matches_tags(_tag_budget(), {"team": "platform", "region": "us"})


def test_budget_matches_tags_rejects_missing_or_invalid_request_tags() -> None:
    budget = _tag_budget()

    assert not budget_matches_tags(budget, None)
    assert not budget_matches_tags(budget, "not-a-dict")  # type: ignore[arg-type]


def test_budget_matches_tags_rejects_inactive_or_empty_tag_budgets() -> None:
    inactive = _tag_budget()
    inactive.is_active = False

    assert not budget_matches_tags(inactive, {"team": "platform"})
    assert not budget_matches_tags(_tag_budget(match_tags={}), {"team": "platform"})
