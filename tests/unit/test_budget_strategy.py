import pytest

from gateway.services.budget_tags import normalize_budget_strategy


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("for_update", "for_update"),
        ("cas", "cas"),
        ("disabled", "disabled"),
        (" CAS ", "cas"),
        ("", "for_update"),
        ("unknown", "for_update"),
    ],
)
def test_normalize_budget_strategy(value: str, expected: str) -> None:
    assert normalize_budget_strategy(value) == expected
