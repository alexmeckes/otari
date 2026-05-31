import pytest

from gateway.services.routing_config_values import (
    float_or_none,
    non_negative_float_or_none,
    score_or_none,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (1.25, 1.25),
        (0, 0.0),
        (True, None),
        (False, None),
        ("1.25", None),
        (None, None),
    ],
)
def test_float_or_none(value: object, expected: float | None) -> None:
    assert float_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (0, 0.0),
        (-1, None),
        (True, None),
        ("1", None),
    ],
)
def test_non_negative_float_or_none(value: object, expected: float | None) -> None:
    assert non_negative_float_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.95, 0.95),
        (1, 1.0),
        (72, 0.72),
        (100, 1.0),
        (101, None),
        (-0.1, None),
        (True, None),
        ("0.95", None),
    ],
)
def test_score_or_none(value: object, expected: float | None) -> None:
    assert score_or_none(value) == expected
