import pytest

from gateway.services.routing_config_values import (
    dict_or_empty,
    float_or_none,
    non_negative_float_or_none,
    score_or_none,
    string_list,
)


def test_dict_or_empty_reuses_dict_by_default() -> None:
    value = {"enabled": True}

    assert dict_or_empty(value) is value


def test_dict_or_empty_can_copy_dict() -> None:
    value = {"enabled": True}
    result = dict_or_empty(value, copy_value=True)

    assert result == value
    assert result is not value


@pytest.mark.parametrize("value", [None, [], "enabled"])
def test_dict_or_empty_returns_empty_dict_for_non_dict(value: object) -> None:
    assert dict_or_empty(value) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" team ", ["team"]),
        (" ", []),
        ([" team ", 42, None, ""], ["team", "42", "None"]),
        (("team",), []),
        (None, []),
    ],
)
def test_string_list(value: object, expected: list[str]) -> None:
    assert string_list(value) == expected


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
