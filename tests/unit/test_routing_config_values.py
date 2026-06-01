import pytest

from gateway.services.routing_config_values import (
    bool_config,
    comma_separated_string_list,
    dict_or_empty,
    float_or_none,
    int_config,
    lower_string_or_none,
    non_negative_float_or_none,
    non_negative_int_config,
    score_or_none,
    string_list,
    string_or_none,
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
        (" alpha, beta ,,gamma ", ["alpha", "beta", "gamma"]),
        (" , ", []),
        ("", []),
        (None, []),
        (["alpha", "beta"], []),
    ],
)
def test_comma_separated_string_list(value: object, expected: list[str]) -> None:
    assert comma_separated_string_list(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" team ", "team"),
        (" ", None),
        (42, None),
        (None, None),
    ],
)
def test_string_or_none(value: object, expected: str | None) -> None:
    assert string_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" Team ", "team"),
        (" ", None),
        (42, None),
        (None, None),
    ],
)
def test_lower_string_or_none(value: object, expected: str | None) -> None:
    assert lower_string_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        ("true", False, False),
        (None, True, True),
    ],
)
def test_bool_config_strict(value: object, default: bool, expected: bool) -> None:
    assert bool_config(value, default) is expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        ("true", False, True),
        (" yes ", False, True),
        ("0", True, False),
        ("off", True, False),
        (" ", True, True),
        ("maybe", True, True),
        (None, False, False),
    ],
)
def test_bool_config_can_coerce_strings(value: object, default: bool, expected: bool) -> None:
    assert bool_config(value, default, coerce_strings=True) is expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (3, 1, 3),
        (0, 1, 1),
        (-1, 1, 1),
        ("3", 1, 1),
    ],
)
def test_int_config(value: object, default: int, expected: int) -> None:
    assert int_config(value, default) == expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (3, 1, 3),
        (0, 1, 0),
        (-1, 1, 1),
        ("3", 1, 1),
    ],
)
def test_non_negative_int_config(value: object, default: int, expected: int) -> None:
    assert non_negative_int_config(value, default) == expected


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
        ("1.25", 1.25),
        (" 72% ", 72.0),
        ("", None),
        ("not-number", None),
        (True, None),
    ],
)
def test_float_or_none_can_coerce_strings(value: object, expected: float | None) -> None:
    assert float_or_none(value, coerce_strings=True, allow_percent=True) == expected


def test_float_or_none_requires_percent_opt_in() -> None:
    assert float_or_none("72%", coerce_strings=True) is None


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
