from gateway.services import routing_policy_shape
from gateway.services.routing_config_values import float_or_none, score_or_none


def test_number_value_reuses_shared_float_parser() -> None:
    assert routing_policy_shape.number_value is float_or_none
    assert routing_policy_shape.number_value(1.25) == 1.25
    assert routing_policy_shape.number_value(True) is None


def test_score_value_reuses_shared_score_parser() -> None:
    assert routing_policy_shape.score_value is score_or_none
    assert routing_policy_shape.score_value(72) == 0.72
    assert routing_policy_shape.score_value(101) is None
