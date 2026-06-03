import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from gateway.services.routing_config_values import coerced_lower_string, dict_or_empty, float_or_none


def policy_match_tags(config: Mapping[str, Any]) -> dict[str, str]:
    tags = dict_or_empty(policy_match_config(config).get("tags"))
    return {str(key): str(value) for key, value in tags.items()}


def policy_match_priority(config: Mapping[str, Any]) -> int:
    priority = policy_match_config(config).get("priority")
    if isinstance(priority, int) and not isinstance(priority, bool):
        return priority
    return 0


def policy_match_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict_or_empty(config.get("match"))


def policy_match_rollout_percentage(config: Mapping[str, Any]) -> float:
    match_config = policy_match_config(config)
    value = match_config.get("rollout_percentage", match_config.get("percentage"))
    parsed = float_or_none(value)
    return 100.0 if parsed is None else min(max(parsed, 0.0), 100.0)


def policy_match_bucket_key(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> str:
    bucket_by = policy_match_config(config).get("bucket_by")
    if isinstance(bucket_by, str) and bucket_by in request_tags:
        return f"{bucket_by}:{request_tags[bucket_by]}"
    if request_tags:
        return json.dumps(sorted(request_tags.items()), separators=(",", ":"))
    return "__empty__"


def policy_match_bucket(config: Mapping[str, Any], policy_id: str, request_tags: Mapping[str, str]) -> float:
    salt = policy_match_config(config).get("salt")
    salt_value = salt if isinstance(salt, str) else policy_id
    bucket_key = policy_match_bucket_key(config, request_tags)
    digest = sha256(f"{salt_value}:{policy_id}:{bucket_key}".encode()).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 100


def policy_rollout_info(
    *,
    policy_id: str,
    config: Mapping[str, Any],
    request_tags: Mapping[str, str],
) -> dict[str, Any] | None:
    match_config = policy_match_config(config)
    if "rollout_percentage" not in match_config and "percentage" not in match_config:
        return None
    percentage = policy_match_rollout_percentage(config)
    bucket_key = policy_match_bucket_key(config, request_tags)
    bucket = policy_match_bucket(config, policy_id, request_tags)
    return {
        "percentage": percentage,
        "bucket": bucket,
        "bucket_key": bucket_key,
        "matched": bucket < percentage,
    }


def _evaluate_tag_condition(condition: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    tag_key = None
    for key in ("tag", "key", "field", "name"):
        value = condition.get(key)
        if isinstance(value, str) and value:
            tag_key = value
            break
    if tag_key is None:
        return False
    operator = condition.get("operator", condition.get("op", "eq"))
    op = coerced_lower_string(operator)
    expected = condition.get("value")
    exists = tag_key in request_tags
    actual = request_tags.get(tag_key)

    if op == "exists":
        expected_exists = expected if isinstance(expected, bool) else True
        return exists is expected_exists
    if actual is None:
        return False

    actual_value = str(actual)
    if op in {"eq", "equals", "=="}:
        return actual_value == str(expected)
    if op in {"ne", "neq", "not_eq", "!="}:
        return actual_value != str(expected)
    if op == "in":
        if isinstance(expected, list | tuple | set):
            return actual_value in {str(item) for item in expected}
        return actual_value == str(expected)
    if op in {"not_in", "nin"}:
        if isinstance(expected, list | tuple | set):
            return actual_value not in {str(item) for item in expected}
        return actual_value != str(expected)
    if op == "contains":
        return str(expected) in actual_value
    if op == "starts_with":
        return actual_value.startswith(str(expected))
    if op == "ends_with":
        return actual_value.endswith(str(expected))

    actual_number = float_or_none(actual_value, coerce_strings=True)
    expected_number = float_or_none(expected, coerce_strings=True)
    if actual_number is None or expected_number is None:
        return False
    if op == "gt":
        return actual_number > expected_number
    if op in {"gte", "ge"}:
        return actual_number >= expected_number
    if op == "lt":
        return actual_number < expected_number
    if op in {"lte", "le"}:
        return actual_number <= expected_number
    return False


def _evaluate_condition_group(
    conditions: Any,
    request_tags: Mapping[str, str],
    *,
    logic: str,
) -> bool:
    if not isinstance(conditions, list) or not conditions:
        return False
    matches = (matches_tag_condition(condition, request_tags) for condition in conditions)
    return any(matches) if logic == "or" else all(matches)


def _condition_group(config: Mapping[str, Any]) -> tuple[Any, str] | None:
    for key, logic in (("any", "or"), ("or", "or"), ("all", "and"), ("and", "and")):
        if key in config:
            return config[key], logic
    return None


def matches_tag_condition(condition: Any, request_tags: Mapping[str, str]) -> bool:
    if not isinstance(condition, dict):
        return False
    group = _condition_group(condition)
    if group is not None:
        conditions, logic = group
        return _evaluate_condition_group(conditions, request_tags, logic=logic)
    return _evaluate_tag_condition(condition, request_tags)


def matches_policy_match_config(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    match_config = policy_match_config(config)
    legacy_tags = policy_match_tags(config)
    if legacy_tags and not all(request_tags.get(key) == value for key, value in legacy_tags.items()):
        return False
    group = _condition_group(match_config)
    if group is not None:
        conditions, logic = group
        return _evaluate_condition_group(conditions, request_tags, logic=logic)
    if "conditions" in match_config:
        logic_value = match_config.get("logic", "and")
        logic = coerced_lower_string(logic_value)
        return _evaluate_condition_group(
            match_config.get("conditions"),
            request_tags,
            logic="or" if logic in {"or", "any"} else "and",
        )
    return bool(legacy_tags)
