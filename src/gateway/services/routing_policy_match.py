import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any


def policy_match_tags(config: Mapping[str, Any]) -> dict[str, str]:
    match_config = config.get("match")
    if not isinstance(match_config, dict):
        return {}
    tags = match_config.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def policy_match_priority(config: Mapping[str, Any]) -> int:
    match_config = config.get("match")
    if not isinstance(match_config, dict):
        return 0
    priority = match_config.get("priority")
    if isinstance(priority, int) and not isinstance(priority, bool):
        return priority
    return 0


def policy_match_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    match_config = config.get("match")
    return match_config if isinstance(match_config, dict) else {}


def policy_match_rollout_percentage(config: Mapping[str, Any]) -> float:
    match_config = policy_match_config(config)
    value = match_config.get("rollout_percentage", match_config.get("percentage"))
    if isinstance(value, bool):
        return 100.0
    if isinstance(value, int | float):
        return min(max(float(value), 0.0), 100.0)
    return 100.0


def policy_match_bucket_key(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> str:
    match_config = policy_match_config(config)
    bucket_by = match_config.get("bucket_by")
    if isinstance(bucket_by, str) and bucket_by in request_tags:
        return f"{bucket_by}:{request_tags[bucket_by]}"
    if request_tags:
        return json.dumps(sorted(request_tags.items()), separators=(",", ":"))
    return "__empty__"


def policy_match_bucket(config: Mapping[str, Any], policy_id: str, request_tags: Mapping[str, str]) -> float:
    match_config = policy_match_config(config)
    salt = match_config.get("salt")
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


def _condition_tag_key(condition: Mapping[str, Any]) -> str | None:
    for key in ("tag", "key", "field", "name"):
        value = condition.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _tag_values(value: Any) -> set[str]:
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value}
    return {str(value)}


def _evaluate_tag_condition(condition: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    tag_key = _condition_tag_key(condition)
    if tag_key is None:
        return False
    operator = condition.get("operator", condition.get("op", "eq"))
    op = str(operator).strip().lower()
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
        return actual_value in _tag_values(expected)
    if op in {"not_in", "nin"}:
        return actual_value not in _tag_values(expected)
    if op == "contains":
        return str(expected) in actual_value
    if op == "starts_with":
        return actual_value.startswith(str(expected))
    if op == "ends_with":
        return actual_value.endswith(str(expected))

    actual_number = _numeric_value(actual_value)
    expected_number = _numeric_value(expected)
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
    results = [matches_tag_condition(condition, request_tags) for condition in conditions]
    return any(results) if logic == "or" else all(results)


def matches_tag_condition(condition: Any, request_tags: Mapping[str, str]) -> bool:
    if not isinstance(condition, dict):
        return False
    for key, logic in (("any", "or"), ("or", "or"), ("all", "and"), ("and", "and")):
        if key in condition:
            return _evaluate_condition_group(condition[key], request_tags, logic=logic)
    return _evaluate_tag_condition(condition, request_tags)


def _matches_condition_config(match_config: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    if "any" in match_config:
        return _evaluate_condition_group(match_config["any"], request_tags, logic="or")
    if "or" in match_config:
        return _evaluate_condition_group(match_config["or"], request_tags, logic="or")
    if "all" in match_config:
        return _evaluate_condition_group(match_config["all"], request_tags, logic="and")
    if "and" in match_config:
        return _evaluate_condition_group(match_config["and"], request_tags, logic="and")
    conditions = match_config.get("conditions")
    if "conditions" not in match_config:
        return False
    logic_value = match_config.get("logic", "and")
    logic = str(logic_value).strip().lower()
    return _evaluate_condition_group(conditions, request_tags, logic="or" if logic in {"or", "any"} else "and")


def _matches_request_tags(policy_tags: Mapping[str, str], request_tags: Mapping[str, str]) -> bool:
    if not policy_tags:
        return False
    return all(request_tags.get(key) == value for key, value in policy_tags.items())


def matches_policy_match_config(config: Mapping[str, Any], request_tags: Mapping[str, str]) -> bool:
    match_config = policy_match_config(config)
    legacy_tags = policy_match_tags(config)
    has_conditions = any(key in match_config for key in ("conditions", "all", "any", "and", "or"))
    if not legacy_tags and not has_conditions:
        return False
    if legacy_tags and not _matches_request_tags(legacy_tags, request_tags):
        return False
    if has_conditions and not _matches_condition_config(match_config, request_tags):
        return False
    return True
