"""Guardrail evaluation and redaction helpers for routing policies."""

import copy
import json
from collections.abc import Mapping
from typing import Any

from gateway.services import routing_guardrail_external
from gateway.services.routing_config_values import bool_config, string_list, string_or_none
from gateway.services.routing_guardrail_external import ExternalClassifierPost
from gateway.services.routing_guardrail_helpers import (
    guardrail_config_value,
    guardrail_violation,
    guardrails_config,
    named_patterns,
    pii_patterns_from_config,
)
from gateway.services.routing_request_analysis import jsonable_text

_GUARDRAIL_ACTIONS = {"block", "observe"}
_PROMPT_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "reveal the system prompt",
    "show me the system prompt",
    "print the system prompt",
    "developer message",
    "jailbreak",
)
_CREDENTIAL_BLOCKED_PATTERNS = (
    {"name": "openai_api_key", "pattern": r"\bsk-[A-Za-z0-9_-]{20,}\b"},
    {"name": "aws_access_key_id", "pattern": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"},
    {"name": "github_token", "pattern": r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"},
    {"name": "slack_token", "pattern": r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"},
    {"name": "private_key_block", "pattern": r"-----BEGIN [A-Z ]*PRIVATE KEY-----"},
)
_GUARDRAIL_PRESETS: dict[str, dict[str, Any]] = {
    "prompt_injection": {"prompt_injection": {"enabled": True}},
    "pii": {"pii": {"enabled": True}},
    "credential_leak": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "secrets": {"blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS)},
    "dlp": {
        "pii": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "baseline": {
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
    "strict": {
        "pii": {"enabled": True},
        "prompt_injection": {"enabled": True},
        "blocked_patterns": list(_CREDENTIAL_BLOCKED_PATTERNS),
    },
}
_GUARDRAIL_PRESET_ALIASES = {
    "credentials": "credential_leak",
    "credential": "credential_leak",
    "secret": "secrets",
    "data_loss_prevention": "dlp",
    "data_loss": "dlp",
    "prompt_shield": "prompt_injection",
    "prompt_injection_detection": "prompt_injection",
}


def _guardrail_list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    item = string_or_none(value)
    if item is not None:
        return [item]
    return []


def _combine_guardrail_list(existing: Any, incoming: Any) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for item in [*_guardrail_list_items(existing), *_guardrail_list_items(incoming)]:
        try:
            key = json.dumps(item, sort_keys=True)
        except TypeError:
            key = str(item)
        if key in seen:
            continue
        seen.add(key)
        values.append(copy.deepcopy(item))
    return values


def _combine_guardrail_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    combined = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in {"blocked_terms", "blocked_patterns", "external_classifiers"}:
            combined[key] = _combine_guardrail_list(combined.get(key), value)
            continue
        if isinstance(combined.get(key), dict) and isinstance(value, dict):
            nested = dict(combined[key])
            nested.update(copy.deepcopy(value))
            combined[key] = nested
            continue
        combined[key] = copy.deepcopy(value)
    return combined


def _guardrail_preset_expansion(
    guardrails: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]] | None]:
    preset_config: dict[str, Any] = {}
    applied: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    presets = guardrails.get("presets")
    if presets is None:
        presets = guardrails.get("managed_presets")
    preset_values: list[Any]
    if isinstance(presets, list):
        preset_values = presets
    else:
        preset = string_or_none(presets)
        preset_values = [preset] if preset is not None else []
    for value in preset_values:
        name: Any = value
        if isinstance(value, dict):
            name = value.get("name") or value.get("preset")
        name_value = string_or_none(name)
        if name_value is None:
            continue
        normalized = name_value.lower().replace("-", "_")
        normalized = _GUARDRAIL_PRESET_ALIASES.get(normalized, normalized)
        if normalized not in _GUARDRAIL_PRESETS:
            ignored.append(normalized)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        applied.append(normalized)
        preset_config = _combine_guardrail_config(preset_config, _GUARDRAIL_PRESETS[normalized])

    if not applied and not ignored:
        return preset_config, None
    metadata = {"applied": applied, "ignored": ignored}
    return preset_config, metadata


def guardrail_action(config: Mapping[str, Any]) -> str:
    action = string_or_none(guardrails_config(config).get("action"))
    if action is not None and action.lower() in _GUARDRAIL_ACTIONS:
        return action.lower()
    return "block"


def _guardrail_request_text(request_body: Mapping[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            jsonable_text(request_body.get("messages")),
            jsonable_text(request_body.get("input")),
            jsonable_text(request_body.get("instructions")),
        )
        if part
    )


def _prompt_injection_violations(guardrails: Mapping[str, Any], normalized_text: str) -> list[dict[str, str]]:
    injection_config = guardrails.get("prompt_injection")
    injection_enabled = bool_config(
        guardrail_config_value(injection_config, "enabled", scalar_value=injection_config),
        False,
    )
    if not injection_enabled:
        return []
    phrases = [*string_list(guardrail_config_value(injection_config, "phrases")), *_PROMPT_INJECTION_PHRASES]
    return [
        guardrail_violation("prompt_injection", phrase)
        for phrase in phrases
        if phrase.lower() in normalized_text
    ]


def _guardrail_result(
    *,
    action: str,
    violations: list[dict[str, str]],
    classifier_results: list[dict[str, Any]],
    checked_text_chars: int,
    preset_metadata: dict[str, list[str]] | None,
) -> dict[str, Any]:
    status_value = "passed"
    if violations:
        status_value = "blocked" if action == "block" else "observed"
    result: dict[str, Any] = {
        "enabled": True,
        "status": status_value,
        "action": action,
        "violations": violations,
        "external_classifiers": classifier_results,
        "checked_text_chars": checked_text_chars,
    }
    if preset_metadata is not None:
        result["presets"] = preset_metadata
    return result


async def evaluate_guardrails(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
    *,
    post_classifier: ExternalClassifierPost | None = None,
) -> dict[str, Any] | None:
    guardrails = guardrails_config(config)
    if not bool_config(guardrails.get("enabled"), bool(guardrails)):
        return None

    preset_config, preset_metadata = _guardrail_preset_expansion(guardrails)
    if preset_config:
        guardrails = _combine_guardrail_config(preset_config, guardrails)
    request_text = _guardrail_request_text(request_body)
    normalized_text = request_text.lower()
    violations: list[dict[str, str]] = []

    violations.extend(
        guardrail_violation("blocked_term", value)
        for value in string_list(guardrails.get("blocked_terms"))
        if value.lower() in normalized_text
    )

    for name, pattern in named_patterns(guardrails.get("blocked_patterns")):
        if pattern.search(request_text):
            violations.append(guardrail_violation("blocked_pattern", name))

    for pii_type, pii_pattern in pii_patterns_from_config(guardrails.get("pii")):
        if pii_pattern.search(request_text):
            violations.append(guardrail_violation("pii", pii_type))

    violations.extend(_prompt_injection_violations(guardrails, normalized_text))

    classifier_post = post_classifier or routing_guardrail_external.post_external_guardrail_classifier
    external_violations, classifier_results = await routing_guardrail_external.evaluate_external_classifiers(
        guardrails=guardrails,
        request_text=request_text,
        post_classifier=classifier_post,
    )
    violations.extend(external_violations)

    action = guardrail_action(config)
    return _guardrail_result(
        action=action,
        violations=violations,
        classifier_results=classifier_results,
        checked_text_chars=len(request_text),
        preset_metadata=preset_metadata,
    )
