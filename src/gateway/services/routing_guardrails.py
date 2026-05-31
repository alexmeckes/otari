"""Guardrail evaluation and redaction helpers for routing policies."""

import copy
import json
from collections.abc import Mapping
from typing import Any

from gateway.services import routing_guardrail_external as _routing_guardrail_external
from gateway.services import routing_guardrail_redactions as _routing_guardrail_redactions
from gateway.services.routing_config_values import string_list
from gateway.services.routing_guardrail_helpers import (
    PII_PATTERNS,
    guardrail_violation,
    guardrails_config,
    named_patterns,
)
from gateway.services.routing_request_analysis import bool_config, jsonable_text

ExternalClassifierPost = _routing_guardrail_external.ExternalClassifierPost
post_external_guardrail_classifier = _routing_guardrail_external.post_external_guardrail_classifier
apply_guardrail_redactions = _routing_guardrail_redactions.apply_guardrail_redactions

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


def _normalize_guardrail_preset_name(value: Any) -> str | None:
    name: Any = value
    if isinstance(value, dict):
        name = value.get("name") or value.get("preset")
    if not isinstance(name, str) or not name.strip():
        return None
    normalized = name.strip().lower().replace("-", "_")
    return _GUARDRAIL_PRESET_ALIASES.get(normalized, normalized)


def _guardrail_preset_values(guardrails: Mapping[str, Any]) -> list[Any]:
    presets = guardrails.get("presets")
    if presets is None:
        presets = guardrails.get("managed_presets")
    if isinstance(presets, list):
        return presets
    if isinstance(presets, str) and presets.strip():
        return [presets]
    return []


def _guardrail_list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
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
    for value in _guardrail_preset_values(guardrails):
        normalized = _normalize_guardrail_preset_name(value)
        if normalized is None:
            continue
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


def _effective_guardrails_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, list[str]] | None]:
    guardrails = guardrails_config(config)
    preset_config, preset_metadata = _guardrail_preset_expansion(guardrails)
    if not preset_config:
        return guardrails, preset_metadata
    return _combine_guardrail_config(preset_config, guardrails), preset_metadata


def _guardrails_enabled(config: Mapping[str, Any]) -> bool:
    guardrails = guardrails_config(config)
    return bool_config(guardrails.get("enabled"), bool(guardrails))


def guardrail_action(config: Mapping[str, Any]) -> str:
    action = guardrails_config(config).get("action")
    if isinstance(action, str) and action.strip().lower() in _GUARDRAIL_ACTIONS:
        return action.strip().lower()
    return "block"


def _request_guardrail_text(request_body: Mapping[str, Any]) -> str:
    parts = [
        jsonable_text(request_body.get("messages")),
        jsonable_text(request_body.get("input")),
        jsonable_text(request_body.get("instructions")),
    ]
    return "\n".join(part for part in parts if part)


async def evaluate_guardrails(
    config: Mapping[str, Any],
    request_body: Mapping[str, Any],
    *,
    post_classifier: ExternalClassifierPost = post_external_guardrail_classifier,
) -> dict[str, Any] | None:
    if not _guardrails_enabled(config):
        return None

    guardrails, preset_metadata = _effective_guardrails_config(config)
    request_text = _request_guardrail_text(request_body)
    normalized_text = request_text.lower()
    violations: list[dict[str, str]] = []
    classifier_results: list[dict[str, Any]] = []

    for term in string_list(guardrails.get("blocked_terms")):
        if term.lower() in normalized_text:
            violations.append(guardrail_violation("blocked_term", term))

    for name, pattern in named_patterns(guardrails.get("blocked_patterns")):
        if pattern.search(request_text):
            violations.append(guardrail_violation("blocked_pattern", name))

    pii_config = guardrails.get("pii")
    pii_enabled = bool_config(pii_config.get("enabled") if isinstance(pii_config, dict) else pii_config, False)
    if pii_enabled:
        type_config = pii_config.get("types") if isinstance(pii_config, dict) else None
        pii_types = string_list(type_config) or sorted(PII_PATTERNS)
        for pii_type in pii_types:
            pii_pattern = PII_PATTERNS.get(pii_type)
            if pii_pattern is not None and pii_pattern.search(request_text):
                violations.append(guardrail_violation("pii", pii_type))

    injection_config = guardrails.get("prompt_injection")
    injection_enabled = bool_config(
        injection_config.get("enabled") if isinstance(injection_config, dict) else injection_config,
        False,
    )
    if injection_enabled:
        phrases = string_list(injection_config.get("phrases") if isinstance(injection_config, dict) else None)
        for phrase in [*phrases, *_PROMPT_INJECTION_PHRASES]:
            if phrase.lower() in normalized_text:
                violations.append(guardrail_violation("prompt_injection", phrase))

    external_violations, classifier_results = await _routing_guardrail_external.evaluate_external_classifiers(
        guardrails=guardrails,
        request_text=request_text,
        post_classifier=post_classifier,
    )
    violations.extend(external_violations)

    action = guardrail_action(config)
    status_value = "passed"
    if violations:
        status_value = "blocked" if action == "block" else "observed"
    result: dict[str, Any] = {
        "enabled": True,
        "status": status_value,
        "action": action,
        "violations": violations,
        "external_classifiers": classifier_results,
        "checked_text_chars": len(request_text),
    }
    if preset_metadata is not None:
        result["presets"] = preset_metadata
    return result
