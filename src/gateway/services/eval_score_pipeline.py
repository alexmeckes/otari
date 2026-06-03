"""Normalize local eval artifacts into routing-policy score updates."""

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gateway.services.routing_config_values import coerced_string_or_none, float_or_none, string_or_none

_ROW_LIST_KEYS = ("scores", "results", "items", "rows", "evals")
_MODEL_KEYS = ("model", "model_key", "candidate_model")
_PROVIDER_KEYS = ("provider", "vendor")
_SCORE_KEYS = ("score", "eval_score", "accuracy", "pass_rate", "mean_score")
_QUALITY_SCORE_KEYS = ("quality_score",)
_BENCHMARK_SCORE_KEYS = ("benchmark_score",)
_METRIC_KEYS = ("metric", "eval", "benchmark", "task")
_SAMPLE_COUNT_KEYS = ("sample_count", "samples", "n", "count")
_KNOWN_KEYS = {
    *_ROW_LIST_KEYS,
    *_MODEL_KEYS,
    *_PROVIDER_KEYS,
    *_SCORE_KEYS,
    *_QUALITY_SCORE_KEYS,
    *_BENCHMARK_SCORE_KEYS,
    *_METRIC_KEYS,
    *_SAMPLE_COUNT_KEYS,
    "metadata",
}


class EvalScorePipelineError(ValueError):
    """Raised when an eval artifact cannot be converted into score rows."""


def _first_float(row: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        parsed = float_or_none(row.get(key), coerce_strings=True, allow_percent=True)
        if parsed is not None:
            return parsed
    return None


def _first_string(row: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        parsed = string_or_none(value)
        if parsed is not None:
            return parsed
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
    return None


def _row_metadata(row: Mapping[str, Any], *, source: str | None, row_number: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    existing = row.get("metadata")
    if isinstance(existing, dict):
        metadata.update(existing)
    for key, value in row.items():
        if key not in _KNOWN_KEYS and value is not None and coerced_string_or_none(value) is not None:
            metadata[key] = value
    if source is not None:
        metadata.setdefault("source", source)
    metadata.setdefault("row_number", row_number)
    return metadata


def _object_rows(rows: list[Any], *, error_message: str) -> list[Mapping[str, Any]]:
    object_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise EvalScorePipelineError(error_message)
        object_rows.append(row)
    return object_rows


def normalize_eval_score_row(
    row: Mapping[str, Any],
    *,
    default_metric: str | None = None,
    source: str | None = None,
    row_number: int = 1,
) -> dict[str, Any]:
    """Normalize one eval artifact row into the routing-policy API shape."""
    model = _first_string(row, _MODEL_KEYS)
    if model is None:
        raise EvalScorePipelineError(f"Eval score row {row_number} is missing a model")

    quality_score = _first_float(row, _QUALITY_SCORE_KEYS)
    benchmark_score = _first_float(row, _BENCHMARK_SCORE_KEYS)
    score = _first_float(row, _SCORE_KEYS)
    if quality_score is None and benchmark_score is None and score is None:
        raise EvalScorePipelineError(
            f"Eval score row {row_number} must include score, quality_score, benchmark_score, "
            "eval_score, accuracy, pass_rate, or mean_score"
        )

    item: dict[str, Any] = {"model": model}
    provider = _first_string(row, _PROVIDER_KEYS)
    if provider is not None:
        item["provider"] = provider
    if quality_score is not None:
        item["quality_score"] = quality_score
    elif benchmark_score is not None:
        item["benchmark_score"] = benchmark_score
    else:
        item["score"] = score

    metric = _first_string(row, _METRIC_KEYS) or string_or_none(default_metric)
    if metric is not None:
        item["metric"] = metric
    sample_count = None
    for key in _SAMPLE_COUNT_KEYS:
        parsed_sample_count = float_or_none(row.get(key), coerce_strings=True, allow_percent=True)
        if parsed_sample_count is not None and parsed_sample_count >= 1:
            sample_count = int(parsed_sample_count)
            break
    if sample_count is not None:
        item["sample_count"] = sample_count

    metadata = _row_metadata(row, source=source, row_number=row_number)
    if metadata:
        item["metadata"] = metadata
    return item


def load_eval_score_rows(path: Path) -> list[Mapping[str, Any]]:
    """Load raw eval score rows from JSON, JSONL/NDJSON, or CSV."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parsed_line = string_or_none(line)
                if parsed_line is None:
                    continue
                parsed = json.loads(parsed_line)
                if not isinstance(parsed, dict):
                    raise EvalScorePipelineError(f"JSONL line {line_number} must be an object")
                rows.append(parsed)
        return rows

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return _object_rows(payload, error_message="JSON eval artifact list must contain objects")
    if not isinstance(payload, dict):
        raise EvalScorePipelineError("JSON eval artifact must be an object or list")
    for key in _ROW_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return _object_rows(value, error_message=f"JSON eval artifact '{key}' list must contain objects")
    if any(key in payload for key in _MODEL_KEYS):
        return [payload]
    raise EvalScorePipelineError("JSON eval artifact must contain scores, results, items, rows, evals, or model")


def build_eval_scores_payload(
    rows: Iterable[Mapping[str, Any]],
    *,
    default_metric: str | None = None,
    change_note: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the `/eval-scores` request payload for a set of raw eval rows."""
    scores = [
        normalize_eval_score_row(
            row,
            default_metric=default_metric,
            source=source,
            row_number=index,
        )
        for index, row in enumerate(rows, start=1)
    ]
    if not scores:
        raise EvalScorePipelineError("Eval artifact did not contain any score rows")
    payload: dict[str, Any] = {"scores": scores}
    change_note_value = string_or_none(change_note)
    if change_note_value is not None:
        payload["change_note"] = change_note_value
    return payload
