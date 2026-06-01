from typing import Any

import pytest

from gateway.services.routing_latency_stats import attach_latency_stats
from gateway.services.routing_policy_service import RoutingCandidate


def _candidate(model: str, *, position: int) -> RoutingCandidate:
    provider, provider_model = model.split(":", 1)
    return RoutingCandidate(
        model=model,
        provider=provider,
        provider_model=provider_model,
        position=position,
        tier=None,
        estimated_cost=None,
        input_price_per_million=None,
        output_price_per_million=None,
        quality_score=None,
        average_latency_ms=None,
        latency_sample_count=0,
        routing_score=None,
        score_components=None,
        provider_health=None,
        metadata={},
    )


class _Trace:
    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        self._attempts = attempts

    def attempt_list(self) -> list[dict[str, Any]]:
        return self._attempts


class _Scalars:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces

    def all(self) -> list[_Trace]:
        return self._traces


class _Result:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces

    def scalars(self) -> _Scalars:
        return _Scalars(self._traces)


class _Db:
    def __init__(self, traces: list[_Trace]) -> None:
        self._traces = traces
        self.statement: Any | None = None

    async def execute(self, statement: Any) -> _Result:
        self.statement = statement
        return _Result(self._traces)


@pytest.mark.asyncio
async def test_attach_latency_stats_uses_configured_limit_and_min_samples() -> None:
    db = _Db(
        [
            _Trace(
                [
                    {"model_key": "openai:gpt-4o", "status": "success", "duration_ms": 100.0},
                    {"model_key": "openai:gpt-4o", "status": "success", "duration_ms": 50.0},
                    {"model_key": "openai:gpt-4o-mini", "status": "success", "duration_ms": 25.0},
                    {"model_key": "openai:gpt-4o-mini", "status": "error", "duration_ms": 5.0},
                ]
            )
        ]
    )

    enriched = await attach_latency_stats(
        db,
        [
            _candidate("openai:gpt-4o", position=1),
            _candidate("openai:gpt-4o-mini", position=2),
        ],
        config={"latency_sample_limit": 7, "latency_min_samples": 2},
    )

    assert db.statement is not None
    assert db.statement._limit_clause.value == 7
    assert enriched[0].average_latency_ms == 75.0
    assert enriched[0].latency_sample_count == 2
    assert enriched[1].average_latency_ms is None
    assert enriched[1].latency_sample_count == 0


@pytest.mark.asyncio
async def test_attach_latency_stats_uses_defaults_for_invalid_config_values() -> None:
    db = _Db(
        [
            _Trace(
                [
                    {"model_key": "openai:gpt-4o", "status": "success", "duration_ms": 42.0},
                ]
            )
        ]
    )

    enriched = await attach_latency_stats(
        db,
        [_candidate("openai:gpt-4o", position=1)],
        config={"latency_sample_limit": 0, "latency_min_samples": "2"},
    )

    assert db.statement is not None
    assert db.statement._limit_clause.value == 200
    assert enriched[0].average_latency_ms == 42.0
    assert enriched[0].latency_sample_count == 1
