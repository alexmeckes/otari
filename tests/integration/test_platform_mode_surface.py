import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


def test_platform_mode_starts_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="platform",
        database_url="postgresql://127.0.0.1:1/does-not-exist",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["mode"] == "platform"
    assert payload["platform_reachable"] in {"yes", "no"}

    reset_config()
    reset_db()


def test_platform_mode_disables_local_management_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="platform",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        responses = [
            client.post("/v1/users", json={"user_id": "u1"}),
            client.get("/v1/users/u1"),
            client.get("/v1/keys"),
            client.delete("/v1/keys/key_123"),
            client.get("/v1/budgets"),
            client.patch("/v1/budgets/budget_123"),
            client.get("/v1/spend"),
            client.get("/v1/spend/projects/project_123"),
        ]

    expected = {"detail": "This endpoint is not available in platform mode. Manage this resource via the platform UI."}
    for response in responses:
        assert response.status_code == 404
        assert response.json() == expected

    reset_config()
    reset_db()


def test_platform_mode_health_reports_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    async def _reachable(_: GatewayConfig) -> bool:
        return True

    monkeypatch.setattr("gateway.api.routes.health._check_platform_reachability", _reachable)

    config = GatewayConfig(
        mode="platform",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")
        readiness_response = client.get("/health/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "mode": "platform", "platform_reachable": "yes"}
    assert readiness_response.status_code == 200
    assert readiness_response.json()["platform"] == "connected"

    reset_config()
    reset_db()


def test_platform_mode_readiness_fails_when_platform_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    async def _unreachable(_: GatewayConfig) -> bool:
        return False

    monkeypatch.setattr("gateway.api.routes.health._check_platform_reachability", _unreachable)

    config = GatewayConfig(
        mode="platform",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["platform"] == "unavailable"

    reset_config()
    reset_db()
