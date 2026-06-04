from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app


def test_root_tutorial_page_is_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-root-test.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Gateway (Proxy Server)" in response.text
    assert "Gateway Quickstart" in response.text
    assert "bootstrap API key" in response.text
    assert "from openai import OpenAI" in response.text
    assert "YOUR_BOOTSTRAP_GATEWAY_KEY" in response.text
    assert "mozilla-ai.github.io/otari/gateway/quickstart" in response.text


def test_admin_dashboard_page_is_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-admin-test.db"
    config = GatewayConfig(
        database_url=f"sqlite:///{database_path}",
        master_key="test-master-key",
        bootstrap_api_key=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/admin")
        slash_response = client.get("/admin/")

    assert response.status_code == 200
    assert slash_response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "text/html" in slash_response.headers["content-type"]
    assert "LLM Routing Gateway Admin" in response.text
    assert slash_response.text == response.text
    assert "/admin/assets/styles.css" in response.text
    assert "/admin/assets/app.js" in response.text
    assert 'type="module"' in response.text
    assert 'data-view="policies"' in response.text


def test_admin_dashboard_assets_are_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-admin-assets-test.db"
    config = GatewayConfig(
        database_url=f"sqlite:///{database_path}",
        master_key="test-master-key",
        bootstrap_api_key=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        css_response = client.get("/admin/assets/styles.css")
        js_responses = {
            asset: client.get(f"/admin/assets/{asset}")
            for asset in ("api.js", "app.js", "dom.js", "format.js", "modal.js", "render.js")
        }
        missing_response = client.get("/admin/assets/missing.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert ".app-shell" in css_response.text
    assert missing_response.status_code == 404
    for js_response in js_responses.values():
        assert js_response.status_code == 200
        assert "text/javascript" in js_response.headers["content-type"]
    assert "/v1/routing-policies" in js_responses["app.js"].text
    assert "/v1/route-traces/summary" in js_responses["app.js"].text
    assert "/v1/usage/summary" in js_responses["app.js"].text
    assert "/v1/budgets/alerts" in js_responses["app.js"].text
    assert "Otari-Key" in js_responses["api.js"].text
    assert "renderOverview" in js_responses["render.js"].text
