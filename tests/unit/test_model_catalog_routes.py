from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.api.routes._model_catalog import (
    CatalogRecord,
    _catalog_record_from_discovered,
    _catalog_record_from_pricing,
    vendor_catalog_from_records,
)
from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.main import create_app
from gateway.models.entities import ModelPricing


@pytest.fixture
def catalog_client(tmp_path: Path) -> Generator[tuple[TestClient, dict[str, str]], None, None]:
    database_path = tmp_path / "catalog.db"
    config = GatewayConfig(
        database_url=f"sqlite:///{database_path}",
        master_key="test-master-key",
        bootstrap_api_key=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-test"}},
    )
    app = create_app(config)
    master_header = {API_KEY_HEADER: "Bearer test-master-key"}

    with TestClient(app) as client:
        yield client, master_header


def _set_price(client: TestClient, master_header: dict[str, str], model_key: str) -> None:
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": model_key,
            "input_price_per_million": 0.15,
            "output_price_per_million": 0.60,
        },
        headers=master_header,
    )
    assert response.status_code == 200, response.text


def test_catalog_record_from_pricing_formats_datetimes() -> None:
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    updated_at = datetime(2026, 1, 3, 3, 4, 5, tzinfo=UTC)
    pricing = ModelPricing(
        model_key="openai:gpt-4o-mini",
        effective_at=created_at,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        created_at=created_at,
        updated_at=updated_at,
    )

    record = _catalog_record_from_pricing(pricing)

    assert record.created == int(created_at.timestamp())
    assert record.created_at == created_at.isoformat()
    assert record.updated_at == updated_at.isoformat()


def test_catalog_record_from_discovered_formats_positive_epoch_timestamp() -> None:
    created = 1_700_000_000
    model = Model(**{"id": "gpt-4o", "object": "model", "created": created, "owned_by": "openai"})

    record = _catalog_record_from_discovered("openai", model, pricing=None)

    expected_timestamp = datetime.fromtimestamp(created, tz=UTC).isoformat().replace("+00:00", "Z")
    assert record.created_at == expected_timestamp
    assert record.updated_at == expected_timestamp


def test_catalog_record_from_discovered_ignores_non_positive_epoch_timestamp() -> None:
    model = Model(**{"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai"})

    record = _catalog_record_from_discovered("openai", model, pricing=None)

    assert record.created_at is None
    assert record.updated_at is None


def test_models_default_response_stays_openai_compatible(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client
    _set_price(client, master_header, "openai:gpt-4o-mini")

    response = client.get("/v1/models", headers=master_header)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["id"] == "openai:gpt-4o-mini"
    assert body["data"][0]["owned_by"] == "openai"


def test_models_gateway_format_returns_vendor_catalog_shape(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client
    _set_price(client, master_header, "openai:gpt-4o-mini")

    response = client.get("/v1/models?format=gateway", headers=master_header)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    model = body["data"][0]
    assert model["model"] == "openai/gpt-4o-mini"
    assert model["provider"] == "openai"
    assert model["display_name"] == "GPT 4o Mini"
    assert model["availability_status"] == "available"
    assert model["vendors"]["openai"]["availability_status"] == "available"
    assert model["vendors"]["openai"]["capabilities"]["streaming"] is True
    assert model["vendors"]["openai"]["pricing"] == {
        "currency": "USD",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    }


def test_models_gateway_query_fetches_single_model_by_slash_selector(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client
    _set_price(client, master_header, "openai:gpt-4o-mini")

    response = client.get("/v1/models?model=openai/gpt-4o-mini", headers=master_header)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["vendors"]["openai"]["pricing"]["input_per_million"] == 0.15


def test_models_gateway_filters_by_vendor(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client
    _set_price(client, master_header, "openai:gpt-4o-mini")
    _set_price(client, master_header, "anthropic:claude-3-5-haiku-latest")

    response = client.get("/v1/models?vendor=anthropic", headers=master_header)

    assert response.status_code == 200
    body = response.json()
    assert [item["model"] for item in body["data"]] == ["anthropic/claude-3-5-haiku-latest"]


def test_vendors_list_and_fetch_single_vendor(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client
    _set_price(client, master_header, "openai:gpt-4o-mini")
    _set_price(client, master_header, "anthropic:claude-3-5-haiku-latest")

    list_response = client.get("/v1/vendors", headers=master_header)

    assert list_response.status_code == 200
    vendors = {item["vendor"]: item for item in list_response.json()["data"]}
    assert vendors["openai"]["name"] == "OpenAI"
    assert vendors["openai"]["models"] == ["openai/gpt-4o-mini"]
    assert vendors["openai"]["supports_byok"] is True
    assert vendors["anthropic"]["models"] == ["anthropic/claude-3-5-haiku-latest"]

    get_response = client.get("/v1/vendors/openai", headers=master_header)
    assert get_response.status_code == 200
    assert get_response.json()["vendor"] == "openai"


def test_vendor_catalog_uses_known_and_fallback_display_names() -> None:
    config = GatewayConfig(
        master_key="test-master-key",
        providers={"custom-provider": {"api_key": "sk-test"}},
    )
    records = [
        CatalogRecord(
            model_key="openai:gpt-4o-mini",
            provider="openai",
            provider_model="gpt-4o-mini",
            created=0,
            created_at=None,
            updated_at=None,
            pricing=None,
        )
    ]

    vendors = {item.vendor: item for item in vendor_catalog_from_records(records, config)}

    assert vendors["openai"].name == "OpenAI"
    assert vendors["custom-provider"].name == "Custom Provider"


def test_vendor_catalog_uses_slash_form_model_ids() -> None:
    config = GatewayConfig(master_key="test-master-key")
    records = [
        CatalogRecord(
            model_key="openai:gpt-4o",
            provider="openai",
            provider_model="gpt-4o",
            created=0,
            created_at=None,
            updated_at=None,
            pricing=None,
        ),
        CatalogRecord(
            model_key="gpt-4o-mini",
            provider="unknown",
            provider_model="gpt-4o-mini",
            created=0,
            created_at=None,
            updated_at=None,
            pricing=None,
        ),
    ]

    vendors = {item.vendor: item for item in vendor_catalog_from_records(records, config)}

    assert vendors["openai"].models == ["openai/gpt-4o"]
    assert vendors["unknown"].models == ["unknown/gpt-4o-mini"]


def test_vendors_unknown_vendor_returns_404(
    catalog_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, master_header = catalog_client

    response = client.get("/v1/vendors/nope", headers=master_header)

    assert response.status_code == 404
