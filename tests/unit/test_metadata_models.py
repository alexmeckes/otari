from gateway.models.entities import APIKey, BudgetAlert, Project, User


def test_api_key_metadata_dict_returns_metadata_when_dict() -> None:
    key = APIKey(metadata_={"env": "dev"})

    assert key.metadata_dict() == {"env": "dev"}
    assert key.to_dict()["metadata"] == {"env": "dev"}


def test_api_key_metadata_dict_returns_empty_dict_for_invalid_metadata() -> None:
    key = APIKey(metadata_=None)  # type: ignore[arg-type]

    assert key.metadata_dict() == {}
    assert key.to_dict()["metadata"] == {}


def test_user_metadata_dict_returns_metadata_when_dict() -> None:
    user = User(metadata_={"plan": "team"})

    assert user.metadata_dict() == {"plan": "team"}
    assert user.to_dict()["metadata"] == {"plan": "team"}


def test_user_metadata_dict_returns_empty_dict_for_invalid_metadata() -> None:
    user = User(metadata_=None)  # type: ignore[arg-type]

    assert user.metadata_dict() == {}
    assert user.to_dict()["metadata"] == {}


def test_project_metadata_dict_returns_metadata_when_dict() -> None:
    project = Project(metadata_={"owner": "platform"})

    assert project.metadata_dict() == {"owner": "platform"}
    assert project.to_dict()["metadata"] == {"owner": "platform"}


def test_project_metadata_dict_returns_empty_dict_for_invalid_metadata() -> None:
    project = Project(metadata_=None)  # type: ignore[arg-type]

    assert project.metadata_dict() == {}
    assert project.to_dict()["metadata"] == {}


def test_budget_alert_metadata_dict_returns_metadata_when_dict() -> None:
    alert = BudgetAlert(metadata_={"threshold": "warning"})

    assert alert.metadata_dict() == {"threshold": "warning"}
    assert alert.to_dict()["metadata"] == {"threshold": "warning"}


def test_budget_alert_metadata_dict_returns_empty_dict_for_invalid_metadata() -> None:
    alert = BudgetAlert(metadata_=None)  # type: ignore[arg-type]

    assert alert.metadata_dict() == {}
    assert alert.to_dict()["metadata"] == {}
