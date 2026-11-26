"""FastAPI service: health, model info, single/batch predictions, validation, reason codes."""

from __future__ import annotations

import json
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import annotated_types
import pytest
from fastapi.testclient import TestClient

from ml_pipeline.api import create_app
from ml_pipeline.config import (
    BinaryFeature,
    CategoricalFeature,
    NumericFeature,
    PipelineConfig,
)
from ml_pipeline.features import ENGINEERED_FEATURES
from ml_pipeline.registry import ModelRegistry
from ml_pipeline.schemas import (
    EXAMPLE_CUSTOMER,
    MAX_BATCH_SIZE,
    CustomerFeatures,
    requests_to_frame,
)
from tests.conftest import TrainedPipeline


@pytest.fixture(scope="module")
def client(trained: TrainedPipeline) -> Iterator[TestClient]:
    app = create_app(config=trained.config, registry_dir=trained.config.paths.registry_dir)
    with TestClient(app) as test_client:
        yield test_client


def customer(**overrides: Any) -> dict[str, Any]:
    return {**EXAMPLE_CUSTOMER, **overrides}


# ---------------------------------------------------------------------------
# Health & model info
# ---------------------------------------------------------------------------
def test_health(client: TestClient, trained: TrainedPipeline) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == trained.report.model_id


def test_service_without_model_reports_unavailable(tmp_path: Path) -> None:
    app = create_app(config=PipelineConfig(), registry_dir=tmp_path / "empty-registry")
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "unavailable"
        assert "no production model" in health.json()["detail"]
        assert test_client.post("/predict", json=customer()).status_code == 503
        assert (
            test_client.post("/predict/batch", json={"instances": [customer()]}).status_code == 503
        )
        assert test_client.get("/model-info").status_code == 503


def test_model_info(client: TestClient, trained: TrainedPipeline) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    metadata = ModelRegistry(trained.config.paths.registry_dir).metadata(trained.report.model_id)
    assert body["model_id"] == metadata.model_id
    assert body["stage"] == "production"
    assert body["artifact_sha256"] == metadata.artifact.sha256
    assert body["threshold"] == metadata.threshold
    assert body["promotion_approved"] is True
    assert [f["name"] for f in body["features"]] == list(trained.config.data_schema.feature_names)
    tenure = next(f for f in body["features"] if f["name"] == "tenure")
    assert tenure["training_min"] == 0.0
    assert tenure["training_max"] <= 72.0


def test_openapi_document(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model-info", "/predict", "/predict/batch"} <= set(paths)


# ---------------------------------------------------------------------------
# Predictions & reason codes
# ---------------------------------------------------------------------------
def test_single_prediction(client: TestClient, trained: TrainedPipeline) -> None:
    response = client.post("/predict", json=customer(customer_id="cust-001"))
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "cust-001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    assert body["model_id"] == trained.report.model_id
    assert body["latency_ms"] >= 0.0
    assert body["warnings"] == []
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0


def test_prediction_matches_the_model(client: TestClient, trained: TrainedPipeline) -> None:
    model, _ = ModelRegistry(trained.config.paths.registry_dir).load(trained.report.model_id)
    frame = requests_to_frame(
        [CustomerFeatures.model_validate(EXAMPLE_CUSTOMER)], trained.config.data_schema
    )
    expected = round(float(model.predict_proba(frame)[0]), 6)
    assert client.post("/predict", json=customer()).json()["churn_probability"] == expected


def test_shap_reason_codes(client: TestClient, trained: TrainedPipeline) -> None:
    body = client.post("/predict", params={"top_k": 5}, json=customer()).json()
    reasons = body["reasons"]
    assert len(reasons) == 5
    allowed = set(trained.config.data_schema.feature_names) | set(ENGINEERED_FEATURES)
    assert {reason["feature"] for reason in reasons} <= allowed
    magnitudes = [abs(reason["contribution"]) for reason in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for reason in reasons:
        sign = "increases_risk" if reason["contribution"] > 0 else "decreases_risk"
        assert reason["direction"] == sign
    contract = next((r for r in reasons if r["feature"] == "Contract"), None)
    if contract is not None:
        assert contract["value"] == "Month-to-month"


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_is_bounded(client: TestClient, top_k: int) -> None:
    assert client.post("/predict", params={"top_k": top_k}, json=customer()).status_code == 422


def test_out_of_training_range_warning(client: TestClient) -> None:
    body = client.post("/predict", json=customer(tenure=110, TotalCharges=9000.0)).json()
    kinds = {(w["feature"], w["kind"]) for w in body["warnings"]}
    assert ("tenure", "out_of_training_range") in kinds
    assert ("TotalCharges", "out_of_training_range") in kinds


def test_batch_matches_single_predictions_in_order(client: TestClient) -> None:
    instances = [
        customer(customer_id="a"),
        customer(customer_id="b", Contract="Two year", tenure=60, TotalCharges=5000.0),
        customer(customer_id="c", InternetService="DSL", MonthlyCharges=45),
    ]
    batch = client.post("/predict/batch", json={"instances": instances})
    assert batch.status_code == 200
    body = batch.json()
    assert body["n_predictions"] == 3
    assert [p["customer_id"] for p in body["predictions"]] == ["a", "b", "c"]
    for instance, predicted in zip(instances, body["predictions"], strict=True):
        single = client.post("/predict", json=instance).json()
        assert predicted["churn_probability"] == single["churn_probability"]
        assert predicted["reasons"] == single["reasons"]


def test_request_id_is_echoed_or_regenerated(client: TestClient) -> None:
    echoed = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert echoed.headers["X-Request-ID"] == "trace-123"
    replaced = client.get("/health", headers={"X-Request-ID": "x" * 300})
    assert replaced.headers["X-Request-ID"] != "x" * 300
    assert len(replaced.headers["X-Request-ID"]) == 32


def test_integer_json_number_is_accepted_for_float_fields(client: TestClient) -> None:
    assert client.post("/predict", json=customer(MonthlyCharges=85)).status_code == 200


# ---------------------------------------------------------------------------
# Malformed payloads -> 422
# ---------------------------------------------------------------------------
def _without(field: str) -> dict[str, Any]:
    payload = customer()
    del payload[field]
    return payload


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_without("tenure"), "missing field"),
        (customer(Contract="Weekly"), "unknown category"),
        (customer(gender="F"), "unknown category"),
        (customer(favourite_colour="blue"), "extra field"),
        (customer(tenure="12"), "string for int (strict)"),
        (customer(tenure=12.5), "float for int"),
        (customer(tenure=121), "above schema bound"),
        (customer(MonthlyCharges=-1.0), "negative charge"),
        (customer(TotalCharges=1e9), "above schema bound"),
        (customer(SeniorCitizen=2), "invalid binary"),
        (customer(SeniorCitizen=True), "bool for binary (strict)"),
        (customer(PhoneService="No"), "phone/lines inconsistency"),
        (customer(InternetService="No"), "internet/add-on inconsistency"),
        (customer(customer_id="bad id!"), "unsafe customer id"),
        ({}, "empty object"),
    ],
)
def test_single_prediction_rejects_malformed_payloads(
    client: TestClient, payload: dict[str, Any], reason: str
) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(client: TestClient, value: float) -> None:
    # Python's json parser accepts the NaN/Infinity tokens, so the model must reject them.
    body = json.dumps(customer(MonthlyCharges=value))
    assert "NaN" in body or "Infinity" in body
    response = client.post(
        "/predict", content=body.encode(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_non_json_body_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"instances": []},
        {"instances": customer()},
        {"instances": [customer(), customer(Contract="Weekly")]},
        {"instances": [customer()] * (MAX_BATCH_SIZE + 1)},
        {"rows": [customer()]},
    ],
    ids=["empty", "not-a-list", "one-bad-row", "too-large", "wrong-key"],
)
def test_batch_rejects_malformed_payloads(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_validation_error_names_the_field(client: TestClient) -> None:
    detail = client.post("/predict", json=customer(Contract="Weekly")).json()["detail"]
    assert any(error["loc"][-1] == "Contract" for error in detail)


# ---------------------------------------------------------------------------
# Contract drift guard
# ---------------------------------------------------------------------------
def test_request_model_matches_the_config_schema(config: PipelineConfig) -> None:
    """Changing the schema in config.py without updating the API model must fail here."""
    fields = CustomerFeatures.model_fields
    schema = config.data_schema
    assert list(fields) == list(schema.feature_names)
    for spec in schema.features:
        field = fields[spec.name]
        if isinstance(spec, CategoricalFeature):
            assert set(typing.get_args(field.annotation)) == set(spec.categories), spec.name
        elif isinstance(spec, BinaryFeature):
            assert set(typing.get_args(field.annotation)) == {0, 1}, spec.name
        else:
            assert isinstance(spec, NumericFeature)
            expected_type = int if spec.integer else float
            assert field.annotation is expected_type, spec.name
            lower = [m.ge for m in field.metadata if isinstance(m, annotated_types.Ge)]
            upper = [m.le for m in field.metadata if isinstance(m, annotated_types.Le)]
            assert lower == [spec.min_value], spec.name
            assert upper == [spec.max_value], spec.name
