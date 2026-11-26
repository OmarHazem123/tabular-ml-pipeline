"""Strict request/response contracts for the inference API and ``ml-pipeline predict``.

``CustomerFeatures`` mirrors ``config.SchemaConfig`` field by field (names, allowed
categories, hard bounds); ``tests/test_api.py`` asserts the two never drift apart. The
cross-field rules are the same ``DEFAULT_CONSISTENCY_RULES`` the training data is
validated against.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml_pipeline.config import DEFAULT_CONSISTENCY_RULES, SchemaConfig, ServingConfig
from ml_pipeline.metrics import ClassificationMetrics, FoldScores
from ml_pipeline.model import InputWarning, Reason

YesNo = Literal["No", "Yes"]
InternetAddon = Literal["No", "No internet service", "Yes"]
MAX_BATCH_SIZE = ServingConfig().max_batch_size

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.5,
    "TotalCharges": 169.0,
}


class CustomerFeatures(BaseModel):
    """One customer's raw attributes (Telco CSV columns, minus ID and target)."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        json_schema_extra={"examples": [EXAMPLE_CUSTOMER]},
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0, le=120, description="Months with the company")]
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: Annotated[
        float, Field(ge=0, le=500, allow_inf_nan=False, description="Current monthly bill")
    ]
    TotalCharges: Annotated[
        float, Field(ge=0, le=100_000, allow_inf_nan=False, description="Lifetime billed")
    ]

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def _reject_boolean(cls, value: object) -> object:
        # Literal[0, 1] matches by equality and True == 1 in Python, so strict mode alone
        # would accept JSON `true`. The contract is the integer 0 or 1.
        if isinstance(value, bool):
            raise ValueError("must be the integer 0 or 1, not a boolean")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        values = self.__dict__
        violated = [
            rule.describe()
            for rule in DEFAULT_CONSISTENCY_RULES
            if (values[rule.left] == rule.left_value) != (values[rule.right] == rule.right_value)
        ]
        if violated:
            raise ValueError("inconsistent fields: " + "; ".join(violated))
        return self


class PredictionRequest(CustomerFeatures):
    customer_id: Annotated[
        str | None,
        Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"),
    ] = None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    instances: Annotated[list[PredictionRequest], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class PredictionResult(BaseModel):
    customer_id: str | None
    churn_probability: float
    churn_prediction: bool
    threshold: float
    reasons: list[Reason]
    warnings: list[InputWarning]


class PredictionResponse(PredictionResult):
    model_id: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    model_id: str
    n_predictions: int
    latency_ms: float
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_id: str | None
    detail: str | None
    uptime_seconds: float


class FeatureInfo(BaseModel):
    name: str
    kind: str
    categories: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    training_min: float | None = None
    training_max: float | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    model_name: str
    stage: str
    created_at: str
    threshold: float
    calibration_method: str
    artifact_sha256: str
    git_commit: str | None
    config_fingerprint: str
    dataset_sha256: str | None
    cv: dict[str, FoldScores]
    validation: ClassificationMetrics
    test: ClassificationMetrics | None
    promotion_approved: bool | None
    features: list[FeatureInfo]


def requests_to_frame(instances: Sequence[CustomerFeatures], schema: SchemaConfig) -> pd.DataFrame:
    """Validated requests -> a frame with the exact training columns and dtypes."""
    columns = list(schema.feature_names)
    records = [instance.model_dump(include=set(columns)) for instance in instances]
    frame = pd.DataFrame.from_records(records, columns=columns)
    dtypes: dict[str, str] = {}
    for spec in schema.features:
        dtypes[spec.name] = "str" if spec.kind == "categorical" else "float64"
    return frame.astype(dtypes)
