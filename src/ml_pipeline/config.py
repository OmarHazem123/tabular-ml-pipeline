"""Typed, immutable pipeline configuration.

A single :class:`PipelineConfig` describes everything that influences a trained model:
dataset provenance, schema & data-quality contract, split policy, feature engineering,
hyperparameter search spaces and promotion thresholds. Instances are frozen and
validated on construction; :meth:`PipelineConfig.fingerprint` gives a stable hash for
lineage metadata.
"""

from __future__ import annotations

import json
import math
import random
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml_pipeline.io_utils import canonical_json, sha256_text

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Dataset provenance
# ---------------------------------------------------------------------------
class DatasetConfig(_Frozen):
    """Pinned public dataset: immutable commit URL + content checksum."""

    name: str = "ibm-telco-customer-churn"
    url: Annotated[str, Field(pattern=r"^https://")] = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
    )
    sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)] = (
        "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    )
    filename: str = "telco_customer_churn.csv"
    license: str = "Apache-2.0 (IBM/telco-customer-churn-on-icp4d)"
    download_timeout_seconds: Annotated[float, Field(gt=0)] = 60.0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class NumericFeature(_Frozen):
    """Continuous or count feature with hard validity bounds (not drift bounds)."""

    kind: Literal["numeric"] = "numeric"
    name: str
    integer: bool = False
    min_value: float
    max_value: float
    max_missing_fraction: UnitInterval = 0.0
    description: str = ""

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if not self.min_value < self.max_value:
            raise ValueError(f"{self.name}: min_value must be < max_value")
        return self


class BinaryFeature(_Frozen):
    """0/1 integer indicator."""

    kind: Literal["binary"] = "binary"
    name: str
    max_missing_fraction: UnitInterval = 0.0
    description: str = ""


class CategoricalFeature(_Frozen):
    """Nominal feature with a closed set of allowed values."""

    kind: Literal["categorical"] = "categorical"
    name: str
    categories: Annotated[tuple[str, ...], Field(min_length=2)]
    max_missing_fraction: UnitInterval = 0.0
    description: str = ""

    @model_validator(mode="after")
    def _check_categories(self) -> Self:
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{self.name}: categories must be unique")
        return self


FeatureSpec = Annotated[
    NumericFeature | BinaryFeature | CategoricalFeature, Field(discriminator="kind")
]

_YES_NO = ("No", "Yes")
_INTERNET_ADDON = ("No", "No internet service", "Yes")

DEFAULT_FEATURES: tuple[NumericFeature | BinaryFeature | CategoricalFeature, ...] = (
    CategoricalFeature(name="gender", categories=("Female", "Male")),
    BinaryFeature(name="SeniorCitizen", description="1 if the customer is 65 or older"),
    CategoricalFeature(name="Partner", categories=_YES_NO),
    CategoricalFeature(name="Dependents", categories=_YES_NO),
    NumericFeature(
        name="tenure",
        integer=True,
        min_value=0,
        max_value=120,
        description="Months with the company",
    ),
    CategoricalFeature(name="PhoneService", categories=_YES_NO),
    CategoricalFeature(name="MultipleLines", categories=("No", "No phone service", "Yes")),
    CategoricalFeature(name="InternetService", categories=("DSL", "Fiber optic", "No")),
    CategoricalFeature(name="OnlineSecurity", categories=_INTERNET_ADDON),
    CategoricalFeature(name="OnlineBackup", categories=_INTERNET_ADDON),
    CategoricalFeature(name="DeviceProtection", categories=_INTERNET_ADDON),
    CategoricalFeature(name="TechSupport", categories=_INTERNET_ADDON),
    CategoricalFeature(name="StreamingTV", categories=_INTERNET_ADDON),
    CategoricalFeature(name="StreamingMovies", categories=_INTERNET_ADDON),
    CategoricalFeature(name="Contract", categories=("Month-to-month", "One year", "Two year")),
    CategoricalFeature(name="PaperlessBilling", categories=_YES_NO),
    CategoricalFeature(
        name="PaymentMethod",
        categories=(
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ),
    ),
    NumericFeature(
        name="MonthlyCharges",
        min_value=0,
        max_value=500,
        description="Current monthly bill (USD)",
    ),
    NumericFeature(
        name="TotalCharges",
        min_value=0,
        max_value=100_000,
        max_missing_fraction=0.01,
        description="Lifetime billed amount (USD)",
    ),
)


class SchemaConfig(_Frozen):
    """Column contract for the raw dataset, in canonical column order."""

    id_column: str = "customerID"
    target_column: str = "Churn"
    positive_label: str = "Yes"
    negative_label: str = "No"
    features: Annotated[tuple[FeatureSpec, ...], Field(min_length=1)] = DEFAULT_FEATURES
    allow_extra_columns: bool = False

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        names = [feature.name for feature in self.features]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {duplicates}")
        reserved = {self.id_column, self.target_column} & set(names)
        if reserved:
            raise ValueError(f"id/target columns cannot also be features: {sorted(reserved)}")
        if self.id_column == self.target_column:
            raise ValueError("id_column and target_column must differ")
        if self.positive_label == self.negative_label:
            raise ValueError("positive_label and negative_label must differ")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def numeric_features(self) -> tuple[NumericFeature, ...]:
        return tuple(f for f in self.features if isinstance(f, NumericFeature))

    @property
    def binary_features(self) -> tuple[BinaryFeature, ...]:
        return tuple(f for f in self.features if isinstance(f, BinaryFeature))

    @property
    def categorical_features(self) -> tuple[CategoricalFeature, ...]:
        return tuple(f for f in self.features if isinstance(f, CategoricalFeature))

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.id_column, *self.feature_names, self.target_column)

    def get(self, name: str) -> NumericFeature | BinaryFeature | CategoricalFeature:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Data quality, splitting, features
# ---------------------------------------------------------------------------
class DataQualityConfig(_Frozen):
    min_rows: Annotated[int, Field(ge=1)] = 500
    min_minority_class_fraction: Annotated[float, Field(gt=0.0, le=0.5)] = 0.05
    # Distinct customers can legitimately share every attribute; only a large share of
    # duplicated feature rows signals an ingestion fault (e.g. a file appended twice).
    max_duplicate_feature_row_fraction: UnitInterval = 0.01


class SplitConfig(_Frozen):
    """Stratified train / validation / test fractions of the full dataset."""

    train_fraction: Fraction = 0.6
    val_fraction: Fraction = 0.2
    test_fraction: Fraction = 0.2

    @model_validator(mode="after")
    def _check_sum(self) -> Self:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"split fractions must sum to 1.0 (got {total:.6f})")
        return self


class FeatureConfig(_Frozen):
    """Domain feature engineering + preprocessing knobs."""

    tenure_column: str = "tenure"
    monthly_charges_column: str = "MonthlyCharges"
    total_charges_column: str = "TotalCharges"
    internet_service_column: str = "InternetService"
    service_columns: tuple[str, ...] = (
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    numeric_impute_strategy: Literal["median", "mean"] = "median"
    clip_quantiles: tuple[UnitInterval, UnitInterval] = (0.005, 0.995)

    @model_validator(mode="after")
    def _check_quantiles(self) -> Self:
        low, high = self.clip_quantiles
        if not low < high:
            raise ValueError("clip_quantiles must satisfy lower < upper")
        return self


# ---------------------------------------------------------------------------
# Training search spaces
# ---------------------------------------------------------------------------
class FloatParam(_Frozen):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if not self.low < self.high:
            raise ValueError("low must be < high")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled ranges require low > 0")
        return self


class IntParam(_Frozen):
    type: Literal["int"] = "int"
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if not self.low < self.high:
            raise ValueError("low must be < high")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled ranges require low > 0")
        return self


ParamValue = str | int | float | bool | None


class CategoricalParam(_Frozen):
    type: Literal["categorical"] = "categorical"
    choices: Annotated[tuple[ParamValue, ...], Field(min_length=1)]


ParamSpec = Annotated[FloatParam | IntParam | CategoricalParam, Field(discriminator="type")]
ModelName = Literal["logistic_regression", "lightgbm", "xgboost"]


class ModelSpec(_Frozen):
    name: ModelName
    is_baseline: bool = False
    n_trials: Annotated[int, Field(ge=1)] = 30
    fixed_params: dict[str, ParamValue] = Field(default_factory=dict)
    search_space: dict[str, ParamSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        overlap = sorted(set(self.fixed_params) & set(self.search_space))
        if overlap:
            raise ValueError(f"{self.name}: params both fixed and searched: {overlap}")
        return self


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="logistic_regression",
        is_baseline=True,
        n_trials=20,
        fixed_params={"max_iter": 5000, "solver": "lbfgs"},
        search_space={
            "C": FloatParam(low=1e-3, high=10.0, log=True),
            "class_weight": CategoricalParam(choices=(None, "balanced")),
        },
    ),
    ModelSpec(
        name="lightgbm",
        n_trials=40,
        fixed_params={
            "objective": "binary",
            "subsample_freq": 1,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": 1,
            "verbose": -1,
        },
        search_space={
            "n_estimators": IntParam(low=100, high=800),
            "learning_rate": FloatParam(low=0.01, high=0.2, log=True),
            "num_leaves": IntParam(low=8, high=64, log=True),
            "min_child_samples": IntParam(low=10, high=120),
            "subsample": FloatParam(low=0.6, high=1.0),
            "colsample_bytree": FloatParam(low=0.5, high=1.0),
            "reg_lambda": FloatParam(low=1e-3, high=10.0, log=True),
        },
    ),
    ModelSpec(
        name="xgboost",
        n_trials=40,
        fixed_params={
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": 1,
        },
        search_space={
            "n_estimators": IntParam(low=100, high=800),
            "learning_rate": FloatParam(low=0.01, high=0.2, log=True),
            "max_depth": IntParam(low=2, high=8),
            "min_child_weight": FloatParam(low=1.0, high=20.0, log=True),
            "subsample": FloatParam(low=0.6, high=1.0),
            "colsample_bytree": FloatParam(low=0.5, high=1.0),
            "reg_lambda": FloatParam(low=1e-3, high=10.0, log=True),
        },
    ),
)


class TrainingConfig(_Frozen):
    models: Annotated[tuple[ModelSpec, ...], Field(min_length=1)] = DEFAULT_MODELS
    cv_folds: Annotated[int, Field(ge=2, le=20)] = 5
    optimize_metric: Literal["average_precision", "roc_auc"] = "average_precision"
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    # F-beta used to pick the decision threshold on the validation split (beta=1 -> F1).
    threshold_beta: Annotated[float, Field(gt=0)] = 1.0

    @model_validator(mode="after")
    def _check_models(self) -> Self:
        names = [model.name for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        if sum(model.is_baseline for model in self.models) != 1:
            raise ValueError("exactly one model must be flagged as the baseline")
        return self


class PromotionConfig(_Frozen):
    """Absolute quality gates a candidate must clear on the hold-out test split."""

    min_roc_auc: UnitInterval = 0.80
    min_pr_auc: UnitInterval = 0.60
    min_f1: UnitInterval = 0.55
    max_brier: UnitInterval = 0.17
    primary_metric: Literal["pr_auc", "roc_auc"] = "pr_auc"
    # Candidate must beat the current production model by at least this margin.
    min_improvement_over_production: Annotated[float, Field(ge=0.0)] = 0.0


class PathsConfig(_Frozen):
    """Output locations, relative to the working directory unless absolute."""

    raw_data_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    registry_dir: Path = Path("artifacts/registry")
    results_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class PipelineConfig(_Frozen):
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    dataset: DatasetConfig = DatasetConfig()
    data_schema: SchemaConfig = SchemaConfig()
    quality: DataQualityConfig = DataQualityConfig()
    split: SplitConfig = SplitConfig()
    features: FeatureConfig = FeatureConfig()
    training: TrainingConfig = TrainingConfig()
    promotion: PromotionConfig = PromotionConfig()
    paths: PathsConfig = PathsConfig()

    @model_validator(mode="after")
    def _check_feature_references(self) -> Self:
        schema = self.data_schema
        known = set(schema.feature_names)
        fc = self.features
        numeric_refs = (fc.tenure_column, fc.monthly_charges_column, fc.total_charges_column)
        for name in numeric_refs:
            if name not in known or not isinstance(schema.get(name), NumericFeature):
                raise ValueError(f"feature engineering expects numeric column '{name}'")
        for name in (*fc.service_columns, fc.internet_service_column):
            if name not in known or not isinstance(schema.get(name), CategoricalFeature):
                raise ValueError(f"feature engineering expects categorical column '{name}'")
        for name in fc.service_columns:
            spec = schema.get(name)
            if isinstance(spec, CategoricalFeature) and "Yes" not in spec.categories:
                raise ValueError(f"service column '{name}' must allow the value 'Yes'")
        return self

    def fingerprint(self) -> str:
        """SHA-256 over every setting that can change a trained model (paths excluded)."""
        payload = self.model_dump(mode="json", exclude={"paths"})
        return sha256_text(canonical_json(payload))


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load a config, applying overrides from a ``.toml`` or ``.json`` file if given.

    Override files may be partial: unspecified sections and fields keep their defaults.
    """
    if path is None:
        return PipelineConfig()
    source = Path(path)
    suffix = source.suffix.lower()
    raw: dict[str, Any]
    if suffix == ".toml":
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    elif suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported config format '{source.suffix}' (use .toml or .json)")
    return PipelineConfig.model_validate(raw)


def seed_everything(seed: int) -> None:
    """Seed the global RNGs. Library objects additionally receive explicit seeds."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeding legacy global state used by third parties
