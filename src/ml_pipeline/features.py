"""Feature engineering and preprocessing, assembled into a single sklearn ``Pipeline``.

Layout of the unified pipeline (serialised as one artifact in later phases)::

    TelcoFeatureEngineer          stateless, row-wise domain features
            │
    ColumnTransformer
      ├─ numeric     SimpleImputer(median) → QuantileClipper → StandardScaler
      ├─ binary      SimpleImputer(most_frequent)
      └─ categorical SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
            │
    estimator (attached in train.py)

Leakage contract: the engineer learns nothing from data, and every stateful step
(imputers, clipper, scaler, encoder) learns its parameters inside ``fit``, so when the
pipeline is fit on a training split / CV fold, nothing from held-out rows can reach it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from ml_pipeline.config import FeatureConfig, PipelineConfig
from ml_pipeline.exceptions import SchemaMismatchError

AVG_MONTHLY_CHARGE = "avg_monthly_charge"
CHARGE_TREND = "charge_trend"
NUM_SERVICES = "num_services"
CHARGE_PER_SERVICE = "charge_per_service"
ENGINEERED_FEATURES: tuple[str, ...] = (
    AVG_MONTHLY_CHARGE,
    CHARGE_TREND,
    NUM_SERVICES,
    CHARGE_PER_SERVICE,
)

_DEFAULTS = FeatureConfig()
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")


def _require_frame(X: object, required: Sequence[str]) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise SchemaMismatchError(
            f"expected a pandas DataFrame with named columns, got {type(X).__name__}"
        )
    missing = [name for name in required if name not in X.columns]
    if missing:
        raise SchemaMismatchError(f"input is missing required column(s): {missing}")
    return X


class TelcoFeatureEngineer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Stateless domain features derived row-by-row (no statistics learned => leak-free).

    * ``avg_monthly_charge``: lifetime charges / tenure (current bill for tenure 0)
    * ``charge_trend``: current bill minus historical average (price-increase signal)
    * ``num_services``: count of subscribed services, including internet
    * ``charge_per_service``: current bill / max(num_services, 1)

    ``fit`` only records the input column layout so ``transform`` can enforce it.
    """

    def __init__(
        self,
        tenure_column: str = _DEFAULTS.tenure_column,
        monthly_charges_column: str = _DEFAULTS.monthly_charges_column,
        total_charges_column: str = _DEFAULTS.total_charges_column,
        internet_service_column: str = _DEFAULTS.internet_service_column,
        service_columns: tuple[str, ...] = _DEFAULTS.service_columns,
    ) -> None:
        self.tenure_column = tenure_column
        self.monthly_charges_column = monthly_charges_column
        self.total_charges_column = total_charges_column
        self.internet_service_column = internet_service_column
        self.service_columns = service_columns

    def _required_columns(self) -> list[str]:
        return [
            self.tenure_column,
            self.monthly_charges_column,
            self.total_charges_column,
            self.internet_service_column,
            *self.service_columns,
        ]

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        frame = _require_frame(X, self._required_columns())
        clashes = sorted(set(ENGINEERED_FEATURES) & set(frame.columns))
        if clashes:
            raise SchemaMismatchError(f"input already contains engineered column(s): {clashes}")
        self.feature_names_in_: npt.NDArray[np.object_] = np.asarray(frame.columns, dtype=object)
        self.n_features_in_: int = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        expected = [str(name) for name in self.feature_names_in_]
        frame = _require_frame(X, expected)
        unexpected = sorted(set(map(str, frame.columns)) - set(expected))
        if unexpected:
            raise SchemaMismatchError(f"input has unexpected column(s): {unexpected}")
        frame = frame[expected]

        tenure = frame[self.tenure_column].astype("float64")
        monthly = frame[self.monthly_charges_column].astype("float64")
        total = frame[self.total_charges_column].astype("float64")
        billed = tenure > 0
        avg_monthly = (total / tenure.where(billed)).where(billed, monthly)

        services = pd.Series(0.0, index=frame.index)
        for name in self.service_columns:
            services += frame[name].eq("Yes").astype("float64")
        internet = frame[self.internet_service_column]
        services += (internet.notna() & internet.ne("No")).astype("float64")

        return frame.assign(
            **{
                AVG_MONTHLY_CHARGE: avg_monthly,
                CHARGE_TREND: monthly - avg_monthly,
                NUM_SERVICES: services,
                CHARGE_PER_SERVICE: monthly / services.clip(lower=1.0),
            }
        )

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray([*self.feature_names_in_, *ENGINEERED_FEATURES], dtype=object)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Winsorise each column to quantile bounds learned during ``fit``.

    Bounds come exclusively from the data passed to ``fit`` (the training split / fold),
    which caps the influence of extreme values at inference time without leaking
    held-out distribution information. NaNs pass through unchanged.
    """

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: object = None) -> Self:
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        values = validate_data(self, X, dtype=np.float64, ensure_all_finite="allow-nan")
        self.lower_bounds_: npt.NDArray[np.float64] = np.nanquantile(
            values, self.lower_quantile, axis=0
        )
        self.upper_bounds_: npt.NDArray[np.float64] = np.nanquantile(
            values, self.upper_quantile, axis=0
        )
        return self

    def transform(self, X: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        values = validate_data(
            self, X, dtype=np.float64, ensure_all_finite="allow-nan", reset=False
        )
        clipped: npt.NDArray[np.float64] = np.clip(values, self.lower_bounds_, self.upper_bounds_)
        return clipped


def onehot_feature_name(feature: str, category: object) -> str:
    """``("Contract", "Month-to-month") -> "Contract__month_to_month"``.

    Restricting names to ``[0-9A-Za-z_]`` keeps them valid for every downstream library
    (LightGBM rejects JSON-special characters; XGBoost rejects ``[``, ``]`` and ``<``).
    Module-level so the fitted encoder stays picklable.
    """
    slug = _NON_ALNUM.sub("_", str(category)).strip("_").lower()
    return f"{feature}__{slug or 'blank'}"


def _check_encoded_names_unique(config: PipelineConfig) -> None:
    for spec in config.data_schema.categorical_features:
        names = [onehot_feature_name(spec.name, category) for category in spec.categories]
        if len(set(names)) != len(names):
            raise ValueError(f"categories of '{spec.name}' collide after name sanitisation")


def build_preprocessor(config: PipelineConfig) -> ColumnTransformer:
    """ColumnTransformer over raw + engineered features (unfitted)."""
    _check_encoded_names_unique(config)
    schema = config.data_schema
    numeric = [spec.name for spec in schema.numeric_features] + list(ENGINEERED_FEATURES)
    binary = [spec.name for spec in schema.binary_features]
    categorical = [spec.name for spec in schema.categorical_features]
    lower, upper = config.features.clip_quantiles

    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(
                            strategy=config.features.numeric_impute_strategy,
                            keep_empty_features=True,
                        ),
                    ),
                    ("clip", QuantileClipper(lower_quantile=lower, upper_quantile=upper)),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if binary:
        transformers.append(
            (
                "binary",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                binary,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float64,
                                feature_name_combiner=onehot_feature_name,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )


def build_feature_engineer(config: PipelineConfig) -> TelcoFeatureEngineer:
    fc = config.features
    return TelcoFeatureEngineer(
        tenure_column=fc.tenure_column,
        monthly_charges_column=fc.monthly_charges_column,
        total_charges_column=fc.total_charges_column,
        internet_service_column=fc.internet_service_column,
        service_columns=fc.service_columns,
    )


def build_feature_pipeline(config: PipelineConfig) -> Pipeline:
    """Unfitted ``engineer → preprocess`` pipeline emitting a float DataFrame."""
    pipeline = Pipeline(
        [
            ("engineer", build_feature_engineer(config)),
            ("preprocess", build_preprocessor(config)),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def build_model_pipeline(config: PipelineConfig, estimator: Any) -> Pipeline:
    """Unfitted ``engineer → preprocess → model`` pipeline: the single serialisable unit."""
    pipeline = Pipeline(
        [
            ("engineer", build_feature_engineer(config)),
            ("preprocess", build_preprocessor(config)),
            ("model", estimator),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline
