"""The deployable model: one serialisable object, used by the CLI and the API.

``ChurnModel`` bundles:

* the fitted ``engineer → preprocess → estimator`` pipeline (fit on the TRAIN split),
* a probability calibrator fit on the VALIDATION split (the pipeline is frozen inside it),
* the decision threshold chosen on out-of-fold calibrated validation probabilities,
* a reference profile of the TRAIN split (for out-of-range / unseen-category warnings),
* a small TRAIN background sample for SHAP.

Reason codes are SHAP values of the *uncalibrated* log-odds score. Calibration is a
monotonic map of that score, so the ranking and direction of contributions carry over to
the calibrated probability. One-hot columns are summed back into their source feature,
which is valid because SHAP values are additive.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ml_pipeline.config import BinaryFeature, NumericFeature, SchemaConfig
from ml_pipeline.exceptions import SchemaMismatchError
from ml_pipeline.features import onehot_feature_name

FloatArray = npt.NDArray[np.float64]
ReasonValue = str | float | None


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NumericRange(_Model):
    min: float
    max: float


class ReferenceProfile(_Model):
    """Envelope of the training split. Inputs outside it get warnings (never rejections:
    schema bounds already reject impossible values, this flags *unfamiliar* ones)."""

    n_rows: int
    numeric: dict[str, NumericRange]
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, schema: SchemaConfig) -> ReferenceProfile:
        numeric: dict[str, NumericRange] = {}
        categories: dict[str, tuple[str, ...]] = {}
        for spec in schema.features:
            column = frame[spec.name].dropna()
            if isinstance(spec, NumericFeature | BinaryFeature):
                numeric[spec.name] = NumericRange(min=float(column.min()), max=float(column.max()))
            else:
                categories[spec.name] = tuple(sorted(str(value) for value in column.unique()))
        return cls(n_rows=len(frame), numeric=numeric, categories=categories)


class Reason(_Model):
    feature: str
    value: ReasonValue
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class InputWarning(_Model):
    feature: str
    kind: Literal["out_of_training_range", "unseen_category"]
    message: str


class Prediction(_Model):
    churn_probability: float
    churn_prediction: bool
    threshold: float
    reasons: tuple[Reason, ...]
    warnings: tuple[InputWarning, ...]


def _display(value: object) -> ReasonValue:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return None if math.isnan(number) else round(number, 4)
    return str(value)


def model_margin(estimator: Any, features: pd.DataFrame) -> FloatArray:
    """Uncalibrated log-odds score, i.e. exactly the quantity SHAP decomposes."""
    if isinstance(estimator, LogisticRegression):
        scores = estimator.decision_function(features)
    elif isinstance(estimator, LGBMClassifier):
        scores = estimator.predict(features, raw_score=True)
    elif isinstance(estimator, XGBClassifier):
        scores = estimator.predict(features, output_margin=True)
    else:
        raise TypeError(f"unsupported estimator type {type(estimator).__name__}")
    return np.asarray(scores, dtype=np.float64)


def source_feature_map(pipeline: Pipeline) -> dict[str, str]:
    """Map every model-input column back to the raw/engineered feature it came from."""
    preprocess = pipeline.named_steps["preprocess"]
    mapping: dict[str, str] = {}
    for _, transformer, columns in preprocess.transformers_:
        if isinstance(transformer, str):  # "drop" / "passthrough" remainder
            continue
        encoder = transformer[-1] if isinstance(transformer, Pipeline) else transformer
        if isinstance(encoder, OneHotEncoder):
            for feature, categories in zip(
                encoder.feature_names_in_, encoder.categories_, strict=True
            ):
                for category in categories:
                    mapping[onehot_feature_name(str(feature), category)] = str(feature)
        else:
            for column in columns:
                mapping[str(column)] = str(column)
    names = [str(name) for name in preprocess.get_feature_names_out()]
    unmapped = [name for name in names if name not in mapping]
    if unmapped:
        raise RuntimeError(f"could not map model inputs to source features: {unmapped}")
    return {name: mapping[name] for name in names}


class ChurnModel:
    """Calibrated churn classifier with reason codes and input-drift warnings."""

    def __init__(
        self,
        *,
        model_name: str,
        pipeline: Pipeline,
        calibrator: CalibratedClassifierCV,
        threshold: float,
        input_columns: tuple[str, ...],
        reference: ReferenceProfile,
        background: pd.DataFrame,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self.model_name = model_name
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.input_columns = input_columns
        self.reference = reference
        self.background = background
        self._explainer: Any = None
        self._source_map: dict[str, str] | None = None

    # -- pickling: explainers are rebuilt lazily, never serialised -------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None
        state["_source_map"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -- core inference ---------------------------------------------------------------
    @property
    def estimator(self) -> Any:
        return self.pipeline.named_steps["model"]

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.input_columns if name not in X.columns]
        if missing:
            raise SchemaMismatchError(f"input is missing required column(s): {missing}")
        return X[list(self.input_columns)]

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Calibrated probability of churn."""
        proba = self.calibrator.predict_proba(self._select(X))[:, 1]
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.bool_]:
        return self.predict_proba(X) >= self.threshold

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model-input features (after engineering + preprocessing)."""
        transformed: pd.DataFrame = self.pipeline[:-1].transform(self._select(X))
        return transformed

    def margin(self, X: pd.DataFrame) -> FloatArray:
        return model_margin(self.estimator, self.transform(X))

    # -- explanations -----------------------------------------------------------------
    @property
    def source_map(self) -> dict[str, str]:
        if self._source_map is None:
            self._source_map = source_feature_map(self.pipeline)
        return self._source_map

    def _get_explainer(self) -> Any:
        if self._explainer is None:
            if isinstance(self.estimator, LogisticRegression):
                # Explicit masker: shap's default silently subsamples the background to
                # 100 rows, which would make explanations depend on a random draw.
                masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
                self._explainer = shap.LinearExplainer(self.estimator, masker)
            else:
                self._explainer = shap.TreeExplainer(self.estimator)
        return self._explainer

    def warm_up(self) -> None:
        """Build caches eagerly (called once at API start-up)."""
        self._get_explainer()
        _ = self.source_map

    def shap_values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, FloatArray]:
        """Per-model-input SHAP values (log-odds) and the per-row base value."""
        features = self.transform(X)
        explanation = self._get_explainer()(features)
        values = np.asarray(explanation.values, dtype=np.float64)
        base = np.asarray(explanation.base_values, dtype=np.float64)
        if values.ndim == 3:  # some explainer/model combos return one slice per class
            values = values[..., 1]
        if base.ndim == 2:
            base = base[:, 1]
        base = np.broadcast_to(base, (len(features),)).astype(np.float64)
        return pd.DataFrame(values, index=features.index, columns=features.columns), base

    def grouped_shap(self, X: pd.DataFrame) -> tuple[pd.DataFrame, FloatArray]:
        """SHAP values summed per source feature (one-hot groups collapsed)."""
        values, base = self.shap_values(X)
        groups = [self.source_map[str(column)] for column in values.columns]
        grouped = values.T.groupby(groups, sort=False).sum().T
        return grouped, base

    def explain(self, X: pd.DataFrame, top_k: int) -> list[tuple[Reason, ...]]:
        frame = self._select(X)
        grouped, _ = self.grouped_shap(frame)
        displayed = self.pipeline.named_steps["engineer"].transform(frame)
        explanations: list[tuple[Reason, ...]] = []
        for position in range(len(frame)):
            row = grouped.iloc[position]
            order = row.abs().sort_values(ascending=False, kind="mergesort").index[:top_k]
            explanations.append(
                tuple(
                    Reason(
                        feature=str(feature),
                        value=_display(displayed.iloc[position][feature]),
                        contribution=round(float(row[feature]), 6),
                        direction="increases_risk" if row[feature] > 0 else "decreases_risk",
                    )
                    for feature in order
                )
            )
        return explanations

    # -- input drift --------------------------------------------------------------------
    def input_warnings(self, X: pd.DataFrame) -> list[tuple[InputWarning, ...]]:
        frame = self._select(X)
        found: list[list[InputWarning]] = [[] for _ in range(len(frame))]
        for feature, bounds in self.reference.numeric.items():
            values = frame[feature].to_numpy(dtype=np.float64)
            for position in np.flatnonzero((values < bounds.min) | (values > bounds.max)):
                found[position].append(
                    InputWarning(
                        feature=feature,
                        kind="out_of_training_range",
                        message=(
                            f"{feature}={values[position]:g} is outside the training range "
                            f"[{bounds.min:g}, {bounds.max:g}]"
                        ),
                    )
                )
        for feature, seen in self.reference.categories.items():
            values_obj = frame[feature].astype(str).to_numpy()
            for position in np.flatnonzero(~np.isin(values_obj, seen)):
                found[position].append(
                    InputWarning(
                        feature=feature,
                        kind="unseen_category",
                        message=f"{feature}={values_obj[position]!r} never occurred in training",
                    )
                )
        return [tuple(items) for items in found]

    # -- combined -----------------------------------------------------------------------
    def predict_records(self, X: pd.DataFrame, top_k: int) -> list[Prediction]:
        frame = self._select(X)
        probabilities = self.predict_proba(frame)
        reasons = self.explain(frame, top_k)
        warnings = self.input_warnings(frame)
        return [
            Prediction(
                churn_probability=round(float(probability), 6),
                churn_prediction=bool(probability >= self.threshold),
                threshold=round(self.threshold, 6),
                reasons=row_reasons,
                warnings=row_warnings,
            )
            for probability, row_reasons, row_warnings in zip(
                probabilities, reasons, warnings, strict=True
            )
        ]
