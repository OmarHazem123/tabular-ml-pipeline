"""Feature engineering + preprocessing contract."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_pipeline.config import (
    DEFAULT_FEATURES,
    CategoricalFeature,
    PipelineConfig,
    SchemaConfig,
)
from ml_pipeline.data import DataSplits
from ml_pipeline.exceptions import SchemaMismatchError
from ml_pipeline.features import (
    AVG_MONTHLY_CHARGE,
    CHARGE_PER_SERVICE,
    CHARGE_TREND,
    ENGINEERED_FEATURES,
    NUM_SERVICES,
    QuantileClipper,
    TelcoFeatureEngineer,
    build_feature_pipeline,
    build_model_pipeline,
    build_preprocessor,
    onehot_feature_name,
)

SAFE_NAME = re.compile(r"^[0-9A-Za-z_]+$")
ADDONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport")
STREAMING = ("StreamingTV", "StreamingMovies")


def _customer(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "tenure": 10.0,
        "MonthlyCharges": 50.0,
        "TotalCharges": 400.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        **dict.fromkeys(ADDONS, "No"),
        **dict.fromkeys(STREAMING, "No"),
    }
    row.update(overrides)
    return row


class TestTelcoFeatureEngineer:
    def test_hand_computed_values(self) -> None:
        frame = pd.DataFrame(
            [
                _customer(),
                _customer(tenure=0.0, TotalCharges=0.0, MonthlyCharges=80.0),
                _customer(
                    PhoneService="No",
                    MultipleLines="No phone service",
                    InternetService="No",
                    **dict.fromkeys((*ADDONS, *STREAMING), "No internet service"),
                ),
                _customer(
                    MultipleLines="Yes",
                    InternetService="Fiber optic",
                    **dict.fromkeys((*ADDONS, *STREAMING), "Yes"),
                ),
            ]
        )
        out = TelcoFeatureEngineer().fit_transform(frame)

        np.testing.assert_allclose(out[AVG_MONTHLY_CHARGE], [40.0, 80.0, 40.0, 40.0])
        np.testing.assert_allclose(out[CHARGE_TREND], [10.0, 0.0, 10.0, 10.0])
        np.testing.assert_allclose(out[NUM_SERVICES], [2.0, 2.0, 0.0, 9.0])
        np.testing.assert_allclose(out[CHARGE_PER_SERVICE], [25.0, 40.0, 50.0, 50.0 / 9.0])

    def test_missing_total_charges_propagates_as_nan_for_pipeline_imputation(self) -> None:
        out = TelcoFeatureEngineer().fit_transform(pd.DataFrame([_customer(TotalCharges=np.nan)]))
        assert np.isnan(out.loc[0, AVG_MONTHLY_CHARGE])
        assert np.isnan(out.loc[0, CHARGE_TREND])

    def test_is_stateless(self, splits: DataSplits) -> None:
        on_train = TelcoFeatureEngineer().fit(splits.X_train).transform(splits.X_val)
        on_test = TelcoFeatureEngineer().fit(splits.X_test).transform(splits.X_val)
        pd.testing.assert_frame_equal(on_train, on_test)

    def test_rejects_engineered_name_clash(self, splits: DataSplits) -> None:
        with pytest.raises(SchemaMismatchError, match=NUM_SERVICES):
            TelcoFeatureEngineer().fit(splits.X_train.assign(**{NUM_SERVICES: 1.0}))

    def test_feature_names_out(self, splits: DataSplits) -> None:
        engineer = TelcoFeatureEngineer().fit(splits.X_train)
        names = list(engineer.get_feature_names_out())
        assert names == [*splits.X_train.columns, *ENGINEERED_FEATURES]


class TestQuantileClipper:
    def test_learns_bounds_and_preserves_nan(self) -> None:
        values = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
        clipper = QuantileClipper(lower_quantile=0.25, upper_quantile=0.75).fit(values)
        np.testing.assert_allclose(clipper.lower_bounds_, [2.0])
        np.testing.assert_allclose(clipper.upper_bounds_, [4.0])
        out = clipper.transform(np.array([[-50.0], [np.nan], [3.5], [1e9]]))
        np.testing.assert_allclose(out, [[2.0], [np.nan], [3.5], [4.0]])

    @pytest.mark.parametrize(("low", "high"), [(0.5, 0.5), (0.9, 0.1), (-0.1, 0.9), (0.1, 1.1)])
    def test_rejects_invalid_quantiles(self, low: float, high: float) -> None:
        with pytest.raises(ValueError, match="quantiles"):
            QuantileClipper(lower_quantile=low, upper_quantile=high).fit(np.ones((3, 1)))

    def test_rejects_feature_count_change(self) -> None:
        clipper = QuantileClipper().fit(np.ones((4, 2)))
        with pytest.raises(ValueError, match="features"):
            clipper.transform(np.ones((4, 3)))


@pytest.mark.parametrize(
    ("feature", "category", "expected"),
    [
        ("Contract", "Month-to-month", "Contract__month_to_month"),
        ("PaymentMethod", "Bank transfer (automatic)", "PaymentMethod__bank_transfer_automatic"),
        ("OnlineSecurity", "No internet service", "OnlineSecurity__no_internet_service"),
        ("X", "!!!", "X__blank"),
    ],
)
def test_onehot_feature_name(feature: str, category: str, expected: str) -> None:
    assert onehot_feature_name(feature, category) == expected


def test_name_collisions_after_sanitisation_are_rejected() -> None:
    features = tuple(
        CategoricalFeature(name="gender", categories=("A-B", "A B"))
        if spec.name == "gender"
        else spec
        for spec in DEFAULT_FEATURES
    )
    config = PipelineConfig(data_schema=SchemaConfig(features=features))
    with pytest.raises(ValueError, match="collide"):
        build_preprocessor(config)


class TestFeaturePipeline:
    def test_output_contract(self, splits: DataSplits, config: PipelineConfig) -> None:
        pipeline = build_feature_pipeline(config)
        train = pipeline.fit_transform(splits.X_train)
        val = pipeline.transform(splits.X_val)

        assert list(train.columns) == list(pipeline.get_feature_names_out())
        assert list(val.columns) == list(train.columns)
        assert train.columns.is_unique
        assert all(SAFE_NAME.match(name) for name in train.columns)
        assert (train.dtypes == np.float64).all()
        assert not train.isna().any().any()
        assert not val.isna().any().any()
        pd.testing.assert_index_equal(val.index, splits.X_val.index)

    def test_imputes_missing_values_at_inference(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        pipeline = build_feature_pipeline(config).fit(splits.X_train)
        holes = splits.X_val.head(5).assign(TotalCharges=np.nan, Contract=np.nan)
        out = pipeline.transform(holes)
        assert np.isfinite(out.to_numpy()).all()

    def test_round_trips_through_joblib(
        self, tmp_path: Path, splits: DataSplits, config: PipelineConfig
    ) -> None:
        pipeline = build_feature_pipeline(config).fit(splits.X_train)
        path = tmp_path / "features.joblib"
        joblib.dump(pipeline, path)
        restored = joblib.load(path)
        pd.testing.assert_frame_equal(
            restored.transform(splits.X_test), pipeline.transform(splits.X_test)
        )


def test_model_pipeline_is_deterministic(splits: DataSplits, config: PipelineConfig) -> None:
    def fit_predict() -> np.ndarray:
        pipeline = build_model_pipeline(config, LogisticRegression(max_iter=5000))
        pipeline.fit(splits.X_train, splits.y_train)
        proba: np.ndarray = pipeline.predict_proba(splits.X_test)[:, 1]
        return proba

    first, second = fit_predict(), fit_predict()
    np.testing.assert_array_equal(first, second)
    assert ((first > 0.0) & (first < 1.0)).all()
