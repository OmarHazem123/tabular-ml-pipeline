"""Zero-leakage guarantees.

1. Splits are disjoint, stratified, deterministic and created before any fitting.
2. Every learned preprocessing statistic comes from the training rows only, demonstrated
   with a *poisoned* hold-out that would visibly shift the statistics if it leaked.
3. Inside cross-validation, each fold's pipeline is fit on exactly that fold's training
   rows (verified by a spy transformer, not assumed).
4. Schema violations at fit/transform time raise clean, named errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Self

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ml_pipeline.config import PipelineConfig
from ml_pipeline.data import DataSplits, PreparedDataset, prepare_dataset, split_dataset
from ml_pipeline.exceptions import DataValidationError, SchemaMismatchError
from ml_pipeline.features import (
    ENGINEERED_FEATURES,
    build_feature_engineer,
    build_feature_pipeline,
    build_model_pipeline,
)

# sklearn clones estimators inside cross_validate, so a spy must record into module
# state rather than onto itself (instance attributes would be lost with the clone).
FIT_LOG: defaultdict[str, list[frozenset[str]]] = defaultdict(list)
TRANSFORM_LOG: defaultdict[str, list[frozenset[str]]] = defaultdict(list)


class FitSpy(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Identity transformer that records which entity IDs it is fit / transformed on."""

    def __init__(self, tag: str = "default") -> None:
        self.tag = tag

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        FIT_LOG[self.tag].append(frozenset(map(str, X.index)))
        self.feature_names_in_: npt.NDArray[np.object_] = np.asarray(X.columns, dtype=object)
        self.n_features_in_: int = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        TRANSFORM_LOG[self.tag].append(frozenset(map(str, X.index)))
        return X

    def get_feature_names_out(self, input_features: object = None) -> npt.NDArray[np.object_]:
        return self.feature_names_in_


def _numeric_columns(config: PipelineConfig) -> list[str]:
    return [spec.name for spec in config.data_schema.numeric_features] + list(ENGINEERED_FEATURES)


def _numeric_steps(pipeline: Pipeline) -> dict[str, Any]:
    numeric: Pipeline = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    return dict(numeric.named_steps)


def _poison(frame: pd.DataFrame) -> pd.DataFrame:
    """In-range but extreme values that would shift every statistic if they leaked."""
    return frame.assign(MonthlyCharges=118.0, tenure=72.0, TotalCharges=8000.0)


# ---------------------------------------------------------------------------
# 1. Splitting
# ---------------------------------------------------------------------------
class TestSplits:
    def test_disjoint_and_exhaustive(self, prepared: PreparedDataset, splits: DataSplits) -> None:
        train, val, test = (
            set(part.index) for part in (splits.X_train, splits.X_val, splits.X_test)
        )
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(prepared.frame.index)

    def test_sizes_follow_config(
        self, prepared: PreparedDataset, splits: DataSplits, config: PipelineConfig
    ) -> None:
        n_rows = len(prepared.frame)
        expected = {
            "train": config.split.train_fraction,
            "val": config.split.val_fraction,
            "test": config.split.test_fraction,
        }
        for name, part in splits.summary().items():
            assert abs(part.n_rows - expected[name] * n_rows) <= 1, name

    def test_stratified(self, prepared: PreparedDataset, splits: DataSplits) -> None:
        overall = float(prepared.target.mean())
        for name, part in splits.summary().items():
            assert abs(part.positive_rate - overall) < 0.01, name

    def test_deterministic(
        self, prepared: PreparedDataset, splits: DataSplits, config: PipelineConfig
    ) -> None:
        assert split_dataset(prepared, config).summary() == splits.summary()

    def test_invariant_to_source_row_order(
        self, raw_frame: pd.DataFrame, splits: DataSplits, config: PipelineConfig
    ) -> None:
        shuffled = raw_frame.sample(frac=1.0, random_state=123).reset_index(drop=True)
        again = split_dataset(prepare_dataset(shuffled, config), config)
        assert again.summary() == splits.summary()

    def test_seed_changes_partition(
        self, prepared: PreparedDataset, splits: DataSplits, config: PipelineConfig
    ) -> None:
        other = split_dataset(prepared, config.model_copy(update={"seed": config.seed + 1}))
        assert other.summary()["test"].ids_sha256 != splits.summary()["test"].ids_sha256

    def test_overlapping_splits_are_rejected(self, splits: DataSplits) -> None:
        with pytest.raises(ValueError, match="overlaps"):
            DataSplits(
                X_train=splits.X_train,
                y_train=splits.y_train,
                X_val=splits.X_train.head(5),
                y_val=splits.y_train.head(5),
                X_test=splits.X_test,
                y_test=splits.y_test,
            )

    def test_invalid_data_never_reaches_splitting(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        raw_frame.loc[0, "tenure"] = "not-a-number"
        with pytest.raises(DataValidationError):
            split_dataset(prepare_dataset(raw_frame, config), config)


# ---------------------------------------------------------------------------
# 2. Learned statistics come from training rows only
# ---------------------------------------------------------------------------
class TestFitOnTrainingOnly:
    def test_statistics_equal_train_values_and_differ_from_full_data(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        holdout = pd.concat([_poison(splits.X_val), _poison(splits.X_test)])
        full = pd.concat([splits.X_train, holdout])

        fitted_on_train = build_feature_pipeline(config).fit(splits.X_train)
        fitted_on_full = build_feature_pipeline(config).fit(full)

        # Recompute every numeric statistic independently from the training rows alone.
        columns = _numeric_columns(config)
        engineered = build_feature_engineer(config).fit(splits.X_train).transform(splits.X_train)
        train_numeric = engineered[columns].to_numpy(dtype=np.float64)
        medians = np.nanmedian(train_numeric, axis=0)
        imputed = np.where(np.isnan(train_numeric), medians, train_numeric)
        low_q, high_q = config.features.clip_quantiles
        lower = np.quantile(imputed, low_q, axis=0)
        upper = np.quantile(imputed, high_q, axis=0)
        clipped = np.clip(imputed, lower, upper)

        steps = _numeric_steps(fitted_on_train)
        np.testing.assert_allclose(steps["impute"].statistics_, medians)
        np.testing.assert_allclose(steps["clip"].lower_bounds_, lower)
        np.testing.assert_allclose(steps["clip"].upper_bounds_, upper)
        np.testing.assert_allclose(steps["scale"].mean_, clipped.mean(axis=0))
        np.testing.assert_allclose(steps["scale"].scale_, clipped.std(axis=0))

        # Sanity check that the poisoned hold-out WOULD have changed them had it leaked.
        leaked = _numeric_steps(fitted_on_full)
        monthly = columns.index("MonthlyCharges")
        assert not np.isclose(leaked["impute"].statistics_[monthly], medians[monthly])
        assert not np.isclose(leaked["scale"].mean_[monthly], clipped.mean(axis=0)[monthly])

    def test_holdout_only_category_is_not_learned(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        unseen = "Mailed check"
        train = splits.X_train[splits.X_train["PaymentMethod"] != unseen]
        pipeline = build_feature_pipeline(config).fit(train)

        encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
        learned = encoder.named_steps["onehot"].categories_
        payment_index = [spec.name for spec in config.data_schema.categorical_features].index(
            "PaymentMethod"
        )
        assert unseen not in set(learned[payment_index])

        holdout = splits.X_test[splits.X_test["PaymentMethod"] == unseen]
        assert not holdout.empty
        encoded = pipeline.transform(holdout)
        payment_columns = [c for c in encoded.columns if c.startswith("PaymentMethod__")]
        assert "PaymentMethod__mailed_check" not in encoded.columns
        assert (encoded[payment_columns] == 0.0).all().all()

    def test_transform_never_mutates_fitted_state(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        pipeline = build_feature_pipeline(config).fit(splits.X_train)
        fingerprint = joblib.hash(pipeline)
        first = pipeline.transform(_poison(splits.X_test))
        second = pipeline.transform(_poison(splits.X_test))
        assert joblib.hash(pipeline) == fingerprint
        pd.testing.assert_frame_equal(first, second)

    def test_fit_does_not_mutate_caller_frame(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        before = splits.X_train.copy(deep=True)
        build_feature_pipeline(config).fit_transform(splits.X_train)
        pd.testing.assert_frame_equal(splits.X_train, before)


# ---------------------------------------------------------------------------
# 3. Cross-validation folds
# ---------------------------------------------------------------------------
def test_each_cv_fold_is_fit_on_its_training_rows_only(
    splits: DataSplits, config: PipelineConfig
) -> None:
    tag = "cv-fold-audit"
    FIT_LOG.pop(tag, None)
    TRANSFORM_LOG.pop(tag, None)
    model_pipeline = build_model_pipeline(config, LogisticRegression(max_iter=5000))
    audited = Pipeline([("spy", FitSpy(tag)), *model_pipeline.steps])
    audited.set_output(transform="pandas")
    folds = StratifiedKFold(n_splits=config.training.cv_folds, shuffle=True, random_state=0)

    result = cross_validate(
        audited,
        splits.X_train,
        splits.y_train,
        cv=folds,
        scoring="roc_auc",
        return_indices=True,
        n_jobs=1,
    )

    ids = splits.X_train.index.astype(str).to_numpy()
    fits = FIT_LOG[tag]
    assert len(fits) == config.training.cv_folds
    outside = set(map(str, splits.X_val.index)) | set(map(str, splits.X_test.index))
    for fold, (train_idx, test_idx) in enumerate(
        zip(result["indices"]["train"], result["indices"]["test"], strict=True)
    ):
        assert fits[fold] == frozenset(ids[train_idx]), f"fold {fold} fit on wrong rows"
        assert fits[fold].isdisjoint(ids[test_idx]), f"fold {fold} saw its validation rows"
        assert fits[fold].isdisjoint(outside), f"fold {fold} saw val/test split rows"
    # Validation rows of every fold were only ever transformed, never fit.
    fold_val_sets = [frozenset(ids[idx]) for idx in result["indices"]["test"]]
    assert all(rows in TRANSFORM_LOG[tag] for rows in fold_val_sets)
    assert np.isfinite(result["test_score"]).all()


# ---------------------------------------------------------------------------
# 4. Schema violations at fit / transform time
# ---------------------------------------------------------------------------
class TestPipelineSchemaGuards:
    @pytest.fixture
    def fitted(self, splits: DataSplits, config: PipelineConfig) -> Pipeline:
        return build_feature_pipeline(config).fit(splits.X_train)

    def test_missing_column_at_transform(self, fitted: Pipeline, splits: DataSplits) -> None:
        with pytest.raises(SchemaMismatchError, match="tenure"):
            fitted.transform(splits.X_test.drop(columns=["tenure"]))

    def test_unexpected_column_at_transform(self, fitted: Pipeline, splits: DataSplits) -> None:
        with pytest.raises(SchemaMismatchError, match="leak"):
            fitted.transform(splits.X_test.assign(leak=1.0))

    def test_non_frame_input(self, fitted: Pipeline, splits: DataSplits) -> None:
        with pytest.raises(SchemaMismatchError, match="DataFrame"):
            fitted.transform(splits.X_test.to_numpy())

    def test_missing_column_at_fit(self, splits: DataSplits, config: PipelineConfig) -> None:
        with pytest.raises(SchemaMismatchError, match="MonthlyCharges"):
            build_feature_pipeline(config).fit(splits.X_train.drop(columns=["MonthlyCharges"]))

    def test_column_order_is_irrelevant(self, fitted: Pipeline, splits: DataSplits) -> None:
        reordered = splits.X_test[list(reversed(splits.X_test.columns))]
        pd.testing.assert_frame_equal(fitted.transform(reordered), fitted.transform(splits.X_test))
