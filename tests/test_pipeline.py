"""End-to-end training, serialisation, SHA-256 verification, determinism and the gate."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from ml_pipeline.config import PathsConfig, PipelineConfig, PromotionConfig
from ml_pipeline.data import DataSplits, PreparedDataset, load_dataset, split_dataset
from ml_pipeline.evaluate import EvaluationReport, run_evaluation
from ml_pipeline.exceptions import PipelineError
from ml_pipeline.features import ENGINEERED_FEATURES, build_model_pipeline
from ml_pipeline.io_utils import sha256_file
from ml_pipeline.metrics import (
    ClassificationMetrics,
    compute_metrics,
    expected_calibration_error,
    reliability_table,
    select_threshold,
)
from ml_pipeline.registry import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    ModelIntegrityError,
    ModelRegistry,
    RegistryError,
    dump_model,
    evaluate_promotion,
    load_model_dir,
    load_verified_model,
    suggest_promotion_floors,
)
from ml_pipeline.train import calibrate, load_candidate, run_training
from tests.conftest import TrainedPipeline

MODEL_NAMES = ("logistic_regression", "lightgbm", "xgboost")


def _test_split(trained: TrainedPipeline) -> DataSplits:
    config = trained.config
    return split_dataset(load_dataset(config, data_path=trained.data_path), config)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TestTraining:
    def test_every_model_is_tuned_and_saved(self, trained: TrainedPipeline) -> None:
        run = trained.run
        assert {summary.model_name for summary in run.candidates} == set(MODEL_NAMES)
        for summary in run.candidates:
            assert len(summary.cv["pr_auc"].folds) == trained.config.training.cv_folds
            assert 0.0 < summary.cv["roc_auc"].mean <= 1.0
            assert 0.0 < summary.threshold < 1.0
            artifact = trained.run_dir / summary.artifact.filename
            assert sha256_file(artifact) == summary.artifact.sha256

    def test_champion_has_best_cv_score(self, trained: TrainedPipeline) -> None:
        run = trained.run
        best = max(summary.cv["pr_auc"].mean for summary in run.candidates)
        assert run.candidate(run.champion).cv["pr_auc"].mean == best

    def test_threshold_is_chosen_on_out_of_fold_validation_probabilities(
        self, trained: TrainedPipeline
    ) -> None:
        summary = trained.run.baseline
        model = load_candidate(trained.run_dir, summary)
        splits = _test_split(trained)
        _, out_of_fold = calibrate(model.pipeline, splits.X_val, splits.y_val, trained.config)
        threshold, _ = select_threshold(
            splits.y_val, out_of_fold, trained.config.training.threshold_beta
        )
        assert threshold == summary.threshold == model.threshold
        assert compute_metrics(splits.y_val, out_of_fold, threshold) == summary.validation

    def test_suggested_floors_follow_the_documented_rule(self, trained: TrainedPipeline) -> None:
        run = trained.run
        baseline = run.baseline
        expected = suggest_promotion_floors(
            cv_roc_auc=baseline.cv["roc_auc"],
            cv_pr_auc=baseline.cv["pr_auc"],
            validation=baseline.validation,
            fold_rows=run.lineage.splits["train"].n_rows // trained.config.training.cv_folds,
            test_rows=run.lineage.splits["test"].n_rows,
        )
        assert run.suggested_floors == expected
        assert expected.min_pr_auc <= baseline.cv["pr_auc"].mean

    def test_lineage_is_recorded(self, trained: TrainedPipeline) -> None:
        lineage = trained.run.lineage
        assert lineage.config_fingerprint == trained.config.fingerprint()
        assert lineage.dataset is not None
        assert lineage.dataset.sha256 == sha256_file(trained.data_path)
        assert set(lineage.splits) == {"train", "val", "test"}
        assert {"python", "scikit-learn", "lightgbm", "xgboost-cpu", "shap"} <= set(
            lineage.environment
        )

    def test_calibration_wraps_the_frozen_train_pipeline(self, trained: TrainedPipeline) -> None:
        model = load_candidate(trained.run_dir, trained.run.baseline)
        frozen = model.calibrator.estimator
        assert isinstance(frozen, FrozenEstimator)
        assert frozen.estimator is model.pipeline

    def test_calibration_never_refits_the_pipeline(
        self, splits: DataSplits, config: PipelineConfig
    ) -> None:
        pipeline = build_model_pipeline(config, LogisticRegression(max_iter=2000))
        pipeline.fit(splits.X_train, splits.y_train)
        before = joblib.hash(pipeline)
        calibrator, out_of_fold = calibrate(pipeline, splits.X_val, splits.y_val, config)
        assert joblib.hash(pipeline) == before
        assert out_of_fold.shape == (len(splits.y_val),)
        assert ((out_of_fold >= 0.0) & (out_of_fold <= 1.0)).all()
        assert len(calibrator.calibrated_classifiers_) == 1

    def test_training_never_reads_the_test_split(
        self,
        trained: TrainedPipeline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replace X_test with garbage: any fit/transform/predict on it would crash."""
        real_split = split_dataset

        def poisoned_split(dataset: PreparedDataset, config: PipelineConfig) -> DataSplits:
            splits = real_split(dataset, config)
            garbage = pd.DataFrame(
                "POISON", index=splits.X_test.index, columns=splits.X_test.columns
            )
            return DataSplits(
                X_train=splits.X_train,
                y_train=splits.y_train,
                X_val=splits.X_val,
                y_val=splits.y_val,
                X_test=garbage,
                y_test=splits.y_test,
            )

        monkeypatch.setattr("ml_pipeline.train.split_dataset", poisoned_split)
        config = trained.config.model_copy(
            update={
                "paths": PathsConfig(runs_dir=tmp_path / "runs", registry_dir=tmp_path / "registry")
            }
        )
        run = run_training(config, data_path=trained.data_path, max_trials=1)
        assert run.champion in MODEL_NAMES


# ---------------------------------------------------------------------------
# Serialisation & integrity
# ---------------------------------------------------------------------------
class TestSerialisation:
    def test_registry_artifact_round_trips(self, trained: TrainedPipeline) -> None:
        registry = ModelRegistry(trained.config.paths.registry_dir)
        model, metadata = registry.load(trained.report.model_id)
        candidate = load_candidate(trained.run_dir, trained.run.candidate(trained.run.champion))
        X_test = _test_split(trained).X_test
        np.testing.assert_array_equal(model.predict_proba(X_test), candidate.predict_proba(X_test))
        artifact = registry.model_dir(metadata.model_id) / metadata.artifact.filename
        assert sha256_file(artifact) == metadata.artifact.sha256
        assert artifact.stat().st_size == metadata.artifact.size_bytes

    def test_tampered_artifact_is_never_deserialised(
        self, trained: TrainedPipeline, tmp_path: Path
    ) -> None:
        source = ModelRegistry(trained.config.paths.registry_dir).model_dir(trained.report.model_id)
        copy = tmp_path / "model"
        shutil.copytree(source, copy)
        artifact = copy / MODEL_FILENAME
        data = bytearray(artifact.read_bytes())
        data[len(data) // 2] ^= 0xFF
        artifact.write_bytes(bytes(data))
        with pytest.raises(ModelIntegrityError, match="refusing to deserialise"):
            load_model_dir(copy)

    def test_missing_artifact_or_metadata(self, trained: TrainedPipeline, tmp_path: Path) -> None:
        source = ModelRegistry(trained.config.paths.registry_dir).model_dir(trained.report.model_id)
        copy = tmp_path / "model"
        shutil.copytree(source, copy)
        (copy / MODEL_FILENAME).unlink()
        with pytest.raises(ModelIntegrityError, match="cannot read model artifact"):
            load_model_dir(copy)
        (copy / METADATA_FILENAME).unlink()
        with pytest.raises(ModelIntegrityError, match="cannot read model metadata"):
            load_model_dir(copy)

    def test_foreign_object_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"weights": [1, 2, 3]}, path)
        with pytest.raises(ModelIntegrityError, match="does not contain a ChurnModel"):
            load_verified_model(path, sha256_file(path))

    def test_explainer_cache_is_not_pickled(self, trained: TrainedPipeline) -> None:
        model = load_candidate(trained.run_dir, trained.run.candidate("lightgbm"))
        model.warm_up()
        restored = pickle.loads(pickle.dumps(model))  # noqa: S301 - our own object
        assert restored._explainer is None
        assert restored._source_map is None

    def test_dump_is_atomic_and_hashed(self, trained: TrainedPipeline, tmp_path: Path) -> None:
        model = load_candidate(trained.run_dir, trained.run.baseline)
        info = dump_model(model, tmp_path / "m.joblib")
        assert info.sha256 == sha256_file(tmp_path / "m.joblib")
        assert [p.name for p in tmp_path.iterdir()] == ["m.joblib"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_retraining_reproduces_params_scores_and_predictions(
    trained: TrainedPipeline, tmp_path: Path
) -> None:
    config = trained.config.model_copy(
        update={"paths": PathsConfig(runs_dir=tmp_path / "runs", registry_dir=tmp_path / "reg")}
    )
    rerun = run_training(config, data_path=trained.data_path)
    X_test = _test_split(trained).X_test
    assert rerun.champion == trained.run.champion
    for original in trained.run.candidates:
        again = rerun.candidate(original.model_name)
        assert again.best_params == original.best_params
        assert again.cv == original.cv
        assert again.threshold == original.threshold
        first = load_candidate(trained.run_dir, original)
        second = load_candidate(config.paths.runs_dir / rerun.run_id, again)
        np.testing.assert_array_equal(first.predict_proba(X_test), second.predict_proba(X_test))


# ---------------------------------------------------------------------------
# SHAP reason codes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_shap_groups_add_up_to_the_model_margin(trained: TrainedPipeline, name: str) -> None:
    model = load_candidate(trained.run_dir, trained.run.candidate(name))
    X = _test_split(trained).X_test.head(40)
    grouped, base = model.grouped_shap(X)
    np.testing.assert_allclose(grouped.sum(axis=1).to_numpy() + base, model.margin(X), atol=1e-5)
    allowed = set(trained.config.data_schema.feature_names) | set(ENGINEERED_FEATURES)
    assert set(grouped.columns) <= allowed


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_reason_codes_are_ranked_and_signed(trained: TrainedPipeline, name: str) -> None:
    model = load_candidate(trained.run_dir, trained.run.candidate(name))
    for reasons in model.explain(_test_split(trained).X_test.head(10), top_k=4):
        assert len(reasons) == 4
        magnitudes = [abs(reason.contribution) for reason in reasons]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for reason in reasons:
            expected = "increases_risk" if reason.contribution > 0 else "decreases_risk"
            assert reason.direction == expected


# ---------------------------------------------------------------------------
# Evaluation & reports
# ---------------------------------------------------------------------------
class TestEvaluation:
    def test_reports_are_written_with_lf(self, trained: TrainedPipeline) -> None:
        results = trained.config.paths.results_dir
        metrics_path, report_path = results / "metrics.json", results / "report.md"
        for path in (metrics_path, report_path):
            assert b"\r" not in path.read_bytes()
        parsed = EvaluationReport.model_validate(json.loads(metrics_path.read_text("utf-8")))
        assert parsed == trained.report
        text = report_path.read_text(encoding="utf-8")
        for heading in ("## Benchmark", "## Promotion gate", "## Global feature importance"):
            assert heading in text

    def test_benchmark_covers_every_model(self, trained: TrainedPipeline) -> None:
        rows = trained.report.benchmark
        assert {row.model_name for row in rows} == set(MODEL_NAMES)
        assert sum(row.is_champion for row in rows) == 1
        assert sum(row.is_baseline for row in rows) == 1
        assert all(row.test.n_rows == trained.run.lineage.splits["test"].n_rows for row in rows)

    def test_champion_was_promoted(self, trained: TrainedPipeline) -> None:
        registry = ModelRegistry(trained.config.paths.registry_dir)
        assert trained.report.promotion.approved
        assert registry.production_id() == trained.report.model_id
        metadata = registry.metadata(trained.report.model_id)
        assert metadata.stage == "production"
        assert metadata.test == trained.report.test

    def test_reevaluation_is_idempotent(self, trained: TrainedPipeline) -> None:
        registry = ModelRegistry(trained.config.paths.registry_dir)
        events_before = registry.index().events
        again = run_evaluation(trained.config, data_path=trained.data_path)
        assert again.promotion == trained.report.promotion
        assert registry.index().events == events_before

    def test_refuses_data_that_differs_from_training(
        self, trained: TrainedPipeline, tmp_path: Path
    ) -> None:
        frame = pd.read_csv(trained.data_path, dtype=str, keep_default_na=False)
        altered = tmp_path / "altered.csv"
        frame.iloc[:-1].to_csv(altered, index=False, lineterminator="\n")
        with pytest.raises(PipelineError, match="differs from the training run"):
            run_evaluation(trained.config, data_path=altered)


# ---------------------------------------------------------------------------
# Registry lifecycle & promotion gate
# ---------------------------------------------------------------------------
def _metrics(**overrides: float) -> ClassificationMetrics:
    base: dict[str, float | int] = {
        "n_rows": 1000,
        "positive_rate": 0.26,
        "threshold": 0.3,
        "roc_auc": 0.85,
        "pr_auc": 0.66,
        "brier": 0.14,
        "log_loss": 0.42,
        "ece": 0.02,
        "f1": 0.62,
        "precision": 0.55,
        "recall": 0.7,
        "accuracy": 0.78,
        "tn": 600,
        "fp": 140,
        "fn": 80,
        "tp": 180,
    }
    return ClassificationMetrics.model_validate({**base, **overrides})


GATES = PromotionConfig(min_roc_auc=0.8, min_pr_auc=0.6, min_f1=0.55, max_brier=0.17)


class TestPromotionGate:
    def test_approves_when_every_check_passes(self) -> None:
        decision = evaluate_promotion(
            candidate_id="c",
            candidate=_metrics(),
            baseline=_metrics(pr_auc=0.65),
            candidate_is_baseline=False,
            production_id=None,
            production=None,
            gates=GATES,
        )
        assert decision.approved
        assert all(check.passed for check in decision.checks)

    @pytest.mark.parametrize(
        ("overrides", "failed"),
        [
            ({"roc_auc": 0.79}, "roc_auc_floor"),
            ({"pr_auc": 0.59}, "pr_auc_floor"),
            ({"f1": 0.5}, "f1_floor"),
            ({"brier": 0.2}, "brier_ceiling"),
        ],
    )
    def test_rejects_below_floors(self, overrides: dict[str, float], failed: str) -> None:
        decision = evaluate_promotion(
            candidate_id="c",
            candidate=_metrics(**overrides),
            baseline=None,
            candidate_is_baseline=True,
            production_id=None,
            production=None,
            gates=GATES,
        )
        assert not decision.approved
        assert [c.name for c in decision.checks if not c.passed] == [failed]
        assert failed in decision.summary()

    def test_rejects_complex_model_clearly_worse_than_baseline(self) -> None:
        decision = evaluate_promotion(
            candidate_id="c",
            candidate=_metrics(pr_auc=0.62),
            baseline=_metrics(pr_auc=0.65),
            candidate_is_baseline=False,
            production_id=None,
            production=None,
            gates=GATES,
        )
        assert [c.name for c in decision.checks if not c.passed] == ["not_worse_than_baseline"]

    def test_rejects_regression_against_production(self) -> None:
        decision = evaluate_promotion(
            candidate_id="c",
            candidate=_metrics(pr_auc=0.66),
            baseline=None,
            candidate_is_baseline=True,
            production_id="prod",
            production=_metrics(pr_auc=0.67),
            gates=GATES,
        )
        assert not decision.approved
        assert decision.production_id == "prod"
        assert [c.name for c in decision.checks if not c.passed] == ["beats_production"]


def test_registry_lifecycle(trained: TrainedPipeline, tmp_path: Path) -> None:
    source = ModelRegistry(trained.config.paths.registry_dir)
    model, metadata = source.load(trained.report.model_id)
    template = metadata.model_dump(mode="json", exclude={"artifact", "stage", "test", "promotion"})
    registry = ModelRegistry(tmp_path / "registry")

    def register(model_id: str) -> None:
        registry.register(model, {**template, "model_id": model_id})

    def decide(model_id: str, approved: bool) -> None:
        decision = evaluate_promotion(
            candidate_id=model_id,
            candidate=_metrics() if approved else _metrics(roc_auc=0.1),
            baseline=None,
            candidate_is_baseline=True,
            production_id=registry.production_id(),
            production=None,
            gates=GATES,
        )
        registry.apply_decision(decision)

    register("a")
    decide("a", approved=True)
    register("b")
    decide("b", approved=True)
    register("c")
    decide("c", approved=False)

    index = registry.index()
    assert index.production == "b"
    assert index.stages == {"a": "archived", "b": "production", "c": "rejected"}
    assert [event.action for event in index.events] == [
        "registered",
        "promoted",
        "registered",
        "archived",
        "promoted",
        "registered",
        "rejected",
    ]
    assert b"\r" not in (tmp_path / "registry" / "registry.json").read_bytes()
    with pytest.raises(RegistryError, match="already registered"):
        register("a")
    with pytest.raises(RegistryError, match="not a candidate"):
        decide("b", approved=True)
    exported = registry.export("b", tmp_path / "export")
    assert load_model_dir(exported)[1].model_id == "b"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
class TestMetricHelpers:
    def test_select_threshold_known_case(self) -> None:
        y = [0, 0, 1, 1, 1]
        p = [0.1, 0.4, 0.35, 0.8, 0.9]
        threshold, f1 = select_threshold(y, p)
        assert threshold == pytest.approx(0.35)
        assert f1 == pytest.approx(2 * 0.75 * 1.0 / 1.75)

    def test_compute_metrics_confusion_matrix(self) -> None:
        metrics = compute_metrics([0, 1, 1, 0], [0.2, 0.7, 0.4, 0.6], threshold=0.5)
        assert (metrics.tn, metrics.fp, metrics.fn, metrics.tp) == (1, 1, 1, 1)
        assert metrics.precision == pytest.approx(0.5)

    def test_perfect_calibration_has_zero_ece(self) -> None:
        y = np.array([0] * 8 + [1] * 2)
        p = np.full(10, 0.2)
        assert expected_calibration_error(y, p) == pytest.approx(0.0)
        bins = reliability_table(y, p)
        assert sum(b.count for b in bins) == 10

    @pytest.mark.parametrize(
        ("y", "p", "message"),
        [
            ([0, 1], [0.5], "equal length"),
            ([0, 2], [0.1, 0.9], "binary"),
            ([0, 1], [-0.1, 0.5], r"\[0, 1\]"),
        ],
    )
    def test_invalid_inputs_raise(self, y: list[int], p: list[float], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            compute_metrics(y, p, threshold=0.5)
