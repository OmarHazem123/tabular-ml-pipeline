"""Multi-model training: tune on TRAIN (CV), calibrate + pick threshold on VALIDATION.

Protocol (the test split is never touched here)::

    for each model (logistic-regression baseline, LightGBM, XGBoost):
        Optuna TPE (seeded, sequential) maximising CV PR-AUC on the TRAIN split
        refit best params on the full TRAIN split
        out-of-fold calibration across VAL folds -> choose F-beta threshold
        final calibrator fit on all of VAL (pipeline frozen)
    champion = best CV mean of the optimisation metric (ties -> the simpler baseline)
    suggest promotion floors from the tuned baseline (documented rule, no test data)
    register champion as a *candidate* in the registry
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml_pipeline.config import (
    FloatParam,
    IntParam,
    ModelName,
    ModelSpec,
    ParamSpec,
    ParamValue,
    PipelineConfig,
    seed_everything,
)
from ml_pipeline.data import DataSplits, load_dataset, split_dataset
from ml_pipeline.exceptions import PipelineError
from ml_pipeline.features import build_model_pipeline
from ml_pipeline.io_utils import atomic_write_json
from ml_pipeline.metrics import ClassificationMetrics, FoldScores, compute_metrics, select_threshold
from ml_pipeline.model import ChurnModel, ReferenceProfile
from ml_pipeline.registry import (
    ArtifactInfo,
    Lineage,
    ModelRegistry,
    SuggestedFloors,
    dump_model,
    environment_info,
    git_info,
    load_verified_model,
    suggest_promotion_floors,
    utc_now,
)

logger = logging.getLogger(__name__)

RUN_FILENAME = "training_run.json"
SCORERS = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
OPTIMIZE_KEY = {"average_precision": "pr_auc", "roc_auc": "roc_auc"}


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateSummary(_Model):
    model_name: ModelName
    is_baseline: bool
    n_trials: int
    best_params: dict[str, ParamValue]
    cv: dict[str, FoldScores]
    calibration_method: str
    threshold: float
    validation: ClassificationMetrics  # out-of-fold calibrated VAL probabilities
    validation_uncalibrated_brier: float
    fit_seconds: float
    artifact: ArtifactInfo


class TrainingRun(_Model):
    run_id: str
    created_at: str
    optimize_metric: str
    champion: ModelName
    champion_model_id: str
    lineage: Lineage
    candidates: tuple[CandidateSummary, ...]
    suggested_floors: SuggestedFloors

    def candidate(self, name: str) -> CandidateSummary:
        for summary in self.candidates:
            if summary.model_name == name:
                return summary
        raise KeyError(name)

    @property
    def baseline(self) -> CandidateSummary:
        return next(summary for summary in self.candidates if summary.is_baseline)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_estimator(name: ModelName, params: Mapping[str, ParamValue], seed: int) -> Any:
    kwargs: dict[str, Any] = dict(params)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **kwargs)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, **kwargs)
    return XGBClassifier(random_state=seed, **kwargs)


def suggest_params(trial: optuna.Trial, space: Mapping[str, ParamSpec]) -> dict[str, ParamValue]:
    params: dict[str, ParamValue] = {}
    for name, spec in space.items():
        if isinstance(spec, FloatParam):
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, IntParam):
            params[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        else:
            params[name] = trial.suggest_categorical(name, list(spec.choices))
    return params


def model_seed(seed: int, name: str) -> int:
    """Stable per-model sampler seed, so models do not share identical draws for
    identically named hyperparameters (e.g. ``n_estimators``)."""
    return (seed + zlib.crc32(name.encode("utf-8"))) % (2**32)


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, ParamValue]
    cv: dict[str, FoldScores]
    n_trials: int


def tune(spec: ModelSpec, splits: DataSplits, config: PipelineConfig) -> TuningResult:
    """Seeded, sequential TPE search; each trial is scored by stratified CV on TRAIN."""
    training = config.training
    folds = StratifiedKFold(n_splits=training.cv_folds, shuffle=True, random_state=config.seed)
    target = OPTIMIZE_KEY[training.optimize_metric]

    def objective(trial: optuna.Trial) -> float:
        params = {**spec.fixed_params, **suggest_params(trial, spec.search_space)}
        pipeline = build_model_pipeline(config, make_estimator(spec.name, params, config.seed))
        scores = cross_validate(
            pipeline,
            splits.X_train,
            splits.y_train,
            cv=folds,
            scoring=SCORERS,
            n_jobs=training.n_jobs,
            error_score="raise",
        )
        for key in SCORERS:
            trial.set_user_attr(f"cv_{key}", [float(v) for v in scores[f"test_{key}"]])
        return float(np.mean(scores[f"test_{target}"]))

    n_trials = spec.n_trials if spec.search_space else 1
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=model_seed(config.seed, spec.name)),
        study_name=spec.name,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_trial
    return TuningResult(
        best_params={**spec.fixed_params, **best.params},
        cv={key: FoldScores.of(best.user_attrs[f"cv_{key}"]) for key in SCORERS},
        n_trials=n_trials,
    )


def calibrate(
    pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series[int], config: PipelineConfig
) -> tuple[CalibratedClassifierCV, np.ndarray[Any, np.dtype[np.float64]]]:
    """Return the final calibrator (fit on all of VAL) and OUT-OF-FOLD calibrated VAL
    probabilities, which are what the threshold is chosen on."""
    method = config.training.calibration_method
    folds = StratifiedKFold(
        n_splits=config.training.calibration_folds, shuffle=True, random_state=config.seed
    )
    out_of_fold = np.empty(len(y_val), dtype=np.float64)
    for fit_idx, held_idx in folds.split(X_val, y_val):
        fold_calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
        fold_calibrator.fit(X_val.iloc[fit_idx], y_val.iloc[fit_idx])
        out_of_fold[held_idx] = fold_calibrator.predict_proba(X_val.iloc[held_idx])[:, 1]
    final = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
    final.fit(X_val, y_val)
    return final, out_of_fold


def _background(pipeline: Pipeline, X_train: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    sample = X_train.sample(n=min(size, len(X_train)), random_state=seed)
    background: pd.DataFrame = pipeline[:-1].transform(sample)
    return background


def train_candidate(
    spec: ModelSpec,
    splits: DataSplits,
    config: PipelineConfig,
    reference: ReferenceProfile,
    run_dir: Path,
) -> tuple[ChurnModel, CandidateSummary]:
    started = time.perf_counter()
    tuning = tune(spec, splits, config)
    estimator = make_estimator(spec.name, tuning.best_params, config.seed)
    pipeline = build_model_pipeline(config, estimator)
    pipeline.fit(splits.X_train, splits.y_train)

    uncalibrated = pipeline.predict_proba(splits.X_val)[:, 1]
    calibrator, out_of_fold = calibrate(pipeline, splits.X_val, splits.y_val, config)
    threshold, _ = select_threshold(splits.y_val, out_of_fold, config.training.threshold_beta)

    model = ChurnModel(
        model_name=spec.name,
        pipeline=pipeline,
        calibrator=calibrator,
        threshold=threshold,
        input_columns=tuple(str(column) for column in splits.X_train.columns),
        reference=reference,
        background=_background(
            pipeline, splits.X_train, config.training.shap_background_size, config.seed
        ),
    )
    artifact = dump_model(model, run_dir / f"{spec.name}.joblib")
    summary = CandidateSummary(
        model_name=spec.name,
        is_baseline=spec.is_baseline,
        n_trials=tuning.n_trials,
        best_params=tuning.best_params,
        cv=tuning.cv,
        calibration_method=config.training.calibration_method,
        threshold=threshold,
        validation=compute_metrics(splits.y_val, out_of_fold, threshold),
        validation_uncalibrated_brier=compute_metrics(splits.y_val, uncalibrated, threshold).brier,
        fit_seconds=round(time.perf_counter() - started, 2),
        artifact=artifact,
    )
    logger.info(
        "%-19s CV PR-AUC %.4f+/-%.4f  CV ROC-AUC %.4f  val(OOF) F1 %.4f @ %.3f  (%d trials, %.1fs)",
        spec.name,
        summary.cv["pr_auc"].mean,
        summary.cv["pr_auc"].std,
        summary.cv["roc_auc"].mean,
        summary.validation.f1,
        threshold,
        tuning.n_trials,
        summary.fit_seconds,
    )
    return model, summary


def cap_trials(config: PipelineConfig, max_trials: int) -> PipelineConfig:
    models = tuple(
        spec.model_copy(update={"n_trials": min(spec.n_trials, max_trials)})
        for spec in config.training.models
    )
    training = config.training.model_copy(update={"models": models})
    return config.model_copy(update={"training": training})


def new_run_id(config: PipelineConfig) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config.fingerprint()[:8]}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_training(
    config: PipelineConfig,
    *,
    data_path: Path | None = None,
    max_trials: int | None = None,
    run_id: str | None = None,
) -> TrainingRun:
    if max_trials is not None:
        config = cap_trials(config, max_trials)
    seed_everything(config.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dataset = load_dataset(config, data_path=data_path)
    splits = split_dataset(dataset, config)
    run_id = run_id or new_run_id(config)
    run_dir = config.paths.runs_dir / run_id
    reference = ReferenceProfile.from_frame(splits.X_train, config.data_schema)
    lineage = Lineage(
        run_id=run_id,
        git=git_info(),
        config_fingerprint=config.fingerprint(),
        seed=config.seed,
        dataset=dataset.source,
        dataset_content_sha256=dataset.report.content_sha256,
        splits=splits.summary(),
        environment=environment_info(),
    )
    logger.info("Run %s: training %d model(s)", run_id, len(config.training.models))

    models: dict[str, ChurnModel] = {}
    summaries: list[CandidateSummary] = []
    for spec in config.training.models:
        model, summary = train_candidate(spec, splits, config, reference, run_dir)
        models[spec.name] = model
        summaries.append(summary)

    target = OPTIMIZE_KEY[config.training.optimize_metric]
    champion = max(summaries, key=lambda s: (s.cv[target].mean, s.is_baseline))
    baseline = next(summary for summary in summaries if summary.is_baseline)
    model_id = f"{run_id}-{champion.model_name}"
    run = TrainingRun(
        run_id=run_id,
        created_at=utc_now(),
        optimize_metric=config.training.optimize_metric,
        champion=champion.model_name,
        champion_model_id=model_id,
        lineage=lineage,
        candidates=tuple(summaries),
        suggested_floors=suggest_promotion_floors(
            cv_roc_auc=baseline.cv["roc_auc"],
            cv_pr_auc=baseline.cv["pr_auc"],
            validation=baseline.validation,
            fold_rows=len(splits.y_train) // config.training.cv_folds,
            test_rows=len(splits.y_test),
        ),
    )
    atomic_write_json(run_dir / RUN_FILENAME, run.model_dump(mode="json"))

    ModelRegistry(config.paths.registry_dir).register(
        models[champion.model_name],
        {
            "model_id": model_id,
            "model_name": champion.model_name,
            "created_at": run.created_at,
            "lineage": lineage.model_dump(mode="json"),
            "data_schema": config.data_schema.model_dump(mode="json"),
            "threshold": champion.threshold,
            "calibration_method": champion.calibration_method,
            "best_params": champion.best_params,
            "cv": {key: scores.model_dump() for key, scores in champion.cv.items()},
            "validation": champion.validation.model_dump(),
        },
    )
    logger.info("Champion %s registered as candidate %s", champion.model_name, model_id)
    return run


def load_training_run(
    config: PipelineConfig, run_id: str | None = None
) -> tuple[TrainingRun, Path]:
    """Load a run summary (the most recent one when ``run_id`` is None)."""
    runs_dir = config.paths.runs_dir
    if run_id is None:
        runs = sorted(path.parent.name for path in runs_dir.glob(f"*/{RUN_FILENAME}"))
        if not runs:
            raise PipelineError(f"no training runs found under {runs_dir.as_posix()}")
        run_id = runs[-1]
    run_dir = runs_dir / run_id
    path = run_dir / RUN_FILENAME
    if not path.exists():
        raise PipelineError(f"training run {run_id} not found under {runs_dir.as_posix()}")
    return TrainingRun.model_validate_json(path.read_text(encoding="utf-8")), run_dir


def load_candidate(run_dir: Path, summary: CandidateSummary) -> ChurnModel:
    return load_verified_model(run_dir / summary.artifact.filename, summary.artifact.sha256)
