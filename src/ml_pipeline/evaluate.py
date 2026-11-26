"""Hold-out evaluation, SHAP summaries, promotion gate and reports.

This is the ONLY place the test split is scored. Model selection already happened in
``train.py`` (CV on train); here every candidate is scored on test for the benchmark
table, the champion goes through the promotion gate, and reports are written to
``results/metrics.json`` and ``results/report.md`` (atomic, LF).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ml_pipeline.config import ParamValue, PipelineConfig
from ml_pipeline.data import DataSplits, SplitSummary, load_dataset, split_dataset
from ml_pipeline.exceptions import PipelineError
from ml_pipeline.io_utils import atomic_write_json, atomic_write_text
from ml_pipeline.metrics import (
    ClassificationMetrics,
    FoldScores,
    ReliabilityBin,
    compute_metrics,
    reliability_table,
)
from ml_pipeline.model import ChurnModel, Reason
from ml_pipeline.registry import ModelRegistry, PromotionDecision, evaluate_promotion
from ml_pipeline.train import TrainingRun, load_candidate, load_training_run

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.json"
REPORT_FILENAME = "report.md"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FeatureImportance(_Model):
    feature: str
    mean_abs_shap: float
    mean_shap: float


class LocalExplanation(_Model):
    label: str
    customer_id: str
    churn_probability: float
    predicted_churn: bool
    actual_churn: bool
    reasons: tuple[Reason, ...]


class BenchmarkRow(_Model):
    model_name: str
    is_baseline: bool
    is_champion: bool
    threshold: float
    best_params: dict[str, ParamValue]
    cv_roc_auc: FoldScores
    cv_pr_auc: FoldScores
    validation: ClassificationMetrics
    test: ClassificationMetrics


class EvaluationReport(_Model):
    run_id: str
    model_id: str
    champion: str
    config_fingerprint: str
    dataset_sha256: str | None
    seed: int
    git_commit: str | None
    git_dirty: bool | None
    splits: dict[str, SplitSummary]
    test: ClassificationMetrics
    reliability: tuple[ReliabilityBin, ...]
    benchmark: tuple[BenchmarkRow, ...]
    shap_global: tuple[FeatureImportance, ...]
    local_explanations: tuple[LocalExplanation, ...]
    promotion: PromotionDecision
    production_model_id: str | None


# ---------------------------------------------------------------------------
# SHAP summaries
# ---------------------------------------------------------------------------
def global_importance(
    model: ChurnModel, X: pd.DataFrame, top_n: int = 15
) -> tuple[FeatureImportance, ...]:
    """Mean |SHAP| per source feature (one-hot groups summed), in log-odds units."""
    grouped, _ = model.grouped_shap(X)
    mean_abs = grouped.abs().mean()
    mean = grouped.mean()
    order = mean_abs.sort_values(ascending=False, kind="mergesort").index[:top_n]
    return tuple(
        FeatureImportance(
            feature=str(feature),
            mean_abs_shap=round(float(mean_abs[feature]), 6),
            mean_shap=round(float(mean[feature]), 6),
        )
        for feature in order
    )


def local_explanations(
    model: ChurnModel, X: pd.DataFrame, y: pd.Series[int], top_k: int
) -> tuple[LocalExplanation, ...]:
    """Highest-risk, lowest-risk and most borderline test customers."""
    proba = pd.Series(model.predict_proba(X), index=X.index)
    distance = (proba - model.threshold).abs()
    picks = {
        "Highest predicted risk": proba.sort_values(ascending=False, kind="mergesort").index[0],
        "Lowest predicted risk": proba.sort_values(kind="mergesort").index[0],
        "Closest to the decision threshold": distance.sort_values(kind="mergesort").index[0],
    }
    rows = X.loc[list(picks.values())]
    reasons = model.explain(rows, top_k)
    return tuple(
        LocalExplanation(
            label=label,
            customer_id=str(customer_id),
            churn_probability=round(float(proba[customer_id]), 6),
            predicted_churn=bool(proba[customer_id] >= model.threshold),
            actual_churn=bool(y[customer_id]),
            reasons=row_reasons,
        )
        for (label, customer_id), row_reasons in zip(picks.items(), reasons, strict=True)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _check_same_data(run: TrainingRun, content_sha256: str | None, splits: DataSplits) -> None:
    """Refuse to evaluate on data/splits that differ from what the run was trained on."""
    if content_sha256 != run.lineage.dataset_content_sha256:
        raise PipelineError(
            "evaluation dataset content differs from the training run "
            f"({content_sha256} != {run.lineage.dataset_content_sha256})"
        )
    current = splits.summary()
    for name, recorded in run.lineage.splits.items():
        if current[name].ids_sha256 != recorded.ids_sha256:
            raise PipelineError(f"the {name} split differs from the training run")


def run_evaluation(
    config: PipelineConfig, *, run_id: str | None = None, data_path: Path | None = None
) -> EvaluationReport:
    run, run_dir = load_training_run(config, run_id)
    dataset = load_dataset(config, data_path=data_path)
    splits = split_dataset(dataset, config)
    _check_same_data(run, dataset.report.content_sha256, splits)
    X_test, y_test = splits.X_test, splits.y_test

    models: dict[str, ChurnModel] = {}
    rows: list[BenchmarkRow] = []
    for summary in run.candidates:
        model = load_candidate(run_dir, summary)
        models[summary.model_name] = model
        test = compute_metrics(y_test, model.predict_proba(X_test), model.threshold)
        rows.append(
            BenchmarkRow(
                model_name=summary.model_name,
                is_baseline=summary.is_baseline,
                is_champion=summary.model_name == run.champion,
                threshold=summary.threshold,
                best_params=summary.best_params,
                cv_roc_auc=summary.cv["roc_auc"],
                cv_pr_auc=summary.cv["pr_auc"],
                validation=summary.validation,
                test=test,
            )
        )
    champion_row = next(row for row in rows if row.is_champion)
    baseline_row = next(row for row in rows if row.is_baseline)
    champion = models[run.champion]
    model_id = run.champion_model_id

    registry = ModelRegistry(config.paths.registry_dir)
    metadata = registry.record_test_metrics(model_id, champion_row.test)
    if metadata.promotion is not None:
        logger.info("Run %s was already decided; reusing the recorded decision", run.run_id)
        decision = metadata.promotion
    else:
        production_id = registry.production_id()
        production_metrics = None
        if production_id is not None:
            production_model, _ = registry.load(production_id)
            production_metrics = compute_metrics(
                y_test, production_model.predict_proba(X_test), production_model.threshold
            )
        decision = evaluate_promotion(
            candidate_id=model_id,
            candidate=champion_row.test,
            baseline=baseline_row.test,
            candidate_is_baseline=champion_row.is_baseline,
            production_id=production_id,
            production=production_metrics,
            gates=config.promotion,
        )
        registry.apply_decision(decision)
    logger.info(decision.summary())

    report = EvaluationReport(
        run_id=run.run_id,
        model_id=model_id,
        champion=run.champion,
        config_fingerprint=run.lineage.config_fingerprint,
        dataset_sha256=run.lineage.dataset.sha256 if run.lineage.dataset else None,
        seed=run.lineage.seed,
        git_commit=run.lineage.git.commit,
        git_dirty=run.lineage.git.dirty,
        splits=run.lineage.splits,
        test=champion_row.test,
        reliability=reliability_table(y_test, champion.predict_proba(X_test)),
        benchmark=tuple(rows),
        shap_global=global_importance(champion, X_test),
        local_explanations=local_explanations(
            champion, X_test, y_test, config.serving.default_top_k
        ),
        promotion=decision,
        production_model_id=registry.production_id(),
    )
    results = config.paths.results_dir
    atomic_write_json(results / METRICS_FILENAME, report.model_dump(mode="json"))
    atomic_write_text(results / REPORT_FILENAME, render_report(report))
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _model_label(row: BenchmarkRow) -> str:
    tags = [tag for tag, on in (("baseline", row.is_baseline), ("champion", row.is_champion)) if on]
    return f"{row.model_name} ({', '.join(tags)})" if tags else row.model_name


def render_report(report: EvaluationReport) -> str:
    test = report.test
    verdict = "APPROVED" if report.promotion.approved else "REJECTED"
    lines = [
        "# Churn model evaluation report",
        "",
        f"- **Run:** `{report.run_id}`",
        f"- **Champion:** `{report.champion}` (model id `{report.model_id}`)",
        f"- **Promotion:** **{verdict}**, production model is now "
        f"`{report.production_model_id or 'none'}`",
        f"- **Lineage:** git `{(report.git_commit or 'unknown')[:12]}`"
        f"{' (with uncommitted changes)' if report.git_dirty else ''}, "
        f"config `{report.config_fingerprint[:12]}`, "
        f"dataset sha256 `{(report.dataset_sha256 or 'unknown')[:12]}`, seed {report.seed}",
        "",
        "## Benchmark",
        "",
        "The champion is chosen by **cross-validated PR-AUC on the train split**. "
        "Validation metrics use out-of-fold calibrated probabilities. Test columns are "
        "hold-out results reported for every model, but they played no part in selection.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "CV ROC-AUC",
            "CV PR-AUC",
            "Val PR-AUC",
            "Val F1",
            "Test ROC-AUC",
            "Test PR-AUC",
            "Test F1",
            "Test Brier",
            "Threshold",
        ],
        [
            [
                _model_label(row),
                f"{_f(row.cv_roc_auc.mean)} ± {_f(row.cv_roc_auc.std)}",
                f"{_f(row.cv_pr_auc.mean)} ± {_f(row.cv_pr_auc.std)}",
                _f(row.validation.pr_auc),
                _f(row.validation.f1),
                _f(row.test.roc_auc),
                _f(row.test.pr_auc),
                _f(row.test.f1),
                _f(row.test.brier),
                _f(row.threshold, 3),
            ]
            for row in report.benchmark
        ],
    )
    lines += ["", "## Promotion gate", ""]
    lines += _table(
        ["Check", "Result", "Detail"],
        [
            [check.name, "pass" if check.passed else "**FAIL**", check.detail]
            for check in report.promotion.checks
        ],
    )
    lines += ["", f"## Hold-out test metrics: {report.champion}", ""]
    lines += _table(
        ["Metric", "Value"],
        [
            ["Rows / positive rate", f"{test.n_rows} / {_f(test.positive_rate)}"],
            ["ROC-AUC", _f(test.roc_auc)],
            ["PR-AUC (average precision)", _f(test.pr_auc)],
            [f"F1 @ threshold {_f(test.threshold, 3)}", _f(test.f1)],
            ["Precision / Recall", f"{_f(test.precision)} / {_f(test.recall)}"],
            ["Accuracy", _f(test.accuracy)],
            ["Brier score", _f(test.brier)],
            ["Log loss", _f(test.log_loss)],
            ["Expected calibration error (10 bins)", _f(test.ece)],
        ],
    )
    lines += ["", "Confusion matrix (rows = actual, columns = predicted):", ""]
    lines += _table(
        ["", "Predicted stay", "Predicted churn"],
        [
            ["**Actual stay**", str(test.tn), str(test.fp)],
            ["**Actual churn**", str(test.fn), str(test.tp)],
        ],
    )
    lines += ["", "## Calibration (reliability, test split)", ""]
    lines += _table(
        ["Predicted bin", "Count", "Mean predicted", "Observed churn rate"],
        [
            [
                f"{bin_.lower:.1f}-{bin_.upper:.1f}",
                str(bin_.count),
                _f(bin_.mean_predicted, 3),
                _f(bin_.observed_rate, 3),
            ]
            for bin_ in report.reliability
            if bin_.count
        ],
    )
    lines += [
        "",
        "## Global feature importance (SHAP, test split)",
        "",
        "Mean |SHAP| in log-odds, with one-hot columns summed back into their source "
        "feature. A positive mean SHAP means the feature pushes risk up on average.",
        "",
    ]
    lines += _table(
        ["Rank", "Feature", "Mean abs SHAP", "Mean SHAP"],
        [
            [str(rank), f"`{item.feature}`", _f(item.mean_abs_shap), f"{item.mean_shap:+.4f}"]
            for rank, item in enumerate(report.shap_global, start=1)
        ],
    )
    lines += ["", "## Local explanations (SHAP reason codes)", ""]
    for example in report.local_explanations:
        lines += [
            f"### {example.label}: `{example.customer_id}`",
            "",
            f"p(churn) = {_f(example.churn_probability)}, predicted "
            f"{'churn' if example.predicted_churn else 'stay'}, actual "
            f"{'churn' if example.actual_churn else 'stay'}",
            "",
        ]
        lines += _table(
            ["Feature", "Value", "Contribution (log-odds)", "Effect"],
            [
                [
                    f"`{reason.feature}`",
                    str(reason.value),
                    f"{reason.contribution:+.4f}",
                    reason.direction.replace("_", " "),
                ]
                for reason in example.reasons
            ],
        )
        lines.append("")
    lines += ["## Splits", ""]
    lines += _table(
        ["Split", "Rows", "Positive rate", "IDs sha256"],
        [
            [name, str(part.n_rows), _f(part.positive_rate), f"`{part.ids_sha256[:16]}`"]
            for name, part in report.splits.items()
        ],
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def summarize_test(metrics: ClassificationMetrics) -> str:
    return (
        f"ROC-AUC {metrics.roc_auc:.4f} | PR-AUC {metrics.pr_auc:.4f} | F1 {metrics.f1:.4f} "
        f"| precision {metrics.precision:.4f} | recall {metrics.recall:.4f} "
        f"| Brier {metrics.brier:.4f} | threshold {metrics.threshold:.3f}"
    )
