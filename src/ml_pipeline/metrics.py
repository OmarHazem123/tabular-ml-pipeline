"""Metric definitions shared by training, evaluation and the registry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FoldScores(_Model):
    mean: float
    std: float
    folds: tuple[float, ...]

    @classmethod
    def of(cls, values: Sequence[float]) -> FoldScores:
        array = np.asarray(values, dtype=np.float64)
        return cls(
            mean=float(array.mean()),
            std=float(array.std()),
            folds=tuple(float(v) for v in array),
        )


class ClassificationMetrics(_Model):
    n_rows: int
    positive_rate: float
    threshold: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ece: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int


class ReliabilityBin(_Model):
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


def _as_arrays(y_true: object, proba: object) -> tuple[IntArray, FloatArray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y_true and proba must be 1-D of equal length ({y.shape} vs {p.shape})")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1")
    if not ((p >= 0.0) & (p <= 1.0)).all():
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def _bin_edges(n_bins: int) -> FloatArray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_index(p: FloatArray, n_bins: int) -> IntArray:
    # Right-closed final bin so that p == 1.0 lands in the last bin.
    return np.minimum((p * n_bins).astype(np.int64), n_bins - 1)


def expected_calibration_error(y_true: object, proba: object, n_bins: int = 10) -> float:
    """Count-weighted mean |observed rate - mean predicted| over equal-width bins."""
    y, p = _as_arrays(y_true, proba)
    index = _bin_index(p, n_bins)
    error = 0.0
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def reliability_table(
    y_true: object, proba: object, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    y, p = _as_arrays(y_true, proba)
    edges = _bin_edges(n_bins)
    index = _bin_index(p, n_bins)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_predicted=float(p[mask].mean()) if count else None,
                observed_rate=float(y[mask].mean()) if count else None,
            )
        )
    return tuple(bins)


def compute_metrics(y_true: object, proba: object, threshold: float) -> ClassificationMetrics:
    """Ranking, calibration and thresholded metrics. Positive iff ``proba >= threshold``."""
    y, p = _as_arrays(y_true, proba)
    predicted = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return ClassificationMetrics(
        n_rows=int(y.size),
        positive_rate=float(y.mean()),
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        ece=expected_calibration_error(y, p),
        f1=float(f1_score(y, predicted, zero_division=0)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        accuracy=float(accuracy_score(y, predicted)),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def select_threshold(y_true: object, proba: object, beta: float = 1.0) -> tuple[float, float]:
    """Return ``(threshold, f_beta)`` maximising F-beta; positive iff ``proba >= threshold``.

    Ties resolve to the lowest qualifying threshold (first argmax), deterministically.
    """
    y, p = _as_arrays(y_true, proba)
    if y.min() == y.max():
        raise ValueError("threshold selection needs both classes present")
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    beta2 = beta**2
    denominator = beta2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, (1 + beta2) * precision * recall / denominator, 0.0)
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])
