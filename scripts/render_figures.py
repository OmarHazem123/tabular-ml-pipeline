#!/usr/bin/env python
"""Render the README figures from the committed evaluation report.

Every number drawn here comes from ``results/metrics.json`` (written by
``ml-pipeline evaluate``), so the figures can be regenerated from a fresh clone without
the dataset, the model artifacts or a retrain::

    uv run python tasks.py figures

The test split is never re-scored here: ``evaluate.py`` stays the only code that
touches it. Output is byte-for-byte deterministic: matplotlib's default style plus the
fixed overrides below, the bundled DejaVu Sans font, a fixed size and DPI, an opaque
background, and no software or timestamp metadata. Files are written atomically.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from matplotlib import style
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ml_pipeline.evaluate import BenchmarkRow, EvaluationReport
from ml_pipeline.io_utils import atomic_write_bytes
from ml_pipeline.metrics import FoldScores
from ml_pipeline.model import ReasonValue

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "results" / "metrics.json"
DEFAULT_OUT_DIR = ROOT / "docs" / "figures"

WIDTH_IN = 8.0
DPI = 200  # 1600 px wide; GitHub scales it down to the README column.
TOP_FEATURES = 10

# Light-surface palette. The three model hues are categorical slots 1-3, which pass the
# colour-vision-deficiency checks for every pair; their contrast against white is below
# 3:1, so every series also carries a legend entry or a direct label. Diverging (risk
# up / risk down) is red/blue and the confusion matrix uses a one-hue blue ramp.
SURFACE = "#ffffff"
INK = "#191919"
INK_MUTED = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
MODEL_COLORS = {
    "xgboost": "#257de0",
    "lightgbm": "#f47230",
    "logistic_regression": "#00b894",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic regression",
}
BLUE_RAMP = ("#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e")
RISK_UP = "#ec4647"
RISK_DOWN = "#257de0"

STYLE: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_MUTED,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_MUTED,
    "ytick.labelcolor": INK_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_MUTED,
    "lines.solid_capstyle": "round",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").capitalize())


def _model_color(name: str) -> str:
    return MODEL_COLORS.get(name, INK_MUTED)


def _row_label(row: BenchmarkRow) -> str:
    tags = [tag for tag, on in (("champion", row.is_champion), ("baseline", row.is_baseline)) if on]
    label = _model_label(row.model_name)
    return f"{label} ({', '.join(tags)})" if tags else label


def _title(fig: Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left")


def _wilson_interval(rate: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def _relative_luminance(color: Any) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(c) for c in to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _format_value(value: ReasonValue) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:g}"
    return value


def _categorical_y(ax: Axes, positions: Sequence[float], labels: Sequence[str]) -> None:
    ax.set_yticks(list(positions), labels=list(labels))
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def model_comparison(report: EvaluationReport) -> Figure:
    """Per-fold CV scores, mean +/- 1 std and the hold-out score for every model."""
    rows = report.benchmark
    fig = Figure(figsize=(WIDTH_IN, 1.5 + 0.62 * len(rows)), layout="constrained")
    _title(fig, "Model benchmark: 5-fold CV on the train split vs. the hold-out test set")
    panels: tuple[
        tuple[str, Callable[[BenchmarkRow], FoldScores], Callable[[BenchmarkRow], float]], ...
    ]
    panels = (
        ("PR-AUC (primary metric)", lambda row: row.cv_pr_auc, lambda row: row.test.pr_auc),
        ("ROC-AUC", lambda row: row.cv_roc_auc, lambda row: row.test.roc_auc),
    )
    positions = [float(len(rows) - 1 - index) for index in range(len(rows))]
    for column, (title, cv_of, test_of) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), column + 1)
        for row, y in zip(rows, positions, strict=True):
            color = _model_color(row.model_name)
            cv, test = cv_of(row), test_of(row)
            cv_y, test_y = y + 0.13, y - 0.2
            ax.scatter(
                cv.folds,
                [cv_y] * len(cv.folds),
                s=26,
                color=color,
                alpha=0.45,
                linewidths=0,
                zorder=2,
            )
            ax.hlines(cv_y, cv.mean - cv.std, cv.mean + cv.std, color=color, linewidth=2, zorder=3)
            ax.scatter(
                [cv.mean], [cv_y], s=80, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4
            )
            ax.scatter(
                [test],
                [test_y],
                marker="D",
                s=40,
                facecolors=SURFACE,
                edgecolors=INK,
                linewidths=1.3,
                zorder=4,
            )
            ax.annotate(
                f"{cv.mean:.3f}",
                (cv.mean, cv_y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_MUTED,
            )
            ax.annotate(
                f"{test:.3f}",
                (test, test_y),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK_MUTED,
            )
        ax.set_title(title)
        ax.set_ylim(-0.65, len(rows) - 0.35)
        ax.margins(x=0.08)
        if column == 0:
            _categorical_y(ax, positions, [_row_label(row) for row in rows])
            for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
                tick.set_fontweight("bold" if row.is_champion else "normal")
        else:
            _categorical_y(ax, positions, ["" for _ in rows])
    fig.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", markersize=5, alpha=0.45, color=INK_MUTED),
            Line2D(
                [],
                [],
                marker="o",
                markersize=8,
                linewidth=2,
                color=INK_MUTED,
                markeredgecolor=SURFACE,
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="D",
                markersize=6,
                markerfacecolor=SURFACE,
                markeredgecolor=INK,
            ),
        ],
        labels=["CV fold", "CV mean ± 1 std", "Hold-out test"],
        loc="outside lower center",
        ncols=3,
    )
    return fig


def _reliability(ax: Axes, report: EvaluationReport) -> None:
    color = _model_color(report.champion)
    points = [
        (bin_.mean_predicted, bin_.observed_rate, bin_.count)
        for bin_ in report.reliability
        if bin_.count and bin_.mean_predicted is not None and bin_.observed_rate is not None
    ]
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    intervals = [_wilson_interval(y, n) for _, y, n in points]
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    ax.vlines(
        xs,
        [low for low, _ in intervals],
        [high for _, high in intervals],
        color=color,
        linewidth=1.2,
        alpha=0.6,
        zorder=2,
    )
    ax.plot(xs, ys, color=color, linewidth=2, zorder=3)
    ax.scatter(
        xs,
        ys,
        s=[16 + 0.22 * n for _, _, n in points],
        color=color,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=4,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted churn probability (0.1-wide bins)")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability")
    ax.text(
        0.03,
        0.97,
        f"ECE {report.test.ece:.3f} · Brier {report.test.brier:.3f}\n"
        "grey diagonal = perfect calibration\n"
        "dot area ∝ customers in the bin\n"
        "whiskers: 95% Wilson interval",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
        linespacing=1.5,
    )


def _confusion(ax: Axes, report: EvaluationReport) -> None:
    test = report.test
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    rows = (
        ("Actual stay", "stayers", ((test.tn, "TN"), (test.fp, "FP"))),
        ("Actual churn", "churners", ((test.fn, "FN"), (test.tp, "TP"))),
    )
    gap = 0.012  # surface gap between cells, in data units
    for r, (_, group, cells) in enumerate(rows):
        total = sum(count for count, _ in cells)
        for c, (count, tag) in enumerate(cells):
            share = count / total if total else 0.0
            face = cmap(share)
            ink = SURFACE if _relative_luminance(face) < 0.4 else INK
            ax.add_patch(
                Rectangle(
                    (c + gap, r + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=face, edgecolor="none"
                )
            )
            ax.text(c + 0.5, r + 0.25, tag, ha="center", va="center", fontsize=8, color=ink)
            ax.text(
                c + 0.5,
                r + 0.5,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=ink,
            )
            ax.text(
                c + 0.5,
                r + 0.75,
                f"{share:.1%} of {group}",
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
            )
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0.5, 1.5], labels=["Predicted stay", "Predicted churn"])
    ax.set_yticks([0.5, 1.5], labels=[label for label, _, _ in rows])
    ax.tick_params(length=0, labelcolor=INK)
    ax.set_xlabel("Shade = share of the actual class")
    ax.set_title(f"Confusion matrix at threshold {test.threshold:.3f}")


def calibration_and_confusion(report: EvaluationReport) -> Figure:
    fig = Figure(figsize=(WIDTH_IN, 4.1), layout="constrained")
    _title(
        fig,
        f"{_model_label(report.champion)} (champion) on the hold-out test set "
        f"({report.test.n_rows:,} customers)",
    )
    grid = fig.add_gridspec(1, 2, width_ratios=(1.1, 1))
    _reliability(fig.add_subplot(grid[0, 0]), report)
    _confusion(fig.add_subplot(grid[0, 1]), report)
    return fig


def shap_importance(report: EvaluationReport) -> Figure:
    items = report.shap_global[:TOP_FEATURES]
    fig = Figure(figsize=(WIDTH_IN, 1.0 + 0.3 * len(items)), layout="constrained")
    _title(
        fig,
        f"Top {len(items)} churn drivers for {_model_label(report.champion)}: "
        "mean |SHAP| on the hold-out test set",
    )
    ax = fig.add_subplot()
    positions = [float(len(items) - 1 - index) for index in range(len(items))]
    values = [item.mean_abs_shap for item in items]
    bars = ax.barh(positions, values, height=0.55, color=_model_color(report.champion))
    ax.bar_label(
        bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=7.5, color=INK_MUTED
    )
    _categorical_y(ax, positions, [item.feature for item in items])
    ax.set_xlim(0, max(values, default=1.0) * 1.12)
    ax.set_xlabel(
        "Mean |SHAP value| in log-odds (one-hot columns summed into their source feature)"
    )
    return fig


def reason_codes(report: EvaluationReport) -> Figure:
    examples = report.local_explanations
    fig = Figure(figsize=(WIDTH_IN, 1.1 + 1.45 * len(examples)), layout="constrained")
    _title(fig, "SHAP reason codes for three hold-out customers (what POST /predict returns)")
    limit = 1.3 * max(
        (abs(reason.contribution) for example in examples for reason in example.reasons),
        default=1.0,
    )
    first: Axes | None = None
    for index, example in enumerate(examples):
        ax = fig.add_subplot(len(examples), 1, index + 1, sharex=first)
        first = first or ax
        reasons = example.reasons
        positions = [float(len(reasons) - 1 - i) for i in range(len(reasons))]
        contributions = [reason.contribution for reason in reasons]
        bars = ax.barh(
            positions,
            contributions,
            height=0.55,
            color=[RISK_UP if value > 0 else RISK_DOWN for value in contributions],
        )
        ax.bar_label(
            bars,
            labels=[f"{value:+.2f}".replace("-", "\N{MINUS SIGN}") for value in contributions],
            padding=4,
            fontsize=7.5,
            color=INK_MUTED,
        )
        _categorical_y(
            ax,
            positions,
            [f"{reason.feature} = {_format_value(reason.value)}" for reason in reasons],
        )
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.6, len(reasons) - 0.4)
        predicted = "churn" if example.predicted_churn else "stay"
        actual = "churned" if example.actual_churn else "stayed"
        ax.set_title(
            f"{example.label} (customer {example.customer_id})\n"
            f"p(churn) = {example.churn_probability:.3f} → predicted {predicted}, "
            f"actually {actual}",
            fontsize=8.5,
            linespacing=1.4,
        )
        if index < len(examples) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Contribution to the model's log-odds score (SHAP)")
    fig.legend(
        handles=[Patch(color=RISK_UP), Patch(color=RISK_DOWN)],
        labels=["Raises churn risk", "Lowers churn risk"],
        loc="outside lower center",
        ncols=2,
    )
    return fig


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
FIGURES: tuple[tuple[str, Callable[[EvaluationReport], Figure]], ...] = (
    ("model_comparison.png", model_comparison),
    ("calibration_confusion.png", calibration_and_confusion),
    ("shap_importance.png", shap_importance),
    ("reason_codes.png", reason_codes),
)


def render_png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    return buffer.getvalue()


def load_report(metrics_path: Path) -> EvaluationReport:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return EvaluationReport.model_validate(payload)


def render_all(metrics_path: Path, out_dir: Path) -> dict[Path, str]:
    """Render every figure into ``out_dir``; return ``{path: sha256}``."""
    report = load_report(metrics_path)
    written: dict[Path, str] = {}
    with style.context(["default", STYLE]):
        for name, build in FIGURES:
            path = out_dir / name
            written[path] = atomic_write_bytes(path, render_png(build(report)))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    metrics_path: Path = args.metrics
    out_dir: Path = args.out_dir
    for path, digest in render_all(metrics_path, out_dir).items():
        print(f"{digest[:12]}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
