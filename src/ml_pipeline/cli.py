"""Command-line interface: ``ml-pipeline <command>`` (or ``python -m ml_pipeline``).

Commands: validate-data, train, evaluate, serve, predict, export-model.
``train`` imports Optuna lazily, so every other command works in the slim serving image.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from pydantic import ValidationError

from ml_pipeline.config import PipelineConfig, load_config, seed_everything
from ml_pipeline.data import load_dataset, split_dataset
from ml_pipeline.exceptions import DataIntegrityError, DataValidationError, PipelineError
from ml_pipeline.io_utils import atomic_write_json, canonical_json

app = typer.Typer(
    name="ml-pipeline",
    help="Reproducible Telco churn pipeline: data validation, training, evaluation, serving.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
logger = logging.getLogger("ml_pipeline.cli")

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config", help="TOML/JSON overrides for the default config.", exists=True, dir_okay=False
    ),
]
DataPathOption = Annotated[
    Path | None,
    typer.Option(
        "--data-path",
        help="Use a local CSV instead of the pinned download (checksum is reported, not enforced).",
        exists=True,
        dir_okay=False,
    ),
]
ModelDirOption = Annotated[
    Path | None,
    typer.Option(
        "--model-dir",
        help="Exported model dir (model.joblib + metadata.json). Default: registry production.",
        exists=True,
        file_okay=False,
    ),
]
RegistryOption = Annotated[
    Path | None,
    typer.Option("--registry-dir", help="Registry root. Default: paths.registry_dir."),
]


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=code)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------
@app.command("validate-data")
def validate_data(
    config_path: ConfigOption = None,
    data_path: DataPathOption = None,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", help="Where to write the JSON report.", dir_okay=False),
    ] = None,
) -> None:
    """Fetch the pinned dataset, validate schema & quality, and report the split layout."""
    config = load_config(config_path)
    seed_everything(config.seed)
    destination = report_path or config.paths.results_dir / "data_validation.json"
    payload: dict[str, Any] = {"config_fingerprint": config.fingerprint(), "seed": config.seed}

    try:
        dataset = load_dataset(config, data_path=data_path)
    except DataIntegrityError as exc:
        raise _fail(f"Data integrity failure: {exc}", code=2) from exc
    except DataValidationError as exc:
        payload |= {
            "passed": False,
            "issues": [issue.model_dump(mode="json") for issue in exc.issues],
        }
        atomic_write_json(destination, payload)
        typer.echo(str(exc), err=True)
        raise _fail(f"Report written to {destination.as_posix()}") from exc

    splits = split_dataset(dataset, config)
    summary = splits.summary()
    payload |= {
        "passed": True,
        "dataset": dataset.source.model_dump(mode="json") if dataset.source else None,
        "quality": dataset.report.model_dump(mode="json"),
        "splits": {name: part.model_dump(mode="json") for name, part in summary.items()},
    }
    atomic_write_json(destination, payload)

    report = dataset.report
    typer.echo(
        f"OK: {report.n_rows} rows, positive rate {report.positive_rate:.4f}, "
        f"{len(report.warnings)} warning(s), content sha256 {report.content_sha256}"
    )
    for name, part in summary.items():
        typer.echo(
            f"  {name:<5} n={part.n_rows:<5} positive_rate={part.positive_rate:.4f} "
            f"ids_sha256={part.ids_sha256[:16]}"
        )
    typer.echo(f"Report written to {destination.as_posix()}")


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------
@app.command()
def train(
    config_path: ConfigOption = None,
    data_path: DataPathOption = None,
    max_trials: Annotated[
        int | None,
        typer.Option("--max-trials", min=1, help="Cap Optuna trials per model (quick runs)."),
    ] = None,
) -> None:
    """Tune all models with CV on train, calibrate + pick thresholds on validation."""
    from ml_pipeline.train import run_training

    config = load_config(config_path)
    try:
        run = run_training(config, data_path=data_path, max_trials=max_trials)
    except PipelineError as exc:
        raise _fail(f"Training failed: {exc}") from exc

    typer.echo(f"\nRun {run.run_id}")
    typer.echo(f"{'model':<21}{'CV PR-AUC':>18}{'CV ROC-AUC':>12}{'val F1':>9}{'thr':>7}")
    for summary in run.candidates:
        pr, roc = summary.cv["pr_auc"], summary.cv["roc_auc"]
        marker = " *" if summary.model_name == run.champion else ""
        typer.echo(
            f"{summary.model_name + marker:<21}{pr.mean:>11.4f} +/- {pr.std:.3f}"
            f"{roc.mean:>12.4f}{summary.validation.f1:>9.4f}{summary.threshold:>7.3f}"
        )
    typer.echo(f"Champion (* best CV PR-AUC): {run.champion} -> candidate {run.champion_model_id}")
    floors, gates = run.suggested_floors, config.promotion
    typer.echo(f"Suggested promotion floors ({floors.rule}):")
    for name, configured in (
        ("min_roc_auc", gates.min_roc_auc),
        ("min_pr_auc", gates.min_pr_auc),
        ("min_f1", gates.min_f1),
        ("max_brier", gates.max_brier),
    ):
        typer.echo(
            f"  {name:<12} suggested {getattr(floors, name):.2f}  configured {configured:.2f}"
        )


@app.command()
def evaluate(
    config_path: ConfigOption = None,
    data_path: DataPathOption = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Training run to evaluate (default: latest).")
    ] = None,
    fail_on_reject: Annotated[
        bool, typer.Option("--fail-on-reject", help="Exit 3 if the promotion gate rejects.")
    ] = False,
) -> None:
    """Score the held-out test split, run the promotion gate, write metrics + report."""
    from ml_pipeline.evaluate import run_evaluation, summarize_test

    config = load_config(config_path)
    try:
        report = run_evaluation(config, run_id=run_id, data_path=data_path)
    except PipelineError as exc:
        raise _fail(f"Evaluation failed: {exc}") from exc

    typer.echo(f"\nTest ({report.champion}): {summarize_test(report.test)}")
    for check in report.promotion.checks:
        typer.echo(f"  [{'pass' if check.passed else 'FAIL'}] {check.name:<24} {check.detail}")
    typer.echo(report.promotion.summary())
    typer.echo(f"Production model: {report.production_model_id or 'none'}")
    results = config.paths.results_dir
    typer.echo(
        f"Wrote {(results / 'metrics.json').as_posix()} and {(results / 'report.md').as_posix()}"
    )
    if fail_on_reject and not report.promotion.approved:
        raise typer.Exit(code=3)


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
@app.command()
def serve(
    config_path: ConfigOption = None,
    model_dir: ModelDirOption = None,
    registry_dir: RegistryOption = None,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Run the FastAPI inference service with uvicorn."""
    import uvicorn

    from ml_pipeline.api import create_app

    factory = functools.partial(
        create_app,
        config=load_config(config_path),
        model_dir=model_dir,
        registry_dir=registry_dir,
    )
    uvicorn.run(factory, factory=True, host=host, port=port, access_log=False, log_config=None)


def _load_serving_model(
    config: PipelineConfig, model_dir: Path | None, registry_dir: Path | None
) -> tuple[Any, Any]:
    from ml_pipeline.registry import ModelRegistry, load_model_dir

    if model_dir is not None:
        return load_model_dir(model_dir)
    loaded = ModelRegistry(registry_dir or config.paths.registry_dir).load_production()
    if loaded is None:
        raise PipelineError("no production model in the registry; run train + evaluate first")
    return loaded


def _read_rows(path: Path, config: PipelineConfig) -> tuple[list[dict[str, Any]], bool]:
    """Return raw rows and whether to validate strictly (JSON) or leniently (CSV)."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("instances") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise PipelineError("JSON input must be a list of objects or {'instances': [...]}")
        return rows, True
    if path.suffix.lower() == ".csv":
        schema, features = config.data_schema, config.features
        frame = pd.read_csv(path).drop(columns=[schema.target_column], errors="ignore")
        if schema.id_column in frame.columns:
            frame = frame.rename(columns={schema.id_column: "customer_id"})
            frame["customer_id"] = frame["customer_id"].astype(str)
        total = features.total_charges_column
        if total in frame.columns:
            # Raw exports leave TotalCharges blank for unbilled customers (tenure == 0):
            # the same row-local rule that data validation applies to training data.
            numeric = pd.to_numeric(frame[total], errors="coerce")
            new_customer = numeric.isna() & frame[features.tenure_column].eq(0)
            frame[total] = numeric.mask(new_customer, 0.0)
        frame = frame.astype(object).where(frame.notna(), None)
        return [{str(k): v for k, v in row.items()} for row in frame.to_dict("records")], False
    raise PipelineError(f"unsupported input format '{path.suffix}' (use .json or .csv)")


@app.command()
def predict(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="JSON (list or {'instances': [...]}) or CSV of customers.", exists=True
        ),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write JSON here instead of stdout.")
    ] = None,
    config_path: ConfigOption = None,
    model_dir: ModelDirOption = None,
    registry_dir: RegistryOption = None,
    top_k: Annotated[int, typer.Option("--top-k", min=1, help="SHAP reason codes per row.")] = 3,
) -> None:
    """Batch predictions with reason codes and drift warnings."""
    from ml_pipeline.schemas import PredictionRequest, PredictionResult, requests_to_frame

    config = load_config(config_path)
    try:
        model, metadata = _load_serving_model(config, model_dir, registry_dir)
        rows, strict = _read_rows(input_path, config)
    except PipelineError as exc:
        raise _fail(str(exc)) from exc

    instances: list[PredictionRequest] = []
    errors: list[str] = []
    for position, row in enumerate(rows):
        try:
            instances.append(PredictionRequest.model_validate(row, strict=strict))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}" for err in exc.errors()
            )
            errors.append(f"row {position}: {details}")
    if errors:
        for line in errors[:20]:
            typer.echo(line, err=True)
        raise _fail(f"{len(errors)} invalid row(s); no predictions written")
    if not instances:
        raise _fail("input contains no rows")

    frame = requests_to_frame(instances, config.data_schema)
    predictions = model.predict_records(frame, top_k)
    payload = {
        "model_id": metadata.model_id,
        "n_predictions": len(predictions),
        "predictions": [
            PredictionResult.model_validate(
                {"customer_id": instance.customer_id, **prediction.model_dump()}
            ).model_dump(mode="json")
            for instance, prediction in zip(instances, predictions, strict=True)
        ],
    }
    if output is None:
        typer.echo(canonical_json(payload), nl=False)
        return
    atomic_write_json(output, payload)
    churners = int(np.sum([p.churn_prediction for p in predictions]))
    typer.echo(
        f"{len(predictions)} prediction(s), {churners} flagged as churn, model "
        f"{metadata.model_id} -> {output.as_posix()}"
    )


@app.command("export-model")
def export_model(
    destination: Annotated[
        Path, typer.Option("--dest", help="Output directory (used by the Docker build).")
    ] = Path("build/model"),
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Default: the production model.")
    ] = None,
    config_path: ConfigOption = None,
    registry_dir: RegistryOption = None,
) -> None:
    """Copy a hash-verified model (artifact + metadata) out of the registry."""
    from ml_pipeline.registry import ModelRegistry

    config = load_config(config_path)
    registry = ModelRegistry(registry_dir or config.paths.registry_dir)
    chosen = model_id or registry.production_id()
    if chosen is None:
        raise _fail("no production model to export; run train + evaluate first")
    try:
        path = registry.export(chosen, destination)
    except PipelineError as exc:
        raise _fail(f"Export failed: {exc}") from exc
    typer.echo(f"Exported {chosen} -> {path.as_posix()}")
