"""Command-line interface.

Phase 1 exposes ``validate-data``; ``train``, ``evaluate``, ``serve`` and ``predict`` are
added alongside their modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from ml_pipeline.config import load_config, seed_everything
from ml_pipeline.data import load_dataset, split_dataset
from ml_pipeline.exceptions import DataIntegrityError, DataValidationError
from ml_pipeline.io_utils import atomic_write_json

app = typer.Typer(
    name="ml-pipeline",
    help="Reproducible Telco churn pipeline: data validation, training, evaluation, serving.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

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


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


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
        typer.echo(f"Data integrity failure: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except DataValidationError as exc:
        payload |= {
            "passed": False,
            "issues": [issue.model_dump(mode="json") for issue in exc.issues],
        }
        atomic_write_json(destination, payload)
        typer.echo(str(exc), err=True)
        typer.echo(f"Report written to {destination.as_posix()}", err=True)
        raise typer.Exit(code=1) from exc

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
