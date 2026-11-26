"""CLI commands against the session-trained registry (no network, no retraining)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ml_pipeline.cli import app
from ml_pipeline.io_utils import atomic_write_json, sha256_file
from ml_pipeline.registry import ModelRegistry, load_model_dir
from ml_pipeline.schemas import EXAMPLE_CUSTOMER
from tests.conftest import TrainedPipeline

runner = CliRunner()


@pytest.fixture(scope="module")
def config_file(trained: TrainedPipeline, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "config.json"
    atomic_write_json(path, trained.config.model_dump(mode="json"))
    return path


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("validate-data", "train", "evaluate", "serve", "predict", "export-model"):
        assert command in result.output


def test_validate_data_writes_lf_report(synthetic_csv: Path, tmp_path: Path) -> None:
    report = tmp_path / "validation.json"
    result = runner.invoke(
        app, ["validate-data", "--data-path", str(synthetic_csv), "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text("utf-8"))["passed"] is True
    assert b"\r" not in report.read_bytes()


def test_validate_data_fails_cleanly_on_bad_data(synthetic_csv: Path, tmp_path: Path) -> None:
    frame = pd.read_csv(synthetic_csv, dtype=str, keep_default_na=False)
    frame.loc[0, "Contract"] = "Weekly"
    bad = tmp_path / "bad.csv"
    frame.to_csv(bad, index=False, lineterminator="\n")
    report = tmp_path / "validation.json"
    result = runner.invoke(app, ["validate-data", "--data-path", str(bad), "--report", str(report)])
    assert result.exit_code == 1
    assert "unknown_category" in result.output
    assert json.loads(report.read_text("utf-8"))["passed"] is False


def test_predict_from_json_file(config_file: Path, tmp_path: Path) -> None:
    source = tmp_path / "customers.json"
    rows = [{**EXAMPLE_CUSTOMER, "customer_id": "a"}, {**EXAMPLE_CUSTOMER, "tenure": 40}]
    source.write_text(json.dumps({"instances": rows}), encoding="utf-8")
    output = tmp_path / "predictions.json"
    result = runner.invoke(
        app, ["predict", str(source), "-o", str(output), "--config", str(config_file)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text("utf-8"))
    assert payload["n_predictions"] == 2
    assert payload["predictions"][0]["customer_id"] == "a"
    assert len(payload["predictions"][1]["reasons"]) == 3
    assert b"\r" not in output.read_bytes()


def test_predict_from_raw_dataset_csv(
    config_file: Path, synthetic_csv: Path, tmp_path: Path
) -> None:
    frame = pd.read_csv(synthetic_csv, dtype=str, keep_default_na=False)
    sample = pd.concat([frame.head(10), frame[frame["tenure"] == "0"].head(2)])
    source = tmp_path / "raw_sample.csv"
    sample.to_csv(source, index=False, lineterminator="\n")
    output = tmp_path / "predictions.json"
    result = runner.invoke(
        app,
        ["predict", str(source), "-o", str(output), "--config", str(config_file), "--top-k", "2"],
    )
    assert result.exit_code == 0, result.output
    predictions = json.loads(output.read_text("utf-8"))["predictions"]
    assert [p["customer_id"] for p in predictions] == sample["customerID"].tolist()
    assert all(len(p["reasons"]) == 2 for p in predictions)


def test_predict_to_stdout(config_file: Path, tmp_path: Path) -> None:
    source = tmp_path / "one.json"
    source.write_text(json.dumps([EXAMPLE_CUSTOMER]), encoding="utf-8")
    result = runner.invoke(app, ["predict", str(source), "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["n_predictions"] == 1


def test_predict_reports_every_invalid_row(config_file: Path, tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    rows = [EXAMPLE_CUSTOMER, {**EXAMPLE_CUSTOMER, "Contract": "Weekly"}, {"tenure": 5}]
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "never.json"
    result = runner.invoke(
        app, ["predict", str(source), "-o", str(output), "--config", str(config_file)]
    )
    assert result.exit_code == 1
    assert "row 1: Contract" in result.output
    assert "row 2:" in result.output
    assert "2 invalid row(s)" in result.output
    assert not output.exists()


def test_export_model_produces_a_verified_copy(
    config_file: Path, trained: TrainedPipeline, tmp_path: Path
) -> None:
    destination = tmp_path / "build" / "model"
    result = runner.invoke(
        app, ["export-model", "--dest", str(destination), "--config", str(config_file)]
    )
    assert result.exit_code == 0, result.output
    model, metadata = load_model_dir(destination)
    assert metadata.model_id == trained.report.model_id
    registry = ModelRegistry(trained.config.paths.registry_dir)
    original = registry.model_dir(metadata.model_id) / metadata.artifact.filename
    assert sha256_file(destination / metadata.artifact.filename) == sha256_file(original)
    assert model.threshold == metadata.threshold


def test_evaluate_rerun_reports_the_decision(config_file: Path, trained: TrainedPipeline) -> None:
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config_file),
            "--data-path",
            str(trained.data_path),
            "--fail-on-reject",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "APPROVED" in result.output
    assert trained.report.model_id in result.output
