"""Schema & data-quality validation: every contract violation fails loudly and cleanly."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config import DatasetConfig, PipelineConfig, SchemaConfig
from ml_pipeline.data import (
    TOTAL_CHARGES_RULE,
    DataQualityReport,
    PreparedDataset,
    fetch_dataset,
    load_dataset,
    prepare_dataset,
    read_raw_csv,
    validate_dataset,
)
from ml_pipeline.exceptions import DataIntegrityError, DataValidationError
from ml_pipeline.io_utils import canonical_json, sha256_bytes


def expect_error(raw: pd.DataFrame, config: PipelineConfig, check: str, column: str | None) -> str:
    """Assert ``prepare_dataset`` fails with ``check`` on ``column``; return the message."""
    with pytest.raises(DataValidationError) as info:
        prepare_dataset(raw, config)
    found = {
        (issue.check, issue.column) for issue in info.value.issues if issue.severity == "error"
    }
    assert (check, column) in found, f"expected {(check, column)}, got {sorted(map(str, found))}"
    message = str(info.value)
    assert message.startswith("Data validation failed with")
    if column is not None:
        assert f"'{column}'" in message
    return message


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestValidData:
    def test_report_passes_without_errors(self, prepared: PreparedDataset) -> None:
        report = prepared.report
        assert report.passed
        assert report.errors == ()
        assert report.n_rows == 1200
        assert report.duplicate_id_count == 0
        assert report.content_sha256 is not None

    def test_output_types(self, prepared: PreparedDataset, config: PipelineConfig) -> None:
        frame = prepared.frame
        schema = config.data_schema
        assert frame.index.name == schema.id_column
        assert frame.index.is_unique
        assert list(frame.columns) == [*schema.feature_names, schema.target_column]
        numeric = [spec.name for spec in schema.numeric_features]
        for name in (*numeric, *(spec.name for spec in schema.binary_features)):
            assert frame[name].dtype == np.float64, name
        for cat in schema.categorical_features:
            assert pd.api.types.is_string_dtype(frame[cat.name]), cat.name
        assert frame[schema.target_column].dtype == np.int64
        assert set(frame[schema.target_column].unique()) == {0, 1}

    def test_zero_tenure_total_charges_rule(
        self, prepared: PreparedDataset, raw_frame: pd.DataFrame
    ) -> None:
        zero_tenure = int((raw_frame["tenure"] == "0").sum())
        assert zero_tenure > 0
        assert prepared.report.rule_applications[TOTAL_CHARGES_RULE] == zero_tenure
        new_customers = prepared.frame[prepared.frame["tenure"] == 0]
        assert (new_customers["TotalCharges"] == 0.0).all()
        assert not prepared.frame["TotalCharges"].isna().any()

    def test_report_round_trips_through_canonical_json(self, prepared: PreparedDataset) -> None:
        text = canonical_json(prepared.report.model_dump(mode="json"))
        assert DataQualityReport.model_validate(json.loads(text)) == prepared.report

    def test_content_hash_is_deterministic(
        self, raw_frame: pd.DataFrame, config: PipelineConfig, prepared: PreparedDataset
    ) -> None:
        again = prepare_dataset(raw_frame, config)
        assert again.report.content_sha256 == prepared.report.content_sha256

    def test_typed_and_text_inputs_parse_identically(
        self, raw_frame: pd.DataFrame, config: PipelineConfig, prepared: PreparedDataset
    ) -> None:
        typed = raw_frame.copy()
        typed["tenure"] = typed["tenure"].astype(int)
        typed["SeniorCitizen"] = typed["SeniorCitizen"].astype(int)
        typed["MonthlyCharges"] = typed["MonthlyCharges"].astype(float)
        again = prepare_dataset(typed, config)
        assert again.report.content_sha256 == prepared.report.content_sha256

    def test_surrounding_whitespace_is_stripped(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        raw_frame.loc[0, "Contract"] = "  One year "
        raw_frame.loc[0, "tenure"] = " 12 "
        dataset = prepare_dataset(raw_frame, config)
        first = dataset.frame.iloc[0]
        assert first["Contract"] == "One year"
        assert first["tenure"] == 12.0

    def test_validate_dataset_reports_without_raising(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        raw_frame.loc[0, "Contract"] = "Weekly"
        report, _ = validate_dataset(raw_frame, config)
        assert not report.passed
        assert report.content_sha256 is None
        assert [issue.check for issue in report.errors] == ["unknown_category"]


# ---------------------------------------------------------------------------
# Schema violations
# ---------------------------------------------------------------------------
class TestSchemaViolations:
    @pytest.mark.parametrize("column", ["customerID", "tenure", "Contract", "Churn"])
    def test_missing_column(
        self, raw_frame: pd.DataFrame, config: PipelineConfig, column: str
    ) -> None:
        expect_error(raw_frame.drop(columns=[column]), config, "missing_column", column)

    def test_unexpected_column_rejected_by_default(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        expect_error(raw_frame.assign(leak="x"), config, "unexpected_column", "leak")

    def test_unexpected_column_allowed_when_configured(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        lenient = config.model_copy(update={"data_schema": SchemaConfig(allow_extra_columns=True)})
        dataset = prepare_dataset(raw_frame.assign(leak="x"), lenient)
        assert "leak" not in dataset.frame.columns
        assert [issue.check for issue in dataset.report.warnings] == ["unexpected_column"]

    def test_duplicate_column(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        doubled = pd.concat([raw_frame, raw_frame[["tenure"]]], axis=1)
        expect_error(doubled, config, "duplicate_column", "tenure")

    @pytest.mark.parametrize("bad_value", ["abc", "inf", "1e400", "12,5"])
    def test_non_numeric_value(
        self, raw_frame: pd.DataFrame, config: PipelineConfig, bad_value: str
    ) -> None:
        raw_frame.loc[3, "MonthlyCharges"] = bad_value
        message = expect_error(raw_frame, config, "type_mismatch", "MonthlyCharges")
        assert repr(bad_value) in message

    @pytest.mark.parametrize(
        ("column", "value"), [("tenure", "-1"), ("tenure", "500"), ("MonthlyCharges", "10000")]
    )
    def test_out_of_range(
        self, raw_frame: pd.DataFrame, config: PipelineConfig, column: str, value: str
    ) -> None:
        raw_frame.loc[5, column] = value
        expect_error(raw_frame, config, "out_of_range", column)

    def test_non_integer_count(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[5, "tenure"] = "12.5"
        expect_error(raw_frame, config, "non_integer", "tenure")

    def test_unknown_category(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[[1, 2], "Contract"] = "Weekly"
        message = expect_error(raw_frame, config, "unknown_category", "Contract")
        assert "'Weekly'" in message
        assert "2 value(s)" in message

    def test_invalid_binary(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[0, "SeniorCitizen"] = "2"
        expect_error(raw_frame, config, "invalid_binary", "SeniorCitizen")

    def test_invalid_target_label(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[0, "Churn"] = "Maybe"
        expect_error(raw_frame, config, "invalid_target_label", "Churn")

    def test_missing_target(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[0, "Churn"] = ""
        expect_error(raw_frame, config, "missing_target", "Churn")

    @pytest.mark.parametrize(
        ("column", "value", "implicated"),
        [
            ("MultipleLines", "No phone service", "MultipleLines"),
            ("TechSupport", "No internet service", "TechSupport"),
        ],
    )
    def test_cross_field_inconsistency(
        self,
        raw_frame: pd.DataFrame,
        config: PipelineConfig,
        column: str,
        value: str,
        implicated: str,
    ) -> None:
        row = raw_frame.index[
            (raw_frame["PhoneService"] == "Yes") & (raw_frame["InternetService"] != "No")
        ][0]
        raw_frame.loc[row, column] = value
        message = expect_error(raw_frame, config, "inconsistent_values", implicated)
        assert "must coincide with" in message

    def test_blank_id(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[0, "customerID"] = "  "
        expect_error(raw_frame, config, "missing_id", "customerID")

    def test_duplicate_id(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame.loc[1, "customerID"] = "0000-SYNTH"  # the ID of row 0
        message = expect_error(raw_frame, config, "duplicate_id", "customerID")
        assert "0000-SYNTH" in message


# ---------------------------------------------------------------------------
# Data-quality violations
# ---------------------------------------------------------------------------
class TestQualityViolations:
    def test_missing_values_above_threshold(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        raw_frame.loc[raw_frame.index[:60], "MonthlyCharges"] = ""
        expect_error(raw_frame, config, "missing_values", "MonthlyCharges")

    def test_missing_values_within_threshold_are_kept_for_pipeline_imputation(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        billed = raw_frame.index[raw_frame["tenure"] != "0"][:6]  # 0.5% < 1% allowance
        raw_frame.loc[billed, "TotalCharges"] = " "
        dataset = prepare_dataset(raw_frame, config)
        warnings = [(issue.check, issue.column) for issue in dataset.report.warnings]
        assert ("missing_values", "TotalCharges") in warnings
        assert int(dataset.frame["TotalCharges"].isna().sum()) == 6

    def test_too_few_rows(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        expect_error(raw_frame.head(100), config, "too_few_rows", None)

    def test_single_class_target(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame["Churn"] = "No"
        expect_error(raw_frame, config, "single_class", "Churn")

    def test_extreme_class_imbalance(self, raw_frame: pd.DataFrame, config: PipelineConfig) -> None:
        raw_frame["Churn"] = "No"
        raw_frame.loc[raw_frame.index[:24], "Churn"] = "Yes"  # 2% < 5% minimum
        expect_error(raw_frame, config, "class_imbalance", "Churn")

    def test_few_duplicate_feature_rows_only_warn(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        copies = raw_frame.head(3).assign(customerID=["DUP-1", "DUP-2", "DUP-3"])
        dataset = prepare_dataset(pd.concat([raw_frame, copies], ignore_index=True), config)
        assert dataset.report.duplicate_feature_row_count == 3
        assert [issue.check for issue in dataset.report.warnings] == ["duplicate_feature_rows"]

    def test_many_duplicate_feature_rows_fail(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        copies = raw_frame.head(60).copy()
        copies["customerID"] = [f"DUP-{i}" for i in range(60)]
        expect_error(
            pd.concat([raw_frame, copies], ignore_index=True),
            config,
            "duplicate_feature_rows",
            None,
        )

    def test_conflicting_duplicate_labels_are_counted(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        twin = raw_frame.head(1).assign(customerID="TWIN-1")
        twin["Churn"] = "Yes" if raw_frame.loc[0, "Churn"] == "No" else "No"
        dataset = prepare_dataset(pd.concat([raw_frame, twin], ignore_index=True), config)
        assert dataset.report.conflicting_duplicate_count == 1

    def test_all_violations_are_reported_together(
        self, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        raw_frame.loc[0, "Contract"] = "Weekly"
        raw_frame.loc[1, "tenure"] = "-3"
        raw_frame.loc[2, "MonthlyCharges"] = "n/a"
        raw_frame.loc[3, "Churn"] = "Maybe"
        with pytest.raises(DataValidationError) as info:
            prepare_dataset(raw_frame.drop(columns=["gender"]), config)
        error = info.value
        assert error.columns >= {"Contract", "tenure", "MonthlyCharges", "Churn", "gender"}
        assert "with 5 error(s)" in str(error)
        assert "Traceback" not in str(error)


# ---------------------------------------------------------------------------
# Ingestion (network is faked)
# ---------------------------------------------------------------------------
class _FakeResponse(io.BytesIO):
    """``urlopen`` stand-in: a readable, closeable context manager."""


def _dataset_config(content: bytes) -> DatasetConfig:
    return DatasetConfig(sha256=sha256_bytes(content), filename="data.csv")


class TestIngestion:
    def test_read_raw_csv_preserves_text(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.csv"
        path.write_bytes(b"id,code,total\n007,NA, \n")
        frame = read_raw_csv(path)
        assert frame.iloc[0].tolist() == ["007", "NA", " "]

    def test_fetch_downloads_and_verifies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = b"a,b\n1,2\n"
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _FakeResponse(content))
        path = fetch_dataset(_dataset_config(content), tmp_path)
        assert path.read_bytes() == content

    def test_fetch_reuses_verified_cache_without_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = b"a,b\n1,2\n"
        (tmp_path / "data.csv").write_bytes(content)

        def offline(*_: object, **__: object) -> _FakeResponse:
            raise AssertionError("network must not be used when the cache is valid")

        monkeypatch.setattr(urllib.request, "urlopen", offline)
        assert fetch_dataset(_dataset_config(content), tmp_path).read_bytes() == content

    def test_fetch_replaces_corrupt_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = b"a,b\n1,2\n"
        (tmp_path / "data.csv").write_bytes(b"corrupted")
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _FakeResponse(content))
        assert fetch_dataset(_dataset_config(content), tmp_path).read_bytes() == content

    def test_fetch_rejects_checksum_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _FakeResponse(b"evil"))
        with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
            fetch_dataset(_dataset_config(b"expected"), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_fetch_wraps_network_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreachable(*_: object, **__: object) -> _FakeResponse:
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(urllib.request, "urlopen", unreachable)
        with pytest.raises(DataIntegrityError, match="failed to download"):
            fetch_dataset(_dataset_config(b"x"), tmp_path)

    def test_load_dataset_reports_unpinned_local_file(
        self, tmp_path: Path, raw_frame: pd.DataFrame, config: PipelineConfig
    ) -> None:
        path = tmp_path / "local.csv"
        raw_frame.to_csv(path, index=False)
        dataset = load_dataset(config, data_path=path)
        assert dataset.source is not None
        assert dataset.source.verified is False
        assert dataset.source.pinned_sha256 == config.dataset.sha256
        assert dataset.report.passed


@pytest.mark.network
def test_pinned_reference_dataset_profile(config: PipelineConfig) -> None:
    """Downloads (or reuses the verified cache of) the real dataset and pins its profile."""
    dataset = load_dataset(config)
    report = dataset.report
    assert dataset.source is not None
    assert dataset.source.verified
    assert report.passed
    assert report.n_rows == 7043
    assert report.positive_count == 1869
    assert report.rule_applications[TOTAL_CHARGES_RULE] == 11
    assert report.duplicate_id_count == 0
    assert [issue.check for issue in report.warnings] == ["duplicate_feature_rows"]
