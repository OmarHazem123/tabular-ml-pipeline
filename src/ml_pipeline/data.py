"""Deterministic ingestion, schema & data-quality validation, and leak-free splitting.

Flow::

    fetch_dataset ──> read_raw_csv ──> prepare_dataset ──> split_dataset
     (pinned URL,      (all columns     (parse + validate,   (stratified, seeded,
      SHA-256 check)    as raw text)     collect ALL issues)  BEFORE any fitting)

Nothing in this module learns statistics from the data. The only value rewrite is a
row-local domain rule (``TotalCharges`` is blank exactly for customers with
``tenure == 0``, i.e. not yet billed, so it is set to 0). Every learned transformation
(imputation, clipping, scaling, encoding) lives in :mod:`ml_pipeline.features` and is fit
on the training split only.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split

from ml_pipeline.config import (
    BinaryFeature,
    CategoricalFeature,
    DatasetConfig,
    NumericFeature,
    PipelineConfig,
)
from ml_pipeline.exceptions import DataIntegrityError, DataValidationError, ValidationIssue
from ml_pipeline.io_utils import atomic_write_chunks, sha256_bytes, sha256_file, sha256_text

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 16
_MAX_EXAMPLES = 5
TOTAL_CHARGES_RULE = "total_charges_zero_for_zero_tenure"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSource(_Model):
    """Where the raw bytes came from and whether they match the pinned checksum."""

    path: str
    sha256: str
    pinned_sha256: str
    verified: bool
    url: str | None = None


class ColumnProfile(_Model):
    name: str
    role: Literal["id", "target", "numeric", "binary", "categorical"]
    dtype: str
    missing_count: int
    missing_fraction: float
    n_unique: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None


class DataQualityReport(_Model):
    n_rows: int
    n_columns: int
    positive_count: int | None
    positive_rate: float | None
    duplicate_id_count: int
    duplicate_feature_row_count: int
    conflicting_duplicate_count: int
    rule_applications: dict[str, int]
    columns: tuple[ColumnProfile, ...]
    issues: tuple[ValidationIssue, ...]
    content_sha256: str | None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PreparedDataset:
    """Validated, typed dataset indexed by entity ID; target encoded as 0/1."""

    frame: pd.DataFrame
    report: DataQualityReport
    target_column: str
    source: DatasetSource | None = None

    @property
    def features(self) -> pd.DataFrame:
        return self.frame.drop(columns=[self.target_column])

    @property
    def target(self) -> pd.Series[int]:
        return self.frame[self.target_column]


class SplitSummary(_Model):
    n_rows: int
    positive_count: int
    positive_rate: float
    ids_sha256: str


@dataclass(frozen=True)
class DataSplits:
    """Disjoint stratified partitions. Constructed only via :func:`split_dataset`."""

    X_train: pd.DataFrame
    y_train: pd.Series[int]
    X_val: pd.DataFrame
    y_val: pd.Series[int]
    X_test: pd.DataFrame
    y_test: pd.Series[int]

    def __post_init__(self) -> None:
        parts = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }
        for name, (features, target) in parts.items():
            if not features.index.equals(target.index):
                raise ValueError(f"{name}: feature and target indices are misaligned")
        seen: set[object] = set()
        for name, (features, _) in parts.items():
            ids = set(features.index)
            overlap = seen & ids
            if overlap:
                raise ValueError(f"{name} split overlaps an earlier split ({len(overlap)} ids)")
            seen |= ids

    def summary(self) -> dict[str, SplitSummary]:
        return {
            "train": _summarise(self.X_train, self.y_train),
            "val": _summarise(self.X_val, self.y_val),
            "test": _summarise(self.X_test, self.y_test),
        }


def _summarise(features: pd.DataFrame, target: pd.Series[int]) -> SplitSummary:
    positives = int(target.sum())
    return SplitSummary(
        n_rows=len(target),
        positive_count=positives,
        positive_rate=positives / len(target) if len(target) else 0.0,
        ids_sha256=ids_fingerprint(features.index),
    )


def ids_fingerprint(index: pd.Index) -> str:
    """Order-independent SHA-256 of a set of entity IDs."""
    return sha256_text("\n".join(sorted(str(value) for value in index)))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def fetch_dataset(config: DatasetConfig, cache_dir: Path, *, force: bool = False) -> Path:
    """Return a local copy of the pinned dataset, downloading it if needed.

    A cached file is reused only if its SHA-256 matches the pinned checksum. Downloads are
    streamed to a temp file and only moved into place once the checksum is verified.
    """
    target = cache_dir / config.filename
    if target.exists() and not force:
        cached = sha256_file(target)
        if cached == config.sha256:
            logger.info("Using cached dataset %s (sha256 verified)", target)
            return target
        logger.warning("Cached dataset checksum mismatch (%s); re-downloading", cached)

    logger.info("Downloading %s", config.url)
    try:
        # The URL scheme is constrained to https:// by DatasetConfig validation.
        with urllib.request.urlopen(  # noqa: S310
            config.url, timeout=config.download_timeout_seconds
        ) as response:
            atomic_write_chunks(
                target,
                iter(lambda: response.read(_DOWNLOAD_CHUNK_BYTES), b""),
                expected_sha256=config.sha256,
            )
    except OSError as exc:
        raise DataIntegrityError(f"failed to download dataset from {config.url}: {exc}") from exc
    logger.info("Saved dataset to %s (sha256 verified)", target)
    return target


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read every column as raw text; typing happens explicitly in :func:`prepare_dataset`."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")


def load_dataset(config: PipelineConfig, *, data_path: Path | None = None) -> PreparedDataset:
    """Fetch (or read a user-supplied file), validate, and return the prepared dataset."""
    if data_path is None:
        path = fetch_dataset(config.dataset, config.paths.raw_data_dir)
        url: str | None = config.dataset.url
    else:
        path, url = data_path, None
    digest = sha256_file(path)
    verified = digest == config.dataset.sha256
    if not verified:
        logger.warning(
            "Dataset %s does not match the pinned checksum; results will not be "
            "reproducible against the reference dataset",
            path,
        )
    source = DatasetSource(
        path=path.as_posix(),
        sha256=digest,
        pinned_sha256=config.dataset.sha256,
        verified=verified,
        url=url,
    )
    dataset = prepare_dataset(read_raw_csv(path), config)
    return PreparedDataset(
        frame=dataset.frame,
        report=dataset.report,
        target_column=dataset.target_column,
        source=source,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _clean_text(series: pd.Series) -> pd.Series[str]:
    """Strip whitespace; blank strings and nulls become NaN."""
    text = series.astype("str").str.strip()
    return text.mask(text == "")


def _parse_numeric(series: pd.Series) -> tuple[pd.Series[float], pd.Series[bool]]:
    """Parse to float64. Returns ``(values, invalid_mask)``; blanks are missing, not invalid."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = series.astype("float64")
        present = values.notna()
    else:
        text = _clean_text(series)
        present = text.notna()
        values = pd.to_numeric(text, errors="coerce").astype("float64")
    invalid = present & ~np.isfinite(values.fillna(0.0))
    invalid |= present & values.isna()
    return values.mask(invalid), invalid


def _examples(values: pd.Series) -> str:
    uniques = sorted({repr(value) for value in values.tolist()})
    shown = ", ".join(uniques[:_MAX_EXAMPLES])
    suffix = ", ..." if len(uniques) > _MAX_EXAMPLES else ""
    return f"[{shown}{suffix}]"


def _error(
    check: str, message: str, column: str | None = None, count: int | None = None
) -> ValidationIssue:
    return ValidationIssue(
        check=check, severity="error", message=message, column=column, count=count
    )


def _warning(
    check: str, message: str, column: str | None = None, count: int | None = None
) -> ValidationIssue:
    return ValidationIssue(
        check=check, severity="warning", message=message, column=column, count=count
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class _Validation:
    """Mutable working state. ``frame`` stays on the raw positional index until the end,
    so every parsed column aligns row-for-row with the raw input."""

    frame: pd.DataFrame
    issues: list[ValidationIssue]
    raw_missing: dict[str, pd.Series[bool]]
    rules: dict[str, int]
    ids: pd.Series[str] | None = None


def _check_structure(raw: pd.DataFrame, config: PipelineConfig) -> list[ValidationIssue]:
    schema = config.data_schema
    issues: list[ValidationIssue] = []
    names = [str(column) for column in raw.columns]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    for name in duplicated:
        issues.append(_error("duplicate_column", "column appears more than once", name))
    for name in schema.required_columns:
        if name not in names:
            issues.append(_error("missing_column", "required column is missing", name))
    for name in names:
        if name not in schema.required_columns:
            make = _warning if schema.allow_extra_columns else _error
            issues.append(make("unexpected_column", "column is not part of the schema", name))
    return issues


def _parse_features(raw: pd.DataFrame, config: PipelineConfig, state: _Validation) -> None:
    for spec in config.data_schema.features:
        if spec.name not in raw.columns:
            continue
        column = raw[spec.name]
        if isinstance(spec, CategoricalFeature):
            text = _clean_text(column)
            state.raw_missing[spec.name] = text.isna()
            unknown = text[text.notna() & ~text.isin(spec.categories)]
            if not unknown.empty:
                state.issues.append(
                    _error(
                        "unknown_category",
                        f"{len(unknown)} value(s) outside {list(spec.categories)}: "
                        f"{_examples(unknown)}",
                        spec.name,
                        len(unknown),
                    )
                )
            state.frame[spec.name] = text
            continue

        values, invalid = _parse_numeric(column)
        state.raw_missing[spec.name] = values.isna() & ~invalid
        if invalid.any():
            state.issues.append(
                _error(
                    "type_mismatch",
                    f"{int(invalid.sum())} non-numeric value(s): {_examples(column[invalid])}",
                    spec.name,
                    int(invalid.sum()),
                )
            )
        present = values.dropna()
        if isinstance(spec, BinaryFeature):
            bad = present[~present.isin((0.0, 1.0))]
            if not bad.empty:
                state.issues.append(
                    _error(
                        "invalid_binary",
                        f"{len(bad)} value(s) not in {{0, 1}}: {_examples(bad)}",
                        spec.name,
                        len(bad),
                    )
                )
        elif isinstance(spec, NumericFeature):
            if spec.integer:
                fractional = present[present != np.floor(present)]
                if not fractional.empty:
                    state.issues.append(
                        _error(
                            "non_integer",
                            f"{len(fractional)} non-integer value(s): {_examples(fractional)}",
                            spec.name,
                            len(fractional),
                        )
                    )
            outside = present[(present < spec.min_value) | (present > spec.max_value)]
            if not outside.empty:
                state.issues.append(
                    _error(
                        "out_of_range",
                        f"{len(outside)} value(s) outside [{spec.min_value:g}, "
                        f"{spec.max_value:g}]: {_examples(outside)}",
                        spec.name,
                        len(outside),
                    )
                )
        state.frame[spec.name] = values


def _apply_domain_rules(config: PipelineConfig, state: _Validation) -> None:
    """Row-local rule: an unbilled new customer (tenure == 0) has TotalCharges == 0."""
    fc = config.features
    frame = state.frame
    if fc.tenure_column not in frame.columns or fc.total_charges_column not in frame.columns:
        return
    rule_mask = frame[fc.total_charges_column].isna() & frame[fc.tenure_column].eq(0.0)
    frame.loc[rule_mask, fc.total_charges_column] = 0.0
    state.raw_missing[fc.total_charges_column] &= ~rule_mask
    state.rules[TOTAL_CHARGES_RULE] = int(rule_mask.sum())


def _check_missing(config: PipelineConfig, state: _Validation, n_rows: int) -> None:
    for spec in config.data_schema.features:
        mask = state.raw_missing.get(spec.name)
        if mask is None or n_rows == 0:
            continue
        count = int(mask.sum())
        fraction = count / n_rows
        if fraction > spec.max_missing_fraction:
            state.issues.append(
                _error(
                    "missing_values",
                    f"{count} missing ({fraction:.2%}) exceeds the allowed "
                    f"{spec.max_missing_fraction:.2%}",
                    spec.name,
                    count,
                )
            )
        elif count:
            state.issues.append(
                _warning(
                    "missing_values",
                    f"{count} missing ({fraction:.2%}); will be imputed inside the pipeline",
                    spec.name,
                    count,
                )
            )


def _parse_ids(raw: pd.DataFrame, config: PipelineConfig, state: _Validation) -> int:
    id_column = config.data_schema.id_column
    if id_column not in raw.columns:
        return 0
    ids = _clean_text(raw[id_column])
    missing = int(ids.isna().sum())
    if missing:
        state.issues.append(_error("missing_id", f"{missing} blank ID(s)", id_column, missing))
    duplicated = ids[ids.notna() & ids.duplicated(keep=False)]
    duplicate_count = int(ids[ids.notna()].duplicated().sum())
    if duplicate_count:
        state.issues.append(
            _error(
                "duplicate_id",
                f"{duplicate_count} duplicated ID(s): {_examples(duplicated)}",
                id_column,
                duplicate_count,
            )
        )
    state.ids = ids
    return duplicate_count


def _parse_target(raw: pd.DataFrame, config: PipelineConfig, state: _Validation) -> None:
    schema = config.data_schema
    name = schema.target_column
    if name not in raw.columns:
        return
    labels = _clean_text(raw[name])
    missing = int(labels.isna().sum())
    if missing:
        state.issues.append(_error("missing_target", f"{missing} blank label(s)", name, missing))
    allowed = (schema.negative_label, schema.positive_label)
    invalid = labels[labels.notna() & ~labels.isin(allowed)]
    if not invalid.empty:
        state.issues.append(
            _error(
                "invalid_target_label",
                f"{len(invalid)} label(s) outside {list(allowed)}: {_examples(invalid)}",
                name,
                len(invalid),
            )
        )
    encoded = labels.map({schema.negative_label: 0, schema.positive_label: 1})
    state.frame[name] = encoded.to_numpy()
    if missing or not invalid.empty:
        return
    state.frame[name] = state.frame[name].astype("int64")
    positives = int(state.frame[name].sum())
    n_rows = len(labels)
    minority = min(positives, n_rows - positives) / n_rows if n_rows else 0.0
    if positives in (0, n_rows):
        state.issues.append(_error("single_class", "target contains only one class", name))
    elif minority < config.quality.min_minority_class_fraction:
        state.issues.append(
            _error(
                "class_imbalance",
                f"minority class fraction {minority:.2%} is below "
                f"{config.quality.min_minority_class_fraction:.2%}",
                name,
            )
        )


def _check_duplicate_rows(config: PipelineConfig, state: _Validation) -> tuple[int, int]:
    schema = config.data_schema
    feature_columns = [name for name in schema.feature_names if name in state.frame.columns]
    if not feature_columns or state.frame.empty:
        return 0, 0
    features = state.frame[feature_columns]
    duplicate_count = int(features.duplicated().sum())
    conflicting = 0
    if duplicate_count and schema.target_column in state.frame.columns:
        grouped = state.frame.reset_index(drop=True).groupby(feature_columns, dropna=False)
        label_counts = grouped[schema.target_column].nunique(dropna=False)
        conflicting = int((label_counts > 1).sum())
    if duplicate_count:
        fraction = duplicate_count / len(features)
        limit = config.quality.max_duplicate_feature_row_fraction
        make = _error if fraction > limit else _warning
        state.issues.append(
            make(
                "duplicate_feature_rows",
                f"{duplicate_count} row(s) ({fraction:.2%}) duplicate another row's features "
                f"(limit {limit:.2%}); {conflicting} group(s) have conflicting labels",
                count=duplicate_count,
            )
        )
    return duplicate_count, conflicting


def _profile(config: PipelineConfig, frame: pd.DataFrame) -> tuple[ColumnProfile, ...]:
    schema = config.data_schema
    profiles: list[ColumnProfile] = []
    n_rows = len(frame)
    index_series = frame.index.to_series()
    profiles.append(
        ColumnProfile(
            name=schema.id_column,
            role="id",
            dtype=str(frame.index.dtype),
            missing_count=int(index_series.isna().sum()),
            missing_fraction=float(index_series.isna().mean()) if n_rows else 0.0,
            n_unique=int(index_series.nunique()),
        )
    )
    for name in (*schema.feature_names, schema.target_column):
        if name not in frame.columns:
            continue
        column = frame[name]
        role: Literal["target", "numeric", "binary", "categorical"]
        role = "target" if name == schema.target_column else schema.get(name).kind
        numeric = role in ("numeric", "binary", "target") and pd.api.types.is_numeric_dtype(column)
        profiles.append(
            ColumnProfile(
                name=name,
                role=role,
                dtype=str(column.dtype),
                missing_count=int(column.isna().sum()),
                missing_fraction=float(column.isna().mean()) if n_rows else 0.0,
                n_unique=int(column.nunique()),
                min=float(column.min()) if numeric and column.notna().any() else None,
                max=float(column.max()) if numeric and column.notna().any() else None,
                mean=float(column.mean()) if numeric and column.notna().any() else None,
            )
        )
    return tuple(profiles)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 over values, index, column names and dtypes of a frame."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    header = "|".join(f"{name}:{dtype}" for name, dtype in frame.dtypes.items())
    return sha256_bytes(header.encode("utf-8") + row_hashes.tobytes())


def validate_dataset(
    raw: pd.DataFrame, config: PipelineConfig
) -> tuple[DataQualityReport, pd.DataFrame]:
    """Parse and validate ``raw`` without raising; returns the report and the typed frame."""
    schema = config.data_schema
    issues = _check_structure(raw, config)
    # Positional index + first occurrence of any duplicated column name, so every
    # column lookup below yields exactly one Series aligned with the raw rows.
    raw = raw.loc[:, ~raw.columns.duplicated()].reset_index(drop=True)
    raw.columns = [str(column) for column in raw.columns]
    state = _Validation(
        frame=pd.DataFrame(index=pd.RangeIndex(len(raw))),
        issues=issues,
        raw_missing={},
        rules={},
    )
    duplicate_ids = _parse_ids(raw, config, state)
    _parse_features(raw, config, state)
    _apply_domain_rules(config, state)
    _check_missing(config, state, len(raw))
    _parse_target(raw, config, state)
    if len(raw) < config.quality.min_rows:
        state.issues.append(
            _error(
                "too_few_rows",
                f"{len(raw)} row(s) is below the minimum of {config.quality.min_rows}",
            )
        )
    duplicate_rows, conflicting = _check_duplicate_rows(config, state)

    ordered = [
        name for name in (*schema.feature_names, schema.target_column) if name in state.frame
    ]
    frame = state.frame[ordered]
    if state.ids is not None:
        frame.index = pd.Index(state.ids.to_numpy(), name=schema.id_column)
    target = frame[schema.target_column] if schema.target_column in frame.columns else None
    target_ok = target is not None and pd.api.types.is_integer_dtype(target)
    positives = int(target.sum()) if target is not None and target_ok else None
    has_errors = any(issue.severity == "error" for issue in state.issues)
    report = DataQualityReport(
        n_rows=len(raw),
        n_columns=len(raw.columns),
        positive_count=positives,
        positive_rate=positives / len(raw) if positives is not None and len(raw) else None,
        duplicate_id_count=duplicate_ids,
        duplicate_feature_row_count=duplicate_rows,
        conflicting_duplicate_count=conflicting,
        rule_applications=dict(state.rules),
        columns=_profile(config, frame),
        issues=tuple(state.issues),
        content_sha256=None if has_errors else frame_fingerprint(frame),
    )
    return report, frame


def prepare_dataset(raw: pd.DataFrame, config: PipelineConfig) -> PreparedDataset:
    """Validate ``raw`` and return a typed dataset, or raise :class:`DataValidationError`."""
    report, frame = validate_dataset(raw, config)
    if not report.passed:
        raise DataValidationError(report.issues)
    for issue in report.warnings:
        logger.warning("Data quality warning %s", issue.render())
    return PreparedDataset(
        frame=frame, report=report, target_column=config.data_schema.target_column
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _stratified_split(
    X: pd.DataFrame, y: pd.Series[int], test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[int], pd.Series[int]]:
    X_keep, X_hold, y_keep, y_hold = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed, shuffle=True
    )
    return X_keep, X_hold, y_keep, y_hold


def split_dataset(dataset: PreparedDataset, config: PipelineConfig) -> DataSplits:
    """Stratified train/val/test split, performed BEFORE any preprocessing is fit.

    Rows are sorted by ID first, so the partition depends only on (data, seed, fractions)
    and not on the row order of the source file.
    """
    frame = dataset.frame.sort_index(kind="mergesort")
    X = frame.drop(columns=[dataset.target_column])
    y: pd.Series[int] = frame[dataset.target_column]
    split = config.split
    X_rest, X_test, y_rest, y_test = _stratified_split(X, y, split.test_fraction, config.seed)
    val_share = split.val_fraction / (split.train_fraction + split.val_fraction)
    X_train, X_val, y_train, y_val = _stratified_split(X_rest, y_rest, val_share, config.seed)
    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
