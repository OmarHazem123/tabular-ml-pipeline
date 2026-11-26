"""Local, file-based model registry with integrity checks and a promotion gate.

Layout::

    <registry>/
      registry.json                  index: production pointer, stages, event log
      models/<model_id>/
        model.joblib                 the ChurnModel artifact
        metadata.json                lineage, metrics, SHA-256, promotion decision

Integrity: the artifact's SHA-256 is recorded at registration and re-verified BEFORE every
unpickle, so a modified or corrupted artifact is never deserialised (``joblib.load`` can
execute code). All JSON is written atomically with LF endings. The registry assumes a
single writer (one training/evaluation process at a time).
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import io
import math
import platform
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import joblib
from pydantic import BaseModel, ConfigDict, Field

from ml_pipeline.config import ParamValue, PromotionConfig
from ml_pipeline.data import DatasetSource, SplitSummary
from ml_pipeline.exceptions import PipelineError
from ml_pipeline.io_utils import atomic_write_bytes, atomic_write_json, sha256_bytes
from ml_pipeline.metrics import ClassificationMetrics, FoldScores
from ml_pipeline.model import ChurnModel

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
INDEX_FILENAME = "registry.json"
Stage = Literal["candidate", "production", "archived", "rejected"]
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost-cpu",
    "shap",
    "pydantic",
    "fastapi",
)


class ModelIntegrityError(PipelineError):
    """Artifact missing, unreadable, or not matching its recorded SHA-256."""


class RegistryError(PipelineError):
    """Invalid registry operation (unknown model, illegal stage transition, ...)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------
class GitInfo(_Model):
    commit: str | None
    dirty: bool | None


def git_info(cwd: Path | None = None) -> GitInfo:
    """Current commit and whether tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git resolved from PATH by design
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return GitInfo(commit=None, dirty=None)
    return GitInfo(commit=commit, dirty=bool(status.strip()))


def environment_info() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Lineage(_Model):
    run_id: str
    git: GitInfo
    config_fingerprint: str
    seed: int
    dataset: DatasetSource | None
    dataset_content_sha256: str | None
    splits: dict[str, SplitSummary]
    environment: dict[str, str]


class ArtifactInfo(_Model):
    filename: str
    sha256: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------
class GateCheck(_Model):
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    detail: str


class PromotionDecision(_Model):
    approved: bool
    candidate_id: str
    production_id: str | None
    primary_metric: str
    checks: tuple[GateCheck, ...]
    decided_at: str

    def summary(self) -> str:
        failed = [check.name for check in self.checks if not check.passed]
        if self.approved:
            return f"APPROVED: {self.candidate_id} passed all {len(self.checks)} checks"
        return f"REJECTED: {self.candidate_id} failed {', '.join(failed)}"


def _floor_check(name: str, value: float, minimum: float) -> GateCheck:
    return GateCheck(
        name=name,
        passed=value >= minimum,
        value=value,
        threshold=minimum,
        detail=f"{value:.4f} >= {minimum:.4f}",
    )


def evaluate_promotion(
    *,
    candidate_id: str,
    candidate: ClassificationMetrics,
    baseline: ClassificationMetrics | None,
    candidate_is_baseline: bool,
    production_id: str | None,
    production: ClassificationMetrics | None,
    gates: PromotionConfig,
) -> PromotionDecision:
    """Pure decision function. Every metric must come from the SAME test split."""
    primary = gates.primary_metric
    value = float(getattr(candidate, primary))
    checks = [
        _floor_check("roc_auc_floor", candidate.roc_auc, gates.min_roc_auc),
        _floor_check("pr_auc_floor", candidate.pr_auc, gates.min_pr_auc),
        _floor_check("f1_floor", candidate.f1, gates.min_f1),
        GateCheck(
            name="brier_ceiling",
            passed=candidate.brier <= gates.max_brier,
            value=candidate.brier,
            threshold=gates.max_brier,
            detail=f"{candidate.brier:.4f} <= {gates.max_brier:.4f}",
        ),
    ]

    if candidate_is_baseline or baseline is None:
        checks.append(
            GateCheck(
                name="not_worse_than_baseline",
                passed=True,
                value=value,
                threshold=None,
                detail="candidate is the baseline model",
            )
        )
    else:
        required = float(getattr(baseline, primary)) - gates.baseline_tolerance
        checks.append(
            GateCheck(
                name="not_worse_than_baseline",
                passed=value >= required,
                value=value,
                threshold=required,
                detail=f"{primary} {value:.4f} >= baseline - tolerance {required:.4f}",
            )
        )

    if production is None:
        checks.append(
            GateCheck(
                name="beats_production",
                passed=True,
                value=value,
                threshold=None,
                detail="no production model yet",
            )
        )
    else:
        required = float(getattr(production, primary)) + gates.min_improvement_over_production
        checks.append(
            GateCheck(
                name="beats_production",
                passed=value >= required,
                value=value,
                threshold=required,
                detail=f"{primary} {value:.4f} >= production + margin {required:.4f}",
            )
        )

    return PromotionDecision(
        approved=all(check.passed for check in checks),
        candidate_id=candidate_id,
        production_id=production_id,
        primary_metric=primary,
        checks=tuple(checks),
        decided_at=utc_now(),
    )


class SuggestedFloors(_Model):
    min_roc_auc: float
    min_pr_auc: float
    min_f1: float
    max_brier: float
    rule: str


def suggest_promotion_floors(
    *,
    cv_roc_auc: FoldScores,
    cv_pr_auc: FoldScores,
    validation: ClassificationMetrics,
    fold_rows: int,
    test_rows: int,
) -> SuggestedFloors:
    """The documented rule that turns the tuned BASELINE's train/val evidence into gates.

    AUC floors are a 2-sigma lower bound on what the baseline would score on a test set
    of ``test_rows``: the CV fold std (measured on folds of ``fold_rows``) is rescaled by
    sqrt(fold_rows / test_rows), since metric noise shrinks with sample size. Only the
    test split's SIZE is used, never its labels or metrics.
    """

    def down(x: float) -> float:
        return math.floor(x * 100) / 100

    def up(x: float) -> float:
        return math.ceil(x * 100) / 100

    scale = math.sqrt(fold_rows / test_rows)
    return SuggestedFloors(
        min_roc_auc=down(cv_roc_auc.mean - 2 * cv_roc_auc.std * scale),
        min_pr_auc=down(cv_pr_auc.mean - 2 * cv_pr_auc.std * scale),
        min_f1=down(validation.f1 - 0.05),
        max_brier=up(validation.brier + 0.02),
        rule=(
            "AUC floor = baseline CV mean - 2 * CV std * sqrt(fold_rows/test_rows), floored; "
            "F1 floor = baseline OOF-val F1 - 0.05, floored; "
            "Brier ceiling = baseline OOF-val Brier + 0.02, ceiled"
        ),
    )


# ---------------------------------------------------------------------------
# Metadata & index
# ---------------------------------------------------------------------------
class ModelMetadata(_Model):
    model_id: str
    model_name: str
    stage: Stage
    created_at: str
    artifact: ArtifactInfo
    lineage: Lineage
    data_schema: dict[str, Any]
    threshold: float
    calibration_method: str
    best_params: dict[str, ParamValue]
    cv: dict[str, FoldScores]
    validation: ClassificationMetrics
    test: ClassificationMetrics | None = None
    promotion: PromotionDecision | None = None


class RegistryEvent(_Model):
    at: str
    action: Literal["registered", "promoted", "rejected", "archived"]
    model_id: str
    detail: str


class RegistryIndex(_Model):
    production: str | None = None
    stages: dict[str, Stage] = Field(default_factory=dict)
    events: tuple[RegistryEvent, ...] = ()


# ---------------------------------------------------------------------------
# Artifact I/O (shared with run directories and exported model dirs)
# ---------------------------------------------------------------------------
def dump_model(model: ChurnModel, path: Path) -> ArtifactInfo:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    data = buffer.getvalue()
    digest = atomic_write_bytes(path, data)
    return ArtifactInfo(filename=path.name, sha256=digest, size_bytes=len(data))


def load_verified_model(path: Path, expected_sha256: str) -> ChurnModel:
    """Verify the SHA-256 of ``path`` and only then deserialise it."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ModelIntegrityError(f"cannot read model artifact {path}: {exc}") from exc
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual}; "
            "refusing to deserialise"
        )
    model = joblib.load(io.BytesIO(data))
    if not isinstance(model, ChurnModel):
        raise ModelIntegrityError(f"{path.name} does not contain a ChurnModel")
    return model


def read_metadata(model_dir: Path) -> ModelMetadata:
    path = model_dir / METADATA_FILENAME
    try:
        return ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelIntegrityError(f"cannot read model metadata {path}: {exc}") from exc


def load_model_dir(model_dir: Path) -> tuple[ChurnModel, ModelMetadata]:
    """Load ``model.joblib`` from a registry or exported model directory, verified."""
    metadata = read_metadata(model_dir)
    model = load_verified_model(model_dir / metadata.artifact.filename, metadata.artifact.sha256)
    return model, metadata


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def _index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    def model_dir(self, model_id: str) -> Path:
        return self.root / "models" / model_id

    def index(self) -> RegistryIndex:
        if not self._index_path.exists():
            return RegistryIndex()
        return RegistryIndex.model_validate_json(self._index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: RegistryIndex) -> None:
        atomic_write_json(self._index_path, index.model_dump(mode="json"))

    def _write_metadata(self, metadata: ModelMetadata) -> None:
        atomic_write_json(
            self.model_dir(metadata.model_id) / METADATA_FILENAME,
            metadata.model_dump(mode="json"),
        )

    @staticmethod
    def _event(action: str, model_id: str, detail: str) -> RegistryEvent:
        return RegistryEvent.model_validate(
            {"at": utc_now(), "action": action, "model_id": model_id, "detail": detail}
        )

    def register(self, model: ChurnModel, metadata: Mapping[str, Any]) -> ModelMetadata:
        """Store a new candidate. ``metadata`` supplies every field except artifact/stage."""
        model_id = str(metadata["model_id"])
        index = self.index()
        if model_id in index.stages:
            raise RegistryError(f"model {model_id} is already registered")
        artifact = dump_model(model, self.model_dir(model_id) / MODEL_FILENAME)
        record = ModelMetadata.model_validate(
            {**metadata, "artifact": artifact.model_dump(), "stage": "candidate"}
        )
        self._write_metadata(record)
        self._write_index(
            index.model_copy(
                update={
                    "stages": {**index.stages, model_id: "candidate"},
                    "events": (
                        *index.events,
                        self._event("registered", model_id, record.model_name),
                    ),
                }
            )
        )
        return record

    def metadata(self, model_id: str) -> ModelMetadata:
        if model_id not in self.index().stages:
            raise RegistryError(f"unknown model {model_id}")
        return read_metadata(self.model_dir(model_id))

    def load(self, model_id: str) -> tuple[ChurnModel, ModelMetadata]:
        if model_id not in self.index().stages:
            raise RegistryError(f"unknown model {model_id}")
        return load_model_dir(self.model_dir(model_id))

    def production_id(self) -> str | None:
        return self.index().production

    def load_production(self) -> tuple[ChurnModel, ModelMetadata] | None:
        production = self.production_id()
        return None if production is None else self.load(production)

    def record_test_metrics(self, model_id: str, test: ClassificationMetrics) -> ModelMetadata:
        updated = self.metadata(model_id).model_copy(update={"test": test})
        self._write_metadata(updated)
        return updated

    def apply_decision(self, decision: PromotionDecision) -> ModelMetadata:
        """Promote (archiving the previous production model) or reject the candidate."""
        model_id = decision.candidate_id
        index = self.index()
        current = self.metadata(model_id)
        if index.stages.get(model_id) != "candidate":
            raise RegistryError(f"{model_id} is not a candidate (stage={current.stage})")

        stages = dict(index.stages)
        events = list(index.events)
        production = index.production
        if decision.approved:
            if production is not None:
                previous = self.metadata(production).model_copy(update={"stage": "archived"})
                self._write_metadata(previous)
                stages[production] = "archived"
                events.append(self._event("archived", production, f"superseded by {model_id}"))
            stages[model_id] = "production"
            production = model_id
            action, stage = "promoted", "production"
        else:
            stages[model_id] = "rejected"
            action, stage = "rejected", "rejected"

        updated = current.model_copy(update={"stage": stage, "promotion": decision})
        self._write_metadata(updated)
        events.append(self._event(action, model_id, decision.summary()))
        self._write_index(RegistryIndex(production=production, stages=stages, events=tuple(events)))
        return updated

    def export(self, model_id: str, destination: Path) -> Path:
        """Copy a verified model directory (artifact + metadata) for packaging."""
        _, metadata = self.load(model_id)  # verifies the hash before copying
        destination.mkdir(parents=True, exist_ok=True)
        source = self.model_dir(model_id)
        for name in (metadata.artifact.filename, METADATA_FILENAME):
            shutil.copyfile(source / name, destination / name)
        load_model_dir(destination)  # verify the copy too
        return destination
