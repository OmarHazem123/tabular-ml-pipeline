"""Exception hierarchy shared across the pipeline.

Every error raised deliberately by this package derives from :class:`PipelineError`, so
callers (CLI, API) can catch domain failures without swallowing programming errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    """A single schema or data-quality finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    severity: Severity
    message: str
    column: str | None = None
    count: int | None = None

    def render(self) -> str:
        location = f" column '{self.column}'" if self.column is not None else ""
        return f"[{self.check}]{location}: {self.message}"


class PipelineError(Exception):
    """Base class for all expected, domain-level pipeline failures."""


class DataIntegrityError(PipelineError):
    """Raised when raw data cannot be obtained or fails checksum verification."""


class SchemaMismatchError(PipelineError, ValueError):
    """Raised when a frame handed to a transformer lacks required columns."""


class DataValidationError(PipelineError):
    """Raised when a dataset violates the schema or data-quality contract.

    All violations are collected before raising, so one run surfaces every problem.
    """

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        errors = [issue for issue in self.issues if issue.severity == "error"]
        lines = [f"Data validation failed with {len(errors)} error(s):"]
        lines.extend(f"  - {issue.render()}" for issue in errors)
        super().__init__("\n".join(lines))

    @property
    def columns(self) -> frozenset[str]:
        """Columns implicated by error-severity issues."""
        return frozenset(
            issue.column
            for issue in self.issues
            if issue.severity == "error" and issue.column is not None
        )
