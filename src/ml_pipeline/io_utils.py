"""Filesystem helpers: atomic, LF-only writes and content hashing.

Every report or artifact written by the pipeline goes through these helpers so that:

* a crash mid-write never leaves a truncated file at the destination (write to a sibling
  temp file, ``fsync``, then ``os.replace``, which is atomic on both POSIX and Windows);
* text output always uses ``\\n`` line endings regardless of the host OS;
* JSON output is canonical (sorted keys, stable indentation, trailing newline), so the
  same content always hashes to the same digest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.exceptions import DataIntegrityError

_HASH_CHUNK_BYTES = 1 << 20
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SECONDS = 0.05


def _make_sibling_temp(path: Path) -> tuple[int, Path]:
    """Create a temp file in the destination directory (same volume => atomic rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return fd, Path(name)


def _replace_with_retry(source: Path, target: Path) -> None:
    """``os.replace`` with a short retry for transient Windows sharing violations.

    Antivirus scanners and search indexers can briefly hold a handle on a freshly written
    file, which surfaces as ``PermissionError`` on Windows. The rename itself stays atomic.
    """
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS * attempt)


def normalize_newlines(text: str) -> str:
    """Convert CRLF / lone CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Atomically write UTF-8 text with LF line endings."""
    target = Path(path)
    fd, tmp = _make_sibling_temp(target)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(normalize_newlines(text))
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def atomic_write_chunks(
    path: Path | str,
    chunks: Iterable[bytes],
    *,
    expected_sha256: str | None = None,
) -> str:
    """Atomically stream bytes to ``path`` and return their SHA-256 hex digest.

    When ``expected_sha256`` is given, the destination is only replaced if the streamed
    content matches it; otherwise :class:`DataIntegrityError` is raised and the existing
    destination (if any) is left untouched.
    """
    target = Path(path)
    digest = hashlib.sha256()
    fd, tmp = _make_sibling_temp(target)
    try:
        with os.fdopen(fd, "wb") as handle:
            for chunk in chunks:
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256.lower():
            raise DataIntegrityError(
                f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual}"
            )
        _replace_with_retry(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return actual


def atomic_write_bytes(path: Path | str, data: bytes) -> str:
    """Atomically write raw bytes and return their SHA-256 hex digest."""
    return atomic_write_chunks(path, (data,))


def _json_default(value: object) -> Any:
    """Serialise common scientific/stdlib types that ``json`` does not handle natively."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, 2-space indent, strict (no NaN/Infinity), LF-terminated."""
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    """Serialise ``payload`` canonically and write it atomically with LF endings.

    Serialisation happens before any file is touched, so an unserialisable payload can
    never clobber an existing report.
    """
    return atomic_write_text(path, canonical_json(payload))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    """Stream a file through SHA-256 without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
