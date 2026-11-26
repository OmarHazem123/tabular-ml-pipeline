"""Atomic, LF-only report writing and hashing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.exceptions import DataIntegrityError
from ml_pipeline.io_utils import (
    atomic_write_bytes,
    atomic_write_chunks,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))


def test_text_is_written_with_lf_only(tmp_path: Path) -> None:
    path = atomic_write_text(tmp_path / "report.md", "a\r\nb\rc\nd")
    assert path.read_bytes() == b"a\nb\nc\nd"


def test_text_is_utf8(tmp_path: Path) -> None:
    path = atomic_write_text(tmp_path / "unicode.md", "naïve → ✓\n")
    assert path.read_bytes() == "naïve → ✓\n".encode()


def test_json_is_canonical(tmp_path: Path) -> None:
    payload = {"b": np.float64(0.5), "a": [np.int64(1), 2], "c": {"z": True, "y": None}}
    path = atomic_write_json(tmp_path / "metrics.json", payload)
    expected = (
        '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 0.5,\n'
        '  "c": {\n    "y": null,\n    "z": true\n  }\n}\n'
    )
    assert path.read_bytes() == expected.encode()


def test_json_rejects_nan_and_leaves_existing_file(tmp_path: Path) -> None:
    path = atomic_write_json(tmp_path / "metrics.json", {"ok": 1})
    with pytest.raises(ValueError, match="JSON compliant"):
        atomic_write_json(path, {"bad": float("nan")})
    assert path.read_text(encoding="utf-8") == canonical_json({"ok": 1})
    assert _leftovers(tmp_path) == []


def test_json_rejects_unknown_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="not JSON serialisable"):
        atomic_write_json(tmp_path / "x.json", {"obj": object()})
    assert list(tmp_path.iterdir()) == []


def test_overwrite_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.txt"
    atomic_write_text(target, "first\n")
    atomic_write_text(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert _leftovers(target.parent) == []


def test_failed_rename_keeps_original_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = atomic_write_text(tmp_path / "out.txt", "original\n")

    def broken_replace(self: Path, other: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "replacement\n")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert _leftovers(tmp_path) == []


def test_transient_permission_error_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_replace = Path.replace
    calls = {"n": 0}

    def flaky_replace(self: Path, other: Path) -> Path:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("file locked by scanner")
        return original_replace(self, other)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    target = atomic_write_text(tmp_path / "out.txt", "ok\n")
    assert target.read_text(encoding="utf-8") == "ok\n"
    assert calls["n"] == 3


def test_interrupted_stream_keeps_original(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    atomic_write_bytes(target, b"original")

    def chunks() -> Iterator[bytes]:
        yield b"partial"
        raise ConnectionError("connection reset")

    with pytest.raises(ConnectionError):
        atomic_write_chunks(target, chunks())
    assert target.read_bytes() == b"original"
    assert _leftovers(tmp_path) == []


def test_checksum_mismatch_never_creates_target(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        atomic_write_chunks(target, [b"tampered"], expected_sha256=sha256_bytes(b"genuine"))
    assert not target.exists()
    assert _leftovers(tmp_path) == []


def test_checksum_match_returns_digest(tmp_path: Path) -> None:
    content = b"genuine"
    digest = atomic_write_chunks(
        tmp_path / "data.csv", [content[:3], content[3:]], expected_sha256=sha256_bytes(content)
    )
    assert digest == hashlib.sha256(content).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    content = bytes(range(256)) * 10_000
    path = tmp_path / "blob.bin"
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
