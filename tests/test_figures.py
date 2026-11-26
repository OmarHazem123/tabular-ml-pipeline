"""README figures: rendered deterministically from metrics.json, and kept in sync with it."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from matplotlib.image import imread

from ml_pipeline.io_utils import sha256_file
from tests.conftest import TrainedPipeline

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_figures.py"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_figures", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figures_are_deterministic_opaque_pngs(
    renderer: ModuleType, trained: TrainedPipeline, tmp_path: Path
) -> None:
    metrics = trained.config.paths.results_dir / "metrics.json"
    first: dict[Path, str] = renderer.render_all(metrics, tmp_path / "first")
    second: dict[Path, str] = renderer.render_all(metrics, tmp_path / "second")

    assert {path.name for path in first} == {name for name, _ in renderer.FIGURES}
    assert {p.name: d for p, d in first.items()} == {p.name: d for p, d in second.items()}
    for path, digest in first.items():
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE)
        assert sha256_file(path) == digest
        assert b"matplotlib" not in data.lower()  # no software/version metadata chunk
        pixels = imread(path)
        if pixels.shape[-1] == 4:  # an alpha channel must be fully opaque (GitHub dark mode)
            assert pixels[..., 3].min() == 1.0


def test_committed_figures_match_committed_metrics(renderer: ModuleType, tmp_path: Path) -> None:
    # Compare decoded pixels, not file bytes: PNG compression can differ between zlib
    # builds on other platforms even when the image itself is identical.
    rendered: dict[Path, str] = renderer.render_all(ROOT / "results" / "metrics.json", tmp_path)
    for path in rendered:
        committed = ROOT / "docs" / "figures" / path.name
        assert committed.is_file(), f"{committed} is missing: run `uv run python tasks.py figures`"
        assert np.array_equal(imread(committed), imread(path)), (
            f"{committed.name} is stale: run `uv run python tasks.py figures`"
        )
