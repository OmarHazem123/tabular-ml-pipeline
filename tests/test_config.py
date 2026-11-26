"""Configuration invariants and reproducibility fingerprint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ml_pipeline.config import (
    DatasetConfig,
    EquivalenceRule,
    FeatureConfig,
    FloatParam,
    IntParam,
    ModelSpec,
    NumericFeature,
    PathsConfig,
    PipelineConfig,
    PromotionConfig,
    SchemaConfig,
    ServingConfig,
    SplitConfig,
    TrainingConfig,
    load_config,
)


def test_defaults_are_valid_and_frozen() -> None:
    config = PipelineConfig()
    with pytest.raises(ValidationError, match="frozen"):
        config.seed = 1  # type: ignore[misc]


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PipelineConfig.model_validate({"sed": 1})


@pytest.mark.parametrize(
    "fractions",
    [(0.7, 0.2, 0.2), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)],
)
def test_split_fractions_must_sum_to_one(fractions: tuple[float, float, float]) -> None:
    train, val, test = fractions
    with pytest.raises(ValidationError):
        SplitConfig(train_fraction=train, val_fraction=val, test_fraction=test)


def test_fingerprint_is_stable_and_sensitive() -> None:
    base = PipelineConfig()
    assert base.fingerprint() == PipelineConfig().fingerprint()
    assert PipelineConfig(seed=7).fingerprint() != base.fingerprint()
    assert (
        PipelineConfig(split=SplitConfig(train_fraction=0.7, val_fraction=0.1)).fingerprint()
        != base.fingerprint()
    )


def test_fingerprint_ignores_settings_that_cannot_change_a_model() -> None:
    reference = PipelineConfig().fingerprint()
    assert PipelineConfig(paths=PathsConfig(results_dir=Path("elsewhere"))).fingerprint() == (
        reference
    )
    assert PipelineConfig(promotion=PromotionConfig(min_pr_auc=0.9)).fingerprint() == reference
    assert PipelineConfig(serving=ServingConfig(max_batch_size=5)).fingerprint() == reference


def test_consistency_rules_must_reference_known_categories() -> None:
    bad_rule = EquivalenceRule(
        left="PhoneService", left_value="Maybe", right="MultipleLines", right_value="No"
    )
    with pytest.raises(ValidationError, match="unknown categorical value"):
        SchemaConfig(consistency_rules=(bad_rule,))


def test_load_config_from_partial_toml(tmp_path: Path) -> None:
    path = tmp_path / "override.toml"
    path.write_text(
        "seed = 7\n[split]\ntrain_fraction = 0.7\nval_fraction = 0.1\n", encoding="utf-8"
    )
    config = load_config(path)
    assert config.seed == 7
    assert config.split.train_fraction == 0.7
    assert config.split.test_fraction == 0.2
    assert config.dataset == DatasetConfig()


def test_load_config_from_json(tmp_path: Path) -> None:
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"promotion": {"min_roc_auc": 0.9}}), encoding="utf-8")
    assert load_config(path).promotion.min_roc_auc == 0.9


def test_load_config_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported config format"):
        load_config(path)


def test_load_config_without_path_returns_defaults() -> None:
    assert load_config() == PipelineConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "http://insecure.example/data.csv"},
        {"sha256": "not-a-checksum"},
    ],
)
def test_dataset_provenance_is_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(**kwargs)


def test_schema_rejects_duplicate_features() -> None:
    feature = NumericFeature(name="x", min_value=0, max_value=1)
    with pytest.raises(ValidationError, match="duplicate feature names"):
        SchemaConfig(features=(feature, feature))


def test_schema_rejects_target_as_feature() -> None:
    with pytest.raises(ValidationError, match="cannot also be features"):
        SchemaConfig(features=(NumericFeature(name="Churn", min_value=0, max_value=1),))


def test_numeric_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="min_value must be < max_value"):
        NumericFeature(name="x", min_value=5, max_value=5)


def test_feature_engineering_must_reference_correct_column_kinds() -> None:
    with pytest.raises(ValidationError, match="expects numeric column 'Contract'"):
        PipelineConfig(features=FeatureConfig(tenure_column="Contract"))


@pytest.mark.parametrize(
    "spec",
    [
        lambda: FloatParam(low=1.0, high=0.5),
        lambda: FloatParam(low=0.0, high=1.0, log=True),
        lambda: IntParam(low=5, high=5),
    ],
)
def test_search_space_ranges_are_validated(spec: Any) -> None:
    with pytest.raises(ValidationError):
        spec()


def test_exactly_one_baseline_model() -> None:
    models = (ModelSpec(name="lightgbm"), ModelSpec(name="xgboost"))
    with pytest.raises(ValidationError, match="exactly one model"):
        TrainingConfig(models=models)


def test_params_cannot_be_both_fixed_and_searched() -> None:
    with pytest.raises(ValidationError, match="both fixed and searched"):
        ModelSpec(
            name="lightgbm",
            fixed_params={"num_leaves": 31},
            search_space={"num_leaves": IntParam(low=8, high=64)},
        )
