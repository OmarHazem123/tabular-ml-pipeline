"""Shared fixtures.

Tests never touch the network: they run on a seeded synthetic dataset that mirrors the
raw Telco CSV exactly (same columns, same text encoding, blank ``TotalCharges`` for
``tenure == 0``) and carries a learnable churn signal so stratification and model fits
behave realistically.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ml_pipeline.config import PipelineConfig
from ml_pipeline.data import DataSplits, PreparedDataset, prepare_dataset, split_dataset

YES_NO = ["No", "Yes"]
RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def make_raw_frame(n_rows: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Synthetic raw frame: every column is text, exactly as ``read_raw_csv`` returns it."""
    rng = np.random.default_rng(seed)

    def pick(values: list[str], p: list[float] | None = None) -> npt.NDArray[np.str_]:
        return rng.choice(np.asarray(values), size=n_rows, p=p)

    tenure = rng.integers(1, 73, size=n_rows)
    tenure[rng.choice(n_rows, size=max(2, n_rows // 100), replace=False)] = 0
    phone = pick(YES_NO, [0.1, 0.9])
    internet = pick(["DSL", "Fiber optic", "No"], [0.35, 0.45, 0.2])

    def addon() -> npt.NDArray[np.str_]:
        return np.where(internet == "No", "No internet service", pick(YES_NO))

    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.2, 0.25])
    payment = pick(
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
        [0.22, 0.22, 0.34, 0.22],
    )
    monthly = np.round(rng.uniform(18.25, 118.75, size=n_rows), 2)
    total = np.round(tenure * monthly * rng.uniform(0.9, 1.1, size=n_rows), 2)

    logit = (
        -1.6
        + 1.6 * (contract == "Month-to-month")
        + 0.9 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
        - 0.035 * tenure
        + rng.normal(0.0, 0.4, size=n_rows)
    )
    churn = np.where(rng.uniform(size=n_rows) < 1.0 / (1.0 + np.exp(-logit)), "Yes", "No")

    frame = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n_rows)],
            "gender": pick(["Female", "Male"]),
            "SeniorCitizen": pick(["0", "1"], [0.84, 0.16]),
            "Partner": pick(YES_NO),
            "Dependents": pick(YES_NO, [0.7, 0.3]),
            "tenure": tenure.astype(str),
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", pick(YES_NO)),
            "InternetService": internet,
            "OnlineSecurity": addon(),
            "OnlineBackup": addon(),
            "DeviceProtection": addon(),
            "TechSupport": addon(),
            "StreamingTV": addon(),
            "StreamingMovies": addon(),
            "Contract": contract,
            "PaperlessBilling": pick(YES_NO, [0.4, 0.6]),
            "PaymentMethod": payment,
            "MonthlyCharges": [f"{value:.2f}" for value in monthly],
            "TotalCharges": [
                " " if t == 0 else f"{v:.2f}" for t, v in zip(tenure, total, strict=True)
            ],
            "Churn": churn,
        }
    )
    return frame[RAW_COLUMNS].astype("str")


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(scope="session")
def _raw_frame_session() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def raw_frame(_raw_frame_session: pd.DataFrame) -> pd.DataFrame:
    """Fresh copy per test, so tests can corrupt it freely."""
    return _raw_frame_session.copy()


@pytest.fixture(scope="session")
def prepared(_raw_frame_session: pd.DataFrame, config: PipelineConfig) -> PreparedDataset:
    return prepare_dataset(_raw_frame_session, config)


@pytest.fixture(scope="session")
def splits(prepared: PreparedDataset, config: PipelineConfig) -> DataSplits:
    return split_dataset(prepared, config)
