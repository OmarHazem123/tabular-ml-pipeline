# tabular-ml-pipeline

A strictly reproducible, end-to-end tabular ML pipeline for **customer churn**
(IBM Telco dataset), ending in a FastAPI inference service.

> **Status: Phase 1 of 2.** The data contract, leakage-safe feature pipeline, and
> tooling are done. Training, evaluation, the model registry, the API, and Docker
> arrive in Phase 2, and this README gains the architecture diagram, benchmark table,
> and API examples along with them.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). No `make` needed: `tasks.py` is a
standard-library task runner, and the `Makefile` only forwards to it.

```bash
uv sync --locked                          # Python 3.12 venv, exact pinned versions
uv run python tasks.py check              # ruff + mypy --strict + LF check + tests
uv run python tasks.py validate-data      # download, verify, validate, split
```

| Task            | What it does                                                    |
|-----------------|-----------------------------------------------------------------|
| `sync`          | Create/refresh `.venv` exactly from `uv.lock`                   |
| `format`        | `ruff format` + safe autofixes                                  |
| `lint`          | `ruff check` + `ruff format --check`                            |
| `typecheck`     | `mypy --strict` over `src/`, `tests/`, `tasks.py`               |
| `eol`           | Fail if any text file contains a CR byte                        |
| `test`          | Offline test suite (synthetic data, no network)                 |
| `test-network`  | Real-dataset test (downloads the pinned CSV)                    |
| `coverage`      | Tests with a coverage report                                    |
| `check`         | `lint` → `typecheck` → `eol` → `test`                           |
| `validate-data` | `ml-pipeline validate-data` → `results/data_validation.json`    |
| `clean`         | Remove tool caches (never touches data, artifacts or results)   |

Pass extra arguments with `--`, for example `uv run python tasks.py test -- -k leakage -x`.
With make: `make test ARGS="-k leakage"`.

## Dataset & provenance

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
(Apache-2.0): 7,043 customers, 19 features, 26.5% churn.

The download is pinned to an **immutable commit URL** and verified against a
**SHA-256 checksum** (`16320c9c…3e55e91`) before it is used. The file is streamed to a
temp file and moved into `data/raw/` only after the checksum matches. A cached copy is
reused only if it still verifies. You can pass `--data-path` to use a local file. Its
checksum is recorded as `verified: false` when it differs from the pinned one.

## Data contract (`config.py`, `data.py`)

Every column is read as raw text, then parsed explicitly against a typed schema. All
violations are collected into one `DataValidationError`, so a single run reports
every problem:

- **Structure:** missing, unexpected, or duplicated columns.
- **Types and domains:**
  - unparseable or non-finite numbers
  - out-of-range values
  - non-integer counts
  - categories outside the allowed set
  - values other than 0/1 in binary columns
- **Quality:**
  - missing-value rate per column
  - blank or duplicate IDs
  - invalid or missing labels
  - single-class or extremely imbalanced target
  - minimum row count
  - duplicated feature rows (a warning below 1%, an error above), with a count of
    duplicate groups that carry conflicting labels

One deterministic, row-local rule is applied: `TotalCharges` is blank **exactly** for
the 11 customers with `tenure == 0` (not yet billed), so it is set to `0`. This learns
nothing from the data, so it cannot leak. Any other blank stays `NaN`, counts against
the missing-value budget, and is imputed *inside* the pipeline.

Findings on the real data: no errors. There is one warning: 40 rows (0.57%) duplicate
another customer's features, and 18 of those groups have **conflicting labels**. That
is irreducible label noise, and it caps the achievable accuracy.

## Zero-leakage design (`features.py`)

1. **Split first.** The stratified 60/20/20 train/val/test split happens on validated
   raw data, before anything is fit. Rows are sorted by ID first, so the partition
   depends only on (data, seed, fractions) and not on the row order of the file.
2. **One pipeline.** `TelcoFeatureEngineer → ColumnTransformer → model` is a single
   sklearn `Pipeline`:
   - numeric: median imputation → quantile clipping → standard scaling
   - binary: most-frequent imputation
   - categorical: most-frequent imputation → one-hot encoding
   Every learned statistic lives inside `fit`, so it only ever sees training rows or
   CV-fold rows.
3. **Domain features are stateless:** `avg_monthly_charge`, `charge_trend` (current bill
   minus historical average), `num_services`, and `charge_per_service`.
4. **Proven, not assumed.** `tests/test_data_leakage.py`:
   - recomputes every imputer, clipper, and scaler statistic from training rows alone
     and asserts an exact match. It also shows that a *poisoned* hold-out **would**
     have shifted them. A mutation check confirmed this test fails when leakage is
     injected.
   - uses a spy transformer inside `cross_validate` to assert that each fold was fit on
     exactly its own training rows, and never on its validation rows or the val/test
     splits.
   - checks that hold-out-only categories are not learned, that `transform` never
     changes fitted state, and that schema mismatches raise named errors.

## Engineering standards

- **Pinned:** Python 3.12, exact `==` pins in `pyproject.toml`, plus `uv.lock`.
  `xgboost-cpu` keeps the Linux image free of CUDA/NCCL wheels.
- **Strict typing:** `mypy --strict` with the pydantic plugin across source, tests,
  and the task runner.
- **LF everywhere:**
  - every report is written by `io_utils.atomic_write_*` with `newline="\n"`
  - `.gitattributes` sets `eol=lf`, overriding Windows `core.autocrlf`
  - `tasks.py eol` fails the build on any CR byte
- **Atomic writes:** write a sibling temp file, `fsync`, then `os.replace`, with a retry
  for transient Windows file locks. A failed or interrupted write never leaves a
  partial file.
- **Deterministic outputs:** JSON is canonical (sorted keys, strict, trailing newline).
  Re-running `validate-data` produces a byte-identical report.
- **Config fingerprint:** a SHA-256 over every setting that can change a model, stored
  in reports for lineage.

## Layout

```
src/ml_pipeline/
  config.py      frozen pydantic config: dataset pin, schema, splits, features,
                 search spaces, promotion thresholds, fingerprint
  data.py        fetch + verify, parse + validate, stratified split
  features.py    domain engineer, QuantileClipper, ColumnTransformer, pipelines
  io_utils.py    atomic LF writes, canonical JSON, SHA-256 helpers
  exceptions.py  PipelineError hierarchy, structured ValidationIssue
  cli.py         typer CLI (validate-data; train/evaluate/serve/predict in Phase 2)
tests/           offline suite on schema-faithful synthetic data, plus 1 network test
results/         committed, deterministic reports
```

## License

MIT © 2026 Omar Hazem
