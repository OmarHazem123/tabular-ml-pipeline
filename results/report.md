# Churn model evaluation report

- **Run:** `20261126T012827Z-37ac9766-8c8c`
- **Champion:** `xgboost` (model id `20261126T012827Z-37ac9766-8c8c-xgboost`)
- **Promotion:** **APPROVED**, production model is now `20261126T012827Z-37ac9766-8c8c-xgboost`
- **Lineage:** git `9963daedeee0`, config `37ac9766d0bf`, dataset sha256 `16320c9c1ec7`, seed 42

## Benchmark

The champion is chosen by **cross-validated PR-AUC on the train split**. Validation metrics use out-of-fold calibrated probabilities. Test columns are hold-out results reported for every model, but they played no part in selection.

| Model | CV ROC-AUC | CV PR-AUC | Val PR-AUC | Val F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Test Brier | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression (baseline) | 0.8530 ± 0.0195 | 0.6675 ± 0.0354 | 0.6053 | 0.6159 | 0.8397 | 0.6445 | 0.6317 | 0.1388 | 0.300 |
| lightgbm | 0.8544 ± 0.0216 | 0.6738 ± 0.0405 | 0.6202 | 0.6102 | 0.8473 | 0.6653 | 0.6352 | 0.1354 | 0.351 |
| xgboost (champion) | 0.8556 ± 0.0199 | 0.6785 ± 0.0371 | 0.6250 | 0.6222 | 0.8469 | 0.6682 | 0.6368 | 0.1372 | 0.262 |

## Promotion gate

| Check | Result | Detail |
|---|---|---|
| roc_auc_floor | pass | 0.8469 >= 0.8200 |
| pr_auc_floor | pass | 0.6682 >= 0.6100 |
| f1_floor | pass | 0.6368 >= 0.5600 |
| brier_ceiling | pass | 0.1372 <= 0.1700 |
| not_worse_than_baseline | pass | pr_auc 0.6682 >= baseline - tolerance 0.6345 |
| beats_production | pass | no production model yet |

## Hold-out test metrics: xgboost

| Metric | Value |
|---|---|
| Rows / positive rate | 1409 / 0.2654 |
| ROC-AUC | 0.8469 |
| PR-AUC (average precision) | 0.6682 |
| F1 @ threshold 0.262 | 0.6368 |
| Precision / Recall | 0.5585 / 0.7406 |
| Accuracy | 0.7757 |
| Brier score | 0.1372 |
| Log loss | 0.4253 |
| Expected calibration error (10 bins) | 0.0566 |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted stay | Predicted churn |
|---|---|---|
| **Actual stay** | 816 | 219 |
| **Actual churn** | 97 | 277 |

## Calibration (reliability, test split)

| Predicted bin | Count | Mean predicted | Observed churn rate |
|---|---|---|---|
| 0.0-0.1 | 530 | 0.078 | 0.042 |
| 0.1-0.2 | 286 | 0.141 | 0.164 |
| 0.2-0.3 | 136 | 0.241 | 0.309 |
| 0.3-0.4 | 94 | 0.348 | 0.447 |
| 0.4-0.5 | 89 | 0.450 | 0.562 |
| 0.5-0.6 | 85 | 0.547 | 0.400 |
| 0.6-0.7 | 85 | 0.652 | 0.588 |
| 0.7-0.8 | 75 | 0.742 | 0.813 |
| 0.8-0.9 | 29 | 0.825 | 0.897 |

## Global feature importance (SHAP, test split)

Mean |SHAP| in log-odds, with one-hot columns summed back into their source feature. A positive mean SHAP means the feature pushes risk up on average.

| Rank | Feature | Mean abs SHAP | Mean SHAP |
|---|---|---|---|
| 1 | `Contract` | 0.6733 | -0.1154 |
| 2 | `tenure` | 0.5133 | -0.1550 |
| 3 | `InternetService` | 0.2515 | -0.0507 |
| 4 | `PaymentMethod` | 0.1731 | -0.0271 |
| 5 | `TechSupport` | 0.1700 | -0.0359 |
| 6 | `OnlineSecurity` | 0.1696 | -0.0366 |
| 7 | `PaperlessBilling` | 0.1571 | -0.0367 |
| 8 | `MultipleLines` | 0.1309 | -0.0016 |
| 9 | `avg_monthly_charge` | 0.1122 | -0.0123 |
| 10 | `StreamingTV` | 0.1018 | -0.0139 |
| 11 | `OnlineBackup` | 0.0915 | -0.0207 |
| 12 | `MonthlyCharges` | 0.0638 | -0.0146 |
| 13 | `TotalCharges` | 0.0620 | -0.0158 |
| 14 | `charge_per_service` | 0.0320 | -0.0171 |
| 15 | `SeniorCitizen` | 0.0156 | -0.0024 |

## Local explanations (SHAP reason codes)

### Highest predicted risk: `5419-JPRRN`

p(churn) = 0.8677, predicted churn, actual churn

| Feature | Value | Contribution (log-odds) | Effect |
|---|---|---|---|
| `tenure` | 1.0 | +0.9074 | increases risk |
| `Contract` | Month-to-month | +0.5163 | increases risk |
| `avg_monthly_charge` | 101.45 | +0.2923 | increases risk |

### Lowest predicted risk: `3173-WSSUE`

p(churn) = 0.0671, predicted stay, actual stay

| Feature | Value | Contribution (log-odds) | Effect |
|---|---|---|---|
| `Contract` | Two year | -1.1476 | decreases risk |
| `tenure` | 71.0 | -0.8195 | decreases risk |
| `InternetService` | No | -0.2325 | decreases risk |

### Closest to the decision threshold: `7246-ZGQDF`

p(churn) = 0.2621, predicted churn, actual stay

| Feature | Value | Contribution (log-odds) | Effect |
|---|---|---|---|
| `tenure` | 1.0 | +0.7990 | increases risk |
| `Contract` | Month-to-month | +0.5009 | increases risk |
| `InternetService` | DSL | -0.2881 | decreases risk |

## Splits

| Split | Rows | Positive rate | IDs sha256 |
|---|---|---|---|
| test | 1409 | 0.2654 | `51e56c56eeb7fef1` |
| train | 4225 | 0.2653 | `41f663c388df453f` |
| val | 1409 | 0.2654 | `a24757280cf26fd2` |
