````markdown
# Alternative Credit Scoring — Leak-Safe, Calibrated, Explainable, Economics-Aware

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

This repository contains a **production-style ML pipeline** that estimates **Probability of Default (PD)** using *alternative* features (no bureau/credit-history inputs) and converts PD to lending decisions using **LGD/EAD economics**.  
It ships with **calibration, explainability (SHAP), fairness auditing, stress testing,** and **optional monotonic constraints** for governance.

---

## ✨ What this build adds

- **ROI in basis points** and **EV per approved loan** in the CLI summary
- **Calibration diagnostics**: *Calibration-in-the-large (CITL)* and *Calibration slope* (logit scale)
- **Global SHAP export**: `artifacts/global_shap_mean_abs.csv`
- All previous features retained: leakage policy, temporal split, calibration, profit curve, stress testing, fairness audit, monotone constraints, and SHAP global & local explanations

---

## 🔢 Data (alt-data only)

**Target construction**
- `default = 1` if `LoanApproved == 0`, else `0`

**Used features (if present)**
- Demographics: `Age`, `MaritalStatus`, `NumberOfDependents`
- Employment & Education: `EmploymentStatus`, `EducationLevel`, `Experience`, `JobTenure`
- Income & Cashflow: `AnnualIncome`, `MonthlyIncome`, `MonthlyDebtPayments`
- Assets & Liabilities: `SavingsAccountBalance`, `CheckingAccountBalance`, `TotalAssets`, `TotalLiabilities`, `NetWorth`
- Loan Terms: `LoanAmount`, `LoanDuration`, `MonthlyLoanPayment`, `LoanPurpose`
- Ratios & Behavior: `TotalDebtToIncomeRatio`, `UtilityBillsPaymentHistory`
- Rates: `BaseInterestRate`, `InterestRate`

**Explicitly excluded (leak/bureau style)**
- `CreditScore`, `NumberOfOpenCreditLines`, `NumberOfCreditInquiries`, `LengthOfCreditHistory`,
  `PaymentHistory`, `PreviousLoanDefaults`, `BankruptcyHistory`, `CreditCardUtilizationRate`, `RiskScore`

> **Note on dataset quirks**  
> If numeric columns are stored as strings with separators or stray characters (e.g., `8.722.406.105.782.900`), clean them before training:
> ```python
> for c in ["MonthlyIncome","BaseInterestRate","InterestRate","MonthlyLoanPayment",
>           "TotalDebtToIncomeRatio","LoanAmount"]:
>     df[c] = (df[c].astype(str)
>                 .str.replace(r"[^0-9\.-]", "", regex=True)
>                 .replace("", np.nan)
>                 .astype(float))
> df["ApplicationDate"] = pd.to_datetime(df["ApplicationDate"], errors="coerce")
> ```

---

## 🧰 Pipeline overview

1. **Split**
   - Stratified split by default, or **temporal split** via `--time_split_col ApplicationDate`.
2. **Preprocessing**
   - Missing values: median (numeric), mode (categorical)
   - Encoding: label encoding (low-cardinality) / frequency encoding (high-cardinality)
   - Scaling: `StandardScaler` on numeric columns
   - Optional leakage policy via `--policy feature_policy.yaml` (+ `--ablation_drop_post`)
3. **Models**
   - **LightGBM** (default, supports **monotone constraints**)
   - **XGBoost** alternative
   - Early stopping on validation fold
4. **Calibration & Reliability**
   - Isotonic (default) or Platt
   - Reliability curve + **Brier score**
   - **CITL** and **slope** (logit scale) reported
5. **Economics**
   - PD–LGD–EAD expected value
   - Revenue:
     - Margin model: `revenue = margin * EAD` (`--margin`)
     - Proxy model: `MonthlyLoanPayment*LoanDuration - LoanAmount` (`--use_proxy_revenue`)
     - Conservative fallback: `0.05 * EAD`
   - **Profit curve** and **optimal approval threshold τ\*** (approve if PD < τ)
   - **ROI** and **EV per approved** in summary
6. **Stress testing**
   - PD multiplier via `--stress_pd_mult`
7. **Explainability (SHAP)**
   - Global: beeswarm + bar, **mean|SHAP| CSV**
   - Local: Top-N highest-PD cases with top contributing features
8. **Fairness audit**
   - Approval rate, mean PD, observed default rate by group (`--fairness_groups`)

---

## 🧪 Example CLI runs & outputs

**1) Baseline calibrated LightGBM**
```bash
python main.py
````

Sample output:

```
ALTERNATIVE CREDIT SCORING — CALIBRATED, EXPLAINABLE & ECONOMICS-AWARE
Model: lightgbm | Calibration: isotonic | SHAP: off
AUC: 0.9560 | PR-AUC: 0.9861 | F1@0.5: 0.9327
Gini: 0.9119 | KS: 0.7788 | Brier: 0.0722 | Slope: 1.100 | CITL: -0.179
Break-even τ: 0.58
Approve τ*: 0.97 | EV*: 66.13 | Approval rate: 52.60% | ROI*: 0.0 bps | EV/approved: 0.03
Review band: [0.30, 0.60]
```

**2) Add economics with margin model**

```bash
python main.py --lgd 0.45 --ead_col LoanAmount --margin 0.10
```

**3) SHAP explanations (global + local files)**

```bash
python main.py --lgd 0.45 --ead_col LoanAmount --margin 0.10 --shap
```

Outputs include:

* `graphs/shap_summary.png`, `graphs/shap_importance_bar.png`
* `artifacts/global_shap_mean_abs.csv`, `artifacts/local_explanations_topN.csv`

**4) Stress testing**

```bash
python main.py --margin 0.08 --stress_pd_mult 1.25
```

Adds stressed τ\*, EV, and approval rate to the summary.

**5) Fairness audit**

```bash
python main.py --fairness_groups EmploymentStatus,EducationLevel
```

Produces `artifacts/group_fairness_metrics.csv`.

**6) Temporal split (backtesting)**

```bash
python main.py --time_split_col ApplicationDate
```

**7) Monotone constraints**

```bash
python main.py --monotone_yaml monotone.yaml
```

**8) Leakage ablation**

```bash
python main.py --policy feature_policy.yaml --ablation_drop_post
```

---

## 📤 Generated outputs

**Graphs** (`graphs/`)

* `confusion_matrix.png`
* `precision_recall_curve.png`
* `roc_curve.png`
* `reliability_curve.png`
* `profit_curve.png`
* `shap_summary.png` & `shap_importance_bar.png` (if `--shap`)

**Artifacts** (`artifacts/`)

* `portfolio_ev_table.csv`
* `global_shap_mean_abs.csv` (if `--shap`)
* `local_explanations_topN.csv` (if `--shap`)
* `group_fairness_metrics.csv` (if `--fairness_groups`)
* `run.json`, `metrics.json`, `thresholds.json`
* `model.pkl`, `calibrator.pkl` (if used)

---

## 🧩 Example: `monotone.yaml`

```yaml
Age: -1
AnnualIncome: -1
Experience: -1
LoanAmount: 1
LoanDuration: 1
MonthlyDebtPayments: 1
TotalDebtToIncomeRatio: 1
SavingsAccountBalance: -1
CheckingAccountBalance: -1
TotalAssets: -1
TotalLiabilities: 1
MonthlyIncome: -1
JobTenure: -1
NetWorth: -1
BaseInterestRate: 1
InterestRate: 1
MonthlyLoanPayment: 1
```

> **Interpretation**: negative means “as the feature increases, PD should *not* increase”; positive means PD should *not* decrease.

---

## 🔒 Example: `feature_policy.yaml`

```yaml
# columns to drop regardless (e.g., potential identifiers/leaks)
drop_always:
  - RiskScore

# features to drop for ablation if --ablation_drop_post is used (post-decision/pricing)
post_decision:
  - InterestRate
  - MonthlyLoanPayment
  - BaseInterestRate

# (Optional) allow-list: if provided, the model uses only these features (intersection with columns)
# allowed:
#   - Age
#   - AnnualIncome
#   - EmploymentStatus
#   - ...
```

---

## ⚙️ Installation

```bash
git clone https://github.com/soumadeepmaiti/AltRiskML.git
cd AltRiskML
pip install -r requirements.txt
```

**requirements.txt**

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.3.0
lightgbm>=3.3.0
xgboost>=1.6.0
shap>=0.42.0
pyyaml
joblib
```

---

## 🗂️ Project structure

```
AltRiskML/
├── main.py
├── requirements.txt
├── Loan new datset.csv
├── monotone.yaml
├── feature_policy.yaml
├── graphs/
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── reliability_curve.png
│   ├── profit_curve.png
│   ├── shap_summary.png
│   └── shap_importance_bar.png
└── artifacts/
    ├── portfolio_ev_table.csv
    ├── global_shap_mean_abs.csv
    ├── local_explanations_topN.csv
    ├── group_fairness_metrics.csv
    ├── run.json
    ├── metrics.json
    ├── thresholds.json
    ├── model.pkl
    └── calibrator.pkl
```

---

## ✅ Good practices embedded

* **No bureau leakage**; optional policy-driven feature ablation
* **Temporal validation** for backtesting (`--time_split_col`)
* **Calibrated PDs** with reliability stats (Brier/CITL/slope)
* **Economics-aware decisions** (PD–LGD–EAD, profit curve, ROI, EV/approved)
* **Stress testing** and **fairness audit** for governance
* **Explainability** with global and local SHAP outputs
* **Monotonicity** for policy-aligned model behavior (LightGBM)

---

## 📄 License & Contributions

* Feel free to open issues/PRs for improvements (feature engineering, hyperparameter sweeps, API deployment, dashboards, etc.).
* Ensure any new features respect the leakage policy and calibration/monitoring hooks already in place.

```
```
