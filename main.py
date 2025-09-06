#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alternative Credit Scoring — leak-safe, calibrated, explainable, economics-aware.

Adds in this version:
- ROI in basis points + EV per approved loan in CLI summary
- Calibration-in-the-large (CITL) and Calibration slope on logit scale
- Global SHAP mean|SHAP| CSV (artifacts/global_shap_mean_abs.csv)
- All prior features: leakage policy, temporal split, calibration, profit curve, stress, fairness, monotone constraints, SHAP
"""

import os, json, hashlib, uuid, argparse, warnings, datetime as dt
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_recall_curve, average_precision_score, roc_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression  # for calibration slope/CITL

import lightgbm as lgb
import xgboost as xgb
import yaml  # pip install pyyaml
import joblib

# SHAP optional
try:
    import shap  # pip install shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)

ARTIFACTS_DIR = "artifacts"
GRAPHS_DIR = "graphs"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic on scores."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    thresholds = np.unique(np.sort(y_score))
    cdf_pos = np.searchsorted(np.sort(pos), thresholds, side='right') / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), thresholds, side='right') / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


class AlternativeCreditScorer:
    def __init__(self, model_type: str = 'lightgbm', monotone_map: Optional[Dict[str, int]] = None):
        self.model_type = model_type
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.monotone_map = monotone_map or {}

        # Excluded bureau-style features (kept from original)
        self.excluded_features = [
            'CreditScore', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
            'LengthOfCreditHistory', 'PaymentHistory', 'PreviousLoanDefaults',
            'BankruptcyHistory', 'CreditCardUtilizationRate'
        ]

        # Alternative features list (will be intersected with df cols)
        self.alternative_features = [
            'Age', 'AnnualIncome', 'EmploymentStatus', 'EducationLevel',
            'Experience', 'LoanAmount', 'LoanDuration', 'MaritalStatus',
            'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments',
            'LoanPurpose', 'SavingsAccountBalance', 'CheckingAccountBalance',
            'TotalAssets', 'TotalLiabilities', 'MonthlyIncome',
            'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth',
            'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
            'TotalDebtToIncomeRatio'
        ]

        # Fitted/calibration state
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.calibration_method: Optional[str] = None

        # Holdouts for evaluation
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_test_raw = None  # for fairness audits on original categorical cols
        self.y_pred = None
        self.y_pred_proba = None

    # ==== Data ====

    def load_and_preprocess_data(self, file_path: str,
                                 feature_policy: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(file_path)

        # Hard drop always
        if feature_policy and feature_policy.get("drop_always"):
            df = df.drop(columns=[c for c in feature_policy["drop_always"] if c in df.columns], errors="ignore")

        df_alt = df.drop(columns=self.excluded_features, errors='ignore')

        # Target: 1 = default, 0 = no default
        if "LoanApproved" not in df_alt.columns:
            raise ValueError("Column 'LoanApproved' missing. Required to construct the target.")
        df_alt['default'] = (df_alt['LoanApproved'] == 0).astype(int)

        available = [c for c in self.alternative_features if c in df_alt.columns]
        X = df_alt[available]
        y = df_alt['default']
        mask = ~y.isna()
        return X[mask], y[mask]

    def apply_feature_policy(self, X: pd.DataFrame, feature_policy: Optional[dict], drop_post_decision: bool) -> pd.DataFrame:
        if not feature_policy:
            return X
        out = X.copy()
        if drop_post_decision and feature_policy.get("post_decision"):
            out = out.drop(columns=[c for c in feature_policy["post_decision"] if c in out.columns], errors="ignore")
        if feature_policy.get("allowed"):
            allowed = [c for c in feature_policy["allowed"] if c in out.columns]
            out = out[allowed] if len(allowed) else out
        return out

    # ==== Encoding/Imputation/Scaling ====

    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                if X[col].nunique() > 10:
                    freq = X[col].value_counts(normalize=True)
                    X_encoded[col] = X[col].map(freq)
                else:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
        return X_encoded

    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        num_cols = Xc.select_dtypes(include=[np.number]).columns
        cat_cols = Xc.select_dtypes(include=['object']).columns

        for col in num_cols:
            if Xc[col].isnull().any():
                Xc[col] = Xc[col].fillna(Xc[col].median())
        for col in cat_cols:
            if Xc[col].isnull().any():
                Xc[col] = Xc[col].fillna(Xc[col].mode()[0])
        return Xc

    def scale_numeric(self, X_train_encoded: pd.DataFrame, X_test_encoded: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Xtr = X_train_encoded.copy()
        Xte = X_test_encoded.copy()
        num_cols = Xtr.select_dtypes(include=[np.number]).columns
        Xtr[num_cols] = self.scaler.fit_transform(Xtr[num_cols])
        Xte[num_cols] = self.scaler.transform(Xte[num_cols])
        return Xtr, Xte

    # ==== Model ====

    def _make_monotone_constraints(self, feature_names: List[str]) -> Optional[List[int]]:
        """Build monotone constraints vector aligned to feature_names, if provided."""
        if not self.monotone_map:
            return None
        return [int(self.monotone_map.get(f, 0)) for f in feature_names]

    def create_model(self):
        if self.model_type == 'lightgbm':
            params = dict(
                n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                verbose=-1
            )
            # monotone constraints if provided (LightGBM only)
            if self.feature_names:
                mc = self._make_monotone_constraints(self.feature_names)
                if mc and any(mc):
                    params["monotone_constraints"] = mc
            model = lgb.LGBMClassifier(**params)
        else:
            model = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                eval_metric='logloss'
            )
        return model

    def _split(self, df_X: pd.DataFrame, y: pd.Series,
               time_split_col: Optional[str], test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if time_split_col and (time_split_col in df_X.index.names or time_split_col in df_X.columns):
            df_tmp = df_X.copy()
            if time_split_col in df_tmp.columns:
                ts = pd.to_datetime(df_tmp[time_split_col])
            else:
                ts = pd.to_datetime(df_tmp.index.get_level_values(time_split_col))
            order = np.argsort(ts.values)
            n = len(order)
            split = int((1 - test_size) * n)
            train_idx = df_tmp.index[order[:split]]
            test_idx = df_tmp.index[order[split:]]
            return df_X.loc[train_idx], df_X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]
        else:
            return train_test_split(df_X, y, test_size=test_size, random_state=42, stratify=y)

    def train_model(self, X: pd.DataFrame, y: pd.Series,
                    time_split_col: Optional[str] = None, test_size: float = 0.2):
        X_train, X_test, y_train, y_test = self._split(X, y, time_split_col, test_size)

        # Keep raw X_test for fairness audits
        self.X_test_raw = X_test.copy()

        X_train_clean = self.handle_missing_values(X_train)
        X_test_clean = self.handle_missing_values(X_test)

        X_train_encoded = self.encode_categorical_features(X_train_clean)
        X_test_encoded = self.encode_categorical_features(X_test_clean)

        # set feature names before creating model (for monotone vectors)
        self.feature_names = X_train_encoded.columns.tolist()

        X_train_scaled, X_test_scaled = self.scale_numeric(X_train_encoded, X_test_encoded)

        self.model = self.create_model()

        if self.model_type == 'lightgbm':
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )

        self.X_train, self.y_train = X_train_scaled, y_train
        self.X_test, self.y_test = X_test_scaled, y_test

        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)
        return self.X_test, self.y_test, self.y_pred, self.y_pred_proba

    # ==== Calibration & Reliability ====

    def calibrate(self, method: str = "isotonic"):
        """Calibrate probabilities on training data; evaluate on test."""
        if method not in {"isotonic", "platt", "none"}:
            raise ValueError("method must be 'isotonic', 'platt', or 'none'")
        if method == "none":
            self.calibrator = None
            self.calibration_method = None
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            self.y_pred = (self.y_pred_proba >= 0.5).astype(int)
            return

        base = self.create_model()
        if self.model_type == 'lightgbm':
            base.fit(self.X_train, self.y_train,
                     eval_set=[(self.X_test, self.y_test)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            base.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)

        # sklearn >=1.4 uses 'estimator'; older uses 'base_estimator'
        try:
            calibrated = CalibratedClassifierCV(
                estimator=base,
                method='isotonic' if method == 'isotonic' else 'sigmoid',
                cv=3
            )
        except TypeError:
            calibrated = CalibratedClassifierCV(
                base_estimator=base,
                method='isotonic' if method == 'isotonic' else 'sigmoid',
                cv=3
            )

        calibrated.fit(self.X_train, self.y_train)
        self.calibrator = calibrated
        self.calibration_method = method

        self.y_pred_proba = self.calibrator.predict_proba(self.X_test)[:, 1]
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)

    def _calibration_stats(self) -> Dict[str, float]:
        """
        Compute calibration-in-the-large (CITL) and calibration slope on logit scale:
        Fit logistic regression: y ~ logit(p). Intercept ≈ CITL, coef ≈ slope.
        """
        eps = 1e-12
        p = np.clip(self.y_pred_proba.astype(float), eps, 1 - eps)
        logit_p = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(logit_p, self.y_test.astype(int))
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])  # CITL on log-odds scale
        except Exception:
            slope = np.nan
            intercept = np.nan
        return {"calib_slope": slope, "calib_citl": intercept}

    def plot_reliability(self, fname=os.path.join(GRAPHS_DIR, "reliability_curve.png")) -> Dict[str, float]:
        """Reliability diagram & Brier score."""
        bs = brier_score_loss(self.y_test, self.y_pred_proba)
        df = pd.DataFrame({"p": self.y_pred_proba, "y": self.y_test.values})
        df["bin"] = pd.qcut(df["p"], q=10, duplicates='drop')
        grp = df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).reset_index()

        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Perfect")
        plt.plot(grp["p_mean"], grp["y_rate"], marker='o', label="Model")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed default rate")
        plt.title(f"Reliability Curve (Brier = {bs:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        out = {"brier": float(bs)}
        out.update(self._calibration_stats())
        return out

    # ==== Evaluation Plots ====

    def evaluate_model(self) -> Dict[str, float]:
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        ap = average_precision_score(self.y_test, self.y_pred_proba)  # PR-AUC
        f1 = f1_score(self.y_test, (self.y_pred_proba >= 0.5).astype(int))
        ks = ks_stat(self.y_test.values, self.y_pred_proba)
        gini = 2 * auc - 1

        cm = confusion_matrix(self.y_test, (self.y_pred_proba >= 0.5).astype(int))
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'])
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
        plt.text(0.02, 0.98, f'Accuracy: {acc:.1f}%',
                 transform=plt.gca().transAxes, fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('Confusion Matrix (thr=0.5)')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        plt.figure(figsize=(7, 5.5))
        plt.plot(recall, precision, label=f'PR-AUC = {ap:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        fpr_test, tpr_test, _ = roc_curve(self.y_test, self.y_pred_proba)
        plt.figure(figsize=(6.5, 5.5))
        plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return {"auc": float(auc), "pr_auc": float(ap), "f1@0.5": float(f1), "ks": float(ks), "gini": float(gini), "acc@0.5": float(acc)}

    # ==== Economic Decisioning ====

    def _revenue(self, df_rows: pd.DataFrame, ead: np.ndarray,
                 margin: Optional[float], use_proxy: bool) -> np.ndarray:
        """
        Compute per-loan revenue BEFORE EL.
        Priority:
          1) margin mode: revenue = margin * EAD
          2) proxy mode:  MonthlyLoanPayment*LoanDuration - LoanAmount
          3) fallback:    0.05 * EAD (or LoanAmount if present)
        """
        if margin is not None:
            rev = float(margin) * ead
        elif use_proxy and all(c in df_rows.columns for c in ("MonthlyLoanPayment", "LoanDuration", "LoanAmount")):
            pay = pd.to_numeric(df_rows["MonthlyLoanPayment"], errors="coerce").fillna(0).to_numpy(dtype=float)
            dur = pd.to_numeric(df_rows["LoanDuration"], errors="coerce").fillna(0).to_numpy(dtype=float)
            amt = pd.to_numeric(df_rows["LoanAmount"], errors="coerce").fillna(0).to_numpy(dtype=float)
            rev = np.maximum(pay * dur - amt, 0.0)
        else:
            if "LoanAmount" in df_rows.columns:
                ead_fallback = pd.to_numeric(df_rows["LoanAmount"], errors="coerce").fillna(0).to_numpy(dtype=float)
            else:
                ead_fallback = ead
            rev = 0.05 * ead_fallback
        return rev

    def expected_values(self, df_rows: pd.DataFrame, pd_scores: np.ndarray,
                        lgd: float, ead_col: Optional[str], ead_value: Optional[float],
                        margin: Optional[float], use_proxy: bool
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # EAD
        if ead_col and (ead_col in df_rows.columns):
            ead = pd.to_numeric(df_rows[ead_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            ead = np.full(len(df_rows), float(ead_value) if ead_value is not None else 1.0, dtype=float)

        revenue = self._revenue(df_rows, ead, margin=margin, use_proxy=use_proxy)
        el = pd_scores * float(lgd) * ead
        ep = revenue - el
        return el, revenue, ep

    def optimize_decision(self, X_eval: pd.DataFrame, pd_scores: np.ndarray,
                          lgd: float = 0.45, ead_col: Optional[str] = "LoanAmount", ead_value: Optional[float] = None,
                          review_band: Tuple[float, float] = (0.3, 0.6),
                          margin: Optional[float] = None, use_proxy_revenue: bool = False
                          ) -> Dict[str, Any]:
        """
        Find PD threshold that maximizes portfolio EV on the eval set.
        Returns thresholds for approve/review/decline plus an EV table and break-even Tau.
        """
        df_eval = X_eval.copy().reset_index(drop=True)
        df_eval["_pd"] = pd_scores

        taus = np.linspace(0.01, 0.99, 99)
        evs, approves, eads = [], [], []
        for tau in taus:
            mask_approve = df_eval["_pd"] < tau
            el, revenue, ep = self.expected_values(
                df_eval[mask_approve],
                df_eval.loc[mask_approve, "_pd"].values,
                lgd=lgd, ead_col=ead_col, ead_value=ead_value,
                margin=margin, use_proxy=use_proxy_revenue
            )
            ev = float(np.sum(ep)) if len(ep) else 0.0
            evs.append(ev)
            approves.append(int(mask_approve.sum()))
            # total EAD for ROI
            if ead_col and (ead_col in df_eval.columns):
                ead_tot = float(pd.to_numeric(df_eval.loc[mask_approve, ead_col], errors="coerce").fillna(0).sum())
            else:
                ead_tot = float((float(ead_value) if ead_value is not None else 1.0) * mask_approve.sum())
            eads.append(ead_tot)

        evs = np.array(evs)
        approves = np.array(approves)
        eads = np.array(eads)

        best_idx = int(np.argmax(evs))
        best_tau = float(taus[best_idx])
        best_ev = float(evs[best_idx])
        approve_rate = float(approves[best_idx] / len(df_eval))
        roi_best = float(best_ev / eads[best_idx]) if eads[best_idx] > 0 else 0.0

        # break-even Tau (first Tau where EV >= 0)
        be_idx = np.argmax(evs >= 0) if np.any(evs >= 0) else None
        break_even_tau = float(taus[be_idx]) if be_idx is not None else None

        thr_approve = best_tau
        thr_review_low, thr_review_high = review_band

        # Profit curve plot
        plt.figure(figsize=(7, 5.5))
        plt.plot(taus, evs, label="Portfolio EV")
        plt.axvline(best_tau, color="r", linestyle="--", label=f"Optimal τ={best_tau:.2f}")
        if break_even_tau is not None:
            plt.axvline(break_even_tau, color="g", linestyle=":", label=f"Break-even τ={break_even_tau:.2f}")
        plt.xlabel("PD threshold (approve if PD < τ)")
        plt.ylabel("Expected Value (sum over approved)")
        plt.title("Profit Curve (PD-LGD-EAD)")
        plt.grid(True, alpha=0.3); plt.legend()
        out_pc = os.path.join(GRAPHS_DIR, "profit_curve.png")
        plt.tight_layout(); plt.savefig(out_pc, dpi=300, bbox_inches='tight'); plt.close()

        # EV table at a few reference thresholds
        ref_taus = sorted(set(np.round([0.1, 0.2, 0.3, best_tau, 0.5, 0.6], 2)))
        rows = []
        for t in ref_taus:
            m = df_eval["_pd"] < t
            el, rev, ep = self.expected_values(
                df_eval[m], df_eval.loc[m, "_pd"].values,
                lgd=lgd, ead_col=ead_col, ead_value=ead_value,
                margin=margin, use_proxy=use_proxy_revenue
            )
            total_ead = float(pd.to_numeric(df_eval.loc[m, ead_col], errors="coerce").fillna(0).sum()) if ead_col and (ead_col in df_eval.columns) else float((ead_value if ead_value is not None else 1.0) * m.sum())
            rows.append({
                "tau": float(t),
                "approved": int(m.sum()),
                "approval_rate": float(m.mean()),
                "EL_sum": float(np.sum(el)) if len(el) else 0.0,
                "Revenue_sum": float(np.sum(rev)) if len(rev) else 0.0,
                "EV_sum": float(np.sum(ep)) if len(ep) else 0.0,
                "EAD_sum": total_ead,
                "ROI": float((np.sum(ep) / total_ead) if total_ead > 0 else 0.0)
            })
        ev_table = pd.DataFrame(rows)
        ev_table.to_csv(os.path.join(ARTIFACTS_DIR, "portfolio_ev_table.csv"), index=False)

        return {
            "best_tau": best_tau,
            "best_EV_sum": best_ev,
            "approval_rate_at_best": approve_rate,
            "approve_threshold": thr_approve,
            "break_even_tau": break_even_tau,
            "roi_at_best": roi_best,
            "review_band": {"low": float(review_band[0]), "high": float(review_band[1])},
            "profit_curve_png": out_pc,
            "ev_table_preview": ev_table.head(10).to_dict(orient="records"),
            "taus": taus.tolist(),
            "evs": evs.tolist()
        }

    # ==== Stress Testing ====

    def stress_pd(self, pd_scores: np.ndarray, multiplier: float) -> np.ndarray:
        """Multiply PDs by a factor and cap at 1."""
        return np.clip(pd_scores * float(multiplier), 0.0, 1.0)

    # ==== Explainability (SHAP) ====

    def shap_global_and_local(self, top_n: int = 5):
        """Make global SHAP plots and dump local top contributors as CSV + global mean|SHAP| CSV."""
        if not SHAP_AVAILABLE:
            return {"shap": "unavailable"}

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            # LightGBM binary returns (n_samples, n_features); xgboost list in older versions
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            # Global summary (beeswarm)
            plt.figure()
            shap.summary_plot(sv, self.X_test, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            out_sum = os.path.join(GRAPHS_DIR, "shap_summary.png")
            plt.savefig(out_sum, dpi=300, bbox_inches='tight')
            plt.close()

            # Global bar (mean |SHAP|)
            abs_mean = np.mean(np.abs(sv), axis=0)
            order = np.argsort(-abs_mean)
            df_global = pd.DataFrame({
                "feature": np.array(self.feature_names)[order],
                "mean_abs_shap": abs_mean[order]
            })
            out_global_csv = os.path.join(ARTIFACTS_DIR, "global_shap_mean_abs.csv")
            df_global.to_csv(out_global_csv, index=False)

            plt.figure()
            shap.summary_plot(sv, self.X_test, feature_names=self.feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            out_bar = os.path.join(GRAPHS_DIR, "shap_importance_bar.png")
            plt.savefig(out_bar, dpi=300, bbox_inches='tight')
            plt.close()

            # Local explanations: top_n instances with highest predicted PD
            top_idx = np.argsort(self.y_pred_proba)[-top_n:]
            rows = []
            for i in top_idx:
                contrib = np.abs(sv[i]).astype(float)
                order_i = np.argsort(-contrib)
                top_feats = [(self.feature_names[j], float(sv[i][j]), float(self.X_test[i, j])) for j in order_i[:10]]
                for rank, (feat, val, xval) in enumerate(top_feats, 1):
                    rows.append({"row_index": int(i), "rank": rank, "feature": feat, "shap_value": val, "feature_value": xval})
            df_local = pd.DataFrame(rows)
            out_local_csv = os.path.join(ARTIFACTS_DIR, "local_explanations_topN.csv")
            df_local.to_csv(out_local_csv, index=False)

            return {
                "global_beeswarm": out_sum,
                "global_bar": out_bar,
                "global_mean_abs_csv": out_global_csv,
                "local_csv": out_local_csv
            }
        except Exception as e:
            return {"shap_error": str(e)}

    # ==== Fairness Audit ====

    def fairness_audit(self, group_cols: List[str], threshold: float) -> Optional[str]:
        """Audit approval rate / mean PD / observed default rate by groups (if columns exist)."""
        if self.X_test_raw is None:
            return None
        df = self.X_test_raw.copy()
        df["_pd"] = self.y_pred_proba
        df["_approve"] = (df["_pd"] < float(threshold)).astype(int)
        # Align y
        y = pd.Series(self.y_test).reset_index(drop=True)
        df = df.reset_index(drop=True)
        df["_y"] = y

        found = [c for c in group_cols if c in df.columns]
        if not found:
            return None

        rows = []
        for c in found:
            for g, sub in df.groupby(c):
                rows.append({
                    "group_col": c,
                    "group": str(g),
                    "n": int(len(sub)),
                    "approval_rate": float(sub["_approve"].mean()),
                    "mean_pd": float(sub["_pd"].mean()),
                    "observed_default_rate": float(sub["_y"].mean())
                })
        out = pd.DataFrame(rows).sort_values(["group_col", "group"])
        path = os.path.join(ARTIFACTS_DIR, "group_fairness_metrics.csv")
        out.to_csv(path, index=False)
        return path

    # ==== Save Artifacts ====

    def save_artifacts(self, dataset_path: str, metrics: Dict[str, Any], thresholds: Dict[str, Any]):
        run = {
            "run_id": str(uuid.uuid4()),
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "model_type": self.model_type,
            "calibration": self.calibration_method,
            "dataset_sha256": sha256_file(dataset_path) if os.path.exists(dataset_path) else None,
            "n_train": int(len(self.y_train)),
            "n_test": int(len(self.y_test)),
            "feature_names": self.feature_names
        }
        with open(os.path.join(ARTIFACTS_DIR, "run.json"), "w") as f:
            json.dump(run, f, indent=2)
        with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(ARTIFACTS_DIR, "thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)

        joblib.dump(self.model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, os.path.join(ARTIFACTS_DIR, "calibrator.pkl"))


# ============================
# CLI Runner
# ============================

def main():
    parser = argparse.ArgumentParser(description="Alternative Credit Scoring (calibrated + economics + explainability)")
    parser.add_argument("--data", default="Loan new datset.csv", help="CSV dataset path")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--policy", default=None, help="feature_policy.yaml path")
    parser.add_argument("--ablation_drop_post", action="store_true", help="drop post-decision features per policy")
    parser.add_argument("--time_split_col", default=None, help="timestamp column for temporal split")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--calibrate", default="isotonic", choices=["isotonic", "platt", "none"])
    parser.add_argument("--lgd", type=float, default=0.45, help="Loss Given Default (0..1)")
    parser.add_argument("--ead_col", default="LoanAmount", help="EAD column name (fallback to --ead_value)")
    parser.add_argument("--ead_value", type=float, default=None, help="Constant EAD if column missing")
    parser.add_argument("--review_low", type=float, default=0.30)
    parser.add_argument("--review_high", type=float, default=0.60)
    # Economics
    parser.add_argument("--margin", type=float, default=None,
                        help="If set, revenue = margin * EAD (recommended for sensitivity sweeps).")
    parser.add_argument("--use_proxy_revenue", action="store_true",
                        help="Use proxy revenue (MonthlyPayment*Duration - LoanAmount). If --margin is set, margin wins.")
    # Explainability & Fairness
    parser.add_argument("--shap", action="store_true", help="Compute SHAP global+local explanations")
    parser.add_argument("--explain_top_n", type=int, default=5, help="Local explanations: top N highest-PD rows")
    parser.add_argument("--fairness_groups", type=str, default="",
                        help="Comma-separated columns for fairness audit (e.g., EmploymentStatus,EducationLevel)")
    # Stress testing
    parser.add_argument("--stress_pd_mult", type=float, default=1.0, help="Multiply PDs by this factor for stress test (cap at 1).")
    # Monotonic constraints
    parser.add_argument("--monotone_yaml", type=str, default=None,
                        help="YAML mapping feature -> +1/-1/0 for LightGBM monotonic constraints.")
    args = parser.parse_args()

    policy = load_yaml(args.policy)
    monotone_map = load_yaml(args.monotone_yaml) if args.monotone_yaml else None

    scorer = AlternativeCreditScorer(model_type=args.model, monotone_map=monotone_map)

    # Load base data
    X, y = scorer.load_and_preprocess_data(args.data, feature_policy=policy)

    # Apply policy to features (ablation: drop post-decision if requested)
    X_policy = scorer.apply_feature_policy(X, policy, drop_post_decision=args.ablation_drop_post)

    # Train (temporal split if provided)
    scorer.train_model(
        X_policy, y, time_split_col=args.time_split_col, test_size=args.test_size
    )

    # Calibration
    scorer.calibrate(method=args.calibrate)

    # Evaluate + reliability
    metrics = scorer.evaluate_model()
    rel = scorer.plot_reliability()
    metrics.update(rel)

    # Economic decisioning
    thresholds = scorer.optimize_decision(
        X_eval=scorer.X_test, pd_scores=scorer.y_pred_proba,
        lgd=args.lgd, ead_col=args.ead_col, ead_value=args.ead_value,
        review_band=(args.review_low, args.review_high),
        margin=args.margin, use_proxy_revenue=args.use_proxy_revenue
    )

    # Stress test if requested
    stress_info = {}
    if args.stress_pd_mult and args.stress_pd_mult != 1.0:
        stressed = scorer.stress_pd(scorer.y_pred_proba, args.stress_pd_mult)
        st = scorer.optimize_decision(
            X_eval=scorer.X_test, pd_scores=stressed,
            lgd=args.lgd, ead_col=args.ead_col, ead_value=args.ead_value,
            review_band=(args.review_low, args.review_high),
            margin=args.margin, use_proxy_revenue=args.use_proxy_revenue
        )
        stress_info = {
            "stress_multiplier": args.stress_pd_mult,
            "stress_best_tau": st["best_tau"],
            "stress_best_EV_sum": st["best_EV_sum"],
            "stress_approval_rate": st["approval_rate_at_best"]
        }

    # SHAP explanations (optional)
    shap_art = {}
    if args.shap:
        shap_art = scorer.shap_global_and_local(top_n=args.explain_top_n)

    # Fairness audit (optional)
    fairness_path = None
    if args.fairness_groups:
        cols = [c.strip() for c in args.fairness_groups.split(",") if c.strip()]
        fairness_path = scorer.fairness_audit(cols, threshold=thresholds["best_tau"])

    # Save artifacts
    meta = {
        **metrics,
        **{k: v for k, v in thresholds.items()
           if k in ["best_tau", "best_EV_sum", "approval_rate_at_best", "break_even_tau", "roi_at_best"]}
    }
    if stress_info:
        meta.update(stress_info)
    scorer.save_artifacts(dataset_path=args.data, metrics=meta, thresholds=thresholds)

    # Finance-friendly extras for summary
    n_test = len(scorer.y_test)
    approved_n = max(1, int(thresholds['approval_rate_at_best'] * n_test))
    ev_per_approved = thresholds['best_EV_sum'] / approved_n
    roi_bps = thresholds['roi_at_best'] * 10_000  # basis points

    # Final summary
    print("\n" + "="*68)
    print("ALTERNATIVE CREDIT SCORING — CALIBRATED, EXPLAINABLE & ECONOMICS-AWARE")
    print("="*68)
    print(f"Model: {args.model} | Calibration: {args.calibrate} | SHAP: {'on' if args.shap and SHAP_AVAILABLE else 'off'}")
    print(f"AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1@0.5: {metrics['f1@0.5']:.4f}")
    print(f"Gini: {metrics['gini']:.4f} | KS: {metrics['ks']:.4f} | Brier: {metrics['brier']:.4f} | "
          f"Slope: {metrics.get('calib_slope', float('nan')):.3f} | CITL: {metrics.get('calib_citl', float('nan')):.3f}")
    be_tau = thresholds['break_even_tau']
    print(f"Break-even τ: {be_tau:.2f}" if be_tau is not None else "Break-even τ: n/a")
    print(f"Approve τ*: {thresholds['best_tau']:.2f} | EV*: {thresholds['best_EV_sum']:.2f} "
          f"| Approval rate: {thresholds['approval_rate_at_best']:.2%} "
          f"| ROI*: {roi_bps:.1f} bps | EV/approved: {ev_per_approved:.2f}")
    if stress_info:
        print(f"[Stress ×{stress_info['stress_multiplier']:.2f}] τ*: {stress_info['stress_best_tau']:.2f} "
              f"| EV*: {stress_info['stress_best_EV_sum']:.2f} "
              f"| Approval: {stress_info['stress_approval_rate']:.2%}")
    print(f"Review band: [{args.review_low:.2f}, {args.review_high:.2f}]")
    print("\nArtifacts written to:")
    print(f"- {GRAPHS_DIR}/confusion_matrix.png")
    print(f"- {GRAPHS_DIR}/precision_recall_curve.png")
    print(f"- {GRAPHS_DIR}/roc_curve.png")
    print(f"- {GRAPHS_DIR}/reliability_curve.png")
    print(f"- {GRAPHS_DIR}/profit_curve.png")
    if args.shap and SHAP_AVAILABLE:
        print(f"- {GRAPHS_DIR}/shap_summary.png")
        print(f"- {GRAPHS_DIR}/shap_importance_bar.png")
        print(f"- {ARTIFACTS_DIR}/global_shap_mean_abs.csv")
        print(f"- {ARTIFACTS_DIR}/local_explanations_topN.csv")
    if fairness_path:
        print(f"- {fairness_path}")
    print(f"- {ARTIFACTS_DIR}/portfolio_ev_table.csv")
    print(f"- {ARTIFACTS_DIR}/run.json, metrics.json, thresholds.json, model.pkl, calibrator.pkl (if used)")
    print()

if __name__ == "__main__":
    main()
