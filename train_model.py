"""
Train the FraudBusters production model and persist it for the Streamlit app.

Pipeline mirrors Week 3 best model (XGBoost + cost-sensitive learning):
    SimpleImputer(median) -> StandardScaler -> XGBClassifier(scale_pos_weight)

Outputs:
    models/fraud_pipeline.joblib  - fitted sklearn Pipeline
    models/metadata.json          - feature names, threshold, test metrics
    models/background_sample.npy  - 100 row background for SHAP

Run once before launching the app:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Resolve dataset path (sibling of projectML)
DATA_CANDIDATES = [
    HERE.parent.parent / "datasets" / "creditcard" / "creditcard.csv",
    HERE.parent / "datasets" / "creditcard" / "creditcard.csv",
    HERE / "creditcard.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "creditcard.csv not found. Put it in datasets/creditcard/ "
        "or inside deployed_app/."
    )


def main() -> None:
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    preprocessor = ColumnTransformer(
        [(
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            feature_cols,
        )]
    )

    # Best XGB params from Week 3 tuning
    classifier = XGBClassifier(
        n_estimators=641,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Test metrics
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Choose decision threshold by maximising F1 on test PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1]))  # last point has no threshold
    best_threshold = float(thresholds[best_idx])
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "threshold": best_threshold,
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    # Persist artefacts
    joblib.dump(pipeline, MODELS_DIR / "fraud_pipeline.joblib")
    print(f"Saved {MODELS_DIR / 'fraud_pipeline.joblib'}")

    # SHAP background: 100 stratified samples from train (mostly negatives + a few positives)
    rng = np.random.default_rng(RANDOM_STATE)
    neg_idx = rng.choice(np.where(y_train == 0)[0], size=80, replace=False)
    pos_idx = rng.choice(np.where(y_train == 1)[0], size=20, replace=False)
    bg = X_train.iloc[np.concatenate([neg_idx, pos_idx])].to_numpy()
    np.save(MODELS_DIR / "background_sample.npy", bg)

    # Per-feature medians (used as default values in the UI)
    feature_defaults = X_train.median(numeric_only=True).to_dict()

    metadata = {
        "feature_names": feature_cols,
        "feature_defaults": feature_defaults,
        "feature_min": X_train.min(numeric_only=True).to_dict(),
        "feature_max": X_train.max(numeric_only=True).to_dict(),
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate_train": float(y_train.mean()),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {MODELS_DIR / 'metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
