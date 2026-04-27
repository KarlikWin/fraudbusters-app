"""
FraudBusters — Streamlit fraud-detection app with SHAP explanations.

Run locally:
    streamlit run app.py

Deploy:
    Streamlit Community Cloud / HuggingFace Spaces / Render — see README.md
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

st.set_page_config(
    page_title="FraudBusters — Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "fraud_pipeline.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    background = np.load(MODELS_DIR / "background_sample.npy")
    return pipeline, metadata, background


@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def build_explainer(_pipeline, _background, feature_names):
    # Pipeline = [imputer, scaler, classifier]. Re-use the first two steps as
    # the "preprocessor" for SHAP input.
    classifier = _pipeline.named_steps["classifier"]
    preprocessor = Pipeline(_pipeline.steps[:-1])
    bg_df = pd.DataFrame(_background, columns=feature_names)
    bg_transformed = preprocessor.transform(bg_df)
    explainer = shap.TreeExplainer(
        classifier,
        data=bg_transformed,
        feature_perturbation="interventional",
        feature_names=feature_names,
    )
    return explainer, preprocessor, classifier


def sidebar_inputs(metadata: dict) -> pd.DataFrame:
    st.sidebar.header("Transaction features")
    st.sidebar.caption(
        "Defaults = training-set medians. V1–V28 are PCA components; "
        "edit Time / Amount or paste a CSV row below."
    )

    paste = st.sidebar.text_area(
        "Paste comma-separated row (Time,V1..V28,Amount)",
        value="",
        height=80,
        help="Optional. Overrides the sliders if 30 numeric values are pasted.",
    )

    feature_names = metadata["feature_names"]
    defaults = metadata["feature_defaults"]
    fmin = metadata["feature_min"]
    fmax = metadata["feature_max"]

    if paste.strip():
        try:
            values = [float(x) for x in paste.replace("\n", ",").split(",") if x.strip()]
            if len(values) != len(feature_names):
                st.sidebar.error(
                    f"Expected {len(feature_names)} values, got {len(values)}."
                )
            else:
                row = dict(zip(feature_names, values))
                return pd.DataFrame([row], columns=feature_names)
        except ValueError:
            st.sidebar.error("Could not parse pasted values as numbers.")

    # Manual sliders — keep UI compact: Time + Amount as numbers, V1–V28 in expander
    row = {}
    row["Time"] = st.sidebar.number_input(
        "Time (seconds since first transaction)",
        value=float(defaults["Time"]),
        step=1.0,
    )
    row["Amount"] = st.sidebar.number_input(
        "Amount (€)",
        value=float(defaults["Amount"]),
        min_value=0.0,
        step=1.0,
    )
    with st.sidebar.expander("V1 – V28 (PCA components)"):
        for f in feature_names:
            if f in ("Time", "Amount"):
                continue
            lo, hi = float(fmin[f]), float(fmax[f])
            row[f] = st.slider(
                f,
                min_value=float(np.floor(lo)),
                max_value=float(np.ceil(hi)),
                value=float(defaults[f]),
                step=0.01,
            )
    return pd.DataFrame([row], columns=feature_names)


def render_prediction(prob: float, threshold: float) -> None:
    is_fraud = prob >= threshold
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud probability", f"{prob*100:.2f}%")
    col2.metric("Decision threshold", f"{threshold*100:.2f}%")
    col3.metric("Predicted class", "FRAUD ⚠" if is_fraud else "Legitimate ✅")

    if is_fraud:
        st.error(
            f"Transaction flagged as **fraud** (probability {prob:.3f} ≥ "
            f"threshold {threshold:.3f}). Review before authorising."
        )
    else:
        st.success(
            f"Transaction predicted **legitimate** (probability {prob:.3f} < "
            f"threshold {threshold:.3f})."
        )


def render_shap(explainer, preprocessor, x_row: pd.DataFrame, feature_names) -> None:
    x_t = preprocessor.transform(x_row)
    shap_values = explainer(x_t)
    # shap_values is an Explanation object with shape (1, n_features)
    sv = shap_values[0]
    # Override displayed feature names (TreeExplainer sometimes loses them)
    sv.feature_names = feature_names

    st.subheader("Why this prediction? — SHAP local explanation")
    st.caption(
        "Red bars push the prediction toward fraud, blue bars push it toward legitimate. "
        "Bar length = magnitude of contribution to the log-odds."
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv, max_display=12, show=False)
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)

    # Top-3 textual summary
    contribs = pd.DataFrame({
        "feature": feature_names,
        "value": x_row.iloc[0].values,
        "shap": sv.values,
    })
    contribs["abs"] = contribs["shap"].abs()
    top = contribs.sort_values("abs", ascending=False).head(3)
    st.markdown("**Top drivers of this decision:**")
    for _, r in top.iterrows():
        direction = "↑ fraud" if r["shap"] > 0 else "↓ legitimate"
        st.markdown(
            f"- `{r['feature']} = {r['value']:.4f}` → SHAP {r['shap']:+.3f} ({direction})"
        )


@st.cache_data(show_spinner="Computing SHAP values for beeswarm...")
def compute_global_shap(_explainer, _preprocessor, background, feature_names, n_samples: int = 200):
    bg_df = pd.DataFrame(background, columns=feature_names)
    bg_t = _preprocessor.transform(bg_df)
    sv = _explainer(bg_t[:n_samples])
    sv.feature_names = feature_names
    return sv


def render_global_beeswarm(sv) -> None:
    with st.expander("Global SHAP — beeswarm (feature impact distribution)"):
        st.caption(
            "Each dot is one transaction from the background sample. "
            "Position on x-axis = SHAP value (impact on log-odds of fraud); "
            "colour = feature value (red = high, blue = low). "
            "Features ordered by overall importance."
        )
        fig = plt.figure(figsize=(9, 6))
        shap.plots.beeswarm(sv, max_display=15, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)


def render_global_importance(sv, feature_names) -> None:
    with st.expander("Global feature importance (mean |SHAP| over background sample)"):
        mean_abs = np.abs(sv.values).mean(axis=0)
        imp = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 15 features (global importance)")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.title("💳 FraudBusters — Credit Card Fraud Detection")
    st.markdown(
        "MSc ML Group Project · Team 19 · "
        "Cost-sensitive XGBoost trained on the Kaggle "
        "[Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset."
    )

    if not (MODELS_DIR / "fraud_pipeline.joblib").exists():
        st.error(
            "Model artefacts not found. Run `python train_model.py` first."
        )
        st.stop()

    pipeline, metadata, background = load_artifacts()
    feature_names = metadata["feature_names"]
    threshold = metadata["metrics"]["threshold"]

    with st.expander("Model performance on hold-out test set"):
        m = metadata["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{m['pr_auc']:.4f}")
        c2.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        c3.metric("F1 (best τ)", f"{m['f1']:.4f}")
        c4.metric("MCC", f"{m['mcc']:.4f}")
        st.caption(
            f"Trained on {metadata['n_train']:,} rows "
            f"(fraud rate {metadata['fraud_rate_train']*100:.3f}%), "
            f"evaluated on {metadata['n_test']:,} rows."
        )

    explainer, preprocessor, _ = build_explainer(pipeline, background, feature_names)
    global_sv = compute_global_shap(explainer, preprocessor, background, feature_names)

    render_global_beeswarm(global_sv)

    x_row = sidebar_inputs(metadata)

    st.subheader("Input transaction")
    st.dataframe(x_row, use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        prob = float(pipeline.predict_proba(x_row)[0, 1])
        render_prediction(prob, threshold)
        render_shap(explainer, preprocessor, x_row, feature_names)

    render_global_importance(global_sv, feature_names)


if __name__ == "__main__":
    main()
