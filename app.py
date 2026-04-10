"""
app.py — Interactive rain prediction demo (Streamlit).

Loads the trained model and preprocessing pipeline from the asset/ directory
and lets users enter raw weather observations to get a next-day rain forecast.

Usage:
    streamlit run app.py
    # or via Makefile:  make app
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Australian Rain Predictor",
    page_icon="🌧️",
    layout="centered",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ASSET_DIR        = Path(__file__).parent / "asset"
MODEL_PATH       = ASSET_DIR / "models" / "rain_model.pkl"
PREPROCESSOR_PATH = ASSET_DIR / "models" / "preprocessor.pkl"
METRICS_PATH     = ASSET_DIR / "reports" / "tables" / "metrics.csv"


# ── Load artifacts (cached so they're only read once) ─────────────────────────
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None
    if not PREPROCESSOR_PATH.exists():
        return None, None
    bundle       = joblib.load(MODEL_PATH)
    prep_bundle  = joblib.load(PREPROCESSOR_PATH)
    return bundle, prep_bundle


bundle, prep_bundle = load_artifacts()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌧️ Australian Rain Predictor")
st.markdown(
    "Enter today's weather observations and the model will estimate the "
    "probability of rain tomorrow."
)

if bundle is None:
    st.error(
        "**Model not found.** "
        "Please run the full pipeline first:\n\n"
        "```bash\nmake all\n```"
    )
    st.stop()

model           = bundle["model"]
algorithm       = bundle.get("algorithm", "Model")
preprocessor    = prep_bundle["preprocessor"]
numeric_cols    = prep_bundle["numeric_cols"]
categorical_cols = prep_bundle["categorical_cols"]

# Read categorical categories directly from the fitted OneHotEncoder
cat_transformer = preprocessor.named_transformers_["cat"]
ohe             = cat_transformer.named_steps["onehot"]
cat_categories  = {
    col: list(cats)
    for col, cats in zip(categorical_cols, ohe.categories_)
}

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    st.markdown(f"**Algorithm:** {algorithm.replace('_', ' ').title()}")

    val_metrics = bundle.get("validation_metrics_at_threshold", {})
    if val_metrics:
        st.markdown(f"**Val ROC-AUC:** {val_metrics.get('roc_auc', 'N/A'):.3f}")
        st.markdown(f"**Val Recall:**  {val_metrics.get('recall',  'N/A'):.3f}")
        st.markdown(f"**Val F1:**      {val_metrics.get('f1',      'N/A'):.3f}")

    if METRICS_PATH.exists():
        st.divider()
        st.subheader("Test Set Results")
        df_m = pd.read_csv(METRICS_PATH)
        test_rows = df_m[df_m["split"] == "test"]
        if not test_rows.empty:
            # Prefer optimal threshold row if available
            if "threshold_type" in test_rows.columns:
                row = test_rows[test_rows["threshold_type"] == "optimal"]
                if row.empty:
                    row = test_rows.iloc[[0]]
            else:
                row = test_rows.iloc[[0]]
            row = row.iloc[0]
            st.markdown(f"**ROC-AUC:** {row.get('roc_auc', 'N/A'):.3f}")
            st.markdown(f"**F1:**      {row.get('f1',      'N/A'):.3f}")
            st.markdown(f"**Recall:**  {row.get('recall',  'N/A'):.3f}")
            thresh = row.get("threshold", 0.5)
            st.markdown(f"**Threshold:** {thresh:.2f}")

    st.divider()
    st.caption(
        "Source: [Australian Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) · "
        "2007–2017 · 145k observations"
    )


# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Enter Today's Weather")

col1, col2 = st.columns(2)

input_values: dict = {}

# Numeric inputs
numeric_defaults = {
    "MinTemp":       15.0,
    "MaxTemp":       25.0,
    "WindGustSpeed": 40.0,
    "WindSpeed9am":  15.0,
    "WindSpeed3pm":  20.0,
    "Humidity9am":   70.0,
    "Humidity3pm":   50.0,
    "Pressure9am":  1015.0,
    "Pressure3pm":  1012.0,
    "Temp9am":       18.0,
    "Temp3pm":       23.0,
}

numeric_ranges = {
    "MinTemp":       (-10.0,  50.0, 0.5),
    "MaxTemp":       (-5.0,   55.0, 0.5),
    "WindGustSpeed": (0.0,   135.0, 1.0),
    "WindSpeed9am":  (0.0,   130.0, 1.0),
    "WindSpeed3pm":  (0.0,   130.0, 1.0),
    "Humidity9am":   (0.0,   100.0, 1.0),
    "Humidity3pm":   (0.0,   100.0, 1.0),
    "Pressure9am":   (970.0, 1040.0, 0.5),
    "Pressure3pm":   (970.0, 1040.0, 0.5),
    "Temp9am":       (-10.0,  50.0, 0.5),
    "Temp3pm":       (-10.0,  55.0, 0.5),
}

# Render numeric sliders across two columns
numeric_in_model = [c for c in numeric_cols if c in numeric_defaults]
mid = len(numeric_in_model) // 2

for i, col_name in enumerate(numeric_in_model):
    mn, mx, step = numeric_ranges.get(col_name, (0.0, 100.0, 1.0))
    default      = numeric_defaults.get(col_name, (mn + mx) / 2)
    target_col   = col1 if i < mid else col2
    label        = col_name.replace("9am", " (9am)").replace("3pm", " (3pm)")
    input_values[col_name] = target_col.slider(
        label, min_value=mn, max_value=mx, value=default, step=step
    )

st.divider()

# Categorical inputs
cat_cols_3 = st.columns(3)
cat_in_model = [c for c in categorical_cols if c in cat_categories]

for i, col_name in enumerate(cat_in_model):
    options = sorted(cat_categories[col_name])
    default_idx = 0
    if col_name == "RainToday":
        default_idx = options.index("No") if "No" in options else 0
    target_col = cat_cols_3[i % 3]
    input_values[col_name] = target_col.selectbox(col_name, options, index=default_idx)

st.divider()


# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Rain Tomorrow", type="primary", use_container_width=True):

    # Build input DataFrame with correct column order expected by preprocessor
    all_input_cols = numeric_cols + categorical_cols
    row_data = {col: [input_values.get(col, np.nan)] for col in all_input_cols}
    X_raw = pd.DataFrame(row_data)

    # Apply saved preprocessor (imputation + encoding)
    try:
        X_encoded = preprocessor.transform(X_raw)
        feature_columns = bundle["feature_columns"]
        X_input = pd.DataFrame(X_encoded, columns=prep_bundle["feature_names"])[feature_columns]

        prob = float(model.predict_proba(X_input)[0, 1])

        # Use optimal threshold from bundle if available
        opt_thresh = val_metrics.get("threshold", 0.5)
        prediction = "Rain" if prob >= opt_thresh else "No Rain"

        st.divider()
        if prediction == "Rain":
            st.error(f"### 🌧️ Rain Tomorrow — {prob * 100:.1f}% probability")
        else:
            st.success(f"### ☀️ No Rain Tomorrow — {prob * 100:.1f}% probability")

        # Probability bar
        st.progress(prob, text=f"Rain probability: {prob:.1%}")

        st.caption(
            f"Decision threshold: {opt_thresh:.2f}  |  "
            f"Algorithm: {algorithm.replace('_', ' ').title()}"
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.exception(exc)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Part of the [Full Stack ML Pipeline](https://github.com) project · "
    "Built with scikit-learn & Streamlit"
)
