import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow import keras

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
DATA_CANDIDATES = [
    "data/cleaned_data.csv",
    "data/creditcard.csv",
    "cleaned_data.csv",
    "creditcard.csv",
]
RANDOM_STATE = 42  # must match 02_Modeling.ipynb for the test split to line up

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    model_path = os.path.join(ARTIFACTS_DIR, "fraud_detection_model.keras")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    threshold_path = os.path.join(ARTIFACTS_DIR, "threshold.json")
    config_path = os.path.join(ARTIFACTS_DIR, "model_config.json")

    for p in [model_path, scaler_path, threshold_path, config_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    with open(config_path) as f:
        config = json.load(f)

    return model, scaler, threshold, config


@st.cache_data(show_spinner=False)
def load_metrics_artifacts():
    out = {}
    for name, fname in [
        ("final_metrics", "final_metrics.csv"),
        ("hyperparameter_results", "hyperparameter_results.csv"),
        ("training_history", "training_history.csv"),
    ]:
        path = os.path.join(ARTIFACTS_DIR, fname)
        out[name] = pd.read_csv(path) if os.path.exists(path) else None
    return out


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_from_disk():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None


@st.cache_data(show_spinner=False)
def build_splits(df: pd.DataFrame):
    """Reproduce the exact train/val/test split used in 02_Modeling.ipynb."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # train/val split isn't needed here, but kept for parity with the notebook
    _, _, _, _ = train_test_split(
        X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=RANDOM_STATE
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


@st.cache_data(show_spinner="Scoring test set...")
def score_test_set(_model, _scaler, X_test, y_test, threshold):
    X_scaled = _scaler.transform(X_test)
    probs = _model.predict(X_scaled, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    results = X_test.copy()
    results["actual"] = y_test.values
    results["fraud_probability"] = probs
    results["predicted"] = preds
    return results


# --------------------------------------------------------------------------
# Load everything up front
# --------------------------------------------------------------------------
try:
    model, scaler, threshold, model_config = load_model_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find `{e.args[0]}`. Make sure you're running "
        "`streamlit run app.py` from the repo root, next to the `artifacts/` folder."
    )
    st.stop()

metrics_artifacts = load_metrics_artifacts()

st.title("🕵️ Credit Card Fraud Detection")
st.caption(
    "A neural-network classifier trained on the anonymized Kaggle credit-card "
    "fraud dataset (Time, V1–V28 PCA components, Amount)."
)

tab_explore, tab_performance = st.tabs(["🔍 Explore Test Transactions", "📊 Model Performance"])

# --------------------------------------------------------------------------
# TAB 1 — Explore Test Transactions
# --------------------------------------------------------------------------
with tab_explore:
    df, source_path = load_dataset_from_disk()

    if df is None:
        st.warning(
            "No dataset found at `data/cleaned_data.csv` (or the other usual spots). "
            "Upload the cleaned dataset to explore real transactions."
        )
        uploaded = st.file_uploader("Upload cleaned_data.csv or creditcard.csv", type="csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.stop()
    else:
        st.caption(f"Loaded dataset from `{source_path}`")

    if "Class" not in df.columns:
        st.error("This dataset needs a `Class` column (0 = legitimate, 1 = fraud) to show ground truth.")
        st.stop()

    X_test, y_test = build_splits(df)
    results = score_test_set(model, scaler, X_test, y_test, threshold)

    st.markdown(
        f"Held-out **test set**: {len(results):,} transactions "
        f"({int(results['actual'].sum())} fraud, {int((results['actual'] == 0).sum()):,} legitimate) "
        "— reproduced with the same 80/20 split used during training, so the model has never seen these rows."
    )

    # --- Filters ---
    col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
    with col_f1:
        class_filter = st.selectbox("Actual class", ["All", "Legitimate", "Fraud"])
    with col_f2:
        amt_min, amt_max = float(results["Amount"].min()), float(results["Amount"].max())
        amount_range = st.slider(
            "Amount range", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max)
        )
    with col_f3:
        st.write("")
        st.write("")
        random_click = st.button("🎲 Random transaction", use_container_width=True)

    filtered = results.copy()
    if class_filter == "Legitimate":
        filtered = filtered[filtered["actual"] == 0]
    elif class_filter == "Fraud":
        filtered = filtered[filtered["actual"] == 1]
    filtered = filtered[
        (filtered["Amount"] >= amount_range[0]) & (filtered["Amount"] <= amount_range[1])
    ]

    if filtered.empty:
        st.info("No transactions match these filters.")
        st.stop()

    if "selected_idx" not in st.session_state or random_click:
        st.session_state.selected_idx = int(np.random.choice(filtered.index))

    if st.session_state.selected_idx not in filtered.index:
        st.session_state.selected_idx = int(filtered.index[0])

    chosen_idx = st.selectbox(
        "Or pick a specific transaction (row index)",
        options=filtered.index.tolist(),
        index=filtered.index.tolist().index(st.session_state.selected_idx),
    )
    st.session_state.selected_idx = chosen_idx

    row = results.loc[chosen_idx]

    st.divider()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Fraud probability", f"{row['fraud_probability']:.2%}")
    col_b.metric("Model prediction", "🚨 Fraud" if row["predicted"] == 1 else "✅ Legitimate")
    col_c.metric("Actual label", "🚨 Fraud" if row["actual"] == 1 else "✅ Legitimate")
    correct = row["predicted"] == row["actual"]
    col_d.metric("Result", "✅ Correct" if correct else "❌ Missed")

    st.progress(min(float(row["fraud_probability"]), 1.0))
    st.caption(f"Decision threshold: {threshold:.4f}")

    # --- Why might this be flagged: compare against class-conditional means ---
    st.subheader("What drove this prediction?")
    st.caption(
        "The V1–V28 columns are anonymized PCA components, so we can't label them "
        "with real-world meaning — but we can show how this transaction's most "
        "fraud-correlated features compare to the typical legitimate vs. fraudulent transaction."
    )

    corr = df.corr(numeric_only=True)["Class"].drop("Class")
    top_features = corr.abs().sort_values(ascending=False).head(6).index.tolist()

    legit_means = df[df["Class"] == 0][top_features].mean()
    fraud_means = df[df["Class"] == 1][top_features].mean()

    compare_df = pd.DataFrame(
        {
            "This transaction": row[top_features].values,
            "Typical legitimate": legit_means.values,
            "Typical fraud": fraud_means.values,
        },
        index=top_features,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    compare_df.plot(kind="barh", ax=ax)
    ax.set_xlabel("Value")
    ax.set_title("Top fraud-correlated features: this transaction vs. class averages")
    ax.invert_yaxis()
    st.pyplot(fig)

    with st.expander("Show raw feature values for this transaction"):
        st.dataframe(row.drop(["actual", "fraud_probability", "predicted"]).to_frame("value"))

# --------------------------------------------------------------------------
# TAB 2 — Model Performance
# --------------------------------------------------------------------------
with tab_performance:
    st.subheader("Final test-set metrics")
    if metrics_artifacts["final_metrics"] is not None:
        fm = metrics_artifacts["final_metrics"].set_index("Metric")["Value"]
        cols = st.columns(len(fm))
        for c, (name, val) in zip(cols, fm.items()):
            c.metric(name, f"{val:.4f}")
    else:
        st.info("`artifacts/final_metrics.csv` not found.")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion matrix (test set)")
        try:
            df_full, _ = load_dataset_from_disk()
            if df_full is not None and "Class" in df_full.columns:
                X_test, y_test = build_splits(df_full)
                scored = score_test_set(model, scaler, X_test, y_test, threshold)
                cm = confusion_matrix(scored["actual"], scored["predicted"])
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
                ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"]).plot(
                    ax=ax_cm, values_format="d", colorbar=False
                )
                ax_cm.set_title(f"Threshold = {threshold:.4f}")
                st.pyplot(fig_cm)

                with st.expander("Classification report"):
                    st.text(
                        classification_report(
                            scored["actual"], scored["predicted"], target_names=["Legitimate", "Fraud"]
                        )
                    )
            else:
                st.info("Dataset not available — upload it in the Explore tab to compute a live confusion matrix.")
        except Exception as e:
            st.info(f"Couldn't compute confusion matrix: {e}")

    with col_right:
        st.subheader("Model configuration")
        st.json(model_config)

    st.divider()

    st.subheader("Training history")
    history = metrics_artifacts["training_history"]
    if history is not None:
        h1, h2, h3 = st.columns(3)

        def plot_curve(container, col_pair, title, ylabel):
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            for col, label in col_pair:
                if col in history.columns:
                    ax.plot(history[col], label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            container.pyplot(fig)

        plot_curve(h1, [("loss", "Train"), ("val_loss", "Validation")], "Loss", "Loss")
        plot_curve(h2, [("pr_auc", "Train"), ("val_pr_auc", "Validation")], "PR-AUC", "PR-AUC")
        plot_curve(h3, [("roc_auc", "Train"), ("val_roc_auc", "Validation")], "ROC-AUC", "ROC-AUC")
    else:
        st.info("`artifacts/training_history.csv` not found.")

    st.divider()

    st.subheader("Hyperparameter search results")
    hp = metrics_artifacts["hyperparameter_results"]
    if hp is not None:
        st.dataframe(
            hp.sort_values("Best Val PR-AUC", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("`artifacts/hyperparameter_results.csv` not found.")
