"""
Valorant rank classifier - Streamlit app.

Upload a test CSV (with the `rating` column still in it), pick one of the five
saved models, and see how well it predicts each player's rank.

The models are only loaded and used for prediction here. Nothing is retrained
on the uploaded file.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Valorant Rank Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return models, encoder, meta


def read_csv(source):
    # keep_default_na=False stops pandas from reading the region "NA" (North America) as missing
    df = pd.read_csv(source, low_memory=False,
                     keep_default_na=False, na_values=["", "NaN", "nan", "null"])
    df.columns = df.columns.str.strip()
    return df


def fix_numeric_columns(df, numeric_cols):
    """The raw Kaggle file stores some numbers as text like "1,506". Make sure they're numbers."""
    df = df.copy()
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False),
                                    errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def evaluate(model_name, X, y_true):
    models, encoder, _ = load_artifacts()
    model = models[model_name]
    n_classes = len(encoder.classes_)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                            labels=list(range(n_classes)))
    except ValueError:
        # this happens when the file doesn't contain every rank
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4.8))
    fig.patch.set_facecolor("white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=2, linecolor="white", ax=ax, annot_kws={"size": 11})
    ax.set_xlabel("Predicted rank")
    ax.set_ylabel("Actual rank")
    ax.tick_params(length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


models, encoder, meta = load_artifacts()
class_names = encoder.classes_.tolist()
target_col = meta["target_col"]
feature_cols = meta["feature_cols"]

st.title("Valorant Rank Classifier")
st.write(
    "Predicts a player's competitive rank (Immortal 1, Immortal 2, Immortal 3 or Radiant) "
    "from their in-game stats: damage per round, headshot %, K/D, win rate, favourite agents and guns. "
    "Five models were trained in `model_training.ipynb`; this app loads them and scores whatever test file you upload."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    use_sample = st.checkbox("Use the bundled test_data.csv", value=uploaded is None)
    if SAMPLE_CSV.exists():
        st.download_button("Download test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

if uploaded is not None:
    data = read_csv(uploaded)
    source_label = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    data = read_csv(SAMPLE_CSV)
    source_label = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV in the sidebar, or tick the box to use the bundled one.")
    st.stop()

# check the file has what the models expect before going any further
missing = [c for c in feature_cols + [target_col] if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

data[target_col] = data[target_col].astype(str).str.strip()
known = data[target_col].isin(class_names)
if not known.all():
    st.warning(f"Skipping {(~known).sum()} row(s) whose rank the models weren't trained on "
               f"(they only know {', '.join(class_names)}).")
    data = data[known]
if data.empty:
    st.error("No usable rows left in the file.")
    st.stop()

X = fix_numeric_columns(data[feature_cols], meta["num_cols"])
y_true = encoder.transform(data[target_col])

st.caption(f"Scoring **{len(data):,} players** from {source_label}")
with st.expander("Preview the data"):
    st.dataframe(data.head(20), width="stretch")

spinner_text = f"Running {model_name}..."
if model_name == "Support Vector Machine":
    spinner_text += " SVM is slow to predict, this can take a few minutes on a big file."
with st.spinner(spinner_text):
    metrics, y_pred = evaluate(model_name, X, y_true)

st.subheader(f"{model_name}: evaluation metrics")
for col, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
    value = metrics[name]
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.3f}")
st.caption("Precision, recall, F1 and AUC are weighted by how many players are in each rank.")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    st.pyplot(plot_confusion_matrix(cm, class_names))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]),
                 width="stretch")

with st.expander("See individual predictions"):
    preds = pd.DataFrame({
        "actual": data[target_col].values,
        "predicted": encoder.inverse_transform(y_pred),
    })
    preds["correct"] = preds["actual"] == preds["predicted"]
    st.dataframe(preds, width="stretch", height=300)
    st.download_button("Download predictions", preds.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].replace('.pkl', '')}.csv",
                       mime="text/csv")

st.divider()
st.subheader("All five models on this file")
st.caption("Runs every model on the file, including SVM, which takes a few minutes. "
           "Results are cached, so switching models afterwards is instant.")
if st.toggle("Compare all models", value=False):
    rows = {}
    progress = st.progress(0.0, text="Scoring models...")
    for i, name in enumerate(MODEL_FILES, start=1):
        rows[name], _ = evaluate(name, X, y_true)
        progress.progress(i / len(MODEL_FILES), text=f"Scored {name}")
    progress.empty()

    comparison = pd.DataFrame(rows).T[METRIC_NAMES]
    best = "background-color: #cfe2f3; color: #0b2545; font-weight: 600"
    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, props=best),
                 width="stretch")
    st.caption("Best score in each column is highlighted.")
