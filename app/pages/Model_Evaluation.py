from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[2]
metrics_path = ROOT / "models" / "evaluation_metrics.csv"
confusion_path = ROOT / "models" / "confusion_matrices.json"

st.title("Model Evaluation")
st.write(
    "Held-out evaluation results for the four saved Bayesian approaches. "
    "These metrics are preserved from the notebook and are not recalculated in Streamlit."
)

if not metrics_path.exists():
    st.error("Evaluation results were not found. Run the notebook first.")
    st.stop()

metrics = pd.read_csv(metrics_path)
display_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]
st.dataframe(
    metrics[display_columns].style.format(
        {column: "{:.3f}" for column in display_columns if column != "Model"}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Performance comparison")
fig, ax = plt.subplots(figsize=(11, 5))
metrics.plot(
    x="Model",
    y=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
    kind="bar",
    ax=ax,
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Held-out performance by model")
ax.tick_params(axis="x", rotation=18)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(
    "The stroke outcome is imbalanced, so accuracy can look high even when a "
    "model misses many stroke cases. Precision measures how many predicted "
    "positive cases were correct, recall measures how many observed stroke "
    "cases were found, and PR-AUC is especially useful for an imbalanced "
    "positive class. These are held-out dataset metrics, not clinical validation."
)

if confusion_path.exists():
    with confusion_path.open(encoding="utf-8") as file:
        confusion_matrices = json.load(file)
    st.subheader("Confusion matrices")
    st.caption("Rows are actual classes [0 = no stroke, 1 = stroke]; columns are predicted classes.")
    matrix_columns = st.columns(2)
    for index, (model, matrix) in enumerate(confusion_matrices.items()):
        with matrix_columns[index % 2]:
            st.markdown(f"**{model}**")
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual no stroke", "Actual stroke"],
                    columns=["Predicted no stroke", "Predicted stroke"],
                ),
                width="stretch",
            )

st.caption("This is an educational comparison, not a medical decision-support tool.")
