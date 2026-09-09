# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# 2. DEFINE PROJECT PATH
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 3. LOAD RESNET-50 DEEP ENSEMBLE RESULTS
# ============================================================

# Load the previously saved ResNet-50 Deep Ensemble results
df = pd.read_csv(PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv")

# Extract the true labels
y_true = df["True Label"].to_numpy()

# Extract the mean predicted probabilities from the Deep Ensemble
y_prob = df["Ensemble Mean Probability"].to_numpy()

# ============================================================
# 4. CALCULATE ROC CURVE
# ============================================================

# Calculate the false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_true, y_prob)

# Calculate the Area Under the Receiver Operating Characteristic Curve
auroc = roc_auc_score(y_true, y_prob)

# ============================================================
# 5. CALCULATE PRECISION-RECALL CURVE
# ============================================================

# Calculate precision and recall values
precision, recall, _ = precision_recall_curve(y_true, y_prob)

# Calculate the Area Under the Precision-Recall Curve
auprc = average_precision_score(y_true, y_prob)

# Calculate the no-skill baseline for the Precision-Recall curve
positive_rate = y_true.mean()

# ============================================================
# 6. CREATE COMBINED ROC AND PRECISION-RECALL FIGURE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================
# 7. PLOT THE ROC CURVE
# ============================================================

axes[0].plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Deep Ensemble (AUROC = {auroc:.4f})"
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="No-skill classifier"
)

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("(a) ROC Curve for ResNet-50 Deep Ensemble")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# ============================================================
# 8. PLOT THE PRECISION-RECALL CURVE
# ============================================================

axes[1].plot(
    recall,
    precision,
    linewidth=2,
    label=f"Deep Ensemble (AUPRC = {auprc:.4f})"
)

axes[1].hlines(
    positive_rate,
    xmin=0,
    xmax=1,
    linestyle="--",
    label="No-skill classifier"
)

axes[1].set_xlabel("Recall (Sensitivity)")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-Recall Curve for ResNet-50 Deep Ensemble")
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

# ============================================================
# 9. SAVE AND DISPLAY THE COMBINED FIGURE
# ============================================================

# Adjust the spacing between the two plots
plt.tight_layout()

# Save the combined ROC and Precision-Recall figure
plt.savefig(
    PROJECT_ROOT / "figures" / "figure6.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the figure
plt.show()