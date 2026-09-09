# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.calibration import calibration_curve

# ============================================================
# 2. DEFINE PROJECT ROOT AND PARAMETERS
# ============================================================

# Define the project root so all files can be accessed consistently
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Number of probability bins used in the reliability diagrams
N_BINS = 10

# ============================================================
# 3. DEFINE INPUT AND OUTPUT PATHS
# ============================================================

# Define the paths to the uncertainty method result files
MC_RESULTS_PATH = PROJECT_ROOT / "results" / "mc_dropout" / "resnet50_mc_dropout_results.csv"

ENSEMBLE_RESULTS_PATH = PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv"

# Define the main figures directory
OUTPUT_DIR = PROJECT_ROOT / "figures"

# Create the figures directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the Figure 7 output file
OUTPUT_PATH = OUTPUT_DIR / "figure7.png"

# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

# Check that the MC Dropout results file exists
if not MC_RESULTS_PATH.exists():
    raise FileNotFoundError(f"MC Dropout results not found: {MC_RESULTS_PATH}")

# Check that the Deep Ensemble results file exists
if not ENSEMBLE_RESULTS_PATH.exists():
    raise FileNotFoundError(f"Deep Ensemble results not found: {ENSEMBLE_RESULTS_PATH}")

# ============================================================
# 5. LOAD MODEL RESULTS
# ============================================================

# Load prediction results for both uncertainty methods
mc_df = pd.read_csv(MC_RESULTS_PATH)
ensemble_df = pd.read_csv(ENSEMBLE_RESULTS_PATH)

print("\nLoaded model results successfully.")

print("\nMC Dropout columns:")
print(mc_df.columns.tolist())

print("\nDeep Ensemble columns:")
print(ensemble_df.columns.tolist())

# ============================================================
# 6. DEFINE REQUIRED COLUMNS
# ============================================================

# Define the expected probability columns for each model
MC_PROBABILITY_COLUMN = "MC Mean Probability"
ENSEMBLE_PROBABILITY_COLUMN = "Ensemble Mean Probability"

# Define the true label column
TRUE_LABEL_COLUMN = "True Label"

# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

# Check that the required MC Dropout columns exist
for column in [TRUE_LABEL_COLUMN, MC_PROBABILITY_COLUMN]:
    if column not in mc_df.columns:
        raise ValueError(f"Column '{column}' was not found in the MC Dropout results.")

# Check that the required Deep Ensemble columns exist
for column in [TRUE_LABEL_COLUMN, ENSEMBLE_PROBABILITY_COLUMN]:
    if column not in ensemble_df.columns:
        raise ValueError(f"Column '{column}' was not found in the Deep Ensemble results.")

# ============================================================
# 8. EXTRACT TRUE LABELS AND PREDICTED PROBABILITIES
# ============================================================

# Extract MC Dropout true labels and mean probabilities
mc_true_labels = mc_df[TRUE_LABEL_COLUMN]
mc_probabilities = mc_df[MC_PROBABILITY_COLUMN]

# Extract Deep Ensemble true labels and mean probabilities
ensemble_true_labels = ensemble_df[TRUE_LABEL_COLUMN]
ensemble_probabilities = ensemble_df[ENSEMBLE_PROBABILITY_COLUMN]

# ============================================================
# 9. CALCULATE CALIBRATION CURVES
# ============================================================

# Calculate observed positive frequency for each probability bin
mc_fraction_positive, mc_mean_predicted = calibration_curve(
    mc_true_labels,
    mc_probabilities,
    n_bins=N_BINS,
    strategy="uniform"
)

ensemble_fraction_positive, ensemble_mean_predicted = calibration_curve(
    ensemble_true_labels,
    ensemble_probabilities,
    n_bins=N_BINS,
    strategy="uniform"
)

# ============================================================
# 10. CREATE FIGURE 7
# ============================================================

# Create two side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================
# 11. CREATE RELIABILITY DIAGRAM
# ============================================================

# Plot the perfectly calibrated reference line
axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfectly calibrated"
)

# Plot the MC Dropout calibration curve
axes[0].plot(
    mc_mean_predicted,
    mc_fraction_positive,
    marker="o",
    label="MC Dropout"
)

# Plot the Deep Ensemble calibration curve
axes[0].plot(
    ensemble_mean_predicted,
    ensemble_fraction_positive,
    marker="s",
    label="Deep Ensemble"
)

# Add labels and title
axes[0].set_title("(a) Reliability Diagram")
axes[0].set_xlabel("Mean Predicted Probability P(GON+)")
axes[0].set_ylabel("Observed Fraction of GON+ Cases")

# Set the probability range
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Add a grid and legend
axes[0].grid(alpha=0.3)
axes[0].legend()

# ============================================================
# 12. CREATE CONFIDENCE HISTOGRAM
# ============================================================

# Define common probability bins for both models
histogram_bins = 20

# Plot the distribution of MC Dropout probabilities
axes[1].hist(
    mc_probabilities,
    bins=histogram_bins,
    alpha=0.6,
    label="MC Dropout"
)

# Plot the distribution of Deep Ensemble probabilities
axes[1].hist(
    ensemble_probabilities,
    bins=histogram_bins,
    alpha=0.6,
    label="Deep Ensemble"
)

# Add labels and title
axes[1].set_title("(b) Predictive Confidence Distribution")
axes[1].set_xlabel("Predicted Probability P(GON+)")
axes[1].set_ylabel("Number of Test Images")

# Set the probability range
axes[1].set_xlim(0, 1)

# Add a grid and legend
axes[1].grid(alpha=0.3)
axes[1].legend()

# ============================================================
# 13. ADD MAIN FIGURE TITLE
# ============================================================

fig.suptitle(
    "Figure 7. Calibration Performance and Predictive Confidence Distribution",
    fontsize=16
)

# ============================================================
# 14. ADJUST FIGURE LAYOUT
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.93])

# ============================================================
# 15. SAVE FIGURE 7
# ============================================================

# Save the figure at high resolution
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

# Display the completed figure
plt.show()

# ============================================================
# 16. PRINT SUMMARY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FIGURE 7 SUMMARY")
print("=" * 70)

print("\nMC Dropout:")
print("Number of test images:", len(mc_df))
print("Mean predicted probability:", f"{mc_probabilities.mean():.6f}")
print("Minimum probability:", f"{mc_probabilities.min():.6f}")
print("Maximum probability:", f"{mc_probabilities.max():.6f}")

print("\nDeep Ensemble:")
print("Number of test images:", len(ensemble_df))
print("Mean predicted probability:", f"{ensemble_probabilities.mean():.6f}")
print("Minimum probability:", f"{ensemble_probabilities.min():.6f}")
print("Maximum probability:", f"{ensemble_probabilities.max():.6f}")

# ============================================================
# 17. PRINT SAVED FILE LOCATION
# ============================================================

print("\nFigure 7 saved to:")
print(OUTPUT_PATH)