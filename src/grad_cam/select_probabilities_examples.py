# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the path to the saved Deep Ensemble results
RESULTS_PATH = PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv"

# ============================================================
# 3. LOAD DEEP ENSEMBLE RESULTS
# ============================================================

# Load the previously saved Deep Ensemble predictions
df = pd.read_csv(RESULTS_PATH)

print("\nLoaded Deep Ensemble results.")
print("Number of test images:", len(df))

# ============================================================
# 4. CHECK REQUIRED PROBABILITY COLUMN
# ============================================================

# Define the column containing the mean ensemble probabilities
probability_column = "Ensemble Mean Probability"

# Ensure the required probability column exists
if probability_column not in df.columns:
    raise ValueError(f"Column '{probability_column}' was not found in the CSV.")

# ============================================================
# 5. SELECT HIGH, MIDDLE AND LOW PROBABILITY EXAMPLES
# ============================================================

# Find the image with the highest predicted probability
high_index = df[probability_column].idxmax()
high_row = df.loc[high_index]

# Find the image with the lowest predicted probability
low_index = df[probability_column].idxmin()
low_row = df.loc[low_index]

# Find the image with a predicted probability closest to 0.5
middle_index = df[probability_column].sub(0.5).abs().idxmin()
middle_row = df.loc[middle_index]

# ============================================================
# 6. DISPLAY SELECTED EXAMPLES
# ============================================================

print("\n" + "=" * 60)
print("HIGH PROBABILITY IMAGE")
print("=" * 60)

print("Image Path:", high_row["Image Path"])
print("True Label:", high_row["True Label"])
print("Ensemble Mean Probability:", high_row[probability_column])
print("Final Prediction:", high_row["Ensemble Final Prediction"])


print("\n" + "=" * 60)
print("MIDDLE PROBABILITY IMAGE")
print("=" * 60)

print("Image Path:", middle_row["Image Path"])
print("True Label:", middle_row["True Label"])
print("Ensemble Mean Probability:", middle_row[probability_column])
print("Final Prediction:", middle_row["Ensemble Final Prediction"])


print("\n" + "=" * 60)
print("LOW PROBABILITY IMAGE")
print("=" * 60)

print("Image Path:", low_row["Image Path"])
print("True Label:", low_row["True Label"])
print("Ensemble Mean Probability:", low_row[probability_column])
print("Final Prediction:", low_row["Ensemble Final Prediction"])

# ============================================================
# 7. CREATE DATAFRAME OF SELECTED EXAMPLES
# ============================================================

# Combine the three selected examples into a new DataFrame
selected_examples = pd.DataFrame([high_row, middle_row, low_row]).copy()

# Add a column identifying the type of each selected example
selected_examples.insert(
    0,
    "Example Type",
    ["High Probability", "Middle Probability", "Low Probability"]
)

# ============================================================
# 8. SAVE SELECTED GRAD-CAM EXAMPLES
# ============================================================

# Define the output path for the selected examples
OUTPUT_PATH = PROJECT_ROOT / "results" / "grad_cam" / "gradcam_probabilities_examples.csv"

# Save the selected examples to a CSV file
selected_examples.to_csv(OUTPUT_PATH, index=False)

print("\nSaved selected Grad-CAM examples to:")
print(OUTPUT_PATH)