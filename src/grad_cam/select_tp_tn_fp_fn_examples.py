# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# 2. DEFINE PROJECT PATH
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 3. LOAD DEEP ENSEMBLE RESULTS
# ============================================================

# Define the path to the previously saved Deep Ensemble results
RESULTS_PATH = (PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv")

# Load the Deep Ensemble results into a DataFrame
df = pd.read_csv(RESULTS_PATH)

# Display information about the loaded results
print("\nLoaded Deep Ensemble results.")
print("Number of test images:", len(df))

# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

# Define the columns required for selecting TP, TN, FP and FN examples
required_columns = [
    "Image Path",
    "True Label",
    "Ensemble Mean Probability",
    "Ensemble Final Prediction",
]

# Check that every required column exists in the results DataFrame
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column not found: {column}")

# ============================================================
# 5. CONVERT LABELS TO INTEGERS
# ============================================================

# Convert the true labels into integer values
df["True Label"] = df["True Label"].astype(int)

# Convert the final ensemble predictions into integer values
df["Ensemble Final Prediction"] = df["Ensemble Final Prediction"].astype(int)

# ============================================================
# 6. CREATE TRUE POSITIVE, TRUE NEGATIVE, FALSE POSITIVE AND FALSE NEGATIVE GROUPS
# ============================================================

# Select correctly predicted positive images
true_positive_df = df[
    (df["True Label"] == 1)
    &
    (df["Ensemble Final Prediction"] == 1)
].copy()

# Select correctly predicted negative images
true_negative_df = df[
    (df["True Label"] == 0)
    &
    (df["Ensemble Final Prediction"] == 0)
].copy()

# Select negative images incorrectly predicted as positive
false_positive_df = df[
    (df["True Label"] == 0)
    &
    (df["Ensemble Final Prediction"] == 1)
].copy()

# Select positive images incorrectly predicted as negative
false_negative_df = df[
    (df["True Label"] == 1)
    &
    (df["Ensemble Final Prediction"] == 0)
].copy()

# ============================================================
# 7. DISPLAY CONFUSION MATRIX GROUP COUNTS
# ============================================================

# Display the number of images in each prediction group
print("\n" + "=" * 60)
print("CONFUSION MATRIX GROUPS")
print("=" * 60)

print("True Positives:", len(true_positive_df))
print("True Negatives:", len(true_negative_df))
print("False Positives:", len(false_positive_df))
print("False Negatives:", len(false_negative_df))

# Display the total number of images across all groups
print(
    "Total:",
    len(true_positive_df)
    + len(true_negative_df)
    + len(false_positive_df)
    + len(false_negative_df)
)

# ============================================================
# 8. EXTRACT IMAGE FILENAMES
# ============================================================

# Extract the filename from the full image path for each prediction group
for group_df in [
    true_positive_df,
    true_negative_df,
    false_positive_df,
    false_negative_df,
]:
    group_df["Image Filename"] = group_df["Image Path"].apply(
        lambda x: Path(str(x)).name
    )

# ============================================================
# 9. DISPLAY CANDIDATE IMAGES FOR EACH GROUP
# ============================================================

# Create a list containing each prediction group and its corresponding name
candidate_groups = [
    ("TRUE POSITIVE CANDIDATES", true_positive_df),
    ("TRUE NEGATIVE CANDIDATES", true_negative_df),
    ("FALSE POSITIVE CANDIDATES", false_positive_df),
    ("FALSE NEGATIVE CANDIDATES", false_negative_df)
]

# Display all available candidate images within each prediction group
for group_name, group_df in candidate_groups:

    print("\n" + "=" * 60)
    print(group_name)
    print("=" * 60)

    # Skip the group if no examples are available
    if len(group_df) == 0:
        print("No examples available.")
        continue

    # Select the relevant columns for displaying candidate examples
    candidates = group_df[
        [
            "Image Filename",
            "True Label",
            "Ensemble Mean Probability",
            "Ensemble Final Prediction"
        ]
    ]

    # Sort the candidate images according to their predicted probabilities
    candidates = candidates.sort_values(by="Ensemble Mean Probability")

    # Display the candidate examples
    print(candidates.to_string(index=False))

# ============================================================
# 10. SELECT ONE EXAMPLE FROM EACH PREDICTION GROUP
# ============================================================

# Create an empty list to store the selected examples
selected_rows = []

# Select the True Positive with the highest predicted probability
if len(true_positive_df) > 0:
    tp_index = true_positive_df["Ensemble Mean Probability"].idxmax()
    tp_row = df.loc[tp_index].copy()
    tp_row["Example Type"] = "True Positive"
    selected_rows.append(tp_row)

# Select the True Negative with the lowest predicted probability
if len(true_negative_df) > 0:
    tn_index = true_negative_df["Ensemble Mean Probability"].idxmin()
    tn_row = df.loc[tn_index].copy()
    tn_row["Example Type"] = "True Negative"
    selected_rows.append(tn_row)

# Select the False Positive with the highest predicted probability
if len(false_positive_df) > 0:
    fp_index = false_positive_df["Ensemble Mean Probability"].idxmax()
    fp_row = df.loc[fp_index].copy()
    fp_row["Example Type"] = "False Positive"
    selected_rows.append(fp_row)

# Select the False Negative with the lowest predicted probability
if len(false_negative_df) > 0:
    fn_index = false_negative_df["Ensemble Mean Probability"].idxmin()
    fn_row = df.loc[fn_index].copy()
    fn_row["Example Type"] = "False Negative"
    selected_rows.append(fn_row)

# ============================================================
# 11. CREATE SELECTED EXAMPLES DATAFRAME
# ============================================================

# Convert the selected examples into a DataFrame
selected_examples_df = pd.DataFrame(selected_rows)

# ============================================================
# 12. DISPLAY SELECTED QUALITATIVE EXAMPLES
# ============================================================

# Display information about the selected examples
print("\n" + "=" * 60)
print("SELECTED QUALITATIVE EXAMPLES")
print("=" * 60)

# Display the details of each selected image
for index, row in selected_examples_df.iterrows():

    # Extract the image filename from the full image path
    image_filename = Path(str(row["Image Path"])).name

    print("\nExample:", row["Example Type"])
    print("Image:", image_filename)
    print("True Label:", row["True Label"])
    print("Ensemble Mean Probability:", row["Ensemble Mean Probability"])
    print("Final Prediction:", row["Ensemble Final Prediction"])

# ============================================================
# 13. SAVE SELECTED EXAMPLES
# ============================================================

# Define the path for saving the selected Grad-CAM examples
OUTPUT_PATH = (PROJECT_ROOT / "results" / "grad_cam" / "gradcam_tp_tn_fp_fn_examples.csv")

# Save the selected examples to a CSV file
selected_examples_df.to_csv(OUTPUT_PATH, index=False)

# Confirm that the selected examples were saved successfully
print("\nSaved TP/TN/FP/FN Grad-CAM examples to:")
print(OUTPUT_PATH)