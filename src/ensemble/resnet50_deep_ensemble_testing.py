# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, accuracy_score, average_precision_score, f1_score, brier_score_loss, log_loss)

# ============================================================
# 2. LOAD THE TEST DATASET
# ============================================================

# Load the previously created held-out test split
test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test_split.csv")

print("\nTesting:")
print("Patients:", test_df["Patient"].nunique())
print("Images:", len(test_df))
print(test_df["Label Encode"].value_counts())

# ============================================================
# 3. PREPARE TEST IMAGE PATHS
# ============================================================

# Convert image paths to strings for the TensorFlow data generator
test_df["Image Path"] = test_df["Image Path"].astype(str)

# ============================================================
# 4. DEFINE IMAGE AND BATCH PARAMETERS
# ============================================================

# Define the image dimensions and batch size used during inference
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# 5. CREATE THE TEST DATA GENERATOR
# ============================================================

# Apply ResNet-50-specific preprocessing to the test images
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="Image Path",
    y_col="Label Encode",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# ============================================================
# 6. RETRIEVE TRUE TEST LABELS
# ============================================================

# Retrieve the true binary labels corresponding to the test images
test_true_labels = test_generator.labels.astype(int)

# ============================================================
# 7. DEFINE THE DEEP ENSEMBLE MODELS
# ============================================================

# Define the file paths of the trained ResNet-50 ensemble members
ensemble_model_files = [
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_1.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_2.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_3.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_4.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_5.keras"
]

# ============================================================
# 8. PERFORM DEEP ENSEMBLE INFERENCE
# ============================================================

# Store predictions from all ensemble members
all_ensemble_predictions = []

# Test each ensemble member
for member_number, model_file in enumerate(
    ensemble_model_files,
    start=1
):

    print("\n" + "=" * 60)

    print(
        f"Testing Deep Ensemble Member "
        f"{member_number}/{len(ensemble_model_files)}"
    )

    print("=" * 60)

    # Load one trained ensemble member
    model = tf.keras.models.load_model(model_file)

    # Reset the generator so testing starts from the first image
    test_generator.reset()

    # Perform normal inference with Dropout disabled
    member_predictions = model.predict(
        test_generator,
        verbose=1
    ).flatten()

    # Check that every test image has one prediction
    if len(member_predictions) != test_generator.samples:
        raise ValueError(
            f"Expected {test_generator.samples} predictions, "
            f"but received {len(member_predictions)}."
        )

    # Store this member's predictions
    all_ensemble_predictions.append(member_predictions)

    print(
        f"\nFinished Ensemble Member "
        f"{member_number}."
    )

    # Clear the model from memory before loading the next member
    del model
    tf.keras.backend.clear_session()

# Convert all ensemble member predictions into one matrix
all_ensemble_predictions = np.array(all_ensemble_predictions)

print("\nFinished Deep Ensemble inference!")

print(
    "Shape of prediction matrix:",
    all_ensemble_predictions.shape
)

# ============================================================
# 9. CALCULATE PREDICTIVE UNCERTAINTY
# ============================================================

# Calculate the mean predicted probability across all ensemble members
ensemble_mean_probabilities = np.mean(
    all_ensemble_predictions,
    axis=0
)

# Calculate prediction variance for each test image
ensemble_variances = np.var(
    all_ensemble_predictions,
    axis=0
)

# Calculate prediction standard deviation for each test image
ensemble_standard_deviations = np.std(
    all_ensemble_predictions,
    axis=0
)

# Convert mean probabilities into final class predictions
ensemble_predictions = (
    ensemble_mean_probabilities >= 0.5
).astype(int)

# ============================================================
# 10. CALCULATE CLASSIFICATION PERFORMANCE METRICS
# ============================================================

# Calculate overall classification accuracy
ensemble_accuracy = accuracy_score(
    test_true_labels,
    ensemble_predictions
)

# Calculate the Area Under the Receiver Operating Characteristic Curve
ensemble_auroc = roc_auc_score(
    test_true_labels,
    ensemble_mean_probabilities
)

# Calculate the Area Under the Precision-Recall Curve
ensemble_auprc = average_precision_score(
    test_true_labels,
    ensemble_mean_probabilities
)

# Create the confusion matrix
ensemble_confusion_matrix = confusion_matrix(
    test_true_labels,
    ensemble_predictions,
    labels=[0, 1]
)

# Extract true negatives, false positives, false negatives and true positives
ensemble_tn, ensemble_fp, ensemble_fn, ensemble_tp = ensemble_confusion_matrix.ravel()

# Calculate sensitivity
ensemble_sensitivity = ensemble_tp / (ensemble_tp + ensemble_fn)

# Calculate specificity
ensemble_specificity = ensemble_tn / (ensemble_tn + ensemble_fp)

# Calculate F1-score
ensemble_f1 = f1_score(
    test_true_labels,
    ensemble_predictions
)

# Calculate balanced accuracy
ensemble_balanced_accuracy = (
    ensemble_sensitivity + ensemble_specificity
) / 2

# Calculate G-Mean
ensemble_g_mean = np.sqrt(
    ensemble_sensitivity * ensemble_specificity
)

# ============================================================
# 11. CALIBRATION ANALYSIS
# ============================================================

# Calculate Brier Score
ensemble_brier_score = brier_score_loss(
    test_true_labels,
    ensemble_mean_probabilities
)

# Calculate Expected Calibration Error (ECE)
N_BINS = 10

bin_edges = np.linspace(
    0.0,
    1.0,
    N_BINS + 1
)

ensemble_ece = 0.0

for i in range(N_BINS):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    # Find predictions belonging to this bin
    if i == 0:
        in_bin = (
            (ensemble_mean_probabilities >= lower)
            &
            (ensemble_mean_probabilities <= upper)
        )
    else:
        in_bin = (
            (ensemble_mean_probabilities > lower)
            &
            (ensemble_mean_probabilities <= upper)
        )

    # Skip empty bins
    if np.sum(in_bin) == 0:
        continue

    # Calculate the mean predicted probability in this bin
    bin_confidence = np.mean(
        ensemble_mean_probabilities[in_bin]
    )

    # Calculate the actual proportion of GON+ cases in this bin
    bin_observed_frequency = np.mean(
        test_true_labels[in_bin]
    )

    # Calculate the proportion of test samples in this bin
    bin_weight = (
        np.sum(in_bin)
        / len(test_true_labels)
    )

    # Add this bin's contribution to ECE
    ensemble_ece += (
        bin_weight
        *
        abs(
            bin_observed_frequency
            - bin_confidence
        )
    )

# Calculate Negative Log-Likelihood (NLL)
ensemble_nll = log_loss(
    test_true_labels,
    ensemble_mean_probabilities
)

# ============================================================
# 12. ERROR-DETECTION ANALYSIS
# ============================================================

# Label incorrect predictions as 1 and correct predictions as 0
ensemble_error_labels = (
    ensemble_predictions != test_true_labels
).astype(int)

# Higher variance indicates greater disagreement between ensemble members
ensemble_error_detection_auroc = roc_auc_score(
    ensemble_error_labels,
    ensemble_variances
)

# ============================================================
# 13. DISPLAY DEEP ENSEMBLE RESULTS
# ============================================================

# Display discrimination and classification performance metrics
print("\nDeep Ensemble Results")

print(f"Accuracy: {ensemble_accuracy:.4f}")
print(f"AUROC: {ensemble_auroc:.4f}")
print(f"AUPRC: {ensemble_auprc:.4f}")
print(f"Sensitivity: {ensemble_sensitivity:.4f}")
print(f"Specificity: {ensemble_specificity:.4f}")
print(f"F1-Score: {ensemble_f1:.4f}")
print(f"Balanced Accuracy: {ensemble_balanced_accuracy:.4f}")
print(f"G-Mean: {ensemble_g_mean:.4f}")

print("\nDeep Ensemble Classification Report:")

print(
    classification_report(
        test_true_labels,
        ensemble_predictions,
        target_names=["GON-", "GON+"]
    )
)

print("\nDeep Ensemble Confusion Matrix:")
print(ensemble_confusion_matrix)

print("\nDeep Ensemble Uncertainty Summary:")
print(f"Mean Variance: {ensemble_variances.mean():.6f}")
print(f"Maximum Variance: {ensemble_variances.max():.6f}")
print(f"Mean Standard Deviation: {ensemble_standard_deviations.mean():.6f}")

print("\nMean Probability for each test image:")
print(ensemble_mean_probabilities)

overall_mean_probability = np.mean(
    ensemble_mean_probabilities
)

print(
    f"\nOverall Mean Probability: "
    f"{overall_mean_probability:.4f}"
)

print("\nCalibration and Error-Detection Results")
print(f"Brier Score: {ensemble_brier_score:.6f}")
print(f"Expected Calibration Error (ECE): {ensemble_ece:.6f}")
print(f"Negative Log-Likelihood (NLL): {ensemble_nll:.6f}")
print(
    f"Error-Detection AUROC: "
    f"{ensemble_error_detection_auroc:.4f}"
)

# ============================================================
# 14. SAVE IMAGE-LEVEL DEEP ENSEMBLE RESULTS
# ============================================================

# Create a DataFrame containing predictions and uncertainty estimates for each test image
ensemble_results_df = test_df.reset_index(drop=True).copy()

ensemble_results_df["True Label"] = test_true_labels

# Save the individual predicted probabilities from each ensemble member
for member_index in range(
    len(ensemble_model_files)
):

    ensemble_results_df[
        f"Ensemble Member "
        f"{member_index + 1} Probability"
    ] = all_ensemble_predictions[
        member_index
    ]

# Save the final ensemble predictions and uncertainty estimates
ensemble_results_df["Ensemble Mean Probability"] = ensemble_mean_probabilities
ensemble_results_df["Ensemble Final Prediction"] = ensemble_predictions
ensemble_results_df["Ensemble Variance"] = ensemble_variances
ensemble_results_df["Ensemble Standard Deviation"] = ensemble_standard_deviations
ensemble_results_df["Prediction Error"] = ensemble_error_labels

# Show whether each prediction was correct
ensemble_results_df["Correct Prediction"] = (
    ensemble_results_df["True Label"]
    == ensemble_results_df["Ensemble Final Prediction"]
)

(PROJECT_ROOT / "results" / "ensemble").mkdir(
    parents=True,
    exist_ok=True
)

ensemble_results_df.to_csv(
    PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv",
    index=False
)

print(
    "\nSaved Deep Ensemble results to "
    "'resnet50_deep_ensemble_results.csv'."
)