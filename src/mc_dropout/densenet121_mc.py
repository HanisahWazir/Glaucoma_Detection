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
from tensorflow.keras.applications.densenet import preprocess_input

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

# Convert image paths to strings and resolve them to absolute paths
test_df["Image Path"] = test_df["Image Path"].astype(str).apply(
    lambda p: str((PROJECT_ROOT / p).resolve()))

# ============================================================
# 4. DEFINE IMAGE AND BATCH PARAMETERS
# ============================================================

# Define the image dimensions and batch size used during inference
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# 5. CREATE THE TEST DATA GENERATOR
# ============================================================

# Apply DenseNet-121-specific preprocessing to the test images
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
# 6. LOAD THE TRAINED DENSENET-121 MODEL
# ============================================================

# Load the previously trained DenseNet-121 baseline model
model = tf.keras.models.load_model(PROJECT_ROOT / "models" / "densenet121_baseline.keras")

print("\nLoaded the saved DenseNet121 baseline model.")
model.summary()

# ============================================================
# 7. RETRIEVE TRUE TEST LABELS
# ============================================================

# Retrieve the true binary labels corresponding to the test images
test_true_labels = test_generator.labels.astype(int)

# ============================================================
# 8. PERFORM MONTE CARLO DROPOUT INFERENCE
# ============================================================

# Define the number of stochastic forward passes
MC_RUNS = 30
all_mc_predictions = []
for run in range(MC_RUNS):
    print(f"\nMonte Carlo Run {run + 1}/{MC_RUNS}")
    # Store predictions from this run
    run_predictions = []
    # Start from the first test batch
    test_generator.reset()
    for batch_number in range(len(test_generator)):
        # Load one batch of test images and labels
        batch_images, batch_labels = (test_generator[batch_number])
        # training=True keeps Dropout active
        batch_predictions = model(
            batch_images,
            training=True
        )
        # Convert predictions into a NumPy array
        batch_predictions = (batch_predictions.numpy().flatten())
        # Store this batch's predictions
        run_predictions.extend(batch_predictions)
    # Convert this run into a NumPy array
    run_predictions = np.array(run_predictions)
    # Check that every test image has one prediction
    if len(run_predictions) != test_generator.samples:
        raise ValueError(
            f"Expected {test_generator.samples} predictions, "
            f"but received {len(run_predictions)}."
        )
    # Store this completed run
    all_mc_predictions.append(run_predictions)
# Convert all 30 runs into one matrix
all_mc_predictions = np.array(all_mc_predictions)

print("\nFinished Monte Carlo Dropout!")
print(
    "Shape of prediction matrix:",
    all_mc_predictions.shape
)

# ============================================================
# 9. CALCULATE PREDICTIVE UNCERTAINTY
# ============================================================

# Calculate the mean predicted probability for each test image
mc_mean_probabilities = np.mean(
    all_mc_predictions,
    axis=0
)

# Variance for each image
mc_variances = np.var(
    all_mc_predictions,
    axis=0
)

# Standard deviation for each image
mc_standard_deviations = np.std(
    all_mc_predictions,
    axis=0
)

# Convert mean probabilities into final classes
mc_predictions = (mc_mean_probabilities >= 0.5).astype(int)

# ============================================================
# 10. CALCULATE CLASSIFICATION PERFORMANCE METRICS
# ============================================================

# Calculate overall classification accuracy
mc_accuracy = accuracy_score(
    test_true_labels,
    mc_predictions
)

# Calculate the Area Under the Receiver Operating Characteristic Curve
mc_auroc = roc_auc_score(
    test_true_labels,
    mc_mean_probabilities
)

# Calculate the Area Under the Precision-Recall Curve
mc_auprc = average_precision_score(
    test_true_labels,
    mc_mean_probabilities
)

# Create the confusion matrix
mc_confusion_matrix = confusion_matrix(
    test_true_labels,
    mc_predictions,
    labels=[0, 1]
)

# Extract true negatives, false positives, false negatives and true positives
mc_tn, mc_fp, mc_fn, mc_tp = (mc_confusion_matrix.ravel())

# Calculate sensitivity
mc_sensitivity = (mc_tp / (mc_tp + mc_fn))

# Calculate specificity
mc_specificity = (mc_tn / (mc_tn + mc_fp))

# Calculate F1-score
mc_f1 = f1_score(
    test_true_labels,
    mc_predictions
)

# Calculate balanced accuracy
mc_balanced_accuracy = (mc_sensitivity + mc_specificity) / 2

# Calculate G-Mean
mc_g_mean = np.sqrt(mc_sensitivity * mc_specificity)

# ============================================================
# 11. CALIBRATION ANALYSIS
# ============================================================

# Calculate Brier Score
mc_brier_score = brier_score_loss(
    test_true_labels,
    mc_mean_probabilities
)

# Calculate Expected Calibration Error (ECE)
N_BINS = 10

bin_edges = np.linspace(
    0.0,
    1.0,
    N_BINS + 1
)

mc_ece = 0.0

for i in range(N_BINS):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    if i == 0:
        in_bin = (
            (mc_mean_probabilities >= lower)
            &
            (mc_mean_probabilities <= upper)
        )
    else:
        in_bin = (
            (mc_mean_probabilities > lower)
            &
            (mc_mean_probabilities <= upper)
        )

    if np.sum(in_bin) == 0:
        continue

    bin_confidence = np.mean(
        mc_mean_probabilities[in_bin]
    )

    bin_observed_frequency = np.mean(
        test_true_labels[in_bin]
    )

    bin_weight = (
        np.sum(in_bin)
        / len(test_true_labels)
    )

    mc_ece += (
        bin_weight
        *
        abs(
            bin_observed_frequency
            - bin_confidence
        )
    )

# Calculate Negative Log-Likelihood (NLL)
mc_nll = log_loss(
    test_true_labels,
    mc_mean_probabilities
)

# ============================================================
# 12. ERROR-DETECTION ANALYSIS
# ============================================================

# Label incorrect predictions as 1 and correct predictions as 0
mc_error_labels = (
    mc_predictions != test_true_labels
).astype(int)

# Higher variance = greater uncertainty
mc_error_detection_auroc = roc_auc_score(
    mc_error_labels,
    mc_variances
)

# ============================================================
# 13. DISPLAY MONTE CARLO DROPOUT RESULTS
# ============================================================

# Display discrimination and classification performance metrics
print("\nMC Dropout Results")

print(f"MC Accuracy: {mc_accuracy:.4f}")
print(f"MC AUROC: {mc_auroc:.4f}")
print(f"MC AUPRC: {mc_auprc:.4f}")
print(f"MC Sensitivity: {mc_sensitivity:.4f}")
print(f"MC Specificity: {mc_specificity:.4f}")
print(f"MC F1-Score: {mc_f1:.4f}")
print(f"MC Balanced Accuracy: {mc_balanced_accuracy:.4f}")
print(f"MC G-Mean: {mc_g_mean:.4f}")

print("\nMC Dropout Classification Report:")
print(
    classification_report(
        test_true_labels,
        mc_predictions,
        target_names=["GON-", "GON+"]
    )
)

print("\nMC Dropout Confusion Matrix:")
print(mc_confusion_matrix)

print("\nUncertainty Summary:")
print(f"Mean Variance: {mc_variances.mean():.6f}")
print(f"Maximum Variance: {mc_variances.max():.6f}")
print(f"Mean Standard Deviation: {mc_standard_deviations.mean():.6f}")

print("\nMean Probability for each test image:")
print(mc_mean_probabilities)

overall_mean_probability = np.mean(mc_mean_probabilities)
print(f"\nOverall Mean Probability: {overall_mean_probability:.4f}")

print("\nCalibration and Error-Detection Results")
print(f"Brier Score: {mc_brier_score:.6f}")
print(f"Expected Calibration Error (ECE): {mc_ece:.6f}")
print(f"Negative Log-Likelihood (NLL): {mc_nll:.6f}")
print(f"Error-Detection AUROC: {mc_error_detection_auroc:.4f}")

# ============================================================
# 14. SAVE IMAGE-LEVEL MONTE CARLO DROPOUT RESULTS
# ============================================================

# Create a DataFrame containing predictions and uncertainty estimates for each test image
mc_results_df = test_df.reset_index(drop=True).copy()

mc_results_df["True Label"] = test_true_labels
mc_results_df["MC Mean Probability"] = mc_mean_probabilities
mc_results_df["MC Final Prediction"] = mc_predictions
mc_results_df["MC Variance"] = mc_variances
mc_results_df["MC Standard Deviation"] = mc_standard_deviations

# Show whether each prediction was correct
mc_results_df["Correct Prediction"] = (
    mc_results_df["True Label"]
    == mc_results_df["MC Final Prediction"]
)

(PROJECT_ROOT / "results" / "mc_dropout").mkdir(parents=True, exist_ok=True)
mc_results_df.to_csv(
    PROJECT_ROOT / "results" / "mc_dropout" / "densenet121_mc_dropout_results.csv",
    index=False
)

print(
    "\nSaved image-level MC Dropout results to "
    "'densenet121_mc_dropout_results.csv'."
)