# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score, brier_score_loss, log_loss

from pathlib import Path

# Define the project root so the script works regardless of where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 2. LOAD AND EXPLORE THE DATASET
# ============================================================

# Load the dataset labels
df = pd.read_csv(PROJECT_ROOT / "data" / "labels.csv")

# Explore the dataset structure and class distribution
df.info()
print(df.head())
print(df["Label"].value_counts())
print(df["Patient"].nunique())

# ============================================================
# 3. LOCATE AND VALIDATE IMAGE FILES
# ============================================================

# Define the directory containing the fundus images
image_location = PROJECT_ROOT / "data" / "images"
df["Image Path"] = df["Image Name"].apply(lambda filename: image_location / filename)
print(df[["Image Name", "Image Path"]].head())

# Verify that every image listed in the dataset exists
df["Image Exists"] = df["Image Path"].apply(lambda path: path.exists())
print(df["Image Exists"].value_counts())

# ============================================================
# 4. VALIDATE AND ENCODE DIAGNOSIS LABELS
# ============================================================

# Convert glaucoma diagnosis labels into binary values
# GON- = 0 and GON+ = 1
print(df["Label"].unique())
df["Label Encode"] = df["Label"].map({"GON-":0, "GON+":1})
print(df["Label Encode"].value_counts())
print(df[["Label", "Label Encode"]].head())

# Check whether every patient has one consistent label
labels_per_patient = df.groupby("Patient")["Label Encode"].nunique()

if labels_per_patient.max() == 1:
    print("\nAll patients have consistent labels.")
else:
    print("\nWarning: some patients have more than one label.")

# Create one row per patient so each patient is assigned to only one data partition
patient_df = df[["Patient", "Label Encode"]].drop_duplicates(subset="Patient")

# Split patients into 70% training and 30% temporary data
# Patient-level splitting prevents data leakage between partitions
train_patients, temp_patients = train_test_split(
    patient_df,
    test_size = 0.30,
    random_state = 1234,
    stratify = patient_df["Label Encode"]
)

# Split the temporary 30% of patients into validation (10% overall) and test (20% overall)
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size = 2 / 3,
    random_state = 1234,
    stratify = temp_patients["Label Encode"] # Preserve class proportions across the data partitions
)

# Retrieve all image rows for each patient split
train_df = df[df["Patient"].isin(train_patients["Patient"])].copy()
val_df = df[df["Patient"].isin(val_patients["Patient"])].copy()
test_df = df[df["Patient"].isin(test_patients["Patient"])].copy()

# Convert Path objects to strings for TensorFlow data generators
train_df["Image Path"] = train_df["Image Path"].astype(str)
val_df["Image Path"] = val_df["Image Path"].astype(str)
test_df["Image Path"] = test_df["Image Path"].astype(str)

# Verify that no patient appears in more than one data partition
train_ids = set(train_df["Patient"])
val_ids = set(val_df["Patient"])
test_ids = set(test_df["Patient"])

print("\nPatient overlap checks:")
print("Train-validation:", train_ids.intersection(val_ids))
print("Train-test:", train_ids.intersection(test_ids))
print("Validation-test:", val_ids.intersection(test_ids))

# Display split summary
print("\nTraining:")
print("Patients:", train_df["Patient"].nunique())
print("Images:", len(train_df))
print(train_df["Label Encode"].value_counts())

print("\nValidation:")
print("Patients:", val_df["Patient"].nunique())
print("Images:", len(val_df))
print(val_df["Label Encode"].value_counts())

print("\nTesting:")
print("Patients:", test_df["Patient"].nunique())
print("Images:", len(test_df))
print(test_df["Label Encode"].value_counts())

# ============================================================
# 5. SUMMARISE IMAGE QUALITY
# ============================================================

# Display the mean and standard deviation of image quality scores
print("\nOverall Quality Score:")
print(f"{df['Quality Score'].mean():.2f} ± {df['Quality Score'].std():.2f}")

print("\nTraining:")
print(f"{train_df['Quality Score'].mean():.2f} ± {train_df['Quality Score'].std():.2f}")

print("\nValidation:")
print(f"{val_df['Quality Score'].mean():.2f} ± {val_df['Quality Score'].std():.2f}")

print("\nTesting:")
print(f"{test_df['Quality Score'].mean():.2f} ± {test_df['Quality Score'].std():.2f}")

# ============================================================
# 6. DEFINE TRAINING PARAMETERS AND RANDOM SEEDS
# ============================================================

# Define image dimensions, batch size, training duration and random seed
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# 7. CREATE IMAGE DATA GENERATORS
# ============================================================

# EfficientNet-B4 includes preprocessing within the model architecture
train_datagen = ImageDataGenerator()
val_test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="Image Path",
    y_col="Label Encode",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE ,
    class_mode="raw",   # Label Encode is already encoded as 0 and 1
    shuffle=True,
    seed = SEED
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="Image Path",
    y_col="Label Encode",
    target_size=IMAGE_SIZE,
    batch_size= BATCH_SIZE ,
    class_mode="raw",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="Image Path",
    y_col="Label Encode",
    target_size=IMAGE_SIZE,
    batch_size= BATCH_SIZE ,
    class_mode="raw",
    shuffle=False
)

# ============================================================
# 8. BUILD THE EFFICIENTNET-B4 BASELINE MODEL
# ============================================================

# Load the ImageNet-pretrained EfficientNet-B4 convolutional backbone
base_model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze the pretrained EfficientNet-B4 layers
base_model.trainable = False

# Add a classification head for binary glaucoma prediction
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer= "adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ============================================================
# 9. TRAIN THE BASELINE MODEL
# ============================================================

# Train the model using the training set and monitor performance on the validation set
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the trained model for reproducibility and later evaluation
(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
model.save(PROJECT_ROOT / "models" / "efficientnetb4_baseline.keras")

print(
    "\nSaved the trained model as "
    "'efficientnetb4_baseline.keras'."
)

# ============================================================
# 10. EVALUATE MODEL PERFORMANCE ON THE TEST SET
# ============================================================

# Evaluate the trained model on the held-out test set
test_loss, test_accuracy, keras_test_auc = model.evaluate(test_generator)

# Generate predicted probabilities and calculate detailed test-set performance metrics using scikit-learn
test_predictions_prob = model.predict(test_generator).flatten()
test_predictions = (test_predictions_prob >= 0.5).astype(int)
test_true_labels = test_generator.labels.astype(int)

# Calculate final test AUROC using scikit-learn
test_auroc = roc_auc_score(
   test_true_labels,
   test_predictions_prob
)

# Calculate AUPRC using predicted probabilities
test_auprc = average_precision_score(
    test_true_labels,
    test_predictions_prob
)

# Create the confusion matrix
test_confusion_matrix = confusion_matrix(
    test_true_labels,
    test_predictions,
    labels=[0, 1]
)

# Extract TN, FP, FN and TP
tn, fp, fn, tp = test_confusion_matrix.ravel()

# Calculate sensitivity
test_sensitivity = tp / (tp + fn)

# Calculate specificity
test_specificity = tn / (tn + fp)

# Calculate F1-score
test_f1 = f1_score(
    test_true_labels,
    test_predictions
)

# Calculate balanced accuracy
test_balanced_accuracy = (test_sensitivity + test_specificity) / 2

# Calculate G-Mean
test_g_mean = np.sqrt(test_sensitivity * test_specificity)

# ============================================================
# 11. CALIBRATION ANALYSIS
# ============================================================

# Calculate Brier Score
brier_score = brier_score_loss(
    test_true_labels,
    test_predictions_prob
)

# Calculate Expected Calibration Error (ECE)
N_BINS = 10

bin_edges = np.linspace(
    0.0,
    1.0,
    N_BINS + 1
)

ece = 0.0

for i in range(N_BINS):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    # Find predictions belonging to this bin
    if i == 0:
        in_bin = (
            (test_predictions_prob >= lower)
            &
            (test_predictions_prob <= upper)
        )
    else:
        in_bin = (
            (test_predictions_prob > lower)
            &
            (test_predictions_prob <= upper)
        )

    # Skip empty bins
    if np.sum(in_bin) == 0:
        continue

    # Average predicted probability in this bin
    bin_confidence = np.mean(
        test_predictions_prob[in_bin]
    )

    # Actual proportion of GON+ cases in this bin
    bin_observed_frequency = np.mean(
        test_true_labels[in_bin]
    )

    # Proportion of test samples in this bin
    bin_weight = (
        np.sum(in_bin)
        / len(test_true_labels)
    )

    # Add this bin's contribution to ECE
    ece += (
        bin_weight
        *
        abs(
            bin_observed_frequency
            - bin_confidence
        )
    )

# Calculate Negative Log-Likelihood (NLL)
nll = log_loss(
    test_true_labels,
    test_predictions_prob
)

# ============================================================
# 12. DISPLAY FINAL TEST-SET RESULTS
# ============================================================

# Display discrimination, classification and calibration performance metrics
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Test Sensitivity: {test_sensitivity:.4f}")
print(f"Test Specificity: {test_specificity:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(
    f"Test Balanced Accuracy: "
    f"{test_balanced_accuracy:.4f}"
)
print(f"Test G-Mean: {test_g_mean:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        test_true_labels,
        test_predictions,
        target_names=["GON-", "GON+"]
    )
)

print("\nConfusion Matrix:")
print(test_confusion_matrix)

print("\nCalibration and Error-Detection Results")

print(f"Brier Score: {brier_score:.6f}")
print(f"Expected Calibration Error (ECE): {ece:.6f}")
print(f"Negative Log-Likelihood (NLL): {nll:.6f}")

# ============================================================
# 13. SAVE IMAGE-LEVEL PREDICTIONS
# ============================================================

# Save predictions and probabilities for each test image to support further analysis and error investigation
baseline_results_df = test_df.reset_index(drop=True).copy()

baseline_results_df["True Label"] = test_true_labels

baseline_results_df["Baseline Probability"] = (
    test_predictions_prob.flatten()
)

baseline_results_df["Baseline Prediction"] = (
    test_predictions.flatten()
)

baseline_results_df["Correct Prediction"] = (
    baseline_results_df["True Label"]
    == baseline_results_df["Baseline Prediction"]
)

(PROJECT_ROOT / "results" / "baseline").mkdir(parents=True, exist_ok=True)
baseline_results_df.to_csv(
    PROJECT_ROOT / "results" / "baseline" / "efficientnetb4_baseline_results.csv",
    index=False
)

print(
    "\nSaved EfficientNet-B4 baseline results to "
    "'efficientnetb4_baseline_results.csv'."
)