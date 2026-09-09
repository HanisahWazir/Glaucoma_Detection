# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ============================================================
# 2. DEFINE PROJECT PATHS AND SETTINGS
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the input image size required by ResNet-50
IMAGE_SIZE = (224, 224)

# Define the image to compare
IMAGE_FILENAME = "206_0.jpg"

# Define the path to the saved Deep Ensemble results
RESULTS_PATH = PROJECT_ROOT / "results" / "ensemble" / "resnet50_deep_ensemble_results.csv"

# Define the path to the selected image
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / IMAGE_FILENAME

# ============================================================
# 3. DEFINE DEEP ENSEMBLE MODEL FILES
# ============================================================

# Store the paths of all five ResNet-50 ensemble members
ensemble_model_files = [
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_1.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_2.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_3.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_4.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_5.keras",
]

# ============================================================
# 4. CHECK SAVED CSV VALUES
# ============================================================

# Load the previously saved Deep Ensemble results
results_df = pd.read_csv(RESULTS_PATH)

# Extract the filename from each stored image path
results_df["Image Filename"] = results_df["Image Path"].apply(lambda x: Path(str(x)).name)

# Find the results corresponding to the selected image
row = results_df[results_df["Image Filename"] == IMAGE_FILENAME]

# Ensure that exactly one result exists for the selected image
if len(row) != 1:
    raise ValueError(f"Expected exactly one row for {IMAGE_FILENAME}, but found {len(row)}.")

row = row.iloc[0]

# ============================================================
# 5. DISPLAY SAVED CSV RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVED CSV RESULTS")
print("=" * 60)

print("Image:", IMAGE_FILENAME)
print("Stored Ensemble Mean Probability:", row["Ensemble Mean Probability"])

# ============================================================
# 6. EXTRACT STORED ENSEMBLE MEMBER PROBABILITIES
# ============================================================

# Store the individual probabilities previously saved for each ensemble member
stored_member_probabilities = []

for i in range(1, 6):
    column_name = f"Ensemble Member {i} Probability"
    probability = float(row[column_name])

    stored_member_probabilities.append(probability)

    print(f"Stored Member {i}: {probability:.8f}")

# Calculate the mean probability from the stored member predictions
stored_member_probabilities = np.array(stored_member_probabilities)
stored_mean = np.mean(stored_member_probabilities)

print("\nMean calculated from stored member probabilities:", f"{stored_mean:.8f}")

# ============================================================
# 7. LOAD AND PREPROCESS THE IMAGE
# ============================================================

# Check that the selected image exists
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Load and resize the image using the same preprocessing approach as testing
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)

# Convert the image into a NumPy array and add a batch dimension
image_array = np.array(image, dtype=np.float32)
image_batch = np.expand_dims(image_array, axis=0)

# Apply ResNet-50 preprocessing
image_batch = preprocess_input(image_batch)

# ============================================================
# 8. GENERATE FRESH PREDICTIONS FROM SAVED MODELS
# ============================================================

# Store newly generated probabilities from each ensemble member
fresh_member_probabilities = []

print("\n" + "=" * 60)
print("FRESH MODEL PREDICTIONS")
print("=" * 60)

# Generate a fresh prediction using each saved ensemble model
for member_number, model_file in enumerate(ensemble_model_files, start=1):

    model = tf.keras.models.load_model(model_file)

    prediction = model(image_batch, training=False)
    probability = float(prediction.numpy().squeeze())

    fresh_member_probabilities.append(probability)

    print(f"Fresh Member {member_number}: {probability:.8f}")

    # Clear the model from memory before loading the next member
    del model
    tf.keras.backend.clear_session()

# ============================================================
# 9. CALCULATE THE FRESH ENSEMBLE MEAN
# ============================================================

# Calculate the mean probability from the newly generated predictions
fresh_member_probabilities = np.array(fresh_member_probabilities)
fresh_mean = np.mean(fresh_member_probabilities)

print("\nFresh Ensemble Mean Probability:", f"{fresh_mean:.8f}")

# ============================================================
# 10. COMPARE STORED AND FRESH PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# Compare the stored and freshly generated probabilities for each member
for i in range(5):

    difference = fresh_member_probabilities[i] - stored_member_probabilities[i]

    print(
        f"Member {i + 1}: "
        f"stored = {stored_member_probabilities[i]:.8f}, "
        f"fresh = {fresh_member_probabilities[i]:.8f}, "
        f"difference = {difference:.8f}"
    )

# Display the difference between the stored and fresh ensemble means
print("\nStored ensemble mean:", f"{stored_mean:.8f}")
print("Fresh ensemble mean:", f"{fresh_mean:.8f}")
print("Difference:", f"{fresh_mean - stored_mean:.8f}")