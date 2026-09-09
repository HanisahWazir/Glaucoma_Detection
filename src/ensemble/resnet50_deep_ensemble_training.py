# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# ============================================================
# 2. LOAD THE DATASETS
# ============================================================

train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train_split.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "validation_split.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test_split.csv")

# ============================================================
# 3. PREPARE IMAGE PATHS
# ============================================================

# Make sure image paths are strings
train_df["Image Path"] = train_df["Image Path"].astype(str)
val_df["Image Path"] = val_df["Image Path"].astype(str)
test_df["Image Path"] = test_df["Image Path"].astype(str)

# ============================================================
# 4. DISPLAY DATASET SPLIT SUMMARY
# ============================================================

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
# 5. DEFINE TRAINING PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

ensemble_seeds = [1234, 2345, 3456, 4567, 5678]

# ============================================================
# 6. CREATE THE VALIDATION DATA GENERATOR
# ============================================================

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="Image Path",
    y_col="Label Encode",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# ============================================================
# 7. TRAIN THE DEEP ENSEMBLE MEMBERS
# ============================================================

for member_number, seed in enumerate(ensemble_seeds, start=1):

    print("\n" + "=" * 60)
    print(f"Training Deep Ensemble Member {member_number}/{len(ensemble_seeds)}")
    print(f"Random Seed: {seed}")
    print("=" * 60)

    # Clear previous TensorFlow state
    tf.keras.backend.clear_session()

    # Set this model's random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Create training data generator
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="Image Path",
        y_col="Label Encode",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="raw",
        shuffle=True,
        seed=seed
    )

    # Build a new ResNet-50 model
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained backbone
    base_model.trainable = False

    # Build binary classifier
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    # ========================================================
    # 8. COMPILE THE MODEL
    # ========================================================

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print(f"\nModel Summary for Ensemble Member {member_number}:")
    model.summary()

    # ========================================================
    # 9. TRAIN THE ENSEMBLE MEMBER
    # ========================================================

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # ========================================================
    # 10. SAVE THE TRAINED ENSEMBLE MEMBER
    # ========================================================

    model_filename = f"resnet50_ensemble_member_{member_number}.keras"

    (PROJECT_ROOT / "models" / "ensemble").mkdir(parents=True, exist_ok=True)

    model.save(PROJECT_ROOT / "models" / "ensemble" / model_filename)

    print(f"\nSaved Ensemble Member {member_number} as '{model_filename}'.")

# ============================================================
# 11. COMPLETE DEEP ENSEMBLE TRAINING
# ============================================================

print("\nFinished training all Deep Ensemble members!")