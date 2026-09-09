# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ============================================================
# 2. DEFINE PROJECT PATHS AND PARAMETERS
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "conv5_block3_out"

# Path to the selected TP, TN, FP and FN examples
SELECTED_EXAMPLES_PATH = PROJECT_ROOT / "results" / "grad_cam" / "gradcam_tp_tn_fp_fn_examples.csv"

# Directory for saving Grad-CAM figures
OUTPUT_DIR = PROJECT_ROOT / "figures" / "gradcam" / "tp_tn_fp_fn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. DEFINE DEEP ENSEMBLE MODEL FILES
# ============================================================

# Paths to all five ResNet-50 ensemble members
ensemble_model_files = [
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_1.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_2.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_3.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_4.keras",
    PROJECT_ROOT / "models" / "ensemble" / "resnet50_ensemble_member_5.keras",
]

# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

for model_file in ensemble_model_files:
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

if not SELECTED_EXAMPLES_PATH.exists():
    raise FileNotFoundError(f"Selected examples CSV not found: {SELECTED_EXAMPLES_PATH}")

# ============================================================
# 5. LOAD SELECTED TP, TN, FP AND FN EXAMPLES
# ============================================================

selected_df = pd.read_csv(SELECTED_EXAMPLES_PATH)

print("\nLoaded TP/TN/FP/FN Grad-CAM examples:")

# ============================================================
# 6. PROCESS EACH SELECTED EXAMPLE
# ============================================================

for index, row in selected_df.iterrows():

    print("\n" + "=" * 70)
    print("Processing:", row["Example Type"])
    print("=" * 70)

    # Extract image and prediction information
    example_type = str(row["Example Type"])
    true_label = int(row["True Label"])
    ensemble_mean_probability = float(row["Ensemble Mean Probability"])
    ensemble_prediction = int(row["Ensemble Final Prediction"])

    # Extract the filename and construct the correct project image path
    image_filename = Path(row["Image Path"]).name
    image_path = PROJECT_ROOT / "data" / "images" / image_filename

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Image:", image_filename)
    print("True label:", true_label)
    print("Stored Ensemble P(GON+):", ensemble_mean_probability)
    print("Deep Ensemble prediction:", ensemble_prediction)

    # ============================================================
    # 7. LOAD AND PREPROCESS THE IMAGE
    # ============================================================

    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    resized_image = original_image.resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)
    image_array = np.array(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    # Apply ResNet-50 preprocessing
    image_batch = preprocess_input(image_batch)

    # ============================================================
    # 8. GENERATE GRAD-CAM FOR ALL ENSEMBLE MEMBERS
    # ============================================================

    all_heatmaps = []
    member_probabilities = []

    for member_number, model_file in enumerate(ensemble_model_files, start=1):

        print(f"\nRunning member {member_number}/5...")

        # Load the ensemble member and extract its ResNet-50 backbone
        model = tf.keras.models.load_model(model_file)
        base_model = model.layers[0]
        last_conv_layer = base_model.get_layer(LAST_CONV_LAYER_NAME)

        # Create a model that provides convolutional features and backbone output
        feature_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        final_dense_layer = model.layers[-1]

        # ============================================================
        # 9. PERFORM FORWARD PASS AND CALCULATE GRADIENTS
        # ============================================================

        with tf.GradientTape() as tape:

            conv_outputs, backbone_output = feature_model(image_batch, training=False)

            # Pass through the classifier head except the final sigmoid layer
            x = backbone_output

            for layer in model.layers[1:-1]:
                x = layer(x, training=False)

            # Calculate the output before applying the sigmoid activation
            logits = tf.matmul(x, final_dense_layer.kernel)

            if final_dense_layer.use_bias:
                logits = tf.nn.bias_add(logits, final_dense_layer.bias)

            # Calculate probability for display purposes
            p_gon_positive = tf.nn.sigmoid(logits)

            # Use the ensemble prediction as the Grad-CAM target
            if ensemble_prediction == 1:
                target_output = logits[:, 0]
                target_name = "GON+"
            else:
                target_output = -logits[:, 0]
                target_name = "GON-"

        # ============================================================
        # 10. CALCULATE THE GRAD-CAM HEATMAP
        # ============================================================

        grads = tape.gradient(target_output, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None.")

        # Calculate importance weights for the convolutional feature maps
        importance_weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        weighted_feature_maps = conv_outputs * importance_weights
        heatmap = tf.reduce_sum(weighted_feature_maps, axis=-1)

        # Apply ReLU and normalise the heatmap
        heatmap = tf.nn.relu(heatmap)
        max_value = tf.reduce_max(heatmap)

        if float(max_value.numpy()) > 0:
            heatmap = heatmap / max_value

        heatmap = heatmap.numpy()
        probability = float(p_gon_positive.numpy()[0, 0])

        all_heatmaps.append(heatmap)
        member_probabilities.append(probability)

        print(f"Member {member_number} P(GON+) = {probability:.6f}")

        # Clear memory before loading the next ensemble member
        del model
        tf.keras.backend.clear_session()

    # ============================================================
    # 11. CALCULATE THE DEEP ENSEMBLE RESULTS
    # ============================================================

    all_heatmaps = np.array(all_heatmaps)
    member_probabilities = np.array(member_probabilities)

    # Calculate the mean probability and average Grad-CAM heatmap
    calculated_ensemble_probability = np.mean(member_probabilities)

    print("\nCalculated Ensemble P(GON+):", calculated_ensemble_probability)

    ensemble_heatmap = np.mean(all_heatmaps, axis=0)
    ensemble_heatmap = np.maximum(ensemble_heatmap, 0)

    # Normalise the averaged heatmap
    max_value = np.max(ensemble_heatmap)

    if max_value > 0: 
        ensemble_heatmap = ensemble_heatmap / max_value

    # ============================================================
    # 12. RESIZE GRAD-CAM HEATMAPS
    # ============================================================

    resized_member_heatmaps = []

    # Resize each individual member heatmap to the original image size
    for heatmap in all_heatmaps:

        heatmap = np.expand_dims(heatmap, axis=-1)

        resized = tf.image.resize(
            heatmap,
            (original_array.shape[0], original_array.shape[1])
        )

        heatmap_resized = resized.numpy().squeeze()
        resized_member_heatmaps.append(heatmap_resized)

    resized_member_heatmaps = np.array(resized_member_heatmaps)

    # Resize the averaged ensemble heatmap
    ensemble_heatmap = np.expand_dims(ensemble_heatmap, axis=-1)

    resized = tf.image.resize(
        ensemble_heatmap,
        (original_array.shape[0], original_array.shape[1])
    )

    ensemble_heatmap_resized = resized.numpy().squeeze()

    # ============================================================
    # 13. CREATE THE GRAD-CAM OVERLAY
    # ============================================================

    # Apply a colour map to the averaged heatmap
    colour_map = plt.get_cmap("jet")
    coloured_heatmap = colour_map(ensemble_heatmap_resized)
    coloured_heatmap = coloured_heatmap[:, :, :3]

    original_normalised = original_array.astype(np.float32) / 255.0

    # Combine the original image and Grad-CAM heatmap
    alpha = 0.4
    ensemble_overlay = (1 - alpha) * original_normalised + alpha * coloured_heatmap

    safe_example_name = example_type.lower().replace(" ", "_")

    # ============================================================
    # 14. CREATE FIGURE FOR ALL ENSEMBLE MEMBERS
    # ============================================================

    # Display all five member heatmaps and the averaged heatmap
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(5):

        axes[i].imshow(
            resized_member_heatmaps[i],
            cmap="jet",
            vmin=0,
            vmax=1
        )

        axes[i].set_title(f"Member {i + 1}\nP(GON+) = {member_probabilities[i]:.4f}")
        axes[i].axis("off")

    axes[5].imshow(
        ensemble_heatmap_resized,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[5].set_title(
        "Average of 5 Grad-CAMs\n"
        f"Ensemble P(GON+) = {calculated_ensemble_probability:.4f}"
    )

    axes[5].axis("off")

    plt.suptitle(f"{example_type} ({image_filename})", fontsize=16)
    plt.tight_layout()

    comparison_path = OUTPUT_DIR / f"{safe_example_name}_all_members_logits.png"

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # 15. CREATE ORIGINAL IMAGE, HEATMAP AND OVERLAY FIGURE
    # ============================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_array)
    axes[0].set_title("Original Fundus Image")
    axes[0].axis("off")

    # Aggregated Deep Ensemble Grad-CAM heatmap
    axes[1].imshow(
        ensemble_heatmap_resized,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(f"Aggregated Deep Ensemble Grad-CAM\nTarget: {target_name}")
    axes[1].axis("off")

    # Grad-CAM overlay
    axes[2].imshow(ensemble_overlay)

    axes[2].set_title(
        "Grad-CAM Overlay\n"
        f"True = {true_label}, Prediction = {ensemble_prediction}, "
        f"P(GON+) = {calculated_ensemble_probability:.4f}"
    )

    axes[2].axis("off")

    plt.suptitle(f"{example_type} ({image_filename})", fontsize=16)
    plt.tight_layout()

    final_path = OUTPUT_DIR / f"{safe_example_name}_gradcam_logits.png"

    plt.savefig(
        final_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved member comparison:", comparison_path)
    print("Saved final Grad-CAM:", final_path)

# ============================================================
# 16. COMPLETE THE GRAD-CAM ANALYSIS
# ============================================================

print("\nFinished TP/TN/FP/FN Deep Ensemble Grad-CAM analysis!")