# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

# Project root so the script works no matter where it is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory containing the individual Grad-CAM example figures
INPUT_DIR = PROJECT_ROOT / "figures" / "gradcam" / "tp_tn_fp_fn"

# Directory where Figure 8 will be saved
OUTPUT_DIR = PROJECT_ROOT / "figures" / "gradcam" / "figure8"

# Create the output directory if it does not already exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. DEFINE INPUT IMAGE FILES
# ============================================================

# Define the paths to the TP, TN, FP and FN Grad-CAM examples
tp_path = INPUT_DIR / "true_positive_gradcam_logits.png"
tn_path = INPUT_DIR / "true_negative_gradcam_logits.png"
fp_path = INPUT_DIR / "false_positive_gradcam_logits.png"
fn_path = INPUT_DIR / "false_negative_gradcam_logits.png"

# ============================================================
# 4. CHECK THAT ALL IMAGE FILES EXIST
# ============================================================

image_paths = [tp_path, tn_path, fp_path, fn_path]

for image_path in image_paths:
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find: {image_path}")

# ============================================================
# 5. LOAD THE GRAD-CAM IMAGES
# ============================================================

tp_image = Image.open(tp_path).convert("RGB")
tn_image = Image.open(tn_path).convert("RGB")
fp_image = Image.open(fp_path).convert("RGB")
fn_image = Image.open(fn_path).convert("RGB")

# ============================================================
# 6. CREATE THE FIGURE LAYOUT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ============================================================
# 7. DISPLAY THE TRUE POSITIVE EXAMPLE
# ============================================================

axes[0, 0].imshow(tp_image)
axes[0, 0].set_title("(a) True Positive\nP(GON+) = 1.0000", fontsize=14)
axes[0, 0].axis("off")

# ============================================================
# 8. DISPLAY THE TRUE NEGATIVE EXAMPLE
# ============================================================

axes[0, 1].imshow(tn_image)
axes[0, 1].set_title("(b) True Negative\nP(GON+) ≈ 0.0000", fontsize=14)
axes[0, 1].axis("off")

# ============================================================
# 9. DISPLAY THE FALSE POSITIVE EXAMPLE
# ============================================================

axes[1, 0].imshow(fp_image)
axes[1, 0].set_title("(c) False Positive\nP(GON+) = 0.9928", fontsize=14)
axes[1, 0].axis("off")

# ============================================================
# 10. DISPLAY THE FALSE NEGATIVE EXAMPLE
# ============================================================

axes[1, 1].imshow(fn_image)
axes[1, 1].set_title("(d) False Negative\nP(GON+) = 0.1930", fontsize=14)
axes[1, 1].axis("off")

# ============================================================
# 11. ADD THE MAIN FIGURE TITLE
# ============================================================

fig.suptitle("Figure 8. Qualitative Examples and Explainability Review", fontsize=18)

# ============================================================
# 12. ADJUST THE FIGURE LAYOUT
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ============================================================
# 13. SAVE AND DISPLAY FIGURE 8
# ============================================================

output_path = OUTPUT_DIR / "figure8_gradcam_tp_tn_fp_fn.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved Figure 8 to:")
print(output_path)