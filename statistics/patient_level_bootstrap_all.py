# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score
)


# ============================================================
# 2. SETTINGS
# ============================================================

# Number of bootstrap resamples.
# Here, the patient-level resampling is repeated 2,000 times.
N_BOOTSTRAPS = 2000

# Random seed used to make the bootstrap results reproducible.
RANDOM_SEED = 42


# ============================================================
# 3. DEFINE MODEL RESULTS
# ============================================================

# Each model uses an existing CSV containing its test-set
# predictions and probabilities.
#
# "probability" = predicted probability used for AUROC/AUPRC
# "prediction" = final 0/1 prediction used for classification metrics

RESULTS = {

    # --------------------------------------------------------
    # Baseline models
    # --------------------------------------------------------

    "ResNet-50": {
        "file": "results/baseline/resnet50_baseline_results.csv",
        "probability": "Baseline Probability",
        "prediction": "Baseline Prediction"
    },

    "DenseNet-121": {
        "file": "results/baseline/densenet121_baseline_results.csv",
        "probability": "Baseline Probability",
        "prediction": "Baseline Prediction"
    },

    "EfficientNet-B4": {
        "file": "results/baseline/efficientnetb4_baseline_results.csv",
        "probability": "Baseline Probability",
        "prediction": "Baseline Prediction"
    },


    # --------------------------------------------------------
    # MC Dropout models
    # --------------------------------------------------------

    "ResNet-50 + MC Dropout": {
        "file": "results/mc_dropout/resnet50_mc_dropout_results.csv",
        "probability": "MC Mean Probability",
        "prediction": "MC Final Prediction"
    },

    "DenseNet-121 + MC Dropout": {
        "file": "results/mc_dropout/densenet121_mc_dropout_results.csv",
        "probability": "MC Mean Probability",
        "prediction": "MC Final Prediction"
    },

    "EfficientNet-B4 + MC Dropout": {
        "file": "results/mc_dropout/efficientnetb4_mc_dropout_results.csv",
        "probability": "MC Mean Probability",
        "prediction": "MC Final Prediction"
    },


    # --------------------------------------------------------
    # Deep Ensemble
    # --------------------------------------------------------

    "ResNet-50 + Deep Ensemble": {
        "file": "results/ensemble/resnet50_deep_ensemble_results.csv",
        "probability": "Ensemble Mean Probability",
        "prediction": "Ensemble Final Prediction"
    }
}


# ============================================================
# 4. DEFINE PROJECT ROOT
# ============================================================

# The script is located inside:
#
# glaucoma_detection/
# └── statistics/
#     └── patient_level_bootstrap_all.py
#
# parents[1] therefore points to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 5. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(y_true, y_prob, y_pred):

    # Calculate the proportion of correctly classified samples.
    accuracy = accuracy_score(y_true, y_pred)

    # AUROC measures how well predicted probabilities
    # distinguish glaucoma from non-glaucoma.
    auroc = roc_auc_score(y_true, y_prob)

    # AUPRC measures performance across the
    # precision-recall trade-off.
    auprc = average_precision_score(y_true, y_prob)

    # Create the confusion matrix.
    #
    # TN = True Negative
    # FP = False Positive
    # FN = False Negative
    # TP = True Positive
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Sensitivity = proportion of actual glaucoma cases
    # correctly identified by the model.
    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    # Specificity = proportion of actual non-glaucoma cases
    # correctly identified by the model.
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    # F1-score combines precision and recall.
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1
    }


# ============================================================
# 6. RUN PATIENT-LEVEL BOOTSTRAP
# ============================================================

def run_bootstrap(model_name, config):

    # Build the full path to the model's results CSV.
    file_path = PROJECT_ROOT / config["file"]

    print("\n" + "=" * 75)
    print(model_name)
    print("=" * 75)

    print("Loading:")
    print(file_path)

    # Load the existing test-set predictions.
    # No model retraining is performed here.
    df = pd.read_csv(file_path)


    # --------------------------------------------------------
    # 6.1 CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    # These columns are required to perform the bootstrap.
    required_columns = [
        "Patient",
        "True Label",
        config["probability"],
        config["prediction"]
    ]

    # Identify any required columns that are missing.
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {model_name}: "
            f"{missing_columns}"
        )

    print(f"Images: {len(df)}")
    print(f"Patients: {df['Patient'].nunique()}")


    # --------------------------------------------------------
    # 6.2 REMOVE MISSING VALUES
    # --------------------------------------------------------

    # Remove rows where any value required for the
    # bootstrap analysis is missing.
    df = df.dropna(
        subset=[
            "Patient",
            "True Label",
            config["probability"],
            config["prediction"]
        ]
    ).copy()


    # --------------------------------------------------------
    # 6.3 CHECK PATIENT LABEL CONSISTENCY
    # --------------------------------------------------------

    # Each patient should have only one true diagnosis label.
    # This is important because patients are the unit being
    # resampled during the bootstrap.
    labels_per_patient = (
        df.groupby("Patient")["True Label"]
        .nunique()
    )

    if labels_per_patient.max() > 1:
        raise ValueError(
            f"{model_name}: at least one patient has "
            f"multiple True Label values."
        )

    patient_ids = df["Patient"].unique()

    print(
        f"Patients available for bootstrap: "
        f"{len(patient_ids)}"
    )


    # ========================================================
    # 7. CALCULATE ORIGINAL TEST-SET PERFORMANCE
    # ========================================================

    # Calculate the original performance before any
    # bootstrap resampling.
    #
    # These are the original point estimates, such as:
    # AUROC = 0.9858
    original_metrics = calculate_metrics(
        df["True Label"].astype(int),
        df[config["probability"]].astype(float),
        df[config["prediction"]].astype(int)
    )

    print("\nOriginal test-set performance:")

    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")


    # ========================================================
    # 8. PERFORM PATIENT-LEVEL BOOTSTRAP
    # ========================================================

    # Create a reproducible random-number generator.
    rng = np.random.default_rng(RANDOM_SEED)

    # Store the metrics from every successful
    # bootstrap sample.
    bootstrap_results = []


    # --------------------------------------------------------
    # 8.1 REPEAT BOOTSTRAP 2,000 TIMES
    # --------------------------------------------------------

    for i in range(N_BOOTSTRAPS):

        # Sample patients WITH replacement.
        #
        # This means a patient can be selected multiple times,
        # while another patient may not be selected at all.
        sampled_patients = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True
        )


        # ----------------------------------------------------
        # 8.2 KEEP ALL IMAGES FOR EACH SELECTED PATIENT
        # ----------------------------------------------------

        # Every image belonging to a selected patient is kept.
        #
        # This is important because your dataset contains
        # multiple images from some patients. The bootstrap
        # therefore resamples patients rather than individual
        # images.
        sampled_data = pd.concat(
            [
                df[df["Patient"] == patient]
                for patient in sampled_patients
            ],
            ignore_index=True
        )


        # ----------------------------------------------------
        # 8.3 EXTRACT LABELS AND MODEL PREDICTIONS
        # ----------------------------------------------------

        y_true = sampled_data["True Label"].astype(int)

        y_prob = sampled_data[
            config["probability"]
        ].astype(float)

        y_pred = sampled_data[
            config["prediction"]
        ].astype(int)


        # ----------------------------------------------------
        # 8.4 CHECK THAT BOTH CLASSES ARE PRESENT
        # ----------------------------------------------------

        # AUROC and AUPRC require both glaucoma and
        # non-glaucoma cases to be present.
        #
        # If a bootstrap sample contains only one class,
        # skip that sample.
        if y_true.nunique() < 2:
            continue


        # Calculate the performance metrics for this
        # particular bootstrap sample.
        metrics = calculate_metrics(
            y_true,
            y_prob,
            y_pred
        )

        bootstrap_results.append(metrics)


    # Convert all bootstrap results into a DataFrame.
    bootstrap_df = pd.DataFrame(bootstrap_results)

    print(
        f"\nSuccessful bootstrap samples: "
        f"{len(bootstrap_df)} / {N_BOOTSTRAPS}"
    )


    # ========================================================
    # 9. CALCULATE 95% CONFIDENCE INTERVALS
    # ========================================================

    # The 2.5th and 97.5th percentiles of the bootstrap
    # distribution are used as the limits of the
    # 95% confidence interval.
    summary = []

    for metric, original_value in original_metrics.items():

        # Get all valid bootstrap values for this metric.
        values = bootstrap_df[metric].dropna()

        # Lower 95% CI limit.
        lower = np.percentile(values, 2.5)

        # Upper 95% CI limit.
        upper = np.percentile(values, 97.5)

        # Mean performance across bootstrap samples.
        bootstrap_mean = values.mean()

        # Standard deviation of the bootstrap estimates.
        bootstrap_sd = values.std(ddof=1)

        summary.append({
            "Model": model_name,
            "Metric": metric,
            "Original": original_value,
            "Bootstrap Mean": bootstrap_mean,
            "Bootstrap SD": bootstrap_sd,
            "95% CI Lower": lower,
            "95% CI Upper": upper
        })


    summary_df = pd.DataFrame(summary)


    # ========================================================
    # 10. SAVE BOOTSTRAP RESULTS
    # ========================================================

    # Create the output folder if it does not already exist.
    output_directory = (
        PROJECT_ROOT / "results" / "bootstrap"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # Create a filename-safe version of the model name.
    safe_name = (
        model_name
        .lower()
        .replace(" + ", "_")
        .replace("-", "")
        .replace(" ", "_")
    )


    # File containing the original performance and
    # bootstrap 95% confidence intervals.
    summary_file = (
        output_directory /
        f"{safe_name}_patient_bootstrap_results.csv"
    )


    # File containing the individual results from
    # every bootstrap sample.
    samples_file = (
        output_directory /
        f"{safe_name}_patient_bootstrap_samples.csv"
    )


    # Save the summary.
    summary_df.to_csv(
        summary_file,
        index=False
    )

    # Save all bootstrap samples.
    bootstrap_df.to_csv(
        samples_file,
        index=False
    )


    # ========================================================
    # 11. DISPLAY BOOTSTRAP RESULTS
    # ========================================================

    print("\nBootstrap 95% confidence intervals:")

    for _, row in summary_df.iterrows():

        print(
            f"{row['Metric']:<15} "
            f"{row['Original']:.4f} "
            f"(95% CI: "
            f"{row['95% CI Lower']:.4f} - "
            f"{row['95% CI Upper']:.4f})"
        )


    print("\nSaved:")
    print(summary_file)
    print(samples_file)

    return summary_df


# ============================================================
# 12. RUN BOOTSTRAP FOR ALL MODELS
# ============================================================

# Run the same patient-level bootstrap procedure for
# all seven models.
all_results = []

for model_name, config in RESULTS.items():

    result = run_bootstrap(
        model_name,
        config
    )

    all_results.append(result)


# ============================================================
# 13. COMBINE ALL MODEL RESULTS
# ============================================================

# Combine the individual model summaries into one DataFrame
# so that all models can be compared in one CSV file.
combined_results = pd.concat(
    all_results,
    ignore_index=True
)


# Location of the combined results file.
combined_file = (
    PROJECT_ROOT /
    "results" /
    "bootstrap" /
    "all_models_patient_bootstrap_results.csv"
)


# Save the combined results.
combined_results.to_csv(
    combined_file,
    index=False
)


# ============================================================
# 14. DISPLAY FINAL COMPARISON TABLE
# ============================================================

print("\n\n")
print("=" * 100)
print("ALL MODELS — PATIENT-LEVEL BOOTSTRAP")
print("=" * 100)


# Print each model and its original performance
# together with its 95% confidence interval.
for model_name in RESULTS.keys():

    model_results = combined_results[
        combined_results["Model"] == model_name
    ]

    print(f"\n{model_name}")

    for _, row in model_results.iterrows():

        print(
            f"  {row['Metric']:<15} "
            f"{row['Original']:.4f} "
            f"(95% CI: "
            f"{row['95% CI Lower']:.4f} - "
            f"{row['95% CI Upper']:.4f})"
        )


# ============================================================
# 15. FINISHED
# ============================================================

print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print("\nCombined results saved to:")
print(combined_file)