# 🧠 Uncertainty-Aware Deep Learning for Gold-Standard Fundus-Based Glaucoma Detection

## 📌 Overview

This project investigates the use of deep learning for automated glaucoma detection from fundus photographs, with a focus on **prediction uncertainty and model interpretability**.

The project evaluates conventional deep learning models alongside uncertainty-aware approaches, including **Monte Carlo (MC) Dropout** and **Deep Ensembles**. Grad-CAM is also used to visualise regions contributing to model predictions.

---

## 🎯 Research Question

**Can deep learning accurately detect glaucoma from fundus photographs while also estimating the uncertainty of its predictions?**

---

## 🔬 Objectives

- Develop deep learning models for fundus-based glaucoma classification.
- Compare different CNN architectures.
- Incorporate uncertainty estimation using MC Dropout and Deep Ensembles.
- Evaluate predictive performance, calibration and uncertainty.
- Use Grad-CAM to investigate model behaviour and visual explanations.

---

## 🗂️ Dataset

The project uses the **Hillel Yaffe Glaucoma Dataset (HYGD)**, a publicly available and de-identified fundus image dataset hosted on PhysioNet.

| Dataset characteristic | Value |
|---|---:|
| Fundus images | 747 |
| Patients | 288 |
| Glaucoma | 548 (73.4%) |
| Non-glaucoma | 199 (26.6%) |

The diagnostic labels are based on comprehensive ophthalmic evaluation.

---

## 🔀 Data Partitioning

To reduce the risk of patient-level data leakage, images were split at the **patient level** rather than the image level.

- **70%** training
- **10%** validation
- **20%** testing
- Stratified by glaucoma status
- `random_state = 1234`
- Images from the same patient were kept within the same partition

---

## 🖼️ Image Preprocessing

- Images resized to **224 × 224 pixels**
- ImageNet-compatible preprocessing
- Batch size of **16**
- No data augmentation
- Labels encoded as:
  - `GON− → 0`
  - `GON+ → 1`

---

## 🤖 Model Development

Three CNN architectures were evaluated:

| Model | Architecture | Key idea | Role |
|---|---|---|---|
| ResNet-50 | Residual CNN | Skip connections | Baseline + Deep Ensemble |
| DenseNet-121 | Dense connectivity CNN | Feature reuse | Baseline + MC Dropout |
| EfficientNet-B4 | Efficient CNN | Efficient scaling | Baseline + MC Dropout |

### Transfer Learning

The models were initialized using **ImageNet-pretrained weights**. The pretrained convolutional backbone was frozen, and a new classification head was trained for binary glaucoma detection.

---

## 🎲 Uncertainty Estimation

### Monte Carlo Dropout

MC Dropout keeps dropout active during inference and performs multiple stochastic forward passes.

In this project:

- One trained model
- **30 stochastic forward passes**
- Mean probability used as the prediction
- Prediction variation used as an uncertainty estimate

Higher variation between predictions indicates greater model uncertainty.

### Deep Ensemble

The deep ensemble consists of **five independently trained ResNet-50 models**.

- 5 ResNet-50 models
- Different random seeds
- Same architecture and experimental setup
- Mean probability used for prediction
- Variation between ensemble members used as uncertainty

This provides a second approach to estimating model uncertainty.

---

## 📊 Model Evaluation

Performance was evaluated using:

- AUROC
- AUPRC
- Accuracy
- Sensitivity
- Specificity
- F1-score
- Balanced accuracy

### Best-performing model

**ResNet-50 + Deep Ensemble**

| Metric | Result |
|---|---:|
| AUROC | **0.9858** |
| AUPRC | **0.9956** |
| Accuracy | **95.71%** |
| Sensitivity | **98.15%** |
| Specificity | **87.50%** |
| F1-score | **0.9725** |
| Balanced accuracy | **0.9282** |

---

## 📏 Calibration & Uncertainty

Calibration was assessed using:

- Brier score
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Error-detection AUROC

| Model | Brier | ECE | NLL | Error AUROC |
|---|---:|---:|---:|---:|
| ResNet-50 | 0.0354 | 0.0408 | 0.1502 | — |
| ResNet-50 + MC Dropout | **0.0341** | 0.0356 | 0.1376 | **0.9334** |
| ResNet-50 + Deep Ensemble | 0.0362 | **0.0353** | **0.1264** | 0.9192 |

The uncertainty estimates generally showed **lower variation for correct predictions and higher variation for incorrect predictions**, although the separation was not perfect.

---

## 🔎 Explainability with Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** was used to visualise regions contributing to model predictions.

Grad-CAM was applied to:

- True positives
- True negatives
- False positives
- False negatives

This provides qualitative insight into **where the model is focusing**, rather than only whether its prediction is correct.

The analysis suggested that correct predictions generally showed more focused activation, while some incorrect predictions showed broader or less consistent activation patterns.

> Grad-CAM provides a qualitative visual explanation and should not be interpreted as proof of clinical reasoning or causal importance.

---

## 🏆 Key Findings

- **ResNet-50 + Deep Ensemble achieved the strongest overall predictive performance.**
- Uncertainty estimates provided additional information about model confidence and prediction consistency.
- MC Dropout and Deep Ensemble approaches demonstrated useful calibration characteristics.
- Grad-CAM provided qualitative insight into model behaviour.

---

## 🧪 Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Grad-CAM
- Git / GitHub

---

## 🔁 Reproducibility

The experiments use fixed random seeds where specified to support reproducibility.

The Deep Ensemble uses the following seeds:

```text
1234
2345
3456
4567
5678
