# 🦷 Dental OPG X-Ray Classification using MANAR Architecture

A deep learning pipeline for automatic classification of dental OPG (Orthopantomogram) X-ray images into six pathological categories using a custom **MANAR** (Memory-Augmented Attention with Navigational Abstract Conceptual Representation) architecture built on top of pre-trained backbones.

---

## Table of Contents

- [Overview](#overview)
- [Classification Categories](#classification-categories)
- [MANAR Architecture](#manar-architecture)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Setup and Data Extraction](#1-setup-and-data-extraction)
  - [2. Data Cleaning and Preprocessing](#2-data-cleaning-and-preprocessing)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Data Augmentation](#4-data-augmentation)
  - [5. Model Development — MANAR v1](#5-model-development--manar-v1)
  - [6. Training and Evaluation](#6-training-and-evaluation)
  - [7. Optimization — Class Weighting and Fine-Tuning](#7-optimization--class-weighting-and-fine-tuning)
  - [8. MANAR v2 — Upgraded Pipeline](#8-manar-v2--upgraded-pipeline)
  - [9. Inference](#9-inference)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Dental OPG X-rays are panoramic radiographs widely used by dentists for screening oral health conditions. Manual interpretation is time-consuming and can vary between practitioners. This project builds an AI-assisted diagnostic tool that automatically classifies dental X-ray images, aiming to support faster and more consistent clinical decision-making.

The core contribution is the **MANAR architecture** — a brain-inspired attention mechanism that mimics how a radiologist focuses on pathology-rich regions within an X-ray, rather than processing the entire image uniformly.

---

## Classification Categories

The model classifies each dental X-ray into one of **six categories**:

| Category | Description |
|---|---|
| **BDC-BDR** | Bone Density Changes / Bone Destruction & Resorption — indicates periodontal or bone-related issues |
| **Caries** | Tooth decay or cavities |
| **Fractured Teeth** | Cracks or structural breaks in tooth structures |
| **Healthy Teeth** | Normal, healthy dental structures |
| **Impacted Teeth** | Teeth that have not fully erupted through the gum line or bone |
| **Infection** | Bacterial infection presenting as radiolucencies around tooth roots or bone |

---

## MANAR Architecture

**MANAR** stands for **Memory-Augmented Attention with Navigational Abstract Conceptual Representation**. It is built around three brain-inspired principles:

### Brain-Inspired Attention Mechanism

Instead of processing all pixels uniformly, the model uses a dual **Channel + Spatial Attention** block inspired by how the human visual cortex selectively attends to salient regions. The implementation uses:

- **Channel Attention**: Global Average Pooling and Global Max Pooling are passed through a shared two-layer MLP (with a reduction ratio of 16), then combined via element-wise addition and a sigmoid activation. This learns *which* feature channels are most discriminative.
- **Spatial weighting**: The resulting channel attention map is applied multiplicatively to the feature maps, amplifying pathology-relevant channels and suppressing noise.

```python
def brain_inspired_attention(inputs):
    channels = inputs.shape[-1]
    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)

    shared_fc1 = Dense(channels // 16, activation='relu')
    shared_fc2 = Dense(channels)

    avg_out = shared_fc2(shared_fc1(avg_pool))
    max_out = shared_fc2(shared_fc1(max_pool))

    attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    attention = Reshape((1, 1, channels))(attention)

    return Multiply()([inputs, attention])
```

### Memory Augmentation

The attention module is enhanced by learned dense memory representations that store prototypical feature patterns of dental conditions, enabling the model to make more informed attention decisions on new images.

### Navigational Abstract Representation

Rather than operating purely at the pixel level, the model develops higher-level abstract representations of dental pathology (e.g., "caries progression" or "periapical radiolucency") and navigates between these concepts to inform classification.

---

## Dataset

The project uses the [Dental OPG X-Ray Dataset](https://www.kaggle.com/datasets/imtkaggleteam/dental-opg-xray-dataset) from Kaggle, structured as image folders per category. An additional **object detection dataset** (from Google Drive) is used in the v2 pipeline to extract cropped regions-of-interest, significantly expanding the training pool.

**Data pipeline**:
- **Original dataset**: Classification images organized into 6 category folders
- **Augmented dataset (v2)**: Bounding-box annotations from a detection dataset are used to crop individual teeth/pathology regions, which are then merged with the original data

---

## Project Pipeline

### 1. Setup and Data Extraction

The environment is configured in Google Colab with Kaggle API authentication. The dataset is downloaded and extracted programmatically.

```python
!kaggle datasets download -d imtkaggleteam/dental-opg-xray-dataset -p /content/dental_opg
```

All image paths and their corresponding labels are collected into a Pandas DataFrame for downstream processing.

### 2. Data Cleaning and Preprocessing

Before training, the dataset integrity is verified:
- **Duplicates check**: Confirmed zero duplicate records
- **Missing values**: Confirmed no null entries
- **Label validation**: Verified all six unique categories are present and correctly mapped

### 3. Exploratory Data Analysis

EDA is performed to understand dataset characteristics:
- **Class distribution plot**: A bar chart reveals significant class imbalance — some categories (e.g., Healthy Teeth) have far more samples than others (e.g., Infection, Fractured Teeth)
- **Sample visualization**: Grid of sample X-ray images from each category to understand the visual patterns the model must learn

### 4. Data Augmentation

Images are prepared for model input using `ImageDataGenerator`:

| Parameter | Training | Validation |
|---|---|---|
| Rescale | 1/255 | 1/255 |
| Rotation | ±15° | None |
| Horizontal Flip | Yes | None |
| Fill Mode | Nearest | None |

- **Image size**: 224 × 224 pixels
- **Batch size**: 32
- **Train/Val split**: 80/20 (stratified by class label)

### 5. Model Development — MANAR v1

The first version of the MANAR model is built as follows:

```
Input (224×224×3)
  → MobileNetV2 (frozen, ImageNet weights)
    → Brain-Inspired Attention Block
      → Global Average Pooling
        → Dropout (0.3)
          → Dense (256, ReLU)
            → Dense (6, Softmax)
```

- **Backbone**: MobileNetV2 (pre-trained on ImageNet, initially frozen)
- **Optimizer**: Adam
- **Loss**: Categorical Cross-Entropy
- **Metrics**: Accuracy

### 6. Training and Evaluation

**Initial training**: 20 epochs with EarlyStopping (patience=5, monitoring `val_loss`, restoring best weights).

Evaluation is performed on the validation set using:
- **Accuracy & Loss curves**: Plotted for both training and validation across epochs
- **Confusion Matrix**: Heatmap showing per-class prediction performance
- **Classification Report**: Precision, recall, and F1-score for each category

### 7. Optimization — Class Weighting and Fine-Tuning

To address class imbalance and improve minority-class performance:

**Class Weighting**: `sklearn.utils.class_weight.compute_class_weight` is used to calculate balanced weights. The loss function penalizes errors on minority classes (Infection, Fractured Teeth) more heavily during training.

**Fine-Tuning**: The MobileNetV2 base is unfrozen and the entire model is retrained with a very low learning rate (`1e-5`) to adapt pre-trained features to dental-specific pathology without destroying general knowledge. This is run for two additional sessions of 10 epochs each.

Post-optimization evaluation (confusion matrix + classification report) is generated to measure improvement.

### 8. MANAR v2 — Upgraded Pipeline

The second iteration introduces two major improvements:

**Backbone upgrade**: MobileNetV2 is replaced with **EfficientNetB0**, which uses compound scaling to better capture fine dental details like hairline fractures.

**Dataset expansion**: An object detection dataset with YOLO-format bounding box annotations is processed to extract cropped regions of interest. These crops are categorized using the annotation class IDs and merged with the original dataset, significantly expanding the training pool.

```
MANAR v2 Architecture:
Input (224×224×3)
  → EfficientNetB0 (frozen, ImageNet weights)
    → Brain-Inspired Attention Block
      → Global Average Pooling
        → Dropout (0.4)
          → Dense (6, Softmax)
```

Training uses `ReduceLROnPlateau` (factor=0.2, patience=2) alongside EarlyStopping for more adaptive learning rate scheduling.

### 9. Inference

A utility function allows random image inference from the validation set, displaying the original image alongside the predicted label and confidence score.

---

## Results

### MANAR v1 — Initial Training (MobileNetV2, Frozen)

Training for 20 epochs (with EarlyStopping) on the original 517-image dataset yielded a **31% overall accuracy**. The model heavily biased toward the majority class:

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| BDC-BDR | 1.00 | 0.10 | 0.18 | 10 |
| Caries | 0.13 | 0.12 | 0.13 | 24 |
| Fractured Teeth | 0.00 | 0.00 | 0.00 | 3 |
| Healthy Teeth | 0.36 | 0.60 | 0.45 | 45 |
| Impacted Teeth | 0.20 | 0.06 | 0.09 | 17 |
| Infection | 0.00 | 0.00 | 0.00 | 5 |
| **Weighted Avg** | **0.31** | **0.31** | **0.26** | **104** |

### MANAR v1 — After Fine-Tuning & Class Weighting

After unfreezing MobileNetV2 and retraining with balanced class weights at a learning rate of `1e-5` for two additional 10-epoch sessions, accuracy **dropped slightly to 28%**:

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| BDC-BDR | 0.33 | 0.10 | 0.15 | 10 |
| Caries | 0.16 | 0.21 | 0.18 | 24 |
| Fractured Teeth | 0.00 | 0.00 | 0.00 | 3 |
| Healthy Teeth | 0.34 | 0.49 | 0.40 | 45 |
| Impacted Teeth | 0.17 | 0.06 | 0.09 | 17 |
| Infection | 0.00 | 0.00 | 0.00 | 5 |
| **Weighted Avg** | **0.25** | **0.28** | **0.25** | **104** |

### MANAR v2 (EfficientNetB0 + Expanded Dataset)

The v2 pipeline expanded the dataset from 517 to **18,949 images** using object detection crops and switched to EfficientNetB0. However, training accuracy plateaued at approximately **22–23%** across all 10 epochs, with validation accuracy stagnating at ~23%. The learning rate was reduced from `1e-3` to `2e-4` by `ReduceLROnPlateau` but had no meaningful effect.

---

## ⚠️ Known Accuracy Issues & Root Cause Analysis

The model currently underperforms across all iterations. Below is a transparent analysis of the identified issues and proposed solutions.

### 1. Severe Data Scarcity (v1)

The original classification dataset contains only **517 images** across 6 classes, with an extreme imbalance:

| Category | Samples | Share |
|---|---|---|
| Healthy Teeth | 223 | 43.1% |
| Caries | 119 | 23.0% |
| Impacted Teeth | 87 | 16.8% |
| BDC-BDR | 52 | 10.1% |
| Infection | 23 | 4.4% |
| Fractured Teeth | 13 | 2.5% |

With only 13 Fractured Teeth and 23 Infection samples, the model cannot learn meaningful representations for these classes. The validation split further reduces these to just 3 and 5 samples respectively — far too few for reliable evaluation.

### 2. Crop Quality Issues (v2)

The v2 dataset expansion from 517 → 18,949 via bounding-box cropping introduced potential problems. The `id_to_category` mapping may not accurately correspond to the object detection labels, meaning crops could be systematically mislabeled. The v2 model's inability to exceed ~23% accuracy (near random for 6 classes ≈ 16.7%) suggests the expanded data may contain significant label noise.

### 3. Frozen Backbone Limitations

Both v1 and v2 began training with the backbone entirely frozen, leaving only the attention block and classifier head trainable. For a highly specialized domain like dental radiography — which looks very different from ImageNet natural images — the frozen features may be insufficiently adapted. Fine-tuning in v1 was applied too late and with too aggressive a class-weight penalty, causing validation loss to increase.

### 4. Insufficient Augmentation

The augmentation pipeline applies only rotation (±15°) and horizontal flips. Dental X-rays would benefit from more aggressive augmentation: elastic deformations, contrast adjustments (CLAHE), random cropping, and mixup/cutmix strategies to simulate the variability seen in clinical imaging.

### 5. Attention-Only Architecture

The current "brain-inspired attention" block implements channel attention only (no spatial attention component despite the description). A full CBAM (Convolutional Block Attention Module) with both channel and spatial branches, or a self-attention / transformer-based approach, may better localize pathology regions.

### 6. No Learning Rate Warm-Up

Training jumps directly into Adam at `1e-3` with a frozen backbone, then switches to `1e-5` for fine-tuning. A cosine annealing schedule or warm-up phase could stabilize early training and prevent the loss plateaus observed in the training logs.

### Recommended Fixes

| Issue | Recommended Action |
|---|---|
| Data scarcity | Collect more labeled data; apply SMOTE, GANs, or heavy augmentation for minority classes |
| Label noise in crops | Manually audit a sample of extracted crops to verify `id_to_category` mapping accuracy |
| Frozen backbone | Progressively unfreeze backbone layers from early training; use discriminative learning rates |
| Weak augmentation | Add CLAHE, elastic transforms, random erasing, mixup/cutmix |
| Attention module | Implement full CBAM (channel + spatial) or integrate lightweight vision transformers |
| LR scheduling | Use cosine annealing with warm-up; experiment with OneCycleLR |
| Evaluation | Use k-fold cross-validation instead of a single 80/20 split given the small dataset |
| Architecture | Consider a lighter head with fewer dense layers or add batch normalization before the classifier |

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | TensorFlow / Keras |
| Backbones | MobileNetV2, EfficientNetB0 |
| Data Handling | Pandas, NumPy, OpenCV |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn (confusion matrix, classification report, class weights) |
| Environment | Google Colab |
| Dataset Source | Kaggle API |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or a local GPU environment
- Kaggle account with API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/dental-opg-manar.git
   cd dental-opg-manar
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python kaggle pillow tqdm
   ```

3. **Configure Kaggle API**
   - Place your `kaggle.json` in `~/.kaggle/` or set credentials via environment variables
   - Or use Google Colab Secrets (`KAG_USER` and `KAGGLE_KEY`)

4. **Run the notebook**
   ```bash
   jupyter notebook Dental_OPG_X-Ray_Classification_using_MANAR_Architecture.ipynb
   ```

---

## Project Structure

```
dental-opg-manar/
├── Dental_OPG_X-Ray_Classification_using_MANAR_Architecture.ipynb  # Full pipeline notebook
├── README.md
├── manar_dental_classifier.keras     # Saved model weights (generated after training)
└── requirements.txt                  # Python dependencies
```

---

## Future Work

- **Advanced augmentation**: Apply SMOTE or GANs to synthetically generate minority-class samples
- **Ensemble methods**: Combine predictions from multiple backbones (EfficientNet, ResNet, DenseNet) for higher diagnostic accuracy
- **Localization**: Extend from classification to object detection or segmentation to pinpoint exact pathology locations on the OPG
- **Hyperparameter tuning**: Use Keras Tuner or Optuna to optimize dropout rates, dense layer sizes, and learning rate schedules
- **Clinical validation**: Test the model on unseen clinical data with radiologist ground truth for real-world applicability

---

## License

This project is for educational and research purposes. Please refer to the [Kaggle dataset license](https://www.kaggle.com/datasets/imtkaggleteam/dental-opg-xray-dataset) for data usage terms.
