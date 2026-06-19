# ISRO Road Segmentation from Satellite Imagery — Documentation

## Project Overview

This project focuses on **road segmentation from satellite imagery** using deep learning. The goal is to automatically detect and segment roads in satellite images using a DeepLabV3+ model with a ResNet34 encoder backbone.

---

## Table of Contents

1. [Dataset Structure](#1-dataset-structure)
2. [Data Exploration](#2-data-exploration)
3. [Model Architecture](#3-model-architecture)
4. [Dataset Class (`RoadDataset`)](#4-dataset-class-roaddataset)
5. [Model Setup — DeepLabV3+](#5-model-setup--deeplabv3)
6. [Training Loop](#6-training-loop)
7. [Inference & Prediction](#7-inference--prediction)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Model Saving](#9-model-saving)
10. [Training Results](#10-training-results)
11. [Dependencies](#11-dependencies)

---

## 1. Dataset Structure

The dataset is organized into three folders:

```
project/
├── train/       ← Training images + masks
├── valid/       ← Validation images + masks
└── test/        ← Test images
```

**File naming convention:**
- Satellite images: `<image_id>_sat.jpg`
- Road masks: `<image_id>_mask.png`

**Dataset statistics:**
- Total matched satellite-mask pairs in `train/`: **6,226 pairs**
- Training subset used: **3,000 images**

---

## 2. Data Exploration

### 2.1 Listing Dataset Contents

```python
import os

print("TRAIN FOLDER")
print(os.listdir("train")[:10])

print("\nVALID FOLDER")
print(os.listdir("valid")[:10])

print("\nTEST FOLDER")
print(os.listdir("test")[:10])
```

### 2.2 Visualizing a Sample Image and Mask

```python
import cv2
import matplotlib.pyplot as plt

image_id = "401242"

img = cv2.imread(f"train/{image_id}_sat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.imread(
    f"train/{image_id}_mask.png",
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Satellite Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Road Mask")
plt.axis("off")

plt.show()
```

This cell loads a satellite image and its corresponding binary road mask and displays them side by side.

### 2.3 Finding Valid Image-Mask Pairs

```python
files = os.listdir("train")

sat_ids = {
    f.replace("_sat.jpg", "")
    for f in files
    if f.endswith("_sat.jpg")
}

mask_ids = {
    f.replace("_mask.png", "")
    for f in files
    if f.endswith("_mask.png")
}

common_ids = list(sat_ids & mask_ids)

print("Pairs:", len(common_ids))
print(common_ids[:5])
```

**Output:**
```
Pairs: 6226
['63019', '735969', '767454', '25742', '364184']
```

---

## 3. Model Architecture

### Initial Prototype — U-Net (not used for training)

A preliminary U-Net model was tested for inspection:

```python
import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model.eval()
print("Model Ready")
```

This was used only to verify the library setup and was **not trained**.

---

## 4. Dataset Class (`RoadDataset`)

### Libraries Used

```python
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, DataLoader
```

### `RoadDataset` Class

```python
class RoadDataset(Dataset):
    def __init__(self, ids, folder="train"):
        self.ids = ids
        self.folder = folder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        # Load and preprocess image
        img = cv2.imread(f"{self.folder}/{image_id}_sat.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        # Load and preprocess mask
        mask = cv2.imread(
            f"{self.folder}/{image_id}_mask.png",
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # ── Data Augmentation ──────────────────────────────────────────
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random Occlusion Simulation (black rectangle patch)
        if np.random.rand() > 0.5:
            x1 = np.random.randint(50, 350)
            y1 = np.random.randint(50, 350)
            w = np.random.randint(40, 120)
            h = np.random.randint(40, 120)
            img[y1:y1+h, x1:x1+w] = 0

        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Binarize mask: pixel > 127 → 1, else → 0
        mask = (mask > 127).astype(np.float32)

        # Convert to PyTorch tensors
        img = torch.tensor(img.transpose(2, 0, 1))    # [C, H, W]
        mask = torch.tensor(mask).unsqueeze(0)         # [1, H, W]

        return img, mask
```

### Key Preprocessing Steps

| Step | Detail |
|------|--------|
| Image resize | 512 × 512 pixels |
| Mask resize | 512 × 512 (nearest-neighbor to preserve binary values) |
| Normalization | Image pixels divided by 255.0 |
| Mask binarization | Threshold at 127 → binary float mask |
| Flip augmentation | 50% horizontal flip, 50% vertical flip |
| Occlusion simulation | 50% chance of random black rectangle patch |

### DataLoader Setup

```python
train_ids = common_ids[:3000]

dataset = RoadDataset(train_ids)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print(len(dataset))  # Output: 3000
```

---

## 5. Model Setup — DeepLabV3+

### Device Configuration

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)  # Output: mps
```

> **Note:** The model was trained on an Apple Silicon GPU using **MPS (Metal Performance Shaders)** backend.

### Model Definition

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model = model.to(device)
```

| Parameter | Value |
|-----------|-------|
| Architecture | DeepLabV3+ |
| Encoder | ResNet34 |
| Pre-trained weights | ImageNet |
| Input channels | 3 (RGB) |
| Output classes | 1 (binary road mask) |

### Loss Functions

```python
# Dice Loss
dice_loss = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)

# Binary Cross-Entropy Loss
bce_loss = torch.nn.BCEWithLogitsLoss()
```

### Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

---

## 6. Training Loop

```python
from tqdm import tqdm

epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0.0
    pbar = tqdm(loader)

    for images, masks in pbar:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        # Verify shape consistency
        assert preds.shape == masks.shape, (
            f"Shape mismatch: preds={preds.shape}, masks={masks.shape}"
        )

        # Combined loss: 50% Dice + 50% BCE
        loss = (
            0.5 * dice_loss(preds, masks)
            +
            0.5 * bce_loss(preds, masks)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch+1} Loss {loss.item():.4f}"
        )

    print(
        f"Epoch {epoch+1} Avg Loss:",
        total_loss / len(loader)
    )
```

### Loss Function Strategy

The training uses a **combined loss**:

```
Total Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```

- **Dice Loss**: Measures overlap between predicted and ground-truth masks. Ideal for imbalanced segmentation tasks.
- **BCE with Logits**: Standard pixel-wise binary cross-entropy loss.

---

## 7. Inference & Prediction

```python
model.eval()

# Pick a test image (index 5000 — outside training set)
test_id = common_ids[5000]

img = cv2.imread(f"train/{test_id}_sat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (512, 512))

# Preprocess
x = img_resized.astype(np.float32) / 255.0
x = torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0)
x = x.to(device)

# Run inference
with torch.no_grad():
    pred = model(x)

# Post-process
pred = torch.sigmoid(pred)
pred = pred.squeeze().cpu().numpy()

# Threshold at 0.2 → binary mask
pred_binary = (pred > 0.2).astype(np.uint8)

# Road coverage percentage
road_pixels = np.sum(pred_binary)
total_pixels = pred_binary.size
print("Road %", 100 * road_pixels / total_pixels)

plt.imshow(pred_binary, cmap="gray")
plt.show()
```

**Output:**
```
Road % 0.5031585693359375
```

> **Threshold:** A value of `0.2` was used (lower than the standard `0.5`) to increase road pixel recall.

---

## 8. Evaluation Metrics

### IoU (Intersection over Union) & Dice Coefficient

```python
from sklearn.metrics import jaccard_score
import numpy as np

# Load ground truth mask
mask = cv2.imread(
    f"train/{test_id}_mask.png",
    cv2.IMREAD_GRAYSCALE
)
mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
mask = (mask > 127).astype(np.uint8)

# IoU Score
iou = jaccard_score(
    mask.flatten(),
    pred_binary.flatten()
)

# Dice Score
intersection = np.sum(mask * pred_binary)
dice = (
    2 * intersection
) / (
    np.sum(mask) +
    np.sum(pred_binary) +
    1e-8
)

print("IoU :", iou)
print("Dice:", dice)
```

**Output:**
```
IoU : 0.3036828963795256
Dice: 0.4658846061755665
```

### Metric Definitions

| Metric | Formula | Result |
|--------|---------|--------|
| **IoU** (Jaccard Index) | Intersection / Union | 0.304 |
| **Dice Coefficient** | 2 × Intersection / (Sum of areas) | 0.466 |

> These scores represent performance on a single test sample after 10 epochs of training on only 3,000 samples.

---

## 9. Model Saving

```python
torch.save(model.state_dict(), "deeplabv3plus_road.pth")
print("Model Saved")
```

The trained model weights are saved to `deeplabv3plus_road.pth` in PyTorch's standard state dictionary format.

---

## 10. Training Results

### Training Loss Per Epoch

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.3761 |
| 2 | 0.3072 |
| 3 | 0.2857 |
| 4 | 0.2675 |
| 5 | 0.2605 |
| 6 | 0.2509 |
| 7 | 0.2439 |
| 8 | 0.2387 |
| 9 | 0.2346 |
| 10 | 0.2319 |

- Training was run for **10 epochs**
- Each epoch processed **750 batches** (3000 images ÷ batch size 4)
- Loss consistently decreased, indicating effective learning
- Average epoch duration: ~12–15 minutes on Apple MPS

### Test Sample Evaluation (After 10 Epochs)

| Metric | Value |
|--------|-------|
| IoU | 0.304 |
| Dice | 0.466 |
| Road Coverage | ~0.5% |

---

## 11. Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning framework (PyTorch) |
| `segmentation_models_pytorch` | Pre-built segmentation model zoo |
| `opencv-python` (`cv2`) | Image loading, resizing, color conversion |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `scikit-image` (`skimage`) | `skeletonize` (imported but not used in training) |
| `sklearn` | `jaccard_score` for IoU metric |
| `tqdm` | Training progress bar |

### Installation

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python
pip install numpy matplotlib scikit-image scikit-learn tqdm
```

---

## Notes & Observations

1. **Dataset Size**: Only 3,000 of 6,226 available training pairs were used. Training on the full dataset may improve metrics.
2. **Threshold**: Inference uses a threshold of 0.2 (instead of 0.5), which increases recall but may reduce precision.
3. **Augmentation**: Horizontal/vertical flips and random occlusion patches help prevent overfitting.
4. **Hardware**: Training ran on Apple MPS (M-series chip). CUDA support is also available.
5. **Model Not Evaluated on Validation Set**: The notebook only evaluates on a single training-set sample. Validation set evaluation would provide more reliable performance metrics.
6. **`skeletonize` Import**: `skimage.morphology.skeletonize` is imported in the dataset class but not used — possibly intended for road centerline extraction in future iterations.
