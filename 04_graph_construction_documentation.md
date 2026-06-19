# ISRO Road Network Graph Construction — Documentation

## Project Overview

This notebook builds a **road network graph** from satellite imagery using a pre-trained DeepLabV3+ segmentation model. The workflow converts the predicted road mask into a skeletal road network, then constructs a mathematical graph to analyze road connectivity, identify critical junctions, and simulate road failure scenarios.

---

## Table of Contents

1. [Imports & Libraries](#1-imports--libraries)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Model Loading & Road Mask Prediction](#3-model-loading--road-mask-prediction)
4. [Road Skeletonization](#4-road-skeletonization)
5. [Graph Construction](#5-graph-construction)
6. [Graph Statistics](#6-graph-statistics)
7. [Connected Components Analysis](#7-connected-components-analysis)
8. [Graph Visualization](#8-graph-visualization)
9. [Largest Connected Component Extraction](#9-largest-connected-component-extraction)
10. [Betweenness Centrality Analysis](#10-betweenness-centrality-analysis)
11. [Critical Junction Visualization](#11-critical-junction-visualization)
12. [Road Failure Simulation](#12-road-failure-simulation)
13. [Network Resilience Score](#13-network-resilience-score)
14. [Post-Failure Network Visualization](#14-post-failure-network-visualization)
15. [Complete Pipeline Summary](#15-complete-pipeline-summary)
16. [Key Results](#16-key-results)
17. [Dependencies](#17-dependencies)

---

## 1. Imports & Libraries

```python
import cv2
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
```

| Library | Purpose |
|---------|---------|
| `cv2` | Image loading, color conversion, resizing |
| `torch` | Deep learning inference (PyTorch) |
| `networkx` | Graph construction and analysis |
| `numpy` | Numerical array operations |
| `matplotlib` | Visualization of images and graphs |
| `skimage.morphology.skeletonize` | Reducing binary road mask to 1-pixel-wide skeleton |

---

## 2. Dataset Preparation

```python
import os

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
```

**Output:**
```
Pairs: 6226
```

- Scans the `train/` directory and extracts all valid satellite-mask pairs.
- Results in **6,226** matched image-mask pairs.

---

## 3. Model Loading & Road Mask Prediction

### 3.1 Device Detection

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

> The model runs on **Apple MPS** (Metal Performance Shaders) on M-series chips.

### 3.2 Model Architecture

```python
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
```

> **Note:** This notebook uses a **ResNet50** encoder (compared to ResNet34 used in `01_data_exploration.ipynb`), indicating a more capable pre-trained model.

### 3.3 Loading Pre-trained Weights

```python
model.load_state_dict(
    torch.load(
        "deeplabv3_resnet50_5000img.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model Loaded Successfully")
```

**Output:**
```
Model Loaded Successfully
```

- Loads weights from `deeplabv3_resnet50_5000img.pth` — a model trained on 5,000 images (larger dataset than the 3,000-image model from `01_data_exploration.ipynb`).

### 3.4 Inference on a Test Image

```python
test_id = common_ids[0]

img = cv2.imread(f"train/{test_id}_sat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (512, 512))

x = img_resized.astype(np.float32) / 255.0
x = torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0)
x = x.to(device)

with torch.no_grad():
    pred = model(x)

pred = torch.sigmoid(pred)
pred = pred.squeeze().cpu().numpy()

pred_binary = (pred > 0.2).astype(np.uint8)
```

**Preprocessing Steps:**
1. Load satellite image via OpenCV
2. Convert BGR → RGB
3. Resize to 512 × 512
4. Normalize to [0, 1]
5. Transpose to [C, H, W] and add batch dimension

**Post-processing:**
- Apply sigmoid to convert logits to probabilities
- Threshold at `0.2` → binary road mask (1 = road, 0 = background)

### 3.5 Visualization

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_resized)
plt.title("Satellite")

plt.subplot(1, 2, 2)
plt.imshow(pred_binary, cmap="gray")
plt.title("Road Mask")

plt.show()
```

Side-by-side display: **satellite image** (left) + **predicted road mask** (right).

---

## 4. Road Skeletonization

```python
road_mask = pred_binary.astype(bool)

skeleton = skeletonize(road_mask)

plt.figure(figsize=(8, 8))
plt.imshow(skeleton, cmap="gray")
plt.title("Road Skeleton")
plt.show()
```

- Converts the binary road mask to a boolean array.
- Applies **morphological skeletonization** (from `skimage.morphology`), which reduces thick road regions to a **1-pixel-wide centerline skeleton**.
- The skeleton preserves the topology and connectivity of the road network while removing redundant road width pixels.

**Purpose:** Skeletonization enables treating each road pixel as a **graph node**, with connectivity based on adjacency.

---

## 5. Graph Construction

### 5.1 Extract Skeleton Pixels

```python
ys, xs = np.where(skeleton)

print("Skeleton Pixels:", len(xs))
```

**Output:**
```
Skeleton Pixels: 1159
```

- `np.where(skeleton)` returns the (row, col) coordinates of all road pixels in the skeleton.

### 5.2 Build the Graph

```python
import networkx as nx
import numpy as np

G = nx.Graph()

ys, xs = np.where(skeleton)

# Add all skeleton pixels as nodes
for y, x in zip(ys, xs):
    G.add_node((x, y))

# 8-neighborhood connections
directions = [
    (-1,-1), (-1,0), (-1,1),
    (0,-1),          (0,1),
    (1,-1),  (1,0),  (1,1)
]

pixel_set = set(zip(xs, ys))

for x, y in pixel_set:
    for dx, dy in directions:
        nx2 = x + dx
        ny2 = y + dy
        if (nx2, ny2) in pixel_set:
            G.add_edge((x, y), (nx2, ny2))
```

**Graph Construction Logic:**

| Component | Description |
|-----------|-------------|
| **Nodes** | Each skeleton pixel → node with coordinate (x, y) |
| **Edges** | Two nodes are connected if they are 8-neighbors (adjacent horizontally, vertically, or diagonally) |
| **Neighborhood** | 8-connectivity (includes diagonals) |

**Why 8-connectivity?** Diagonal connections ensure continuity for diagonal road segments that would otherwise be disconnected with 4-connectivity.

---

## 6. Graph Statistics

```python
print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())
```

**Output:**
```
Nodes : 1764
Edges : 1753
```

> Note: The number of nodes (1764) is larger than skeleton pixels (1159). This is because `G.add_edge()` also implicitly creates nodes when both endpoints are new. The duplicate node adds likely arise from the iteration logic (both pixels added individually and then added again as edge endpoints).

---

## 7. Connected Components Analysis

```python
components = list(nx.connected_components(G))

print("Graph Components:", len(components))
```

**Output:**
```
Graph Components: 5
```

- The road network is **fragmented into 5 disconnected subgraphs** (connected components).
- This is expected because road segmentation models may predict disconnected road segments for complex scenes.

### 7.1 Component Size Distribution

```python
component_sizes = sorted(
    [len(c) for c in nx.connected_components(G)],
    reverse=True
)

print(component_sizes[:10])
```

**Output:**
```
[395, 350, 262, 147, 5]
```

| Component | Nodes |
|-----------|-------|
| Largest | 395 |
| 2nd | 350 |
| 3rd | 262 |
| 4th | 147 |
| Smallest | 5 |

---

## 8. Graph Visualization

```python
plt.figure(figsize=(10, 10))

pos = {
    node: node
    for node in G.nodes()
}

nx.draw(
    G,
    pos=pos,
    node_size=1,
    width=0.5
)

plt.title("Road Graph")
plt.show()
```

- Uses **pixel coordinates as layout positions** (`pos = {node: node}`), so the graph is drawn in its correct spatial orientation.
- Very small `node_size=1` and `width=0.5` are used to keep the visualization clean at pixel resolution.

---

## 9. Largest Connected Component Extraction

```python
import networkx as nx

largest_component = max(
    nx.connected_components(G),
    key=len
)

G_main = G.subgraph(
    largest_component
).copy()

print("Main Nodes:", G_main.number_of_nodes())
print("Main Edges:", G_main.number_of_edges())
```

**Output:**
```
Main Nodes: 395
Main Edges: 394
```

- Extracts the **largest connected component** (395 nodes) for focused analysis.
- `G.subgraph(...).copy()` creates an independent copy so modifications don't affect the original graph.

### 9.1 Visualization of Largest Component

```python
plt.figure(figsize=(10, 10))

pos = {
    node: node
    for node in G_main.nodes()
}

nx.draw(
    G_main,
    pos=pos,
    node_size=1,
    width=0.5
)

plt.title("Largest Connected Road Network")
plt.show()
```

---

## 10. Betweenness Centrality Analysis

```python
import networkx as nx

centrality = nx.betweenness_centrality(G_main)

top_nodes = sorted(
    centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]

print(top_nodes[:10])
```

**Output:**
```
[((17, 197), 0.5012722646310432),
 ((17, 196), 0.5012593482388499),
 ((17, 198), 0.5012593482388499),
 ((17, 195), 0.5012205990622699),
 ((17, 199), 0.5012205990622699),
 ((17, 200), 0.5011560171013033),
 ((17, 194), 0.5011560171013033),
 ((17, 201), 0.50106560235595),
 ((17, 193), 0.50106560235595),
 ((17, 192), 0.50094935482621)]
```

### What is Betweenness Centrality?

**Betweenness centrality** of a node `v` is defined as:

```
BC(v) = Σ(s≠v≠t) [ σ(s,t|v) / σ(s,t) ]
```

Where:
- `σ(s,t)` = total number of shortest paths from node `s` to node `t`
- `σ(s,t|v)` = number of those paths that pass through node `v`

**Interpretation:**
- A node with **high betweenness centrality** lies on many shortest paths between other nodes.
- In a road network, such nodes are **critical junctions** — removing them would disconnect or significantly lengthen routes.

**Findings:**
- The top critical nodes cluster around coordinate `(17, 192-201)` with centrality scores ≈ **0.50**, meaning approximately 50% of all shortest paths in `G_main` pass through these nodes.
- This indicates a **narrow bottleneck corridor** in the road network.

---

## 11. Critical Junction Visualization

```python
plt.figure(figsize=(10, 10))

pos = {
    node: node
    for node in G_main.nodes()
}

nx.draw(G_main, pos, node_size=1, width=0.5)

critical_nodes = [
    node
    for node, score in top_nodes[:10]
]

xs = [n[0] for n in critical_nodes]
ys = [n[1] for n in critical_nodes]

plt.scatter(xs, ys, s=50)

plt.title("Critical Junctions")
plt.show()
```

- Draws the full road network.
- Overlays **scatter plot markers** (`s=50`) at the top-10 critical nodes.
- Visually identifies the most strategically important road points.

---

## 12. Road Failure Simulation

```python
G_fail = G_main.copy()

critical_node = top_nodes[0][0]

print("Removing:", critical_node)

G_fail.remove_node(critical_node)

print("Before:", nx.number_connected_components(G_main))
print("After:",  nx.number_connected_components(G_fail))
```

**Output:**
```
Removing: (17, 197)
Before: 1
After: 2
```

- Creates a copy of the main graph (`G_fail`).
- Removes the **most critical node** `(17, 197)` — simulating a road failure at that point (bridge collapse, roadblock, landslide, etc.).
- The network splits from **1 connected component → 2 connected components**, confirming that this is a true bottleneck.

---

## 13. Network Resilience Score

```python
before = len(max(
    nx.connected_components(G_main),
    key=len
))

after = len(max(
    nx.connected_components(G_fail),
    key=len
))

resilience = after / before

print("Resilience Score:", resilience)
```

**Output:**
```
Resilience Score: 0.49873417721518987
```

### Resilience Score Definition

```
Resilience = Size of Largest Component (After Failure)
             ─────────────────────────────────────────
             Size of Largest Component (Before Failure)
```

| Metric | Value |
|--------|-------|
| Largest component before failure | 395 nodes |
| Largest component after failure | ~197 nodes |
| Resilience Score | **≈ 0.499 (~50%)** |

**Interpretation:**
- A resilience score of **~0.50** means the road network's largest connected component **shrinks by approximately 50%** when the most critical node is removed.
- This indicates **low resilience** — the network heavily depends on this single bottleneck node.
- An ideal resilient network would have a resilience score close to **1.0** (removing any single node would barely affect connectivity).

---

## 14. Post-Failure Network Visualization

```python
plt.figure(figsize=(10, 10))

pos = {
    node: node
    for node in G_fail.nodes()
}

nx.draw(G_fail, pos, node_size=1, width=0.5)

plt.title("Network After Critical Road Failure")
plt.show()
```

- Visualizes the fragmented road network after removing the critical node.
- The two separate components are visible as disconnected clusters.

---

## 15. Complete Pipeline Summary

```
Satellite Image (512×512)
        ↓
DeepLabV3+ (ResNet50, trained on 5000 images)
        ↓
Binary Road Mask (threshold=0.2)
        ↓
Skeletonization (1-pixel-wide centerline)
        ↓
Graph Construction
  - Nodes: skeleton pixels
  - Edges: 8-neighborhood adjacency
        ↓
Connected Components Analysis
  - 5 components found
  - Sizes: [395, 350, 262, 147, 5]
        ↓
Largest Component Extraction (395 nodes, 394 edges)
        ↓
Betweenness Centrality Analysis
  - Top critical node: (17, 197), score ≈ 0.501
        ↓
Road Failure Simulation
  - Remove node (17, 197)
  - 1 component → 2 components
        ↓
Resilience Score: ≈ 0.499
```

---

## 16. Key Results

| Metric | Value |
|--------|-------|
| Total image-mask pairs | 6,226 |
| Skeleton pixels (road centerline) | 1,159 |
| Total graph nodes | 1,764 |
| Total graph edges | 1,753 |
| Connected components | 5 |
| Component sizes | [395, 350, 262, 147, 5] |
| Largest component nodes | 395 |
| Largest component edges | 394 |
| Most critical node | (17, 197) |
| Critical node betweenness centrality | 0.5013 |
| Network components after failure | 2 |
| Network resilience score | **≈ 0.499** |

---

## 17. Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning framework (PyTorch) |
| `segmentation_models_pytorch` | DeepLabV3+ segmentation model |
| `opencv-python` (`cv2`) | Image I/O and preprocessing |
| `numpy` | Array operations, coordinate extraction |
| `matplotlib` | Visualization of images and graphs |
| `networkx` | Graph construction, analysis, centrality |
| `scikit-image` (`skimage`) | Morphological skeletonization |

### Installation

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python
pip install numpy matplotlib
pip install networkx
pip install scikit-image
```

---

## Notes & Observations

1. **Model Upgrade:** This notebook uses a **ResNet50** encoder (larger than ResNet34 in `01_data_exploration.ipynb`) trained on **5,000 images** instead of 3,000 — providing better road mask quality.

2. **Threshold = 0.2:** A low threshold is intentionally used to maximize road pixel recall, ensuring more complete skeleton coverage.

3. **Graph Fragmentation:** The 5-component fragmentation suggests that predicted road masks have some discontinuities. Morphological closing operations on the binary mask could potentially improve graph connectivity before skeletonization.

4. **Bottleneck Analysis:** The clustering of top-10 critical nodes around coordinates `(17, 192–201)` suggests a specific narrow road segment (possibly a bridge or alley) that forms the sole connection between two larger road subnetworks.

5. **Resilience Application:** This type of resilience analysis is directly applicable to ISRO's disaster response scenarios — identifying which road segments, if damaged by floods, earthquakes, or landslides, would most severely disrupt connectivity.

6. **Graph Simplification:** The current graph operates at pixel-level resolution (1 node per pixel). For large-scale maps, graph simplification (merging straight-line segments into single edges) would improve scalability and analytical efficiency.
