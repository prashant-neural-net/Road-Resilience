# 🛰️ ISRO Road Network Resilience — Satellite Imagery Analysis

A deep-learning pipeline for **road segmentation**, **topology extraction**, **graph construction**, and **network resilience analysis** from satellite imagery using DeepLabV3+ and NetworkX.

---

## 📌 Project Overview

This project processes satellite images to:
1. **Segment roads** using a fine-tuned DeepLabV3+ model
2. **Extract road topology** via morphological skeletonization and junction detection
3. **Build a road network graph** with 8-connectivity adjacency
4. **Analyse network resilience** using betweenness centrality and failure simulation
5. **Serve results via a FastAPI backend** with endpoints for prediction, routing, and blockage simulation

---

## 🗂️ Project Structure

```
ISRO/
├── 01_data_exploration.ipynb         # Dataset exploration, U-Net prototype, DeepLabV3+ training
├── 03_topology_extraction.ipynb      # Skeletonization, junction detection, connected components
├── 04_graph_construction.ipynb       # Graph building, centrality analysis, resilience scoring
├── 05_graph_intelligence.ipynb       # Advanced graph intelligence
├── backend/
│   ├── main.py                       # FastAPI app (predict, route, blockage simulation)
│   ├── model.py                      # Model loading and inference
│   ├── graph_engine.py               # Graph construction and analysis engine
│   └── requirements.txt             # Python dependencies
├── documentation.md                  # Notebook 01 documentation
├── 04_graph_construction_documentation.md  # Notebook 04 documentation
├── outputs/                          # Generated output images (git-ignored)
└── README.md
```

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| Architecture | DeepLabV3+ |
| Encoder | ResNet-50 |
| Pre-trained weights | ImageNet |
| Input | 512 × 512 RGB satellite image |
| Output | Binary road mask |
| Training images | 5,000 |
| Inference threshold | 0.2 |
| Hardware | Apple MPS / CUDA / CPU |

---

## 🔬 Pipeline

```
Satellite Image (512×512)
        ↓
DeepLabV3+ Inference  →  Binary Road Mask
        ↓
binary_dilation (gap filling)
        ↓
Morphological Skeletonization  →  1-pixel road centerline
        ↓
remove_small_objects  →  Clean Skeleton
        ↓
Convolution-based Junction Detection  →  Road Junctions
        ↓
NetworkX Graph Construction
  Nodes  = skeleton pixels
  Edges  = 8-neighbor adjacency
        ↓
Connected Components Analysis
        ↓
Largest Component  →  Betweenness Centrality
        ↓
Critical Node Identification
        ↓
Failure Simulation  →  Resilience Score
```

---

## 📊 Key Results (Sample Image `63019`)

| Metric | Value |
|--------|-------|
| Road pixels (mask) | 5,696 |
| Skeleton pixels (raw) | 736 |
| Clean skeleton pixels | 454 |
| Junctions detected | 432 |
| Connected components | 4 |
| Most critical node | (17, 197) |
| Betweenness centrality | 0.5013 |
| Resilience score | **≈ 0.499** |

---

## 🚀 FastAPI Backend

### Run the server

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Server status |
| GET | `/graph-stats` | Node/edge counts |
| GET | `/critical-junctions` | Top-10 critical nodes with scores |
| GET | `/resilience` | Network resilience metrics |
| POST | `/predict` | Upload image → road mask + graph analysis |
| POST | `/route` | Find shortest path between two points |
| POST | `/simulate-blockage` | Simulate road failure at a junction |

### Example: Predict road mask

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@satellite_image.jpg"
```

### Example: Simulate blockage

```bash
curl -X POST "http://localhost:8000/simulate-blockage" \
     -H "Content-Type: application/json" \
     -d '{"junction_x": 17, "junction_y": 197}'
```

---

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/isro-road-resilience.git
cd isro-road-resilience

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

### Additional dependencies for notebooks

```bash
pip install jupyter notebook
pip install scikit-image scipy networkx
```

---

## 📦 Dataset

This project uses the **[DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)**.

- **6,226** matched satellite-mask pairs
- Image size: 1024 × 1024 (resized to 512 × 512 for training)
- File naming: `<id>_sat.jpg` / `<id>_mask.png`

> ⚠️ Dataset and model weights are **not included** in this repository (too large for GitHub). Download the dataset from Kaggle and place in `train/`, `test/`, `valid/` directories.

---

## 📁 Model Weights

Download pre-trained weights separately and place in the project root:

| File | Encoder | Training Images |
|------|---------|----------------|
| `deeplabv3_resnet50_5000img.pth` | ResNet-50 | 5,000 |
| `deeplabv3plus_road.pth` | ResNet-34 | 3,000 |

---

## 📚 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Data loading, U-Net baseline, DeepLabV3+ training (10 epochs), IoU/Dice evaluation |
| `03_topology_extraction.ipynb` | Gap filling, skeletonization, junction detection, connected component counting |
| `04_graph_construction.ipynb` | Full graph pipeline, betweenness centrality, critical node analysis, resilience scoring |
| `05_graph_intelligence.ipynb` | Advanced graph-based intelligence and routing |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, segmentation-models-pytorch |
| Image Processing | OpenCV, scikit-image, SciPy |
| Graph Analysis | NetworkX |
| Backend API | FastAPI, Uvicorn |
| Visualisation | Matplotlib |
| Hardware | Apple MPS (M-series) / CUDA |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
