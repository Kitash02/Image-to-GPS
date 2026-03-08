# Image-to-GPS: High-Precision Campus Geolocalization

A rigorous Master's-level deep learning project for predicting precise GPS coordinates (Latitude, Longitude) from campus images using transfer learning and multiple backbone architectures. Engineered for micro-geographic accuracy in a constrained study area (~100m × 30m) with scientific integrity and zero data leakage.

---

## Pre-trained Models & Dataset

**Important:** Due to file size limits, the full Image Dataset, the Ground Truth file (gt.csv), and the Trained Model Weights (.pth & .pkl files) are hosted on the University Shared Drive. Access them via the following link:

**Shared Drive Link:** [Ben-Gurion University Shared Drive - Image-to-GPS Project](https://drive.bgu.ac.il/folder/XXXXXXX) (Placeholder - Replace with actual link)

Download and place the files in the following structure:
```
C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\
├── images/                  # Full image dataset (~1900 images)
├── gt.csv                   # Ground truth with GPS coordinates
└── models/                  # Pre-trained weights (best_model.pth, scaler.pkl)
```

---

## Project Overview

**Objective:** Predict absolute GPS coordinates from RGB images with sub-5-meter accuracy using single-task CNN regression.

**Study Area:** Ben-Gurion University of the Negev, Beer-Sheva, Israel (~100m × 30m geographic footprint)

**Dataset:** ~1900 geotagged images (after duplicate cleanup) across day/night conditions

**Key Innovation:** Rigorous scientific methodology with zero data leakage, micro-geographic precision, and folder anonymity through generic prefixing.

---

## Scientific Core Features

### 1. Zero Data Leakage: "Split-then-Fit" Logic

**Problem:** Standard ML pipelines fit the scaler on the entire dataset, then split it. This causes **data leakage** — validation and test statistics influence the scaling parameters learned during training.

**Solution:** We implement strict temporal separation:

```
1. Load raw dataset (no scaling applied)
2. Split indices into Train (70%) / Val (15%) / Test (15%)
3. Fit MinMaxScaler ONLY on training split coordinates
   scaler.fit(coords[train_indices])
4. Apply fitted scaler to Val and Test (never refits)
5. Save scaler.pkl for inference (scaler learns nothing from Val/Test)
```

**Implementation Details:**
- Location: `src/train.py` lines 95-110
- The test set remains truly blind — model never "sees" its statistical properties during training
- Scaler parameters: `feature_range=(-1, 1)`, fitted on `np.float64` coordinates
- Validation loss selection is based on scaled targets (consistent with training), but scaler remains frozen

**Scientific Impact:** Ensures reported test metrics are not biased by training data normalization.

---

### 2. Micro-Geographic Precision: WGS-84 & Double Precision

**Problem:** Small study area (100m × 30m) requires sub-meter accuracy. Standard single-precision (float32) Haversine loses precision in our range.

**Solution:** High-precision Haversine formula with WGS-84 ellipsoid:

$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\lat}{2}\right) + \cos(\lat_1)\cos(\lat_2)\sin^2\left(\frac{\Delta\lon}{2}\right)}\right)$$

Where:
- $R = 6378137.0$ meters (WGS-84 ellipsoid equatorial radius at ~31°N latitude)
- All calculations use `np.float64` (double precision)
- No intermediate rounding before final mean/median computation
- Input: degrees; Output: meters

**Implementation:**
```python
# src/metrics.py: haversine_distance()
lat1 = np.asarray(lat1, dtype=np.float64)  # Force double precision
lon1 = np.asarray(lon1, dtype=np.float64)
# ... Haversine formula in radians ...
distance = R * c  # R = 6378137.0 meters
```

**Accuracy Guarantee:** Sub-centimeter precision for distances in the 0-100m range.

---

### 3. Folder Anonymity: Generic Prefixing System

**Problem:** Original folder structure contains metadata (e.g., "Day", "Night"). The model should predict GPS based on **pixels only**, not folder context. Additionally, duplicate filenames across folders cause collisions (IMG_0001.jpg appears in both Day/ and Night/).

**Solution:** Generic prefix system neutralizes folder-name bias while preventing collisions:

**Before Prefixing:**
```
data/images/
├── Day/
│   ├── IMG_0001.jpg     ← Same name in multiple folders
│   ├── IMG_0002.jpg
│   └── ...
├── Night/
│   ├── IMG_0001.jpg     ← Collision!
│   ├── IMG_0003.jpg
│   └── ...
```

**After Prefixing:**
```
Master CSV: gt.csv
image_name,Latitude,Longitude
f1_IMG_0001.jpg,31.263456,-34.784123
f1_IMG_0002.jpg,31.263789,-34.784456
f2_IMG_0001.jpg,31.264012,-34.784789  ← No collision!
f2_IMG_0003.jpg,31.264345,-34.785012
```

**Mapping Logic:**
```
f1_ → Day/           (first subfolder scanned)
f2_ → Night/         (second subfolder scanned)
f3_ → Other/         (any additional folders)
```

**Benefits:**
- ✅ Eliminates filename collisions
- ✅ Removes metadata bias (model is "blind" to Day/Night labels)
- ✅ Forces model to learn GPS prediction from image content alone
- ✅ Maintains scientific objectivity

**Implementation:**
- Location: `src/generate_gt_csv.py` (prefixing during CSV creation)
- Resolution: `src/dataset.py` (internal mapping dictionary resolves prefixed names back to disk paths during data loading)

---

### 4. Internal Path Mapping

**Problem:** CSV stores prefixed names (e.g., "f1_IMG_0001.jpg"), but actual files exist on disk under original folder structure (e.g., "Day/IMG_0001.jpg").

**Solution:** Dataset class maintains internal mapping dictionary:

```python
# src/dataset.py: __init__()
self.path_mapping = {}
for prefixed_name in df["image_name"]:
    prefix = prefixed_name.split("_")[0] + "_"  # Extract f1_, f2_, etc.
    original_name = prefixed_name[len(prefix):]  # Remove prefix
    
    # Search subfolders to resolve actual path
    for subfolder in images_root.rglob("*"):
        candidate = subfolder / original_name
        if candidate.exists():
            self.path_mapping[prefixed_name] = candidate
            break
```

**Usage during inference:**
```python
# src/dataset.py: __getitem__()
image_path = self.path_mapping.get(prefixed_name, fallback)
image = Image.open(image_path).convert("RGB")
```

**Result:** Model sees prefixed name in outputs but loads correct image from disk.

---

## Project Structure

```
Image-to-GPS/
├── src/
│   ├── __init__.py
│   ├── generate_gt_csv.py       # Recursive scan, prefixing, EXIF GPS extraction
│   ├── dataset.py               # GPSDataset: augmentations, internal path mapping
│   ├── model.py                 # Multi-backbone CNN regression (ResNet/EfficientNet/ConvNeXt)
│   ├── metrics.py               # Haversine (float64, WGS-84), MDE, MedDE, Acc@threshold
│   ├── train.py                 # Training: split→fit scaler→train/val/test, timestamped outputs
│   └── inference.py             # predict_gps API: uint8→float32 GPS prediction
├── models/
│   ├── resnet50_20260305_1430/      # Timestamped experiment (YYYYMMDD_HHMM)
│   │   ├── best_model.pth           # Best weights (lowest validation loss)
│   │   ├── scaler.pkl               # MinMaxScaler fitted on training split
│   │   ├── config.json              # Training hyperparameters
│   │   ├── history.csv              # Per-epoch metrics
│   │   ├── test_results.csv         # Per-image predictions + errors
│   │   └── test_results.xlsx        # Same, Excel format
│   ├── efficientnet_b0_20260305_1530/
│   └── convnext_tiny_20260305_1630/
├── data/
│   └── dataset_root/
│       ├── images/                  # (Download from Shared Drive)
│       │   ├── Day/
│       │   │   ├── IMG_0001.jpg     (with GPS EXIF)
│       │   │   └── ...
│       │   ├── Night/
│       │   │   ├── IMG_0001.jpg     (with GPS EXIF)
│       │   │   └── ...
│       │   └── [other folders]/
│       └── gt.csv                   # Ground truth (Download from Shared Drive)
├── run_experiments.py               # Automated multi-backbone runner
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File | Purpose |
|------|---------|
| **src/generate_gt_csv.py** | Recursively scans image subfolders, assigns generic prefixes (f1_, f2_, ...), extracts GPS from EXIF metadata, creates gt.csv with ~1900 images after duplicate cleanup. |
| **src/dataset.py** | `GPSDataset` class: loads images, applies day/night augmentations (ColorJitter, GaussianBlur, RandomGrayscale, etc.), maintains internal path mapping for prefixed-name resolution, returns (image, GPS) pairs. |
| **src/model.py** | `GPSRegressionModel`: Single-task architecture with ResNet50/EfficientNet/ConvNeXt backbone + regression head outputting 2D GPS predictions. Uses `weights="DEFAULT"` for pre-trained initialization. |
| **src/metrics.py** | High-precision Haversine distance (float64, WGS-84 R=6378137.0m), Mean Distance Error (MDE), Median Distance Error (MedDE), Accuracy@threshold calculations. |
| **src/train.py** | Main training pipeline: implements split→fit scaler logic, creates timestamped experiment folders, trains with early stopping, performs batch-based TTA on test set, saves history and per-image results. |
| **src/inference.py** | `predict_gps` API: Accepts uint8 RGB numpy array, returns float32 [Latitude, Longitude]. Loads model/scaler globally for fast inference. |
| **run_experiments.py** | Runs `train.py` sequentially for ResNet50, EfficientNet-B0, and ConvNeXt-Tiny with consistent hyperparameters. |

---

## Setup & Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (optional; CPU fully supported)
- ~20GB disk space for models and results
- 8GB+ RAM

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download Dataset & Models

Download the full dataset, gt.csv, and pre-trained weights from the University Shared Drive (link above) and place them in the specified directories.

### Step 3: Verify Paths

Default paths are hardcoded in the scripts. Update if your directory structure differs:

**generate_gt_csv.py (lines 57-58):**
```python
DEFAULT_INPUT = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images")
DEFAULT_OUTPUT = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\gt.csv")
```

**train.py (lines 69-77):**
```python
--csv_path: r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\gt.csv"
--images_dir: r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images"
```

---

## Usage Instructions

### Running Inference with predict_gps

To evaluate a single image using the trained model:

```python
import numpy as np
from src.inference import load_model_and_scaler, predict_gps

# Load model and scaler once (global)
load_model_and_scaler(
    model_path="models/resnet50_20260305_1430/best_model.pth",
    scaler_path="models/resnet50_20260305_1430/scaler.pkl"
)

# Load an image as uint8 RGB numpy array (H, W, 3)
image = np.load("path/to/image.npy")  # Or use cv2.imread() and convert

# Predict GPS coordinates
gps_coords = predict_gps(image)  # Returns float32 array [Latitude, Longitude]
print(f"Predicted GPS: {gps_coords}")
```

**API Compliance:**
- Input: `np.ndarray` uint8 RGB (0-255)
- Output: `np.ndarray` float32 (2,) [Lat, Lon]
- No prints or interactive inputs during execution

---

## Execution Workflow

### Step 1: Generate Ground Truth CSV

```bash
python src/generate_gt_csv.py
```

**Output:** gt.csv with prefixed image names and GPS coordinates.

### Step 2: Run Training Experiments

```bash
python run_experiments.py
```

**Output:** Timestamped folders with trained models and results.

### Step 3: Evaluate with predict_gps

Use the API as shown above for individual image predictions.

---

## Metrics: Mean & Median Distance Error

### Mean Distance Error (MDE)

$$\text{MDE} = \frac{1}{N} \sum_{i=1}^{N} d_i$$

**Purpose:** Overall average localization accuracy  
**Interpretation:** "On average, predictions are within MDE meters"

### Median Distance Error (MedDE)

$$\text{MedDE} = \text{median}(d_1, d_2, \ldots, d_N)$$

**Purpose:** Robustness metric (unaffected by outliers)  
**Interpretation:** "50% of predictions are within MedDE meters"  
**Why Critical:** In small study areas, a few worst-case errors can inflate MDE. MedDE reveals typical performance.

### Accuracy @ Threshold

$$\text{Acc@}t = \frac{\#\{i : d_i \leq t \text{ meters}\}}{N} \times 100\%$$

**Common thresholds:**
- Acc@5m: Percentage of predictions within 5 meters
- Acc@10m: Percentage of predictions within 10 meters

---

## Target Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Mean Distance Error (MDE)** | <5m | Project specification |
| **Median Distance Error (MedDE)** | <2.5m | Robust typical performance |
| **Accuracy @ 5m** | >80% | Practical localization utility |

---

## References

- **Haversine Formula:** https://en.wikipedia.org/wiki/Haversine_formula
- **WGS-84 Geodetic System:** https://en.wikipedia.org/wiki/World_Geodetic_System
- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling" (ICML 2019)
- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s" (CVPR 2022)

---

## License

Academic use only. Part of BGU (Ben-Gurion University) graduate coursework.