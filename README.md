# Image-to-GPS Regression (Campus Localization)

This repository predicts GPS coordinates (latitude, longitude) from a single campus image.
It includes dataset generation from EXIF, training, fine-tuning, evaluation, and single-image inference.

## Reproducibility First

This README is written to satisfy the requirement: "README.md must explain how to reproduce results".
All commands below are runnable from the repository root and use CLI arguments instead of hardcoded paths.

## Environment Setup

Requirements:
- Python 3.9+
- Optional CUDA GPU (CPU also works)

Install:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Repository Layout

```text
Image-to-GPS-main/
  src/
    generate_gt_csv.py
    dataset.py
    model.py
    metrics.py
    train.py
    inference.py
  data/
    photos_gt.csv
    uni_gt.csv
    report_run_summary.csv
    report_finetune_deltas.csv
  models/
    ... experiment folders ...
  predict_image.py
  run_experiments.py
  README.md
```

## 1) Build Ground Truth CSV From Images

Generate CSV from EXIF GPS (JPG/JPEG/PNG/HEIC supported):

```bash
python src/generate_gt_csv.py --input_dir "<IMAGES_ROOT>" --output data/photos_gt.csv
```

Optional extra day images:

```bash
python src/generate_gt_csv.py --input_dir "<EXTRA_DAY_IMAGES_ROOT>" --output data/uni_gt.csv
```

Expected CSV columns:
- `image_name`
- `Latitude`
- `Longitude`
- `relative_path` (when available)

Optional (all labeled data in one CSV):
```bash
python -c "import pandas as pd; a=pd.read_csv('data/photos_gt.csv'); b=pd.read_csv('data/uni_gt.csv'); pd.concat([a,b], ignore_index=True).drop_duplicates(subset=['image_name']).to_csv('data/all_labeled_gt.csv', index=False)"
```

## 2) Reproduce Baseline Backbone Experiments

Use the same split file across backbones for fair comparison:

```bash
python src/train.py --backbone resnet50 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
python src/train.py --backbone efficientnet_b0 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
python src/train.py --backbone convnext_tiny --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
```

Notes:
- `train.py` creates a timestamped experiment folder.
- Each folder contains `best_model.pth`, `scaler*.pkl`, `history.csv`, `test_results.csv`, `split_manifest.json`.
- Scaler is fit on train split only (split-then-fit, no leakage).

## 3) Reproduce Fine-Tuning From Best Baseline Model

If your best baseline is `models/resnet50_20260305_2009`:

### 3.1 Baseline-on-same-split evaluation (before fine-tuning)

```bash
python src/train.py --backbone resnet50 --epochs 0 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models/finetune_photos_runs --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/finetune_shared_split_photos_seed42.json --resume_checkpoint "models/resnet50_20260305_2009/best_model.pth" --resume_scaler_path "models/resnet50_20260305_2009/scaler (1).pkl" --use_pretrained 0
```

### 3.2 Fine-tuning run (after)

```bash
python src/train.py --backbone resnet50 --epochs 12 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models/finetune_photos_runs --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/finetune_shared_split_photos_seed42.json --resume_checkpoint "models/resnet50_20260305_2009/best_model.pth" --resume_scaler_path "models/resnet50_20260305_2009/scaler (1).pkl" --use_pretrained 0
```

## 4) Check Reported Results

Saved summary files:
- `data/report_run_summary.csv`
- `data/report_finetune_deltas.csv`

Current key fine-tuning delta (ft1272):
- Test MDE improvement: `-1.5285 m`
- Test MedDE improvement: `-1.6044 m`
- Accuracy@5m improvement: `+10.00 pp`
- Accuracy@10m improvement: `+6.32 pp`

## 5) Single-Image Prediction (Updated `predict_image.py`)

### Quick use (auto-discover latest valid model)

```bash
python predict_image.py --image "<PATH_TO_IMAGE>"
```

### Use a specific model directory

```bash
python predict_image.py --image "<PATH_TO_IMAGE>" --model-dir "models/finetune_photos_runs/resnet50_20260315_171307"
```

### Include true GPS from CSV (if EXIF is missing)

```bash
python predict_image.py --image "<PATH_TO_IMAGE>" --model-dir "models/finetune_photos_runs/resnet50_20260315_171307" --truth-csv data/photos_gt.csv
```

Output includes:
- Predicted latitude/longitude
- True latitude/longitude (from EXIF or CSV when available)
- Error in meters (Haversine)

## 6) Programmatic Inference API

```python
import numpy as np
from PIL import Image
from src.inference import load_model_and_scaler, predict_gps

load_model_and_scaler(
    "models/finetune_photos_runs/resnet50_20260315_171307/best_model.pth",
    "models/finetune_photos_runs/resnet50_20260315_171307/scaler.pkl",
)

image_np = np.array(Image.open("path/to/image.jpg").convert("RGB"), dtype=np.uint8)
pred = predict_gps(image_np)
print(pred)  # [lat, lon]
```

Input contract:
- `np.ndarray`, `uint8`, shape `(H, W, 3)`, RGB

Output contract:
- `np.ndarray`, `float32`, shape `(2,)` as `[Latitude, Longitude]`

## 7) Important Notes

- `run_experiments.py` currently contains machine-specific dataset defaults. For strict reproducibility, use `src/train.py` commands above with explicit `--csv_path` and `--images_dir`.
- HEIC support requires `pillow-heif` (already listed in `requirements.txt`).
- For report consistency, keep seed/split file fixed when comparing backbones or before-vs-after fine-tuning.

## License

Academic project use.
