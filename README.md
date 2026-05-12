# Image-to-GPS Regression (Campus Localization)

This repository predicts GPS coordinates (latitude, longitude) from a single campus image.
It includes dataset generation from EXIF, training multiple CNN backbones, fine-tuning
from a best-baseline checkpoint, deterministic evaluation, and single-image inference
(including HEIC support).

## Dataset & Model Weights

Due to file size constraints, the full image dataset and trained model weights (`.pth`, `.pkl`)
are hosted on Google Drive:

**[Google Drive — Dataset & Models](https://drive.google.com/drive/folders/1a57ViDiH0pJI-2yFKobmQAdQr0JqUISF)**

Download and place files under:
```
dataset_root/
├── images/       ← full image dataset (~1,900 images)
└── gt.csv        ← ground truth (image_name, Latitude, Longitude)
```
Trained model weights go under `models/finetune_photos_runs/resnet50_20260315_171307/`.

## Reproducibility First

This README is written to satisfy the course requirement *"README.md must explain how
to reproduce results"*. All commands below are runnable from the repository root and
use explicit CLI arguments instead of hardcoded paths. A fixed random seed and a
shared split manifest are used so that all backbones are evaluated on the exact same
test images.

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
Image-to-GPS/
  src/
    generate_gt_csv.py       # Recursive scan, generic-prefix anonymization, EXIF GPS extraction
    dataset.py               # GPSDataset: 3-tier path resolution, augmentations
    model.py                 # Multi-backbone regression head (use_pretrained flag)
    metrics.py               # WGS-84 Haversine in float64 (unchanged across branches)
    train.py                 # split -> fit scaler on train only -> train/val/test, shared_split_path, resume_checkpoint
    inference.py             # predict_gps API (uint8 HxWx3 -> float32 [lat,lon])
  data/
    photos_gt.csv            # Primary ground-truth CSV used by the final fine-tune
    uni_gt.csv               # Extra day images ground-truth
    finetune_dataset.csv     # Curated subset used for fine-tune stage
    report_run_summary.csv   # Per-run test metrics (headline numbers)
    report_finetune_deltas.csv     # Before / after fine-tune deltas
    report_day_night_labeled_counts.csv
  models/
    resnet50_20260305_2009/              # 100-epoch ResNet50 baseline
    efficientnet_b0_20260306_2115/       # 100-epoch EfficientNet-B0 baseline
    convnext_tiny_20260307_1319/         # 100-epoch ConvNeXt-Tiny baseline
    finetune_runs/                        # Earlier fine-tune pair (ft245)
    finetune_photos_runs/                 # Final fine-tune pair (ft1272)
    finetune_shared_split_seed42.json
    finetune_shared_split_photos_seed42.json
    archive/pre_shared_split/             # Historical early runs retained as evidence (see README there)
  predict_image.py
  run_experiments.py
  README.md
```

## 1) Build Ground Truth CSV From Images

Generate CSV from EXIF GPS (JPG / JPEG / PNG / HEIC supported):

```bash
python src/generate_gt_csv.py --input_dir "<IMAGES_ROOT>" --output data/photos_gt.csv
```

Optional extra day images:

```bash
python src/generate_gt_csv.py --input_dir "<EXTRA_DAY_IMAGES_ROOT>" --output data/uni_gt.csv
```

Expected CSV columns:

- `image_name` (generic-prefixed, e.g. `f1_IMG_0001.jpg`)
- `Latitude`
- `Longitude`
- `relative_path` (when available, used as the preferred path key in `dataset.py`)

Optional merge of all labeled data:

```bash
python -c "import pandas as pd; a=pd.read_csv('data/photos_gt.csv'); b=pd.read_csv('data/uni_gt.csv'); pd.concat([a,b], ignore_index=True).drop_duplicates(subset=['image_name']).to_csv('data/all_labeled_gt.csv', index=False)"
```

## 2) Reproduce Baseline Backbone Experiments

Use the same split manifest across backbones so the test set is literally the same:

```bash
python src/train.py --backbone resnet50        --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
python src/train.py --backbone efficientnet_b0 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
python src/train.py --backbone convnext_tiny   --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/shared_split_seed42_v15_t15.json
```

Notes:

- `train.py` creates a timestamped experiment folder `<backbone>_<YYYYMMDD_HHMMSS>` under `--model_dir`.
- Each folder contains `best_model.pth`, `scaler*.pkl`, `history.csv`, `test_results.csv`, `split_manifest.json`.
- The scaler is fit on the train split **only** (split-then-fit, no leakage — see "Scientific Design" below).

## 3) Reproduce Fine-Tuning From Best Baseline Model

Given the best baseline run at `models/resnet50_20260305_2009/`:

### 3.1 Before fine-tuning (baseline evaluated on the fine-tune split)

```bash
python src/train.py --backbone resnet50 --epochs 0 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models/finetune_photos_runs --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/finetune_shared_split_photos_seed42.json --resume_checkpoint "models/resnet50_20260305_2009/best_model.pth" --resume_scaler_path "models/resnet50_20260305_2009/scaler (1).pkl" --use_pretrained 0
```

### 3.2 Fine-tune run (after)

```bash
python src/train.py --backbone resnet50 --epochs 12 --csv_path data/photos_gt.csv --images_dir "<IMAGES_ROOT>" --model_dir models/finetune_photos_runs --seed 42 --val_split 0.15 --test_split 0.15 --shared_split_path models/finetune_shared_split_photos_seed42.json --resume_checkpoint "models/resnet50_20260305_2009/best_model.pth" --resume_scaler_path "models/resnet50_20260305_2009/scaler (1).pkl" --use_pretrained 0
```

(The reported final run used 5 effective epochs before early-stopping / best-val selection.)

## 4) Achieved Performance (measured, not target)

Headline test metrics (from `data/report_run_summary.csv`):

| Run | N_test | Test MDE (m) | Test MedDE (m) | Acc@5m | Acc@10m |
|------|-------:|-------------:|---------------:|-------:|--------:|
| ResNet50 (100 ep, baseline) | 231 | 14.06 | 10.58 | 17.32% | 47.19% |
| EfficientNet-B0 (72 ep, early-stopped) | 231 | 17.59 | 14.10 | 12.99% | 35.50% |
| ConvNeXt-Tiny (59 ep, early-stopped) | 231 | 15.87 | 11.49 | 17.32% | 42.86% |
| Final fine-tune **ft1272** (before FT) | 190 | 9.17 | 5.65 | 45.26% | 74.21% |
| Final fine-tune **ft1272** (after FT, 5 ep) | 190 | **7.64** | **4.04** | **55.26%** | **80.53%** |

Fine-tune delta (from `data/report_finetune_deltas.csv`):

- Test MDE: **-1.5285 m**
- Test MedDE: **-1.6044 m**
- Acc@5m: **+10.00 pp**
- Acc@10m: **+6.32 pp**

These are the numbers that should be cited in the report and the slide deck. No
"sub-meter" or "< 5 m MDE" claim is supported by the current pipeline on the full
test split. Individual images can land in that range (visible in
`test_results.csv`), but the aggregate is 7.64 m mean / 4.04 m median.

## 5) Single-Image Prediction (`predict_image.py`)

### Auto-discover the latest valid model

```bash
python predict_image.py --image "<PATH_TO_IMAGE>"
```

### Use a specific model directory

```bash
python predict_image.py --image "<PATH_TO_IMAGE>" --model-dir "models/finetune_photos_runs/resnet50_20260315_171307"
```

### Include true GPS from CSV (when EXIF is missing)

```bash
python predict_image.py --image "<PATH_TO_IMAGE>" --model-dir "models/finetune_photos_runs/resnet50_20260315_171307" --truth-csv data/photos_gt.csv
```

Output includes predicted latitude / longitude, true latitude / longitude (from EXIF
or CSV when available), and Haversine error in meters.

## 6) Programmatic Inference API (course-required signature)

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

Input contract: `np.ndarray`, `uint8`, shape `(H, W, 3)`, RGB.

Output contract: `np.ndarray`, `float32`, shape `(2,)` as `[Latitude, Longitude]`.

---

## 7) Scientific Design

### 7.1 Split-then-Fit (no data leakage)

Naive pipelines fit a scaler on the entire dataset and split afterwards, which leaks
validation and test statistics into the scaler the model is trained against. This
repository splits first, then fits:

```text
1. Load dataset without any scaling (GPSDataset(..., fit_scaler=False)).
2. Build index split from --seed (or load from --shared_split_path if given):
     train_indices, val_indices, test_indices
3. Fit MinMaxScaler(feature_range=(-1, 1)) on coords[train_indices] only.
4. Apply the already-fit scaler to val and test (never refit).
5. Persist scaler as scaler*.pkl inside the experiment folder for inference.
```

The code path is `src/train.py` (see `build_split_indices`, then the scaler fit block
that runs on train indices). Consequence: reported test metrics are not biased by
having seen val/test coordinates during normalization.

### 7.2 Metrics: WGS-84 Haversine, MDE, MedDE, Acc@t

Distances are computed in float64 so rounding does not dominate over small
intra-campus displacements:

$$d = 2R \arcsin\left(\sqrt{\sin^2\!\left(\tfrac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\!\left(\tfrac{\Delta\lambda}{2}\right)}\right)$$

with $R = 6{,}378{,}137$ m (WGS-84 equatorial radius), inputs in degrees, output
in meters.

Aggregate metrics on the test split:

$$\text{MDE} = \tfrac{1}{N}\sum_{i=1}^{N} d_i \qquad
\text{MedDE} = \operatorname{median}(d_1, \ldots, d_N) \qquad
\text{Acc@}t = \frac{|\{i : d_i \le t\}|}{N} \times 100\%$$

In this report `Acc@5m` is treated as the *effective perfect-localization* metric
because the ground truth itself came from consumer-phone EXIF GPS, whose
quantization is of that order.

### 7.3 Generic-Prefix Folder Anonymity

Original folder structure contains metadata (e.g. `Day/`, `Night/`) and duplicate
filenames across subfolders. To prevent any folder-name signal from leaking into
the model, and to avoid filename collisions, `src/generate_gt_csv.py` assigns a
generic prefix (`f1_`, `f2_`, `f3_`, ...) to every image while building the CSV:

```text
Before:  Day/IMG_0001.jpg            collides with Night/IMG_0001.jpg
After:   f1_IMG_0001.jpg (from Day), f2_IMG_0001.jpg (from Night)    -> disjoint
```

`src/dataset.py` resolves these prefixed names back to disk using a 3-tier path
resolver:

1. If the CSV has a `relative_path` column, that path is preferred.
2. Otherwise, a recomputed `prefix -> folder` mapping (same sorted-recursive scan as
   `generate_gt_csv.py`) is used to locate the file.
3. As a last resort, a unique basename match under `images_root` is accepted.
   Ambiguous or missing entries raise `FileNotFoundError` - the dataset never
   silently points at the wrong file.

## 8) Archive Folder (Historical Evidence)

The folder `models/archive/pre_shared_split/` preserves three earlier runs
(`resnet50_early/`, `efficientnet_b0_early/`, `convnext_tiny_early/`) plus an
earlier `config.json` that documents a short-lived `gps_weight=0.8 /
area_weight=0.2` composite-loss experiment that was later dropped in favour of
pure Huber-on-coordinates. Their `failure_summary.txt` files include a
per-device (iPhone / Android) breakdown used in the report's phone-robustness
discussion. See `models/archive/pre_shared_split/README.md` for details.

Do not use these as the current benchmark; they predate the deterministic
`--seed` / `--shared_split_path` protocol and therefore are not strictly
cross-backbone fair.

## 9) Important Notes

- `run_experiments.py` contains machine-specific dataset defaults (hardcoded
  Windows paths). For strict reproducibility, prefer the `src/train.py` commands
  above with explicit `--csv_path` and `--images_dir`.
- HEIC support requires `pillow-heif` (listed in `requirements.txt`). Both
  training and inference register the HEIC opener on import.
- For report consistency, keep `--seed 42` and the shared-split manifest fixed
  when comparing backbones or before-vs-after fine-tuning.

## References

- Haversine formula: <https://en.wikipedia.org/wiki/Haversine_formula>
- WGS-84 geodetic system: <https://en.wikipedia.org/wiki/World_Geodetic_System>
- ResNet: He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016)
- EfficientNet: Tan & Le, *EfficientNet: Rethinking Model Scaling* (ICML 2019)
- ConvNeXt: Liu et al., *A ConvNet for the 2020s* (CVPR 2022)

## License

Academic use only. Part of BGU graduate coursework.
