import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ExifTags

# Register HEIC support when available.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


GPS_IFD_TAG = 34853
GPS_TAGS = ExifTags.GPSTAGS


def _to_float(value):
    try:
        return float(value)
    except Exception:
        if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
            return value[0] / value[1]
    return None


def _dms_to_decimal(dms, ref):
    if not dms or len(dms) != 3:
        return None
    deg = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])
    if deg is None or minutes is None or seconds is None:
        return None
    value = deg + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        value = -value
    return value


def _extract_true_gps_from_exif(image_path: Path):
    with Image.open(image_path) as image:
        exif = image.getexif()
        if not exif:
            return None, None

        gps_info = None
        if hasattr(exif, "get_ifd"):
            try:
                gps_info = exif.get_ifd(GPS_IFD_TAG)
            except Exception:
                gps_info = None
        if not gps_info:
            raw = exif.get(GPS_IFD_TAG)
            if isinstance(raw, dict):
                gps_info = raw
        if not gps_info:
            return None, None

        gps = {GPS_TAGS.get(k, k): v for k, v in gps_info.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
        return lat, lon


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6378137.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat * 0.5) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * (math.sin(dlon * 0.5) ** 2)
    )
    c = 2.0 * math.asin(math.sqrt(a))
    return radius * c


def _lookup_true_gps_from_csv(image_path: Path, truth_csv: Path):
    """
    Try to find ground-truth GPS in CSV using either:
    - relative_path (preferred)
    - image_name (fallback)
    """
    if not truth_csv.exists():
        return None, None

    image_name = image_path.name
    image_path_norm = str(image_path).replace("\\", "/")

    with open(truth_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None, None

    # 1) Match by relative_path suffix if present.
    for row in rows:
        rel = row.get("relative_path")
        if not rel:
            continue
        rel_norm = rel.replace("\\", "/")
        if image_path_norm.endswith(rel_norm):
            try:
                return float(row["Latitude"]), float(row["Longitude"])
            except Exception:
                return None, None

    # 2) Match by image_name as fallback.
    matches = [r for r in rows if (r.get("image_name") or "").strip() == image_name]
    if len(matches) == 1:
        row = matches[0]
        try:
            return float(row["Latitude"]), float(row["Longitude"])
        except Exception:
            return None, None

    return None, None


def _find_scaler_in_dir(model_dir: Path) -> Path:
    preferred = model_dir / "scaler.pkl"
    if preferred.exists():
        return preferred

    pkl_files = sorted(model_dir.glob("*.pkl"))
    if len(pkl_files) == 1:
        return pkl_files[0]
    if len(pkl_files) > 1:
        # Keep deterministic behavior while preferring scaler-like names.
        scaler_like = [p for p in pkl_files if "scaler" in p.name.lower()]
        if scaler_like:
            return sorted(scaler_like)[0]
        return pkl_files[0]

    raise FileNotFoundError(f"No scaler .pkl file found in {model_dir}")


def _discover_latest_model_dir(models_root: Path) -> Path:
    if not models_root.exists():
        raise FileNotFoundError(f"Models directory not found: {models_root}")

    candidates = []
    for best_model in models_root.rglob("best_model.pth"):
        model_dir = best_model.parent
        try:
            _find_scaler_in_dir(model_dir)
        except FileNotFoundError:
            continue
        candidates.append(model_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No valid model directory found under {models_root} "
            "(must contain best_model.pth and a scaler .pkl)."
        )

    # Pick the most recently modified model checkpoint.
    return max(candidates, key=lambda p: (p / "best_model.pth").stat().st_mtime)


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict GPS (lat/lon) for one image using your fine-tuned model."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (absolute or relative to repository root).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Optional model folder containing best_model.pth and scaler .pkl. "
            "If omitted, latest valid model under --models-root is used."
        ),
    )
    parser.add_argument(
        "--models-root",
        default="models",
        help="Root folder for model auto-discovery (default: models).",
    )
    parser.add_argument(
        "--truth-csv",
        default=None,
        help=(
            "Optional CSV for true GPS lookup (columns: Latitude, Longitude and "
            "either relative_path or image_name). If omitted, EXIF GPS is used."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        import inference
    except Exception as exc:
        raise RuntimeError(f"Failed importing src/inference.py: {exc}") from exc

    image_path = _resolve_path(args.image, repo_root)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.model_dir is not None:
        model_dir = _resolve_path(args.model_dir, repo_root)
    else:
        models_root = _resolve_path(args.models_root, repo_root)
        model_dir = _discover_latest_model_dir(models_root)

    best_model_path = model_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")
    scaler_path = _find_scaler_in_dir(model_dir)

    # Load image and run prediction.
    image_np = np.array(Image.open(image_path).convert("RGB"))
    inference.load_model_and_scaler(str(best_model_path), str(scaler_path))
    pred = inference.predict_gps(image_np)

    print(f"Image: {image_path}")
    print(f"Model dir: {model_dir}")
    print(f"Checkpoint: {best_model_path.name}")
    print(f"Scaler: {scaler_path.name}")
    print(f"Predicted latitude: {float(pred[0]):.8f}")
    print(f"Predicted longitude: {float(pred[1]):.8f}")

    true_lat, true_lon = _extract_true_gps_from_exif(image_path)
    if (true_lat is None or true_lon is None) and args.truth_csv is not None:
        truth_csv = _resolve_path(args.truth_csv, repo_root)
        true_lat, true_lon = _lookup_true_gps_from_csv(image_path, truth_csv)
        if true_lat is not None and true_lon is not None:
            print(f"True GPS source: CSV ({truth_csv})")
    elif true_lat is not None and true_lon is not None:
        print("True GPS source: EXIF")

    if true_lat is not None and true_lon is not None:
        error_m = _haversine_meters(float(pred[0]), float(pred[1]), true_lat, true_lon)
        print(f"True latitude: {true_lat:.8f}")
        print(f"True longitude: {true_lon:.8f}")
        print(f"Error (meters): {error_m:.4f}")
    else:
        print("True GPS: not found in EXIF metadata for this image.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
