import argparse
import os
from pathlib import Path
import pandas as pd 
from PIL import Image, ExifTags

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("⚠️  pillow-heif not installed. HEIC support disabled.")

# EXIF Tag Constants
TAGS = ExifTags.TAGS
GPSTAGS = ExifTags.GPSTAGS

def _to_float(x):
    """Convert EXIF rational / int / float to float safely."""
    try:
        return float(x)
    except Exception:
        if isinstance(x, tuple) and len(x) == 2 and x[1] != 0:
            return x[0] / x[1]
        return None

def dms_to_decimal(dms, ref):
    """Convert EXIF GPS DMS to decimal degrees."""
    if not dms or len(dms) != 3:
        return None
    deg = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])
    if deg is None or minutes is None or seconds is None:
        return None
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal

def extract_exif_fields(image_path: Path):
    """Extracts GPS metadata from image."""
    image_name = image_path.name
    
    data = {
        "image_name": image_name,
        "Latitude": None,
        "Longitude": None,
    }
    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            if not exif: 
                return data
            gps_info = None
            if hasattr(exif, "get_ifd"):
                try: 
                    gps_info = exif.get_ifd(34853)
                except Exception: 
                    gps_info = None
            if not gps_info:
                raw_gps = exif.get(34853)
                if isinstance(raw_gps, dict): 
                    gps_info = raw_gps
            if not gps_info: 
                return data

            gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            data["Latitude"] = dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            data["Longitude"] = dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            return data
    except Exception:
        return data

def main():
    # Use absolute paths with raw strings for Windows compatibility
    DEFAULT_INPUT = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images")
    DEFAULT_OUTPUT = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\gt.csv")

    parser = argparse.ArgumentParser(description="Extract GPS metadata into gt.csv format.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT, help="Path to images root directory")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Recursively scan for images and assign generic prefixes to subfolders
    images = []
    folder_mapping = {}  # {prefix: actual_folder_path}
    prefix_counter = 1
    
    # Find all subfolders containing images, ignore hidden files
    for subfolder in sorted(input_dir.rglob("*")):
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            folder_images = []
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.heic", "*.HEIC"):
                folder_images.extend([p for p in subfolder.glob(ext) if not p.name.startswith('.')])
            if folder_images:
                prefix = f"f{prefix_counter}_"
                folder_mapping[prefix] = subfolder
                for img_path in folder_images:
                    images.append((img_path, prefix))
                prefix_counter += 1

    # Process all images and extract metadata
    rows = []
    for img_path, prefix in images:
        data = extract_exif_fields(img_path)
        # Replace image_name with prefixed version
        data["image_name"] = prefix + img_path.name
        rows.append(data)
    
    # Create DataFrame with exactly 3 columns: image_name, Latitude, Longitude
    df = pd.DataFrame(rows, columns=["image_name", "Latitude", "Longitude"])
    
    # Remove rows where GPS could not be extracted
    df = df.dropna(subset=["Latitude", "Longitude"])
    
    # Drop duplicates based on image_name to fix 1900 images vs 3000 rows issue
    df = df.drop_duplicates(subset=['image_name'])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    
    # Print summary statistics
    print(f"✅ Success! Saved {len(df)} entries to: {out_path.resolve()}")
    print(f"\nDataset Statistics:")
    print(f"  Total images with GPS: {len(df)}")
    print(f"  Folders mapped: {len(folder_mapping)} (generic prefixes for anonymity)")

if __name__ == "__main__":
    main()