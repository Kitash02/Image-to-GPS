import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def run_experiment(
    backbone: str,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    shared_split_path: Optional[Path] = None,
) -> bool:
    """Run a single training experiment."""
    print(f"\n{'=' * 70}")
    print(f"Starting experiment: {backbone}")
    print(f"{'=' * 70}\n")

    csv_path = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\campus_gps_master_dataset.csv")
    images_dir = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images")

    if not csv_path.exists():
        print(f"[ERROR] CSV file not found at {csv_path}")
        print("        Run: python src/generate_gt_csv.py")
        return False
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found at {images_dir}")
        return False

    cmd = [
        sys.executable,
        "src/train.py",
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--val_split", str(val_split),
        "--test_split", str(test_split),
        "--csv_path", str(csv_path),
        "--images_dir", str(images_dir),
    ]
    if shared_split_path is not None:
        cmd.extend(["--shared_split_path", str(shared_split_path)])

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print(f"\n[OK] {backbone} experiment completed successfully.")

        model_dir = Path(__file__).parent / "models"
        backbone_dirs = [d for d in model_dir.glob(f"{backbone}_*") if d.is_dir()]
        if not backbone_dirs:
            print(f"No {backbone} directories found in models/.")
            return True

        latest_dir = max(backbone_dirs, key=lambda x: x.stat().st_mtime)
        test_csv = latest_dir / "test_results.csv"
        if not test_csv.exists():
            print(f"No test_results.csv found for {backbone} in {latest_dir}")
            return True

        try:
            df = pd.read_csv(test_csv)
            if "error_meters" in df.columns and len(df) > 0:
                medde = float(df["error_meters"].median())
                print(f"{backbone} Test Median Distance Error (MedDE): {medde:.2f} meters")
            else:
                print(f"{backbone} test_results.csv found but no 'error_meters' column or it is empty.")
        except Exception:
            print(f"Could not read {test_csv}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {backbone} experiment failed with error code: {e.returncode}")
        return False


def main() -> None:
    """Run all experiments sequentially with a shared deterministic split."""
    backbones = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    epochs = 100
    batch_size = 32
    seed = 42
    val_split = 0.15
    test_split = 0.15
    shared_split_path = Path(__file__).parent / "models" / f"shared_split_seed{seed}_v15_t15.json"

    print(f"\n{'=' * 70}")
    print("IMAGE-TO-GPS MULTI-EXPERIMENT RUNNER")
    print(f"{'=' * 70}")
    print(f"Backbones: {', '.join(backbones)}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"Val/Test split: {val_split}/{test_split}")
    print(f"Shared split file: {shared_split_path}")
    print("Data Directory: C:\\Users\\shaha\\Desktop\\DEEP_LEARNING_PROJECT\\data")
    print("Master Dataset: campus_gps_master_dataset.csv\n")

    results = {}
    for backbone in backbones:
        success = run_experiment(
            backbone=backbone,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            val_split=val_split,
            test_split=test_split,
            shared_split_path=shared_split_path,
        )
        results[backbone] = "[OK] PASSED" if success else "[ERROR] FAILED"

    print(f"\n{'=' * 70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 70}\n")
    for backbone, status in results.items():
        print(f"{backbone:<25} {status}")

    print(f"\n{'=' * 70}")
    print("All experiments completed.")
    print(f"Models saved in: {Path(__file__).parent / 'models'}")
    print("Dataset file: campus_gps_master_dataset.csv")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
