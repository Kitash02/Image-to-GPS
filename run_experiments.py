import subprocess
import sys
from pathlib import Path
import pandas as pd

def run_experiment(backbone: str, epochs: int = 100, batch_size: int = 32):
    """Run a single training experiment."""
    print(f"\n{'='*70}")
    print(f"Starting experiment: {backbone}")
    print(f"{'='*70}\n")
    
    # Use absolute paths with raw strings
    csv_path = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\campus_gps_master_dataset.csv")
    images_dir = Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images")
    
    # Verify data directory exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        print(f"   Run: python src/generate_gt_csv.py")
        return False
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found at {images_dir}")
        return False
    
    cmd = [
        sys.executable,
        "src/train.py",
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--csv_path", str(csv_path),
        "--images_dir", str(images_dir),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print(f"\n✅ {backbone} experiment completed successfully!")
        # Attempt to read test_results.csv to report median error
        # Note: Since folders are timestamped, we need to find the latest one
        model_dir = Path(__file__).parent / "models"
        backbone_dirs = [d for d in model_dir.glob(f"{backbone}_*") if d.is_dir()]
        if backbone_dirs:
            latest_dir = max(backbone_dirs, key=lambda x: x.stat().st_mtime)
            test_csv = latest_dir / "test_results.csv"
            if test_csv.exists():
                try:
                    df = pd.read_csv(test_csv)
                    if "error_meters" in df.columns and len(df) > 0:
                        medde = float(df["error_meters"].median())
                        print(f"{backbone} Test Median Distance Error (MedDE): {medde:.2f} meters")
                    else:
                        print(f"{backbone} test_results.csv found but no 'error_meters' column or empty.")
                except Exception:
                    print(f"Could not read {test_csv}")
            else:
                print(f"No test_results.csv found for {backbone} in {latest_dir}")
        else:
            print(f"No {backbone} directories found in models/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {backbone} experiment failed with error code: {e.returncode}")
        return False

def main():
    """Run all experiments sequentially."""
    backbones = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    epochs = 100
    batch_size = 32
    
    print(f"\n{'='*70}")
    print(f"IMAGE-TO-GPS MULTI-EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Backbones: {', '.join(backbones)}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Data Directory: C:\\Users\\shaha\\Desktop\\DEEP_LEARNING_PROJECT\\data")
    print(f"Master Dataset: campus_gps_master_dataset.csv\n")
    
    results = {}
    
    for backbone in backbones:
        success = run_experiment(backbone, epochs=epochs, batch_size=batch_size)
        results[backbone] = "✅ PASSED" if success else "❌ FAILED"
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    for backbone, status in results.items():
        print(f"{backbone:<25} {status}")
    
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Models saved in: {Path(__file__).parent / 'models'}")
    print(f"Dataset file: campus_gps_master_dataset.csv")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
