import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

class GPSDataset(Dataset):
    """Dataset class with advanced augmentations for day/night robustness."""
    
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        input_size: int = 224,
        is_train: bool = True,
        scaler=None,
        fit_scaler: bool = False,
    ):
        """
        Args:
            csv_path: Path to campus_gps_master_dataset.csv file (can be str or Path)
            images_dir: Path to images root directory (can be str or Path)
            input_size: Input image size for the model
            is_train: If True, apply augmentations; otherwise use clean validation transforms
            scaler: MinMaxScaler instance for GPS coordinates
            fit_scaler: If True, fit the scaler on this dataset
        """
        # Convert to Path objects if strings
        self.csv_path = Path(csv_path)
        self.images_root = Path(images_dir)
        
        self.df = pd.read_csv(self.csv_path)
        self.input_size = input_size
        self.is_train = is_train
        self.scaler = scaler
        
        # Create internal mapping {prefix_filename: full_path} for anonymity
        self.path_mapping = {}
        for idx, row in self.df.iterrows():
            prefixed_name = row["image_name"]
            # Extract prefix (e.g., f1_IMG_0001.jpg -> f1_)
            prefix = prefixed_name.split("_")[0] + "_"
            # Resolve by searching subfolders under images_root
            actual_path = None
            for subfolder in self.images_root.rglob("*"):
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    candidate = subfolder / prefixed_name[len(prefix):]  # Remove prefix to get original name
                    if candidate.exists():
                        actual_path = candidate
                        break
            if actual_path:
                self.path_mapping[prefixed_name] = actual_path
            else:
                # Fallback if not found
                self.path_mapping[prefixed_name] = self.images_root / prefixed_name
        
        # Extract GPS targets (raw absolute coordinates)
        self.gps_targets = self.df[["Latitude", "Longitude"]].values.astype(np.float32)
        
        # Fit or use provided scaler
        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.gps_targets = self.scaler.fit_transform(self.gps_targets).astype(np.float32)
        elif self.scaler is not None:
            self.gps_targets = self.scaler.transform(self.gps_targets).astype(np.float32)
        
        # Define transforms based on train/val split
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get augmentations for training or clean transforms for validation."""
        if self.is_train:
            # Aggressive augmentations for training (day/night robustness)
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Stronger color jitter for lighting variations
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.15
                ),
                # Grayscale for night images
                transforms.RandomGrayscale(p=0.2),
                # Motion blur and focus variations
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Sharpness variations
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                # Rotation for camera angle variations
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Clean transforms for validation/test
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefixed_name = row["image_name"]
        # Resolve to actual path using mapping
        image_path = self.path_mapping.get(prefixed_name, self.images_root / prefixed_name)
        
        # Load and convert image to RGB
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        gps = torch.tensor(self.gps_targets[idx], dtype=torch.float32)
        
        return image, gps
    
    def get_scaler(self):
        """Return the scaler for inference."""
        return self.scaler
