import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import pickle

# Register HEIC support when pillow-heif is available.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Global variables for model and scaler (loaded once)
MODEL = None
SCALER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load model and scaler globally."""
    global MODEL, SCALER
    if MODEL is None:
        # Support both direct `import inference` and package import `from src import inference`.
        try:
            from model import GPSRegressionModel
        except ImportError:
            from .model import GPSRegressionModel
        MODEL = GPSRegressionModel(backbone_name="resnet50", use_pretrained=False)  # Default backbone; adjust if needed
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
    if SCALER is None:
        with open(scaler_path, "rb") as f:
            SCALER = pickle.load(f)

def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS coordinates from a single image.
    
    Args:
        image: uint8 numpy array of shape (H, W, 3) in RGB format (range 0-255)
    
    Returns:
        float32 numpy array of shape (2,) containing [Latitude, Longitude]
    """
    # Ensure global loading (call once before using)
    if MODEL is None or SCALER is None:
        raise RuntimeError("Call load_model_and_scaler() first to initialize model and scaler.")
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    
    # Preprocessing: Resize to 224x224, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = transform(pil_image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        scaled_pred = MODEL(tensor_image).cpu().numpy().squeeze()  # Shape (2,)
    
    # Inverse transform to absolute coordinates
    absolute_pred = SCALER.inverse_transform(scaled_pred.reshape(1, -1)).squeeze()
    
    return absolute_pred.astype(np.float32)
