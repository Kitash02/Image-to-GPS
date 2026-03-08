import torch
import torch.nn as nn
import torchvision.models as models

class GPSRegressionModel(nn.Module):
    """Single-task GPS regression model."""
    
    BACKBONE_CONFIGS = {
        "resnet50": {
            "model_fn": lambda: models.resnet50(weights="DEFAULT"),
            "output_features": 2048,
            "input_size": 224,
        },
        "resnet101": {
            "model_fn": lambda: models.resnet101(weights="DEFAULT"),
            "output_features": 2048,
            "input_size": 224,
        },
        "efficientnet_b0": {
            "model_fn": lambda: models.efficientnet_b0(weights="DEFAULT"),
            "output_features": 1280,
            "input_size": 224,
        },
        "efficientnet_b1": {
            "model_fn": lambda: models.efficientnet_b1(weights="DEFAULT"),
            "output_features": 1280,
            "input_size": 240,
        },
        "convnext_tiny": {
            "model_fn": lambda: models.convnext_tiny(weights="DEFAULT"),
            "output_features": 768,
            "input_size": 224,
        },
        "vit_b16": {
            "model_fn": lambda: models.vit_b_16(weights="DEFAULT"),
            "output_features": 768,
            "input_size": 224,
        },
    }
    
    def __init__(self, backbone_name: str = "resnet50", dropout_rate: float = 0.3):
        """
        Args:
            backbone_name: Name of backbone architecture
            dropout_rate: Dropout probability (default: 0.3)
        """
        super(GPSRegressionModel, self).__init__()
        
        if backbone_name not in self.BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.backbone_name = backbone_name
        config = self.BACKBONE_CONFIGS[backbone_name]
        
        # Load pre-trained backbone
        self.backbone = config["model_fn"]()
        output_features = config["output_features"]
        
        # Remove classification head
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "heads"):
            self.backbone.heads = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        
        # Shared -> GPS head (single-task)
        self.shared_head = nn.Sequential(
            nn.Linear(output_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.gps_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Output: [Latitude, Longitude]
        )
    
    def forward(self, x):
        """
        Forward pass through backbone and dual heads.
        
        Returns:
            gps: Tensor of shape (batch_size, 2) - scaled GPS coordinates
        """
        features = self.backbone(x)
        
        # Handle different feature output shapes
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() > 2:
            features = features.mean(dim=(2, 3))  # Global average pooling
        
        shared_features = self.shared_head(features)
        gps = self.gps_head(shared_features)
        
        return gps
    
    @staticmethod
    def get_input_size(backbone_name: str) -> int:
        """Get recommended input image size for a backbone."""
        if backbone_name not in GPSRegressionModel.BACKBONE_CONFIGS:
            return 224
        return GPSRegressionModel.BACKBONE_CONFIGS[backbone_name]["input_size"]
