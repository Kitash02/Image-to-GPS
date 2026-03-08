import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import pickle
import json
import warnings
import csv
from datetime import datetime
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

from dataset import GPSDataset
from model import GPSRegressionModel
from metrics import mean_distance_error, median_distance_error, accuracy_within_threshold, haversine_distance

warnings.filterwarnings("ignore")

class EarlyStopping:
    """Early stopping callback."""
    def __init__(self, patience: int = 12, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def tta_batch_predict(model, images_norm, device, brightness=0.3):
    """
    images_norm: Tensor (B,C,H,W) normalized with ImageNet mean/std (float32)
    Returns: numpy array (B,2) of scaled predictions (same scale as training)
    """
    model.eval()
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    
    # ensure float32
    imgs = images_norm.to(device)
    
    # Unnormalize -> pixel range approx [0,1]
    imgs_unn = imgs * std + mean  # still float32
    
    # prepare variants
    img_orig = torch.clamp(imgs_unn, 0.0, 1.0)
    img_flip = torch.flip(img_orig, dims=[3])
    img_bright = torch.clamp(img_orig * (1.0 + brightness), 0.0, 1.0)
    
    # Renormalize
    v1 = (img_orig - mean) / std
    v2 = (img_flip - mean) / std
    v3 = (img_bright - mean) / std
    
    batch_all = torch.cat([v1, v2, v3], dim=0)  # (3B, C, H, W)
    with torch.no_grad():
        preds = model(batch_all.float())  # (3B,2)
    preds = preds.cpu().numpy().reshape(3, -1, 2)  # (3, B, 2)
    avg = preds.mean(axis=0)  # (B,2)
    return avg

def main():
    parser = argparse.ArgumentParser(description="Train GPS regression model")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\campus_gps_master_dataset.csv"),
        help="Path to campus_gps_master_dataset.csv"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path(r"C:\Users\shaha\Desktop\DEEP_LEARNING_PROJECT\data\dataset_root\images"),
        help="Path to images root directory"
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Base directory to save models"
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=["resnet50", "resnet101", "efficientnet_b0", "efficientnet_b1", "convnext_tiny", "vit_b16"],
        help="Backbone architecture"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for head")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--huber_delta", type=float, default=0.1, help="Huber loss delta parameter")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create unique timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_model_dir = Path(args.model_dir) / f"{args.backbone}_{timestamp}"
    exp_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.backbone} ({timestamp})")
    print(f"Model directory: {exp_model_dir}")
    print(f"{'='*60}\n")
    
    # Load dataset WITHOUT fitting scaler
    full_dataset = GPSDataset(args.csv_path, args.images_dir, input_size=GPSRegressionModel.get_input_size(args.backbone), is_train=False, fit_scaler=False)
    
    total_size = len(full_dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    test_size = int(total_size * args.test_split)
    val_size = int(total_size * args.val_split)
    train_size = total_size - test_size - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Fit scaler only on training coordinates (abs lat/lon) - ZERO DATA LEAKAGE
    coords = full_dataset.df[["Latitude","Longitude"]].values.astype(np.float64)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(coords[train_indices])
    
    # Save scaler
    with open(exp_model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Create datasets using fitted scaler
    train_dataset = GPSDataset(args.csv_path, args.images_dir, input_size=GPSRegressionModel.get_input_size(args.backbone), is_train=True, scaler=scaler, fit_scaler=False)
    val_dataset = GPSDataset(args.csv_path, args.images_dir, input_size=GPSRegressionModel.get_input_size(args.backbone), is_train=False, scaler=scaler, fit_scaler=False)
    test_dataset = GPSDataset(args.csv_path, args.images_dir, input_size=GPSRegressionModel.get_input_size(args.backbone), is_train=False, scaler=scaler, fit_scaler=False)
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = GPSRegressionModel(backbone_name=args.backbone, dropout_rate=args.dropout).to(device)
    lr_backbone = args.lr_head * 0.1
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.shared_head.parameters()) + list(model.gps_head.parameters())
    optimizer = Adam([{"params": backbone_params, "lr": lr_backbone}, {"params": head_params, "lr": args.lr_head}])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    gps_criterion = nn.HuberLoss(delta=args.huber_delta)
    early_stopping = EarlyStopping(patience=12, verbose=True)
    
    # history CSV header updated to include Val MedDE
    history_path = exp_model_dir / "history.csv"
    with open(history_path, "w", newline="") as hf:
        hw = csv.writer(hf)
        hw.writerow(["Epoch","Train Loss","Val Loss","Val MDE (meters)","Val MedDE (meters)","Accuracy@5m"])
    
    best_val_loss = float("inf")
    best_model_path = exp_model_dir / "best_model.pth"
    
    # Training loop
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val MDE (m)':<14} {'Val MedDE (m)':<14} {'Acc@5m':<8}")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, gps in train_loader:
            images = images.to(device)
            gps = gps.to(device)
            optimizer.zero_grad()
            gps_pred = model(images)
            loss = gps_criterion(gps_pred, gps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for images, gps in val_loader:
                images = images.to(device)
                gps = gps.to(device)
                gps_pred = model(images)
                loss = gps_criterion(gps_pred, gps)
                val_loss += loss.item() * images.size(0)
                val_preds.append(gps_pred.cpu().numpy())
                val_trues.append(gps.cpu().numpy())
        val_loss /= len(val_dataset)
        val_preds = np.concatenate(val_preds, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_mde = mean_distance_error(val_preds, val_trues, scaler)
        val_medde = median_distance_error(val_preds, val_trues, scaler)
        # compute acc@5m using high-precision haversine
        unscaled_preds = scaler.inverse_transform(val_preds)
        unscaled_trues = scaler.inverse_transform(val_trues)
        val_distances = haversine_distance(unscaled_preds[:,0], unscaled_preds[:,1], unscaled_trues[:,0], unscaled_trues[:,1])
        val_acc5 = accuracy_within_threshold(val_distances, threshold_meters=5.0)
        
        # append to history
        with open(history_path, "a", newline="") as hf:
            hw = csv.writer(hf)
            hw.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_mde:.4f}", f"{val_medde:.4f}", f"{val_acc5:.2f}"])
        
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {val_mde:<14.4f} {val_medde:<14.4f} {val_acc5:<8.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation on test set using batch TTA
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    all_preds = []
    all_trues = []
    all_names = []
    # test_dataset.df subset indices mapping to names
    for batch_images, batch_gps in test_loader:
        # batch_images normalized tensors
        preds_scaled = tta_batch_predict(model, batch_images, device, brightness=0.3)  # (B,2) scaled
        all_preds.append(preds_scaled)
        all_trues.append(batch_gps.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    # gather image names from full_dataset using test_indices
    test_names = full_dataset.df.iloc[test_indices]["image_name"].values
    
    # distances
    unscaled_preds = scaler.inverse_transform(all_preds)
    unscaled_trues = scaler.inverse_transform(all_trues)
    distances = haversine_distance(unscaled_preds[:,0], unscaled_preds[:,1], unscaled_trues[:,0], unscaled_trues[:,1])
    test_mde = float(np.mean(distances))
    test_medde = float(np.median(distances))
    test_acc5 = float((distances <= 5.0).mean() * 100.0)
    test_acc10 = float((distances <= 10.0).mean() * 100.0)
    
    print("\nFINAL TEST RESULTS")
    print(f"Test MDE: {test_mde:.4f} m")
    print(f"Test MedDE: {test_medde:.4f} m")
    print(f"Test Acc@5m: {test_acc5:.2f}%")
    
    # Save detailed CSV/XLSX
    results_df = pd.DataFrame({
        "image_name": test_names,
        "true_latitude": unscaled_trues[:,0],
        "true_longitude": unscaled_trues[:,1],
        "predicted_latitude": unscaled_preds[:,0],
        "predicted_longitude": unscaled_preds[:,1],
        "error_meters": distances
    })
    results_df.to_csv(exp_model_dir / "test_results.csv", index=False)
    results_df.to_excel(exp_model_dir / "test_results.xlsx", index=False)
    
    print(f"\nSaved test results to {exp_model_dir / 'test_results.csv'}")

if __name__ == "__main__":
    main()
