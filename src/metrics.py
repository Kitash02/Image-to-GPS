import numpy as np

def haversine_distance(lat1: np.ndarray, lon1: np.ndarray,
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    High-precision Haversine using np.float64 and WGS-84 radius ~6378137.0 m.
    Inputs in degrees; outputs in meters (np.float64).
    """
    # ensure double precision
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    
    # Earth radius (WGS-84)
    R = 6378137.0  # meters
    
    # convert to radians
    lat1_r = np.deg2rad(lat1)
    lon1_r = np.deg2rad(lon1)
    lat2_r = np.deg2rad(lat2)
    lon2_r = np.deg2rad(lon2)
    
    # high-precision haversine
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    sin_dlat = np.sin(dlat * 0.5)
    sin_dlon = np.sin(dlon * 0.5)
    a = sin_dlat * sin_dlat + np.cos(lat1_r) * np.cos(lat2_r) * (sin_dlon * sin_dlon)
    c = 2.0 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def mean_distance_error(predictions: np.ndarray, ground_truth: np.ndarray, scaler=None) -> float:
    """
    Mean haversine distance in meters (float64).
    predictions/ground_truth: shape (N,2), may be scaled -> pass scaler to inverse_transform.
    """
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        ground_truth = scaler.inverse_transform(ground_truth)
    d = haversine_distance(predictions[:,0], predictions[:,1], ground_truth[:,0], ground_truth[:,1])
    return float(np.mean(d))

def median_distance_error(predictions: np.ndarray, ground_truth: np.ndarray, scaler=None) -> float:
    """
    Median haversine distance in meters (float64).
    """
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        ground_truth = scaler.inverse_transform(ground_truth)
    d = haversine_distance(predictions[:,0], predictions[:,1], ground_truth[:,0], ground_truth[:,1])
    return float(np.median(d))

def accuracy_within_threshold(distances: np.ndarray, threshold_meters: float = 5.0) -> float:
    distances = np.asarray(distances, dtype=np.float64)
    return float((distances <= threshold_meters).mean() * 100.0)
