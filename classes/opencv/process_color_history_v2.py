import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import colorsys
from queue import Queue
from classes.utils.logger import add_log
from typing import List, Tuple, Optional

def queue_to_array(rgb_queue: Queue) -> np.ndarray:
    """Convert a Queue of arrays to numpy array, extracting only RGB values."""
    items = []
    temp_queue = Queue()
    
    while not rgb_queue.empty():
        item = rgb_queue.get()
        rgb_values = item[:3] if len(item) >= 3 else item
        items.append(rgb_values)
        temp_queue.put(item)
    
    # Restore original queue
    while not temp_queue.empty():
        rgb_queue.put(temp_queue.get())
    
    return np.array(items)

def rgb_to_robust_hsv(rgb_array: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV with robust handling of degraded colors."""
    rgb_norm = np.clip(rgb_array / 255.0, 0, 1)
    hsv_array = np.zeros_like(rgb_norm)
    
    for i, (r, g, b) in enumerate(rgb_norm):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # Scale and handle edge cases
        hsv_array[i] = [
            h * 360,  # Hue: 0-360
            s * 100,  # Saturation: 0-100
            v * 100   # Value: 0-100
        ]
    
    return hsv_array

def calculate_perceptual_signature(colors: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Calculate perceptual signature focusing on hue changes and brightness patterns."""
    hsv = rgb_to_robust_hsv(colors)
    
    # Separate HSV components
    hue = hsv[:, 0]
    sat = hsv[:, 1] 
    val = hsv[:, 2]
    
    # Handle hue discontinuity (0° = 360°)
    hue_unwrapped = np.unwrap(np.radians(hue)) * 180 / np.pi
    
    # Calculate gradients for each component
    hue_grad = np.gradient(hue_unwrapped)
    sat_grad = np.gradient(sat)
    val_grad = np.gradient(val)
    
    # Weight components based on perceptual importance
    # Hue changes are most important, then brightness, then saturation
    signature = (
        0.5 * np.abs(hue_grad) +      # Hue changes (most important)
        0.3 * np.abs(val_grad) +      # Brightness changes  
        0.2 * np.abs(sat_grad)        # Saturation changes (least important for degraded video)
    )
    
    # Apply Gaussian smoothing instead of simple moving average
    if len(signature) >= window_size:
        sigma = window_size / 3.0
        signature = signal.gaussian_filter1d(signature, sigma)
    
    return signature

def adaptive_cross_correlation(sig1: np.ndarray, sig2: np.ndarray, max_shift: Optional[int] = None) -> Tuple[int, float]:
    """Improved cross-correlation with adaptive normalization."""
    if max_shift is None:
        max_shift = min(len(sig1), len(sig2)) // 4  # Reduced from //3 for efficiency
    
    # Robust normalization to handle different signal ranges
    def robust_normalize(signal):
        # Use median-based normalization to handle outliers
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))  # Median Absolute Deviation
        if mad == 0:
            mad = np.std(signal)
        if mad == 0:
            return signal - median
        return (signal - median) / mad
    
    sig1_norm = robust_normalize(sig1)
    sig2_norm = robust_normalize(sig2)
    
    # Use scipy's correlate for efficiency with large signals
    if len(sig1_norm) > 500 or len(sig2_norm) > 500:
        # For large signals, use FFT-based correlation
        correlation = signal.correlate(sig1_norm, sig2_norm, mode='full')
        lags = signal.correlation_lags(len(sig1_norm), len(sig2_norm), mode='full')
        
        # Limit to max_shift range
        valid_indices = np.where(np.abs(lags) <= max_shift)[0]
        correlation = correlation[valid_indices]
        lags = lags[valid_indices]
        
        best_idx = np.argmax(np.abs(correlation))
        optimal_shift = -lags[best_idx]  # Negative because of correlation definition
        correlation_strength = correlation[best_idx] / len(sig1_norm)  # Normalize
    else:
        # For smaller signals, use manual calculation
        correlations = []
        shifts = range(-max_shift, max_shift + 1)
        
        for shift in shifts:
            if shift >= 0:
                min_len = min(len(sig1_norm) - shift, len(sig2_norm))
                if min_len > 10:  # Minimum overlap requirement
                    corr = np.corrcoef(sig1_norm[shift:shift+min_len], sig2_norm[:min_len])[0, 1]
                else:
                    corr = 0
            else:
                shift_abs = abs(shift)
                min_len = min(len(sig1_norm), len(sig2_norm) - shift_abs)
                if min_len > 10:
                    corr = np.corrcoef(sig1_norm[:min_len], sig2_norm[shift_abs:shift_abs+min_len])[0, 1]
                else:
                    corr = 0
            correlations.append(corr if not np.isnan(corr) else 0)
        
        best_idx = np.argmax(np.abs(correlations))
        optimal_shift = shifts[best_idx]
        correlation_strength = correlations[best_idx]
    
    return int(optimal_shift), float(correlation_strength)

def align_and_trim_signals(sig1: np.ndarray, sig2: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """Align signals and ensure optimal overlap length."""
    if shift > 0:
        # sig1 leads sig2
        aligned_sig1 = sig1[shift:]
        aligned_sig2 = sig2
    elif shift < 0:
        # sig2 leads sig1
        shift_abs = abs(shift)
        aligned_sig1 = sig1
        aligned_sig2 = sig2[shift_abs:]
    else:
        aligned_sig1 = sig1
        aligned_sig2 = sig2
    
    # Ensure same length and sufficient overlap
    min_len = min(len(aligned_sig1), len(aligned_sig2))
    if min_len < 20:  # Minimum required for meaningful correlation
        return np.array([]), np.array([])
    
    return aligned_sig1[:min_len], aligned_sig2[:min_len]

def calculate_similarity_metrics(sig1: np.ndarray, sig2: np.ndarray) -> dict:
    """Calculate multiple similarity metrics for robust comparison."""
    if len(sig1) < 2 or len(sig2) < 2:
        return {'correlation': 0, 'p_value': 1, 'dtw_distance': float('inf'), 
                'cosine_similarity': 0, 'normalized_rmse': 1}
    
    # Pearson correlation
    try:
        correlation, p_value = pearsonr(sig1, sig2)
        if np.isnan(correlation):
            correlation, p_value = 0, 1
    except:
        correlation, p_value = 0, 1
    
    # Cosine similarity
    norm1 = np.linalg.norm(sig1)
    norm2 = np.linalg.norm(sig2)
    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = np.dot(sig1, sig2) / (norm1 * norm2)
    
    # Normalized RMSE
    rmse = np.sqrt(np.mean((sig1 - sig2) ** 2))
    signal_range = max(np.max(np.abs(sig1)), np.max(np.abs(sig2)))
    normalized_rmse = rmse / (signal_range + 1e-8)
    
    # Simple DTW-like distance (efficient approximation)
    dtw_distance = np.mean(np.abs(sig1 - sig2))
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'dtw_distance': dtw_distance,
        'cosine_similarity': cosine_similarity,
        'normalized_rmse': normalized_rmse
    }

def compare_color_fluctuations(queue1: Queue, queue2: Queue, 
                              similarity_threshold: float = 0.6,  # Lowered for degraded video
                              max_time_shift: Optional[int] = None,
                              fluctuation_window: int = 7,
                              min_data_points: int = 50) -> dict:  # Reduced minimum
    """
    Optimized comparison of RGB queues for similar color fluctuation patterns.
    
    Args:
        queue1, queue2: Queues containing RGB arrays
        similarity_threshold: Minimum correlation for similarity (0-1)
        max_time_shift: Maximum allowed time shift (None for auto)
        fluctuation_window: Window size for smoothing fluctuation signature
        min_data_points: Minimum required data points
    
    Returns:
        Dictionary with comprehensive comparison results
    """
    # Convert queues to arrays
    colors1 = queue_to_array(queue1)
    colors2 = queue_to_array(queue2)
    
    # Check minimum data requirements
    if len(colors1) < min_data_points or len(colors2) < min_data_points:
        return {
            'similar': False,
            'error': f'Insufficient data points (need at least {min_data_points})',
            'time_shift': 0,
            'correlation': 0.0,
            'queue1_length': len(colors1),
            'queue2_length': len(colors2)
        }
    
    # Calculate perceptual signatures
    signature1 = calculate_perceptual_signature(colors1, fluctuation_window)
    signature2 = calculate_perceptual_signature(colors2, fluctuation_window)
    
    if len(signature1) < 10 or len(signature2) < 10:
        return {
            'similar': False,
            'error': 'Insufficient signature data after processing',
            'time_shift': 0,
            'correlation': 0.0
        }
    
    # Find optimal time shift
    optimal_shift, shift_correlation = adaptive_cross_correlation(
        signature1, signature2, max_time_shift
    )
    
    # Align signatures
    aligned_sig1, aligned_sig2 = align_and_trim_signals(signature1, signature2, optimal_shift)
    
    if len(aligned_sig1) == 0 or len(aligned_sig2) == 0:
        return {
            'similar': False,
            'error': 'No valid alignment found',
            'time_shift': optimal_shift,
            'correlation': shift_correlation
        }
    
    # Calculate comprehensive similarity metrics
    metrics = calculate_similarity_metrics(aligned_sig1, aligned_sig2)
    
    # Multi-criteria similarity decision
    # Relaxed criteria for degraded video comparison
    is_similar = (
        abs(metrics['correlation']) >= similarity_threshold and
        metrics['p_value'] < 0.1 and  # Relaxed p-value
        metrics['normalized_rmse'] < 0.7 and  # Relaxed RMSE threshold
        abs(metrics['cosine_similarity']) >= similarity_threshold * 0.8  # Additional metric
    )
    
    # Compile comprehensive results
    result = {
        'similar': is_similar,
        'time_shift': optimal_shift,
        'correlation': metrics['correlation'],
        'p_value': metrics['p_value'],
        'normalized_rmse': metrics['normalized_rmse'],
        'cosine_similarity': metrics['cosine_similarity'],
        'dtw_distance': metrics['dtw_distance'],
        'shift_correlation': shift_correlation,
        'aligned_length': len(aligned_sig1),
        'signature1_length': len(signature1),
        'signature2_length': len(signature2),
        'confidence_score': (
            abs(metrics['correlation']) * 0.4 +
            abs(metrics['cosine_similarity']) * 0.3 +
            (1 - metrics['normalized_rmse']) * 0.2 +
            (1 - metrics['p_value']) * 0.1
        )
    }
    
    return result

# Enhanced test function for validation
def create_realistic_test_queues():
    """Create test queues that simulate real webcam vs stream scenarios."""
    np.random.seed(42)  # For reproducible results
    
    # Base pattern with more realistic color transitions
    base_colors = []
    for i in range(200):  # Longer sequence
        # Simulate scene changes with smoother transitions
        scene_factor = np.sin(i * 0.02) * 0.5 + 0.5  # Slow scene changes
        
        r = int(100 + 80 * scene_factor + 20 * np.sin(i * 0.1) + np.random.normal(0, 5))
        g = int(120 + 60 * np.cos(i * 0.08) + 15 * scene_factor + np.random.normal(0, 5))
        b = int(90 + 70 * np.sin(i * 0.12) + 30 * scene_factor + np.random.normal(0, 5))
        
        base_colors.append([
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        ])
    
    # Queue 1: "Stream" - original quality
    queue1 = Queue()
    for color in base_colors:
        queue1.put(color)
    
    # Queue 2: "Webcam" - degraded with latency
    queue2 = Queue()
    latency = 12  # Frames of delay
    
    # Add latency padding
    for i in range(latency):
        queue2.put([128, 128, 128])
    
    # Add degraded colors
    for i, color in enumerate(base_colors[:-latency]):
        # Simulate webcam degradation
        degraded_color = [
            max(0, min(255, int(color[0] * 0.85 + 20))),  # Brightness/contrast shift
            max(0, min(255, int(color[1] * 0.9 + 10))),   # Desaturation
            max(0, min(255, int(color[2] * 0.95 + 5)))    # Color shift
        ]
        # Add noise
        noise = np.random.normal(0, 8, 3)
        degraded_color = [max(0, min(255, int(c + n))) for c, n in zip(degraded_color, noise)]
        
        queue2.put(degraded_color)
    
    return queue1, queue2

# Usage example
if __name__ == "__main__":
    # Test with realistic data
    test_queue1, test_queue2 = create_realistic_test_queues()
    result = compare_color_fluctuations(test_queue1, test_queue2)
    
    print("Comparison Results:")
    print(f"Similar: {result['similar']}")
    print(f"Time shift: {result['time_shift']} frames")
    print(f"Correlation: {result['correlation']:.3f}")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"RMSE: {result['normalized_rmse']:.3f}")