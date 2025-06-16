import numpy as np
from scipy.stats import pearsonr
import colorsys
from queue import Queue
from classes.utils.logger import add_log
from typing import List, Tuple, Optional

def queue_to_array(rgb_queue: Queue) -> np.ndarray:
    """Convert a Queue of arrays to numpy array, extracting only RGB values.
    
    Args:
        rgb_queue: Queue containing arrays where RGB values are the first 3 elements
        
    Returns:
        numpy array of shape (n, 3) containing only RGB values
    """
    items = []
    temp_queue = Queue()
    
    # Extract items while preserving queue
    while not rgb_queue.empty():
        item = rgb_queue.get()
        # Extract only RGB values (first 3 elements)
        rgb_values = item[:3] if len(item) >= 3 else item
        items.append(rgb_values)
        temp_queue.put(item)
    
    # Restore original queue
    while not temp_queue.empty():
        rgb_queue.put(temp_queue.get())
    
    return np.array(items)

def rgb_to_perceptual(rgb_array: np.ndarray) -> np.ndarray:
    """Convert RGB to perceptually uniform representation (LAB-like)."""
    # Normalize RGB to 0-1
    rgb_norm = rgb_array / 255.0
    
    # Convert to HSV for better perceptual comparison
    hsv_array = np.zeros_like(rgb_norm)
    for i, (r, g, b) in enumerate(rgb_norm):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_array[i] = [h * 360, s * 100, v * 100]  # Scale to typical ranges
    
    return hsv_array

def calculate_fluctuation_signature(colors: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Calculate color fluctuation signature using gradients and variance."""
    # Convert to perceptual space
    perceptual = rgb_to_perceptual(colors)
    
    # Calculate gradients (rate of change)
    gradients = np.gradient(perceptual, axis=0)
    gradient_magnitude = np.linalg.norm(gradients, axis=1)
    
    # Smooth the signature with moving average
    if len(gradient_magnitude) >= window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(gradient_magnitude, kernel, mode='valid')
    else:
        smoothed = gradient_magnitude
    
    return smoothed

def find_optimal_time_shift(sig1: np.ndarray, sig2: np.ndarray, max_shift: Optional[int] = None) -> Tuple[int, float]:
    """Find optimal time shift using cross-correlation."""
    if max_shift is None:
        max_shift = min(len(sig1), len(sig2)) // 3
    
    # Normalize signals
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)
    
    # Calculate cross-correlation using numpy
    correlations = []
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            # Ensure both arrays have the same length
            min_len = min(len(sig1_norm) - shift, len(sig2_norm))
            if min_len > 0:
                corr = np.corrcoef(sig1_norm[shift:shift+min_len], sig2_norm[:min_len])[0, 1]
            else:
                corr = 0
        else:
            # Ensure both arrays have the same length
            min_len = min(len(sig1_norm), len(sig2_norm) + shift)
            if min_len > 0:
                corr = np.corrcoef(sig1_norm[:min_len], sig2_norm[-shift:-shift+min_len])[0, 1]
            else:
                corr = 0
        correlations.append(corr)
    
    # Find the best correlation
    best_idx = np.argmax(np.abs(correlations))
    optimal_shift = best_idx - max_shift
    correlation_strength = correlations[best_idx]
    
    return int(optimal_shift), float(correlation_strength)

def align_signals(sig1: np.ndarray, sig2: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """Align two signals based on the calculated shift."""
    if shift > 0:
        # sig2 is delayed relative to sig1
        aligned_sig1 = sig1[shift:]
        aligned_sig2 = sig2[:len(aligned_sig1)]  # Ensure same length as sig1
    elif shift < 0:
        # sig1 is delayed relative to sig2
        shift = abs(shift)
        aligned_sig2 = sig2[shift:]
        aligned_sig1 = sig1[:len(aligned_sig2)]  # Ensure same length as sig2
    else:
        # No shift needed, use minimum length
        min_len = min(len(sig1), len(sig2))
        aligned_sig1 = sig1[:min_len]
        aligned_sig2 = sig2[:min_len]
    
    # Final length check to ensure both arrays are exactly the same length
    min_len = min(len(aligned_sig1), len(aligned_sig2))
    return aligned_sig1[:min_len], aligned_sig2[:min_len]

def compare_color_fluctuations(queue1: Queue, queue2: Queue, 
                              similarity_threshold: float = 0.7,
                              max_time_shift: Optional[int] = None,
                              fluctuation_window: int = 5) -> dict:
    """
    Compare two RGB queues for similar color fluctuation patterns.
    
    Args:
        queue1, queue2: Queues containing RGB arrays
        similarity_threshold: Minimum correlation for considering signals similar (0-1)
        max_time_shift: Maximum allowed time shift (None for auto)
        fluctuation_window: Window size for smoothing fluctuation signature
    
    Returns:
        Dictionary with comparison results
    """
    # Convert queues to arrays
    colors1 = queue_to_array(queue1)
    colors2 = queue_to_array(queue2)
    
    # Check if both queues have at least 100 RGB values
    if len(colors1) < 100 or len(colors2) < 100:
        return {
            'similar': False,
            'error': 'Insufficient data points (need at least 100 RGB values)',
            'time_shift': 0,
            'correlation': 0.0,
            'queue1_length': len(colors1),
            'queue2_length': len(colors2)
        }
    
    # Calculate fluctuation signatures
    signature1 = calculate_fluctuation_signature(colors1, fluctuation_window)
    signature2 = calculate_fluctuation_signature(colors2, fluctuation_window)
    
    if len(signature1) < 2 or len(signature2) < 2:
        return {
            'similar': False,
            'error': 'Insufficient signature data',
            'time_shift': 0,
            'correlation': 0.0
        }
    
    # Find optimal time shift
    optimal_shift, shift_correlation = find_optimal_time_shift(
        signature1, signature2, max_time_shift
    )
    
    # Align signatures
    aligned_sig1, aligned_sig2 = align_signals(signature1, signature2, optimal_shift)
    
    # Debug logging
    print(f"Original lengths - sig1: {len(signature1)}, sig2: {len(signature2)}")
    print(f"Aligned lengths - sig1: {len(aligned_sig1)}, sig2: {len(aligned_sig2)}")
    print(f"Optimal shift: {optimal_shift}")
    
    if len(aligned_sig1) < 2 or len(aligned_sig2) < 2:
        return {
            'similar': False,
            'error': 'Insufficient aligned data',
            'time_shift': optimal_shift,
            'correlation': shift_correlation
        }
    
    # Ensure both signals have exactly the same length
    min_len = min(len(aligned_sig1), len(aligned_sig2))
    aligned_sig1 = aligned_sig1[:min_len]
    aligned_sig2 = aligned_sig2[:min_len]
    
    # Calculate final correlation after alignment
    final_correlation, p_value = pearsonr(aligned_sig1, aligned_sig2)
    
    # Additional similarity metrics
    # Normalized cross-correlation
    ncc = np.corrcoef(aligned_sig1, aligned_sig2)[0, 1]
    
    # Root mean square error (normalized)
    rmse = np.sqrt(np.mean((aligned_sig1 - aligned_sig2) ** 2))
    max_val = max(np.max(aligned_sig1), np.max(aligned_sig2))
    normalized_rmse = rmse / (max_val + 1e-8)
    
    # Determine similarity
    is_similar = (abs(final_correlation) >= similarity_threshold and 
                  p_value < 0.05 and 
                  normalized_rmse < 0.5)
    
    return {
        'similar': is_similar,
        'time_shift': optimal_shift,
        'correlation': final_correlation,
        'p_value': p_value,
        'normalized_rmse': normalized_rmse,
        'cross_correlation': ncc,
        'shift_correlation': shift_correlation,
        'aligned_length': len(aligned_sig1),
        'signature1_length': len(signature1),
        'signature2_length': len(signature2)
    }

# Example usage and test function
def create_test_queues():
    """Create test queues with similar fluctuation patterns."""
    # Base pattern
    base_colors = []
    for i in range(50):
        # Create a fluctuating pattern
        r = int(128 + 50 * np.sin(i * 0.3) + 30 * np.sin(i * 0.1))
        g = int(128 + 40 * np.cos(i * 0.25) + 20 * np.sin(i * 0.15))
        b = int(128 + 60 * np.sin(i * 0.2) + 25 * np.cos(i * 0.12))
        base_colors.append([max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))])
    
    # Queue 1: Original pattern
    queue1 = Queue()
    for color in base_colors:
        queue1.put(color)
    
    # Queue 2: Time-shifted and slightly modified pattern
    queue2 = Queue()
    shift = 5  # 5-frame delay
    for i in range(shift):
        queue2.put([128, 128, 128])  # Neutral color for delay
    
    for color in base_colors[:-shift]:
        # Add some brightness/contrast variation
        modified_color = [
            max(0, min(255, int(color[0] * 1.1 + 10))),  # Brighter red
            max(0, min(255, int(color[1] * 0.9 - 5))),   # Darker green
            max(0, min(255, int(color[2] * 1.05)))       # Slightly brighter blue
        ]
        queue2.put(modified_color)
    
    return queue1, queue2
