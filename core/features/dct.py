"""
DCT Feature Extractor for Deepfake Detection
=============================================

Extracts frequency-domain features that capture compression artifacts
and GAN fingerprints common in deepfakes.
"""

import numpy as np
from scipy.fftpack import dct
from pathlib import Path
from typing import Optional, Tuple
import cv2


def dct2d(block: np.ndarray) -> np.ndarray:
    """2D DCT on a block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def extract_dct_features(
    image: np.ndarray,
    block_size: int = 8,
    n_bands: int = 8
) -> np.ndarray:
    """
    Extract DCT features from an image.
    
    Args:
        image: RGB or grayscale image
        block_size: DCT block size (8 = JPEG standard)
        n_bands: Number of frequency bands to compute energy
    
    Returns:
        Feature vector capturing frequency statistics
    """
    # Convert to grayscale (luminance)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # Crop to multiple of block_size
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    gray = gray[:h, :w]
    
    # Compute DCT on all blocks
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    
    dct_coeffs = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = gray[i*block_size:(i+1)*block_size, 
                        j*block_size:(j+1)*block_size]
            dct_block = dct2d(block)
            dct_coeffs.append(dct_block.flatten())
    
    dct_coeffs = np.array(dct_coeffs)  # (n_blocks, block_size^2)
    
    # === Feature extraction ===
    features = []
    
    # 1. Per-coefficient statistics (mean, std) - skip DC
    coeff_means = dct_coeffs[:, 1:].mean(axis=0)  # skip DC
    coeff_stds = dct_coeffs[:, 1:].std(axis=0)
    features.extend(coeff_means[:32])  # first 32 AC coefficients
    features.extend(coeff_stds[:32])
    
    # 2. Frequency band energies (zigzag order approximation)
    band_energies = []
    for band in range(n_bands):
        # Simple band selection (distance from DC)
        band_mask = np.zeros(block_size * block_size)
        for idx in range(block_size * block_size):
            row, col = idx // block_size, idx % block_size
            dist = row + col
            if band <= dist < band + 2:
                band_mask[idx] = 1
        
        if band_mask.sum() > 0:
            band_energy = (dct_coeffs ** 2 * band_mask).sum(axis=1).mean()
            band_energies.append(np.log1p(band_energy))
    
    features.extend(band_energies)
    
    # 3. High-frequency energy ratio (artifact indicator)
    total_energy = (dct_coeffs[:, 1:] ** 2).sum()
    high_freq_energy = (dct_coeffs[:, block_size*block_size//2:] ** 2).sum()
    hf_ratio = high_freq_energy / (total_energy + 1e-10)
    features.append(hf_ratio)
    
    # 4. Block variance statistics (blocking artifact detection)
    block_variances = dct_coeffs.var(axis=1)
    features.append(block_variances.mean())
    features.append(block_variances.std())
    features.extend(np.percentile(block_variances, [25, 50, 75, 90]))
    
    return np.array(features, dtype=np.float32)


def extract_dct_from_video(
    video_path: str,
    max_frames: int = 32,
    block_size: int = 8
) -> np.ndarray:
    """
    Extract aggregated DCT features from a video.
    
    Args:
        video_path: Path to video file
        max_frames: Max frames to sample
        block_size: DCT block size
    
    Returns:
        Aggregated feature vector (mean + std over frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    
    frame_features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feat = extract_dct_features(frame, block_size=block_size)
        frame_features.append(feat)
    
    cap.release()
    
    if len(frame_features) == 0:
        raise ValueError(f"Could not read frames from {video_path}")
    
    frame_features = np.array(frame_features)
    
    # Aggregate: mean + std + percentiles
    agg_features = np.concatenate([
        frame_features.mean(axis=0),
        frame_features.std(axis=0),
        np.percentile(frame_features, 25, axis=0),
        np.percentile(frame_features, 75, axis=0),
    ])
    
    return agg_features


def extract_dct_from_image(image_path: str, block_size: int = 8) -> np.ndarray:
    """Extract DCT features from a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_dct_features(image, block_size=block_size)

