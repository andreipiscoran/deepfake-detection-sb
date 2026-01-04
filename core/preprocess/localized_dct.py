"""
Localized DCT Feature Extraction for Deepfake Detection
========================================================

Key insight: Deepfakes fail LOCALLY, not globally.
This extractor analyzes many small regions independently.

Instead of: face → DCT → aggregate → one big vector
We do:      face → patches → DCT per patch → score per patch → aggregate scores

This preserves local artifacts that get washed out by global averaging.
"""

import numpy as np
import cv2
from scipy.fftpack import dct
from scipy import stats
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PatchFeatures:
    """Features and metadata for a single patch."""
    features: np.ndarray  # Feature vector for this patch
    location: Tuple[int, int]  # (row, col) in patch grid
    position: Tuple[int, int, int, int]  # (y, x, h, w) in pixels


class LocalizedDCTExtractor:
    """
    Localized frequency analysis for deepfake detection.
    
    Pipeline:
    1. Divide face into patches (grid or semantic)
    2. Compute DCT per patch
    3. Extract local statistics per patch
    4. Score each patch independently
    5. Aggregate patch scores (top-k, max, etc.)
    
    This captures local artifacts that global methods miss.
    """
    
    def __init__(
        self,
        patch_size: int = 32,
        patch_stride: int = 16,  # Overlap for better coverage
        dct_block_size: int = 8,
        n_coeffs: int = 15,  # Per DCT block (skip DC)
        n_bands: int = 3,  # low/mid/high frequency bands
    ):
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dct_block_size = dct_block_size
        self.n_coeffs = n_coeffs
        self.n_bands = n_bands
        
        # Zig-zag indices for DCT coefficients (skip DC at index 0)
        self.zigzag = self._make_zigzag(dct_block_size)
        
        # Define frequency bands
        band_size = n_coeffs // n_bands
        self.bands = [
            (1, 1 + band_size),  # Low (skip DC)
            (1 + band_size, 1 + 2 * band_size),  # Mid
            (1 + 2 * band_size, n_coeffs + 1),  # High
        ]
    
    def _make_zigzag(self, size: int) -> np.ndarray:
        """Generate zigzag scan indices for NxN block."""
        indices = []
        for s in range(2 * size - 1):
            if s % 2 == 0:
                for i in range(min(s, size - 1), max(-1, s - size), -1):
                    indices.append(i * size + (s - i))
            else:
                for i in range(max(0, s - size + 1), min(s + 1, size)):
                    indices.append(i * size + (s - i))
        return np.array(indices)
    
    def extract_patches(self, image: np.ndarray) -> List[PatchFeatures]:
        """
        Step 1 & 2: Divide image into patches and compute DCT features per patch.
        
        Returns list of PatchFeatures, each containing:
        - features: local frequency statistics
        - location: grid position
        - position: pixel coordinates
        """
        # Convert to YCbCr (artifacts often strongest in chroma)
        if image.ndim == 3 and image.shape[2] == 3:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        else:
            ycbcr = image.astype(np.float32)
            if ycbcr.ndim == 2:
                ycbcr = ycbcr[:, :, np.newaxis]
        
        h, w = ycbcr.shape[:2]
        patches = []
        
        row_idx = 0
        for y in range(0, h - self.patch_size + 1, self.patch_stride):
            col_idx = 0
            for x in range(0, w - self.patch_size + 1, self.patch_stride):
                patch = ycbcr[y:y + self.patch_size, x:x + self.patch_size]
                features = self._extract_patch_features(patch)
                
                patches.append(PatchFeatures(
                    features=features,
                    location=(row_idx, col_idx),
                    position=(y, x, self.patch_size, self.patch_size)
                ))
                col_idx += 1
            row_idx += 1
        
        return patches
    
    def _extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Step 3: Extract local frequency statistics from a single patch.
        
        For each channel, computes:
        - Band energies (low/mid/high)
        - Coefficient variance per band
        - Kurtosis (tail heaviness - artifacts often have heavy tails)
        - Inter-band ratios
        """
        all_features = []
        bs = self.dct_block_size
        
        for ch in range(patch.shape[2]):
            channel = patch[:, :, ch]
            coeffs_list = []
            
            # Block-wise DCT within patch
            for i in range(0, self.patch_size - bs + 1, bs):
                for j in range(0, self.patch_size - bs + 1, bs):
                    block = channel[i:i + bs, j:j + bs]
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    flat = dct_block.flatten()[self.zigzag]
                    coeffs_list.append(flat[1:self.n_coeffs + 1])  # Skip DC
            
            if not coeffs_list:
                # Patch too small for DCT blocks
                all_features.extend([0] * (self.n_bands * 5 + 2))
                continue
            
            coeffs = np.array(coeffs_list)  # (n_blocks, n_coeffs)
            
            # Features per band
            band_features = []
            band_energies = []
            
            for lo, hi in self.bands:
                lo_idx, hi_idx = lo - 1, min(hi - 1, coeffs.shape[1])
                band_coeffs = coeffs[:, lo_idx:hi_idx].flatten()
                
                if len(band_coeffs) == 0:
                    band_features.extend([0, 0, 0, 0])
                    band_energies.append(1e-8)
                    continue
                
                # 1. Band energy (mean squared magnitude)
                energy = np.mean(band_coeffs ** 2)
                band_energies.append(energy + 1e-8)
                
                # 2. Variance within band
                var = np.var(band_coeffs)
                
                # 3. Kurtosis (heavy tails = potential artifacts)
                kurt = stats.kurtosis(band_coeffs) if len(band_coeffs) > 4 else 0
                
                # 4. Coefficient of variation
                cv = np.std(band_coeffs) / (np.mean(np.abs(band_coeffs)) + 1e-8)
                
                band_features.extend([
                    np.log1p(energy),  # Log energy (more stable)
                    np.log1p(var),
                    np.clip(kurt, -10, 10),  # Clip extreme kurtosis
                    cv
                ])
            
            all_features.extend(band_features)
            
            # Inter-band ratios (artifacts often affect specific bands)
            total_energy = sum(band_energies)
            for e in band_energies:
                all_features.append(e / total_energy)
            
            # High/low ratio (deepfakes often have unusual high-freq content)
            all_features.append(band_energies[-1] / band_energies[0])
            
            # Overall coefficient statistics for this channel
            all_coeffs = coeffs.flatten()
            all_features.append(stats.skew(all_coeffs) if len(all_coeffs) > 4 else 0)
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[PatchFeatures]]:
        """
        Full extraction pipeline for a single image.
        
        Returns:
        - image_features: aggregated feature vector for the image
        - patches: list of per-patch features (for detailed analysis)
        """
        patches = self.extract_patches(image)
        
        if not patches:
            return np.zeros(self.feature_dim, dtype=np.float32), []
        
        # Stack all patch features
        patch_matrix = np.stack([p.features for p in patches])
        
        # Image-level aggregation (preserve local info via statistics)
        image_features = self._aggregate_patches(patch_matrix)
        
        return image_features, patches
    
    def _aggregate_patches(self, patch_matrix: np.ndarray) -> np.ndarray:
        """
        Aggregate patch features into image-level representation.
        
        Preserves local variation by computing:
        - Mean (central tendency)
        - Std (variation across patches)
        - Max (most extreme patch)
        - Percentiles (distribution shape)
        """
        if len(patch_matrix) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        features = []
        
        # Per-feature statistics across patches
        features.append(np.mean(patch_matrix, axis=0))
        features.append(np.std(patch_matrix, axis=0))
        features.append(np.max(patch_matrix, axis=0))
        features.append(np.percentile(patch_matrix, 90, axis=0))
        features.append(np.percentile(patch_matrix, 10, axis=0))
        
        # Spatial consistency: variance of neighboring patches
        # High variance = potential artifact boundary
        
        return np.concatenate(features).astype(np.float32)
    
    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = 30,
        fps: int = 3
    ) -> np.ndarray:
        """Extract features from video, aggregating across frames."""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_idx = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Extract from each frame
        frame_features = []
        for frame in frames:
            feat, _ = self.extract_image(frame)
            frame_features.append(feat)
        
        features = np.array(frame_features)
        
        # Temporal aggregation: mean + std across frames
        return np.concatenate([features.mean(0), features.std(0)])
    
    @property
    def patch_feature_dim(self) -> int:
        """Feature dimension per patch."""
        # Per channel: (n_bands * 4 stats) + n_bands ratios + high/low ratio + skew
        per_channel = self.n_bands * 4 + self.n_bands + 1 + 1
        return per_channel * 3  # 3 channels (Y, Cb, Cr)
    
    @property
    def feature_dim(self) -> int:
        """Total output feature dimension for video (with temporal aggregation)."""
        # 5 aggregation stats * patch_dim * 2 (mean + std over frames)
        return self.patch_feature_dim * 5 * 2


class PatchScorer:
    """
    Score patches using a reference distribution of "real" patches.
    
    Implements the key insight: score each patch independently,
    then aggregate scores to detect localized artifacts.
    """
    
    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        self.mean_real = None
        self.cov_inv = None
        self.fitted = False
    
    def fit(self, real_patches: np.ndarray):
        """
        Fit reference distribution from real patch features.
        
        Args:
            real_patches: (n_patches, n_features) array of real patch features
        """
        self.mean_real = real_patches.mean(axis=0)
        
        # Shrinkage covariance for stability
        cov = np.cov(real_patches, rowvar=False)
        cov_shrunk = (1 - self.shrinkage) * cov + self.shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        
        self.cov_inv = np.linalg.inv(cov_shrunk)
        self.fitted = True
    
    def score_patch(self, patch_features: np.ndarray) -> float:
        """
        Compute anomaly score for a single patch.
        
        Uses Mahalanobis distance to real patch distribution.
        High score = patch looks unusual compared to real patches.
        """
        if not self.fitted:
            raise RuntimeError("PatchScorer not fitted. Call fit() first.")
        
        diff = patch_features - self.mean_real
        mahal = np.sqrt(diff @ self.cov_inv @ diff)
        return mahal
    
    def score_image(
        self,
        patches: List[PatchFeatures],
        aggregation: str = "top_k",
        k: int = 5
    ) -> Tuple[float, Dict]:
        """
        Score an image by aggregating patch scores.
        
        Args:
            patches: List of PatchFeatures from LocalizedDCTExtractor
            aggregation: How to combine scores:
                - "max": Most suspicious patch
                - "top_k": Average of k most suspicious patches
                - "mean": Average all patches (not recommended)
                - "threshold": Count patches above threshold
            k: Number of patches for top_k aggregation
        
        Returns:
            - score: Overall image score
            - details: Dict with per-patch scores and locations
        """
        if not patches:
            return 0.0, {"patch_scores": [], "locations": []}
        
        # Score each patch
        patch_scores = np.array([self.score_patch(p.features) for p in patches])
        locations = [p.location for p in patches]
        
        # Aggregate
        if aggregation == "max":
            score = np.max(patch_scores)
        elif aggregation == "top_k":
            k = min(k, len(patch_scores))
            top_k_idx = np.argsort(patch_scores)[-k:]
            score = np.mean(patch_scores[top_k_idx])
        elif aggregation == "mean":
            score = np.mean(patch_scores)
        elif aggregation == "threshold":
            # Threshold at 95th percentile of training scores
            threshold = getattr(self, 'threshold_', 3.0)
            score = np.mean(patch_scores > threshold)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return score, {
            "patch_scores": patch_scores.tolist(),
            "locations": locations,
            "max_score": float(np.max(patch_scores)),
            "max_location": locations[np.argmax(patch_scores)],
        }


def extract_all_patches(
    extractor: LocalizedDCTExtractor,
    video_paths: List[str],
    labels: List[int],
    max_frames: int = 10,
    fps: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patch-level features from multiple videos.
    
    Returns:
        patch_features: (n_total_patches, feature_dim) array
        patch_labels: (n_total_patches,) array of labels
    """
    all_features = []
    all_labels = []
    
    for i, (path, label) in enumerate(zip(video_paths, labels)):
        print(f"  [{i+1}/{len(video_paths)}] {Path(path).name}", end="\r")
        
        cap = cv2.VideoCapture(path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(video_fps / fps))
        
        frame_idx = 0
        n_frames = 0
        
        while n_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame = cv2.resize(frame, (256, 256))
                patches = extractor.extract_patches(frame)
                for p in patches:
                    all_features.append(p.features)
                    all_labels.append(label)
                n_frames += 1
            frame_idx += 1
        
        cap.release()
    
    print()
    return np.array(all_features), np.array(all_labels)

