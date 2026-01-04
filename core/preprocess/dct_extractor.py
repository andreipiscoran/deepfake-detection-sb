"""
DCT Feature Extraction for Deepfake Detection
Extracts frequency-domain features that expose deepfake artifacts.
"""

import numpy as np
import cv2
from scipy.fftpack import dct
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Zig-zag scan order for 8x8 DCT block
ZIGZAG = np.array([
    0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
])


def _process_video(args: Tuple[str, int]) -> Tuple[Optional[np.ndarray], int, str, Optional[str]]:
    """Worker function for parallel video processing. Returns (features, label, filename, error)."""
    video_path, label = args
    extractor = DCTExtractor()
    try:
        feat = extractor.extract_from_video(video_path)
        return (feat, label, Path(video_path).name, None)
    except Exception as e:
        return (None, label, Path(video_path).name, str(e))


class DCTExtractor:
    """Extract DCT features from images for deepfake detection."""
    
    def __init__(self, block_size: int = 8, n_coeffs: int = 21, n_bins: int = 32):
        self.block_size = block_size
        self.n_coeffs = n_coeffs
        self.n_bins = n_bins
        self.zigzag_idx = ZIGZAG[:n_coeffs]
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract DCT histogram features from an image."""
        # Convert to YCbCr (artifacts strongest in chroma)
        if image.ndim == 3:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycbcr = image[:, :, np.newaxis]
        
        h, w = ycbcr.shape[:2]
        bs = self.block_size
        features = []
        
        for ch in range(ycbcr.shape[2]):
            channel = ycbcr[:, :, ch].astype(np.float32)
            coeffs_list = []
            
            # Block-wise DCT
            for i in range(0, h - bs + 1, bs):
                for j in range(0, w - bs + 1, bs):
                    block = channel[i:i+bs, j:j+bs]
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    coeffs_list.append(dct_block.flatten()[self.zigzag_idx])
            
            coeffs = np.array(coeffs_list)
            
            # Histogram per coefficient
            for k in range(self.n_coeffs):
                vals = coeffs[:, k]
                hist, _ = np.histogram(vals, bins=self.n_bins, 
                                       range=(np.percentile(vals, 2), np.percentile(vals, 98)))
                features.append(hist / (hist.sum() + 1e-8))
        
        return np.concatenate(features).astype(np.float32)
    
    def extract_from_video(self, video_path: str, max_frames: int = 30, fps: int = 3) -> np.ndarray:
        """Extract and aggregate DCT features from video frames."""
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
        
        frame_features = [self.extract(f) for f in frames]
        features = np.array(frame_features)
        return np.concatenate([features.mean(0), features.std(0)])
    
    @property
    def feature_dim(self) -> int:
        """Output feature dimension (for video: mean + std)."""
        return self.n_coeffs * self.n_bins * 3 * 2  # 3 channels, 2 for mean+std


class DeepfakeDataset(Dataset):
    """PyTorch dataset for deepfake detection."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.real_idx = (self.labels == 0).nonzero(as_tuple=True)[0]
        self.fake_idx = (self.labels == 1).nonzero(as_tuple=True)[0]
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
    
    def get_real(self) -> torch.Tensor:
        return self.features[self.real_idx]
    
    def get_fake(self) -> torch.Tensor:
        return self.features[self.fake_idx]
    
    @property
    def num_real(self) -> int:
        return len(self.real_idx)
    
    @property
    def num_fake(self) -> int:
        return len(self.fake_idx)
    
    @classmethod
    def from_directory(cls, data_dir: str, extractor: DCTExtractor) -> "DeepfakeDataset":
        """Load dataset from directory structure: data_dir/{real,fake}/*.mp4"""
        data_dir = Path(data_dir)
        features, labels = [], []
        
        for label, folder in enumerate(["real", "fake"]):
            folder_path = data_dir / folder
            if not folder_path.exists():
                continue
            for video_path in folder_path.glob("*.mp4"):
                try:
                    feat = extractor.extract_from_video(str(video_path))
                    features.append(feat)
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping {video_path}: {e}")
        
        return cls(np.array(features), np.array(labels))
    
    @classmethod
    def from_ffc23(cls, data_dir: str, extractor: DCTExtractor, 
                   fake_types: Optional[List[str]] = None,
                   max_per_class: Optional[int] = None,
                   n_workers: Optional[int] = None) -> "DeepfakeDataset":
        """
        Load FF-C23 dataset structure:
            data_dir/original/*.mp4           -> real (label=0)
            data_dir/{fake_type}/*.mp4        -> fake (label=1)
        
        Args:
            data_dir: Path to FaceForensics++ C23 dataset
            extractor: DCTExtractor instance
            fake_types: List of fake folders to include. Default: all fake types
            max_per_class: Max samples per folder (for faster testing)
            n_workers: Number of parallel workers. Default: CPU count
        """
        data_dir = Path(data_dir)
        
        # FF-C23 fake method folders
        all_fake_types = [
            "DeepFakeDetection", "Deepfakes", "Face2Face", 
            "FaceShifter", "FaceSwap", "NeuralTextures"
        ]
        fake_types = fake_types or all_fake_types
        
        # Collect all video tasks: (video_path, label)
        tasks = []
        
        # Real videos from 'original' folder
        real_path = data_dir / "original"
        if real_path.exists():
            video_files = list(real_path.glob("*.mp4"))
            if max_per_class:
                video_files = video_files[:max_per_class]
            tasks.extend([(str(v), 0) for v in video_files])
            print(f"Found {len(video_files)} real videos from 'original'")
        
        # Fake videos from each manipulation type
        for fake_type in fake_types:
            fake_path = data_dir / fake_type
            if not fake_path.exists():
                print(f"Warning: {fake_type} folder not found")
                continue
            video_files = list(fake_path.glob("*.mp4"))
            if max_per_class:
                video_files = video_files[:max_per_class]
            tasks.extend([(str(v), 1) for v in video_files])
            print(f"Found {len(video_files)} fake videos from '{fake_type}'")
        
        features, labels = [], []
        errors = 0
        
        n_workers = n_workers or min(os.cpu_count() or 4, 10)
        print(f"\nProcessing {len(tasks)} videos with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_video, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures), 1):
                feat, label, filename, error = future.result()
                
                if error:
                    errors += 1
                    print(f"  [{i}/{len(tasks)}] Error: {filename}: {error}")
                else:
                    features.append(feat)
                    labels.append(label)
                    print(f"  [{i}/{len(tasks)}] Done ({errors} errors)", end="\r")
        
        print(f"\nTotal: {labels.count(0)} real, {labels.count(1)} fake ({errors} errors)")
        return cls(np.array(features), np.array(labels))
