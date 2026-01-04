"""
CNN Feature Extraction for Deepfake Detection
Uses pretrained EfficientNet for powerful learned features.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class CNNExtractor:
    """Extract CNN features from images using pretrained EfficientNet."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load pretrained EfficientNet-B0
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove classifier, keep feature extractor
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # ImageNet normalization
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.feature_dim = 1280  # EfficientNet-B0 output dim
    
    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract CNN features from a single image."""
        # Convert BGR to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(x).squeeze().cpu().numpy()
        return features
    
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract CNN features from a batch of images."""
        batch = []
        for img in images:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(self.transform(img))
        
        x = torch.stack(batch).to(self.device)
        features = self.model(x).squeeze(-1).squeeze(-1).cpu().numpy()
        return features
    
    def extract_from_video(self, video_path: str, max_frames: int = 16, fps: int = 2) -> np.ndarray:
        """Extract and aggregate CNN features from video frames."""
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
                frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Batch extract features
        features = self.extract_batch(frames)
        
        # Aggregate: mean + std across frames
        return np.concatenate([features.mean(0), features.std(0)])


def _process_video_cnn(args: Tuple[str, int]) -> Tuple[Optional[np.ndarray], int, str, Optional[str]]:
    """Worker function for CNN video processing."""
    video_path, label = args
    extractor = CNNExtractor(device="cpu")
    try:
        feat = extractor.extract_from_video(video_path)
        return (feat, label, Path(video_path).name, None)
    except Exception as e:
        return (None, label, Path(video_path).name, str(e))

