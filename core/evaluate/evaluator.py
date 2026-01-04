"""
Evaluation module for GMM-based deepfake detector.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from core.models import SchrodingerBridge
from core.preprocess import DeepfakeDataset


class Evaluator:
    """Evaluate the GMM-based detector."""
    
    def __init__(self, model: SchrodingerBridge, **kwargs):
        self.model = model
    
    def compute_metrics(self, dataset: DeepfakeDataset) -> Dict:
        """Compute detection metrics on a dataset."""
        features = dataset.features
        labels = dataset.labels.numpy()
        
        with torch.no_grad():
            scores, _ = self.model.detect(features)
        scores = scores.cpu().numpy()
        
        # Find optimal threshold via ROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # EER: where FPR = 1 - TPR
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        optimal_threshold = thresholds[eer_idx]
        
        predictions = (scores > optimal_threshold).astype(int)
        
        return {
            "roc_auc": roc_auc_score(labels, scores),
            "eer": eer,
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "threshold": optimal_threshold,
        }
    
    def compute_bridge_complexity(self, dataset: DeepfakeDataset, n_samples: int = 200) -> Dict:
        """
        Compute GMM-based complexity metrics.
        """
        real = dataset.get_real()[:n_samples]
        fake = dataset.get_fake()[:n_samples]
        
        # Log-likelihoods
        ll_real_on_real, ll_fake_on_real = self.model.compute_log_likelihood(real.numpy())
        ll_real_on_fake, ll_fake_on_fake = self.model.compute_log_likelihood(fake.numpy())
        
        return {
            "ll_real_on_real_gmm": ll_real_on_real.mean(),
            "ll_real_on_fake_gmm": ll_fake_on_real.mean(),
            "ll_fake_on_real_gmm": ll_real_on_fake.mean(),
            "ll_fake_on_fake_gmm": ll_fake_on_fake.mean(),
            "ll_gap_real": (ll_real_on_real - ll_fake_on_real).mean(),
            "ll_gap_fake": (ll_fake_on_fake - ll_real_on_fake).mean(),
        }
    
    def plot_results(self, dataset: DeepfakeDataset, save_path: Optional[str] = None):
        """Generate visualization of detection results."""
        features = dataset.features
        labels = dataset.labels.numpy()
        
        with torch.no_grad():
            scores, _ = self.model.detect(features)
        scores = scores.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Score distribution
        axes[0].hist(scores[labels == 0], bins=30, alpha=0.7, label="Real", density=True, color='green')
        axes[0].hist(scores[labels == 1], bins=30, alpha=0.7, label="Fake", density=True, color='red')
        axes[0].axvline(self.model.threshold.item(), color='black', linestyle='--', 
                       label=f'Threshold: {self.model.threshold.item():.3f}')
        axes[0].set_xlabel("Log-Likelihood Ratio (fake - real)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Score Distribution")
        axes[0].legend()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1].fill_between(fpr, tpr, alpha=0.2)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        
        return fig
