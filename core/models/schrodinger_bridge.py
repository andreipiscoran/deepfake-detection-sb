"""
Robust Gaussian Schrödinger Bridge
==================================

Proper implementation with:
1. Shrinkage covariance (no singularity)
2. Whitening (equalized support)
3. Distribution validation (overlap, condition, stability)
"""

import numpy as np
from typing import Tuple, Dict


class SchrodingerBridge:
    """
    Gaussian SB with proper regularization and validation.
    """
    
    def __init__(self, shrinkage: float = 0.1):
        """
        Args:
            shrinkage: λ in Σ = (1-λ)Σ̂ + λI, range [0.05, 0.2]
        """
        self.shrinkage = shrinkage
        
        self.mu_R = None
        self.mu_F = None
        self.whitening = None  # Σ_pooled^{-1/2}
        self.mu_R_white = None
        self.mu_F_white = None
    
    def shrinkage_cov(self, X: np.ndarray) -> np.ndarray:
        """Compute shrinkage covariance: (1-λ)Σ̂ + λ·trace(Σ̂)/d·I"""
        cov = np.cov(X.T)
        d = cov.shape[0]
        trace = np.trace(cov) / d
        return (1 - self.shrinkage) * cov + self.shrinkage * trace * np.eye(d)
    
    def fit(self, real: np.ndarray, fake: np.ndarray) -> Dict:
        """
        Fit Gaussian SB with validation.
        
        Returns diagnostic dict.
        """
        d = real.shape[1]
        
        # === Step 1: Means ===
        self.mu_R = real.mean(axis=0)
        self.mu_F = fake.mean(axis=0)
        
        # === Step 2: Shrinkage covariances ===
        cov_R = self.shrinkage_cov(real)
        cov_F = self.shrinkage_cov(fake)
        
        # === Step 3: Pooled covariance for whitening ===
        cov_pooled = 0.5 * (cov_R + cov_F)
        
        # === Step 4: Compute whitening transform ===
        eigvals, eigvecs = np.linalg.eigh(cov_pooled)
        eigvals = np.maximum(eigvals, 1e-6)  # numerical stability
        self.whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # === Step 5: Whiten the means ===
        self.mu_R_white = self.whitening @ self.mu_R
        self.mu_F_white = self.whitening @ self.mu_F
        
        # === Validation checks ===
        diagnostics = self._validate(cov_R, cov_F, cov_pooled, real, fake)
        
        return diagnostics
    
    def _validate(self, cov_R, cov_F, cov_pooled, real, fake) -> Dict:
        """Run diagnostic checks."""
        d = cov_R.shape[0]
        
        # 1. Mahalanobis distance between means
        diff = self.mu_R - self.mu_F
        mahal_dist = np.sqrt(diff @ np.linalg.solve(cov_pooled, diff))
        
        # 2. Condition numbers
        kappa_R = np.linalg.cond(cov_R)
        kappa_F = np.linalg.cond(cov_F)
        kappa_pooled = np.linalg.cond(cov_pooled)
        
        # 3. Bootstrap stability (quick check)
        n_boot = 5
        mu_R_boots = []
        mu_F_boots = []
        for _ in range(n_boot):
            idx_r = np.random.choice(len(real), len(real), replace=True)
            idx_f = np.random.choice(len(fake), len(fake), replace=True)
            mu_R_boots.append(real[idx_r].mean(0))
            mu_F_boots.append(fake[idx_f].mean(0))
        
        mu_R_std = np.std(mu_R_boots, axis=0).mean()
        mu_F_std = np.std(mu_F_boots, axis=0).mean()
        
        # Print diagnostics
        print(f"\n=== Distribution Diagnostics ===")
        print(f"Dimension: {d}")
        print(f"Samples: {len(real)} real, {len(fake)} fake")
        print(f"\nMahalanobis distance (μ_R ↔ μ_F): {mahal_dist:.2f}")
        if mahal_dist < 1:
            print(f"  ⚠️  LOW: distributions heavily overlap, SB may not help")
        elif mahal_dist > 10:
            print(f"  ⚠️  HIGH: distributions far apart, trivial classifier may suffice")
        else:
            print(f"  ✓ Good separation")
        
        print(f"\nCondition numbers:")
        print(f"  κ(Σ_R) = {kappa_R:.1e}")
        print(f"  κ(Σ_F) = {kappa_F:.1e}")
        print(f"  κ(Σ_pooled) = {kappa_pooled:.1e}")
        if max(kappa_R, kappa_F, kappa_pooled) > 1e4:
            print(f"  ⚠️  HIGH: increase shrinkage or reduce dimension")
        else:
            print(f"  ✓ Well-conditioned")
        
        print(f"\nBootstrap stability (mean std):")
        print(f"  Real: {mu_R_std:.4f}")
        print(f"  Fake: {mu_F_std:.4f}")
        if max(mu_R_std, mu_F_std) > 0.5:
            print(f"  ⚠️  HIGH: reduce dimension or get more data")
        else:
            print(f"  ✓ Stable")
        
        return {
            "mahal_dist": mahal_dist,
            "kappa_R": kappa_R,
            "kappa_F": kappa_F,
            "kappa_pooled": kappa_pooled,
            "mu_stability": max(mu_R_std, mu_F_std)
        }
    
    def detect(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify in whitened space.
        
        In whitened space, both distributions have identity covariance,
        so Mahalanobis distance = Euclidean distance.
        """
        # Whiten input
        x_white = x @ self.whitening.T
        
        # Euclidean distances to whitened means
        d_R = np.sum((x_white - self.mu_R_white) ** 2, axis=1)
        d_F = np.sum((x_white - self.mu_F_white) ** 2, axis=1)
        
        # Score: positive = closer to real
        scores = d_F - d_R
        preds = (scores <= 0).astype(int)  # 0=real, 1=fake
        
        return scores, preds
    
    def evaluate(self, real: np.ndarray, fake: np.ndarray) -> Dict:
        """Evaluate balanced accuracy."""
        _, pred_r = self.detect(real)
        _, pred_f = self.detect(fake)
        
        acc_r = (pred_r == 0).mean() * 100
        acc_f = (pred_f == 1).mean() * 100
        balanced = (acc_r + acc_f) / 2
        
        print(f"Accuracy: real={acc_r:.1f}%, fake={acc_f:.1f}%, balanced={balanced:.1f}%")
        return {"real": acc_r, "fake": acc_f, "balanced": balanced}
