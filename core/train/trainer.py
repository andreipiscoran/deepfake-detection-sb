"""
Robust Gaussian SB Trainer
==========================

1. Global normalization (before split)
2. PCA to controlled dimension
3. Fit with validation
4. Baseline comparison
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from core.models import SchrodingerBridge


class Trainer:
    def __init__(self, dataset, pca_dim: int = 64, val_split: float = 0.2, 
                 shrinkage: float = 0.1, **kwargs):
        features = dataset.features.numpy()
        labels = dataset.labels.numpy()
        
        # === Step 1: Global normalization (before split!) ===
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8
        features = (features - self.mean) / self.std
        
        # === Step 2: PCA to controlled dimension ===
        print(f"PCA: {features.shape[1]} → {pca_dim}")
        self.pca = PCA(n_components=pca_dim, random_state=42)
        features = self.pca.fit_transform(features)
        print(f"  Variance explained: {self.pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        # === Step 3: Stratified split ===
        real_idx = np.where(labels == 0)[0]
        fake_idx = np.where(labels == 1)[0]
        np.random.seed(42)
        np.random.shuffle(real_idx)
        np.random.shuffle(fake_idx)
        
        n_val_r = max(1, int(len(real_idx) * val_split))
        n_val_f = max(1, int(len(fake_idx) * val_split))
        
        self.train_real = features[real_idx[n_val_r:]]
        self.train_fake = features[fake_idx[n_val_f:]]
        self.val_real = features[real_idx[:n_val_r]]
        self.val_fake = features[fake_idx[:n_val_f]]
        
        # Report class balance
        n_real = len(real_idx)
        n_fake = len(fake_idx)
        print(f"\nData: {n_real} real ({100*n_real/(n_real+n_fake):.1f}%), "
              f"{n_fake} fake ({100*n_fake/(n_real+n_fake):.1f}%)")
        print(f"Train: {len(self.train_real)} real, {len(self.train_fake)} fake")
        print(f"Val:   {len(self.val_real)} real, {len(self.val_fake)} fake")
        
        self.model = SchrodingerBridge(shrinkage=shrinkage)
    
    def train(self, save_dir: str = None, **kwargs) -> dict:
        """Fit, validate, and compare to baseline."""
        
        # === Baseline: Balanced Logistic Regression ===
        print("\n" + "="*50)
        print("BASELINE: Logistic Regression (balanced)")
        print("="*50)
        
        X_train = np.vstack([self.train_real, self.train_fake])
        y_train = np.array([0]*len(self.train_real) + [1]*len(self.train_fake))
        
        lr = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
        lr.fit(X_train, y_train)
        
        pred_val_r = lr.predict(self.val_real)
        pred_val_f = lr.predict(self.val_fake)
        lr_real = (pred_val_r == 0).mean() * 100
        lr_fake = (pred_val_f == 1).mean() * 100
        lr_balanced = (lr_real + lr_fake) / 2
        print(f"Val: real={lr_real:.1f}%, fake={lr_fake:.1f}%, balanced={lr_balanced:.1f}%")
        
        # === Gaussian SB ===
        print("\n" + "="*50)
        print("GAUSSIAN SCHRÖDINGER BRIDGE")
        print("="*50)
        
        diagnostics = self.model.fit(self.train_real, self.train_fake)
        
        print("\n--- Train ---")
        self.model.evaluate(self.train_real, self.train_fake)
        
        print("\n--- Val ---")
        metrics = self.model.evaluate(self.val_real, self.val_fake)
        
        # === Summary ===
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"LogReg balanced acc:     {lr_balanced:.1f}%")
        print(f"Gaussian SB balanced acc: {metrics['balanced']:.1f}%")
        
        if metrics['balanced'] < lr_balanced - 2:
            print("\n⚠️  SB underperforms LogReg - features may not suit Gaussian assumption")
        elif metrics['balanced'] > lr_balanced + 2:
            print("\n✓ SB outperforms LogReg")
        else:
            print("\n≈ Similar performance")
        
        if save_dir:
            self._save(save_dir)
        
        return {"val_acc": [metrics["balanced"]], "diagnostics": diagnostics}
    
    def _save(self, save_dir: str):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "checkpoint.npz",
            mu_R=self.model.mu_R,
            mu_F=self.model.mu_F,
            whitening=self.model.whitening,
            mu_R_white=self.model.mu_R_white,
            mu_F_white=self.model.mu_F_white,
            mean=self.mean,
            std=self.std,
            pca_components=self.pca.components_,
            pca_mean=self.pca.mean_,
        )
        print(f"\nSaved to {path / 'checkpoint.npz'}")
