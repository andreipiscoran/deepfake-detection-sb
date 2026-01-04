#!/usr/bin/env python3
"""
Test Localized DCT vs Global Features
=====================================

Quick comparison of localized patch-based DCT vs global features
on existing preprocessed data.

Usage:
    python test_localized.py data/cnn
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: str):
    """Load features and labels."""
    data_path = Path(data_dir)
    features = np.load(data_path / "features.npy")
    labels = np.load(data_path / "labels.npy")
    return features, labels


def evaluate_method(name, y_true, y_pred, y_train_true=None, y_train_pred=None):
    """Print evaluation metrics."""
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    real_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]) * 100 if (y_true == 0).any() else 0
    fake_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]) * 100 if (y_true == 1).any() else 0
    
    print(f"\n{name}:")
    if y_train_true is not None and y_train_pred is not None:
        train_bal = balanced_accuracy_score(y_train_true, y_train_pred) * 100
        print(f"  Train: balanced={train_bal:.1f}%")
    print(f"  Val:   real={real_acc:.1f}%, fake={fake_acc:.1f}%, balanced={bal_acc:.1f}%")
    
    return bal_acc


def evaluate_with_threshold_tuning(name, model, X_val, y_val, X_train=None, y_train=None):
    """Evaluate with optimal threshold selection."""
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val)[:, 1]
        train_probs = model.predict_proba(X_train)[:, 1] if X_train is not None else None
    else:
        return None
    
    # Find optimal threshold on training data (or val if no train)
    best_thresh = 0.5
    best_bal_acc = 0
    
    search_probs = train_probs if train_probs is not None else probs
    search_labels = y_train if y_train is not None else y_val
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (search_probs >= thresh).astype(int)
        bal = balanced_accuracy_score(search_labels, preds)
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_thresh = thresh
    
    # Apply optimal threshold to val
    y_pred = (probs >= best_thresh).astype(int)
    bal_acc = balanced_accuracy_score(y_val, y_pred) * 100
    real_acc = accuracy_score(y_val[y_val == 0], y_pred[y_val == 0]) * 100 if (y_val == 0).any() else 0
    fake_acc = accuracy_score(y_val[y_val == 1], y_pred[y_val == 1]) * 100 if (y_val == 1).any() else 0
    
    print(f"\n{name} (tuned thresh={best_thresh:.2f}):")
    print(f"  Val:   real={real_acc:.1f}%, fake={fake_acc:.1f}%, balanced={bal_acc:.1f}%")
    
    return bal_acc


def mahalanobis_classifier(X_train, y_train, X_val, shrinkage=0.1):
    """Simple Mahalanobis-based classifier."""
    real_train = X_train[y_train == 0]
    fake_train = X_train[y_train == 1]
    
    mu_r = real_train.mean(axis=0)
    mu_f = fake_train.mean(axis=0)
    
    # Shrinkage covariance
    def shrunk_cov(X):
        cov = np.cov(X, rowvar=False)
        return (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    
    cov_pooled = shrunk_cov(X_train)
    cov_inv = np.linalg.inv(cov_pooled)
    
    # Predict based on distance to centroids
    preds = []
    for x in X_val:
        d_r = np.sqrt((x - mu_r) @ cov_inv @ (x - mu_r))
        d_f = np.sqrt((x - mu_f) @ cov_inv @ (x - mu_f))
        preds.append(0 if d_r < d_f else 1)
    
    return np.array(preds)


def test_classifiers(features, labels, pca_dims=[32, 64, 128, 256], val_split=0.2, balance=True):
    """Test multiple classifiers at different PCA dimensions."""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=val_split, random_state=42, stratify=labels
    )
    
    print(f"\nData: {len(X_train)} train, {len(X_val)} val")
    print(f"Train: {(y_train == 0).sum()} real, {(y_train == 1).sum()} fake")
    print(f"Val:   {(y_val == 0).sum()} real, {(y_val == 1).sum()} fake")
    
    # Balance training data by undersampling majority class
    if balance:
        real_idx = np.where(y_train == 0)[0]
        fake_idx = np.where(y_train == 1)[0]
        
        n_minority = min(len(real_idx), len(fake_idx))
        
        np.random.seed(42)
        if len(real_idx) > len(fake_idx):
            real_idx = np.random.choice(real_idx, n_minority, replace=False)
        else:
            fake_idx = np.random.choice(fake_idx, n_minority, replace=False)
        
        balanced_idx = np.concatenate([real_idx, fake_idx])
        np.random.shuffle(balanced_idx)
        
        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]
        
        print(f"Balanced: {(y_train == 0).sum()} real, {(y_train == 1).sum()} fake")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    results = {}
    
    for dim in pca_dims:
        if dim > min(X_train_scaled.shape):
            continue
            
        print(f"\n{'='*60}")
        print(f"PCA DIM: {dim}")
        print(f"{'='*60}")
        
        # PCA
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"Variance explained: {var_explained:.1f}%")
        
        # Compute Mahalanobis distance for diagnostics
        real_train = X_train_pca[y_train == 0]
        fake_train = X_train_pca[y_train == 1]
        mu_diff = real_train.mean(0) - fake_train.mean(0)
        cov_pooled = np.cov(X_train_pca, rowvar=False)
        cov_pooled += 0.1 * np.eye(dim) * np.trace(cov_pooled) / dim
        try:
            mahal = np.sqrt(mu_diff @ np.linalg.inv(cov_pooled) @ mu_diff)
        except:
            mahal = 0
        print(f"Mahalanobis distance: {mahal:.2f}")
        
        dim_results = {}
        
        # 1. Logistic Regression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(X_train_pca, y_train)
        y_pred_train = lr.predict(X_train_pca)
        y_pred = lr.predict(X_val_pca)
        dim_results['LogReg'] = evaluate_method("Logistic Regression", y_val, y_pred, y_train, y_pred_train)
        
        # Also try with threshold tuning
        tuned_acc = evaluate_with_threshold_tuning("LogReg+Tuned", lr, X_val_pca, y_val, X_train_pca, y_train)
        if tuned_acc:
            dim_results['LogReg+T'] = tuned_acc
        
        # 2. LDA
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_pca, y_train)
            y_pred_train = lda.predict(X_train_pca)
            y_pred = lda.predict(X_val_pca)
            dim_results['LDA'] = evaluate_method("LDA", y_val, y_pred, y_train, y_pred_train)
        except Exception as e:
            print(f"\nLDA: failed ({e})")
        
        # 3. QDA (handles different covariances)
        try:
            qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
            qda.fit(X_train_pca, y_train)
            y_pred_train = qda.predict(X_train_pca)
            y_pred = qda.predict(X_val_pca)
            dim_results['QDA'] = evaluate_method("QDA", y_val, y_pred, y_train, y_pred_train)
        except Exception as e:
            print(f"\nQDA: failed ({e})")
        
        # 4. Mahalanobis classifier
        y_pred_train = mahalanobis_classifier(X_train_pca, y_train, X_train_pca)
        y_pred = mahalanobis_classifier(X_train_pca, y_train, X_val_pca)
        dim_results['Mahalanobis'] = evaluate_method("Mahalanobis", y_val, y_pred, y_train, y_pred_train)
        
        results[dim] = dim_results
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY (Balanced Accuracy %)")
    print(f"{'='*60}")
    
    methods = ['LogReg', 'LogReg+T', 'LDA', 'Mahalanobis']
    header = f"{'Dim':>6} " + " ".join(f"{m:>10}" for m in methods)
    print(header)
    print("-" * len(header))
    
    for dim in pca_dims:
        if dim not in results:
            continue
        row = f"{dim:>6} "
        for m in methods:
            val = results[dim].get(m, 0)
            row += f"{val:>10.1f} "
        print(row)
    
    # Best result
    best_acc = 0
    best_config = None
    for dim, dim_results in results.items():
        for method, acc in dim_results.items():
            if acc > best_acc:
                best_acc = acc
                best_config = (dim, method)
    
    if best_config:
        print(f"\n→ Best: {best_config[1]} @ {best_config[0]}D = {best_acc:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test classifiers on feature data")
    parser.add_argument("data_dir", help="Directory with features.npy and labels.npy")
    parser.add_argument("--pca_dims", type=int, nargs="+", default=[32, 64, 128, 256],
                       help="PCA dimensions to test")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--no-balance", action="store_true", 
                       help="Disable undersampling (default: balance classes)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}")
    features, labels = load_data(args.data_dir)
    
    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print(f"Classes: {(labels == 0).sum()} real, {(labels == 1).sum()} fake")
    
    test_classifiers(features, labels, args.pca_dims, args.val_split, balance=not args.no_balance)


if __name__ == "__main__":
    main()

