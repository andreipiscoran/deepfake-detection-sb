#!/usr/bin/env python3
"""
Data Analysis Script for Deepfake Detection
============================================

Analyzes statistical properties of feature data to guide method selection.

Usage:
    python analyze_data.py data/cnn
    python analyze_data.py data/test --pca_dims 32 64 128
    python analyze_data.py data/cnn --save_plots
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: str):
    """Load features and labels from directory."""
    data_path = Path(data_dir)
    features = np.load(data_path / "features.npy")
    labels = np.load(data_path / "labels.npy")
    
    # Load class names if available
    class_names_path = data_path / "class_names.npy"
    class_names = None
    if class_names_path.exists():
        class_names = np.load(class_names_path, allow_pickle=True)
    
    return features, labels, class_names


def basic_stats(features, labels, class_names=None):
    """Basic dataset statistics."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    n_samples, n_features = features.shape
    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()
    
    print(f"\nSamples:    {n_samples}")
    print(f"Features:   {n_features}")
    print(f"Real:       {n_real} ({100*n_real/n_samples:.1f}%)")
    print(f"Fake:       {n_fake} ({100*n_fake/n_samples:.1f}%)")
    print(f"Imbalance:  {max(n_real, n_fake) / min(n_real, n_fake):.1f}:1")
    
    # Per-class breakdown if available
    if class_names is not None:
        unique_classes = np.unique(class_names)
        print(f"\nClass breakdown:")
        for cls in unique_classes:
            count = (class_names == cls).sum()
            print(f"  {cls}: {count}")
    
    # Feature value ranges
    print(f"\nFeature value range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Mean feature norm:   {np.linalg.norm(features, axis=1).mean():.3f}")
    
    # Check for NaN/Inf
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"\n⚠️  WARNING: {nan_count} NaN, {inf_count} Inf values!")
    
    return {"n_samples": n_samples, "n_features": n_features, 
            "n_real": n_real, "n_fake": n_fake}


def distribution_analysis(features, labels):
    """Analyze distribution properties of real vs fake."""
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    
    # Compute means and covariances
    mu_r, mu_f = real_feat.mean(axis=0), fake_feat.mean(axis=0)
    
    # Use shrinkage for numerical stability
    def shrunk_cov(X, shrinkage=0.1):
        cov = np.cov(X, rowvar=False)
        return (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    
    cov_r = shrunk_cov(real_feat)
    cov_f = shrunk_cov(fake_feat)
    cov_pooled = (cov_r * len(real_feat) + cov_f * len(fake_feat)) / len(features)
    
    # Mahalanobis distance between class means
    try:
        cov_inv = np.linalg.inv(cov_pooled)
        diff = mu_r - mu_f
        mahal = np.sqrt(diff @ cov_inv @ diff)
    except:
        mahal = np.nan
    
    print(f"\nMahalanobis distance (μ_R ↔ μ_F): {mahal:.3f}")
    if mahal < 1.0:
        print("  ⚠️  LOW (<1): Heavy overlap, linear methods may struggle")
    elif mahal < 2.0:
        print("  ⚡ MODERATE (1-2): Some separation, SB/LDA may help")
    else:
        print("  ✓ HIGH (>2): Good separation, most methods should work")
    
    # Euclidean distance between means (normalized)
    euclidean_dist = np.linalg.norm(mu_r - mu_f)
    pooled_std = np.sqrt(np.diag(cov_pooled).mean())
    normalized_dist = euclidean_dist / pooled_std
    print(f"\nNormalized Euclidean distance: {normalized_dist:.3f}")
    
    # Bhattacharyya distance (measures distribution overlap)
    try:
        cov_avg = (cov_r + cov_f) / 2
        diff = mu_r - mu_f
        term1 = 0.125 * diff @ np.linalg.inv(cov_avg) @ diff
        term2 = 0.5 * np.log(np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov_r) * np.linalg.det(cov_f)))
        bhatt = term1 + term2
        print(f"Bhattacharyya distance: {bhatt:.3f}")
        if bhatt < 0.5:
            print("  ⚠️  Very high overlap")
        elif bhatt < 1.0:
            print("  ⚡ Moderate overlap")
        else:
            print("  ✓ Low overlap")
    except:
        print("Bhattacharyya distance: (failed to compute)")
    
    return {"mahalanobis": mahal, "euclidean_norm": normalized_dist}


def covariance_analysis(features, labels):
    """Analyze covariance structure."""
    print("\n" + "="*60)
    print("COVARIANCE ANALYSIS")
    print("="*60)
    
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    
    def analyze_cov(X, name):
        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]  # Filter numerical zeros
        
        condition = eigvals.max() / eigvals.min() if len(eigvals) > 0 else np.inf
        eff_rank = np.exp(stats.entropy(eigvals / eigvals.sum())) if len(eigvals) > 0 else 0
        var_90 = np.searchsorted(np.cumsum(eigvals[::-1]) / eigvals.sum(), 0.9) + 1
        
        print(f"\n{name}:")
        print(f"  Condition number: {condition:.2e}")
        print(f"  Effective rank:   {eff_rank:.1f} / {len(eigvals)}")
        print(f"  Dims for 90% var: {var_90}")
        
        if condition > 1e6:
            print(f"  ⚠️  ILL-CONDITIONED: Consider regularization/PCA")
        
        return {"condition": condition, "eff_rank": eff_rank, "var_90": var_90}
    
    real_stats = analyze_cov(real_feat, "Real covariance")
    fake_stats = analyze_cov(fake_feat, "Fake covariance")
    
    # Compare covariance structures
    cov_r = np.cov(real_feat, rowvar=False)
    cov_f = np.cov(fake_feat, rowvar=False)
    
    # Relative Frobenius distance
    frob_dist = np.linalg.norm(cov_r - cov_f, 'fro') / np.linalg.norm(cov_r + cov_f, 'fro')
    print(f"\nCovariance similarity (Frobenius): {1 - frob_dist:.3f}")
    if frob_dist < 0.2:
        print("  Similar covariances → homoscedastic assumption OK")
    else:
        print("  Different covariances → consider heteroscedastic methods")
    
    return {"real": real_stats, "fake": fake_stats, "cov_similarity": 1 - frob_dist}


def feature_importance(features, labels, top_k=20):
    """Find most discriminative features."""
    print("\n" + "="*60)
    print("FEATURE DISCRIMINABILITY")
    print("="*60)
    
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    
    n_features = features.shape[1]
    
    # T-test for each feature
    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    effect_sizes = np.zeros(n_features)  # Cohen's d
    
    for i in range(n_features):
        t, p = stats.ttest_ind(real_feat[:, i], fake_feat[:, i])
        t_stats[i] = abs(t)
        p_values[i] = p
        
        # Cohen's d
        pooled_std = np.sqrt((real_feat[:, i].var() + fake_feat[:, i].var()) / 2)
        if pooled_std > 0:
            effect_sizes[i] = abs(real_feat[:, i].mean() - fake_feat[:, i].mean()) / pooled_std
    
    # Sort by effect size
    top_idx = np.argsort(effect_sizes)[::-1][:top_k]
    
    print(f"\nTop {top_k} most discriminative features:")
    print(f"{'Idx':>6} {'Cohen d':>10} {'t-stat':>10} {'p-value':>12}")
    print("-" * 42)
    
    significant = 0
    for i, idx in enumerate(top_idx):
        sig = "*" if p_values[idx] < 0.05 / n_features else ""  # Bonferroni
        if sig:
            significant += 1
        print(f"{idx:>6} {effect_sizes[idx]:>10.3f} {t_stats[idx]:>10.2f} {p_values[idx]:>12.2e} {sig}")
    
    # Summary
    n_sig = (p_values < 0.05 / n_features).sum()
    n_large = (effect_sizes > 0.5).sum()
    n_medium = ((effect_sizes > 0.3) & (effect_sizes <= 0.5)).sum()
    
    print(f"\nSummary:")
    print(f"  Significant (Bonferroni): {n_sig}/{n_features}")
    print(f"  Large effect (d>0.5):     {n_large}")
    print(f"  Medium effect (d>0.3):    {n_medium}")
    
    if n_large < 5:
        print("  ⚠️  Few discriminative features - consider different representation")
    
    return {"top_features": top_idx.tolist(), "n_significant": int(n_sig)}


def pca_analysis(features, labels, dims_to_try=[16, 32, 64, 128, 256]):
    """Analyze PCA projections."""
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    
    # Standardize first
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Full PCA to get variance explained
    max_dim = min(features.shape[0], features.shape[1])
    pca_full = PCA(n_components=min(max_dim, 512))
    pca_full.fit(features_scaled)
    
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    print(f"\nVariance explained:")
    for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
        n_dims = np.searchsorted(cumvar, threshold) + 1
        if n_dims <= len(cumvar):
            print(f"  {threshold*100:.0f}%: {n_dims} dims")
    
    # Test different dimensions
    print(f"\nSeparability vs PCA dimension:")
    print(f"{'Dims':>6} {'Var%':>8} {'Mahal':>8} {'LDA Acc':>10} {'Silhouette':>12}")
    print("-" * 48)
    
    results = []
    for dim in dims_to_try:
        if dim > max_dim:
            continue
            
        pca = PCA(n_components=dim)
        X_pca = pca.fit_transform(features_scaled)
        var_explained = pca.explained_variance_ratio_.sum()
        
        # Mahalanobis in PCA space
        real_pca = X_pca[labels == 0]
        fake_pca = X_pca[labels == 1]
        
        mu_r, mu_f = real_pca.mean(axis=0), fake_pca.mean(axis=0)
        cov_pooled = np.cov(X_pca, rowvar=False)
        try:
            cov_pooled += 0.1 * np.eye(dim) * np.trace(cov_pooled) / dim
            cov_inv = np.linalg.inv(cov_pooled)
            mahal = np.sqrt((mu_r - mu_f) @ cov_inv @ (mu_r - mu_f))
        except:
            mahal = np.nan
        
        # LDA accuracy (leave-one-out approximation)
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_pca, labels)
            lda_acc = lda.score(X_pca, labels)
        except:
            lda_acc = np.nan
        
        # Silhouette score (cluster quality)
        try:
            sil = silhouette_score(X_pca, labels)
        except:
            sil = np.nan
        
        print(f"{dim:>6} {var_explained*100:>7.1f}% {mahal:>8.2f} {lda_acc*100:>9.1f}% {sil:>12.3f}")
        results.append({"dim": dim, "var": var_explained, "mahal": mahal, 
                       "lda_acc": lda_acc, "silhouette": sil})
    
    # Recommendation
    best = max(results, key=lambda x: x["mahal"] if not np.isnan(x["mahal"]) else 0)
    print(f"\n→ Best separation at {best['dim']} dims (Mahal={best['mahal']:.2f})")
    
    return results


def method_recommendations(stats):
    """Recommend methods based on analysis."""
    print("\n" + "="*60)
    print("METHOD RECOMMENDATIONS")
    print("="*60)
    
    mahal = stats.get("mahalanobis", 0)
    cov_sim = stats.get("cov_similarity", 1)
    n_sig = stats.get("n_significant", 0)
    
    print("\nBased on the analysis:\n")
    
    if mahal < 1.0:
        print("⚠️  LOW SEPARABILITY detected")
        print("   - Linear methods (LogReg, LDA, SB) will struggle")
        print("   - Consider: different features, nonlinear methods, or domain adaptation")
        print()
    
    if cov_sim < 0.8:
        print("📊 DIFFERENT COVARIANCE structures")
        print("   - QDA may outperform LDA")
        print("   - Heteroscedastic SB variants could help")
        print()
    
    if n_sig < 10:
        print("🔍 FEW DISCRIMINATIVE features")
        print("   - Current features may not capture real/fake differences")
        print("   - Try: different CNN backbone, frequency-domain features, artifact detectors")
        print()
    
    print("Suggested methods by separability:\n")
    
    if mahal < 1.0:
        print("  1. Deep learning (if you have enough data)")
        print("  2. Feature engineering (try different representations)")
        print("  3. Ensemble methods")
        print("  4. Data augmentation / domain adaptation")
    elif mahal < 2.0:
        print("  1. LDA (simple, often works)")
        print("  2. Schrödinger Bridge (your current approach)")
        print("  3. SVM with RBF kernel")
        print("  4. Logistic Regression with regularization")
    else:
        print("  1. Any linear classifier should work well")
        print("  2. LDA / Logistic Regression (simplest)")
        print("  3. SB for interpretable transport")


def visualize(features, labels, class_names=None, save_path=None):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping visualization")
        return
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. PCA projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    ax = axes[0, 0]
    colors = ['#2ecc71', '#e74c3c']
    for label, color, name in zip([0, 1], colors, ['Real', 'Fake']):
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, 
                  alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Projection (2D)')
    ax.legend()
    
    # 2. t-SNE
    print("  Computing t-SNE...")
    # Use PCA first for speed
    X_pca50 = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca50)
    
    ax = axes[0, 1]
    for label, color, name in zip([0, 1], colors, ['Real', 'Fake']):
        mask = labels == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                  alpha=0.6, s=20)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection')
    ax.legend()
    
    # 3. Feature distributions (top discriminative)
    ax = axes[1, 0]
    real_feat = features[labels == 0]
    fake_feat = features[labels == 1]
    
    # Find most discriminative feature
    t_stats = []
    for i in range(features.shape[1]):
        t, _ = stats.ttest_ind(real_feat[:, i], fake_feat[:, i])
        t_stats.append(abs(t))
    best_feat = np.argmax(t_stats)
    
    ax.hist(real_feat[:, best_feat], bins=50, alpha=0.6, color=colors[0], 
            label='Real', density=True)
    ax.hist(fake_feat[:, best_feat], bins=50, alpha=0.6, color=colors[1],
            label='Fake', density=True)
    ax.set_xlabel(f'Feature {best_feat}')
    ax.set_ylabel('Density')
    ax.set_title(f'Most Discriminative Feature (idx={best_feat})')
    ax.legend()
    
    # 4. Variance explained curve
    ax = axes[1, 1]
    pca_full = PCA(n_components=min(100, X.shape[1]))
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax.plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90%')
    ax.axhline(y=0.95, color='orange', linestyle='--', label='95%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature data for deepfake detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_data.py data/cnn
    python analyze_data.py data/test --pca_dims 32 64 128 256
    python analyze_data.py data/cnn --save_plots --output analysis_cnn.png
        """
    )
    parser.add_argument("data_dir", help="Directory containing features.npy and labels.npy")
    parser.add_argument("--pca_dims", type=int, nargs="+", default=[16, 32, 64, 128, 256],
                       help="PCA dimensions to analyze (default: 16 32 64 128 256)")
    parser.add_argument("--top_features", type=int, default=20,
                       help="Number of top features to show (default: 20)")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save visualization plots")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for plots (default: data_dir/analysis.png)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from {args.data_dir}")
    features, labels, class_names = load_data(args.data_dir)
    
    # Run analyses
    basic = basic_stats(features, labels, class_names)
    dist = distribution_analysis(features, labels)
    cov = covariance_analysis(features, labels)
    feat = feature_importance(features, labels, args.top_features)
    pca = pca_analysis(features, labels, args.pca_dims)
    
    # Collect stats for recommendations
    all_stats = {
        "mahalanobis": dist.get("mahalanobis", 0),
        "cov_similarity": cov.get("cov_similarity", 1),
        "n_significant": feat.get("n_significant", 0),
    }
    method_recommendations(all_stats)
    
    # Visualization
    if args.save_plots:
        save_path = args.output or str(Path(args.data_dir) / "analysis.png")
        visualize(features, labels, class_names, save_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

