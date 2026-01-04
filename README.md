# Deepfake Detection via Frequency Analysis & Schrödinger Bridges

Statistical deepfake detection using frequency-domain features and optimal transport modeling.

---

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download FF-C23 dataset (optional - uses Kaggle)
python main.py download

# 3. Extract features (choose one)
python main.py preprocess --features cnn --output data/cnn           # CNN features (EfficientNet)
python main.py preprocess --features localized --output data/local   # Localized DCT (patch-based)
python main.py preprocess --features dct --output data/dct           # Global DCT

# 4. Analyze data quality (recommended before training)
python analyze_data.py data/local --save_plots

# 5. Train Schrödinger Bridge
python main.py train --data_dir data/local --pca_dim 64

# 6. Test different classifiers
python test_localized.py data/local --pca_dims 32 64 128
```

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `main.py` | Main CLI: download, preprocess, train, evaluate, demo |
| `analyze_data.py` | Statistical analysis of feature data (distributions, separability) |
| `test_localized.py` | Quick classifier comparison (LogReg, LDA, QDA, Mahalanobis) |

### main.py Commands

```bash
# Download FF-C23 dataset from Kaggle
python main.py download

# Extract features from videos
python main.py preprocess --data_dir /path/to/videos --output data/processed \
    --features localized \           # cnn | dct | localized
    --fake_types Deepfakes,Face2Face \  # Subset of fake types (optional)
    --max_samples 100          # Limit samples per class (optional)

# Train Gaussian Schrödinger Bridge
python main.py train --data_dir data/local \
    --pca_dim 64 \             # PCA dimensions
    --shrinkage 0.1 \          # Covariance regularization
    --val_split 0.2

# Evaluate trained model
python main.py evaluate --data_dir data/local \
    --checkpoint experiments/checkpoints/checkpoint_best.pt \
    --n_samples 500            # Subsample for faster eval (optional)

# Single video demo
python main.py demo --video path/to/video.mp4 \
    --checkpoint experiments/checkpoints/checkpoint_best.pt
```

### analyze_data.py

Analyze statistical properties of your feature data before choosing a method:

```bash
python analyze_data.py data/local --save_plots --pca_dims 32 64 128 256
```

**Output includes:**
- Basic statistics (samples, features, class balance)
- Distribution analysis (Mahalanobis distance, Bhattacharyya distance)
- Covariance analysis (condition numbers, effective rank)
- Feature discriminability (t-tests, effect sizes)
- PCA analysis (variance explained, separability vs dimension)
- Method recommendations based on data properties

### test_localized.py

Quick comparison of different classifiers:

```bash
python test_localized.py data/local --pca_dims 32 64 128 256 --val_split 0.2
```

Tests: Logistic Regression, LDA, QDA, Mahalanobis classifier

---

## Feature Extraction Methods

### 1. CNN Features (`--features cnn`)

Uses pretrained EfficientNet-B0 to extract semantic features.

- **Dimension**: 2560 (before PCA)
- **Pros**: Captures high-level artifacts, good baseline
- **Cons**: May not capture frequency-specific artifacts

### 2. Global DCT (`--features dct`)

Traditional DCT histogram features across the whole image.

- **Dimension**: ~4000 (configurable)
- **Pros**: Fast, interpretable, frequency-focused
- **Cons**: Averages out localized artifacts

### 3. Localized DCT (`--features localized`) ⭐ Recommended

**Patch-based frequency analysis** — the key innovation.

```
Image → Patches (32×32) → DCT per patch → Local statistics → Aggregate
```

- **Dimension**: ~4000 (patch_size and stride configurable)
- **Pros**: Preserves local artifacts that global methods miss
- **Cons**: Higher dimensional, needs more samples

**Why localized?** Deepfakes introduce **spatially localized artifacts** (around eyes, mouth, hairline). Global averaging washes these out. Localized modeling detects "any suspicious region" rather than "overall suspicious image."

---

## Theory: Why This Approach?

### The Problem with Classifiers

Standard classifiers learn a decision boundary between "real" and "fake". This is brittle:
- Fails on new deepfake methods
- Sensitive to distribution shift
- Easily fooled by adversarial perturbations

### Schrödinger Bridge Perspective

Instead of classification, we model the **optimal transport** between distributions:

```
P_real ←──── Schrödinger Bridge ────→ P_fake
```

The Schrödinger Bridge finds the minimum-energy path transforming one distribution into another. For detection:

1. Fit Gaussian distributions to real and fake samples
2. Compute the optimal transport map
3. Score samples by their transport cost / distributional fit

**Key insight**: If real and fake occupy different regions of feature space, the transport cost reveals this. Unlike classifiers, this is **distribution-aware** rather than point-based.

### Gaussian Schrödinger Bridge

For Gaussian distributions, the Schrödinger Bridge has closed-form solutions:

```
μ_t = (1-t)μ_0 + t μ_1
Σ_t = interpolation of Σ_0 and Σ_1
```

This enables:
- No neural network training needed
- Fast inference via Mahalanobis distance
- Interpretable transport paths

### Why Frequency Features?

Deepfakes fail to replicate natural image statistics in the frequency domain:

- **Compression artifacts**: GAN-generated faces have different JPEG statistics
- **Blending boundaries**: Face swaps create edge artifacts visible in high frequencies
- **Texture inconsistencies**: Neural textures lack fine-grained noise patterns

DCT (Discrete Cosine Transform) exposes these artifacts that pixel-domain analysis misses.

---

## Understanding Your Data

Before training, **always run analyze_data.py**. Key metrics:

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| Mahalanobis distance | >2.0 | <1.0 | Distribution separation |
| Condition number | <1e4 | >1e6 | Numerical stability |
| Significant features | >50 | <10 | Discriminative signal |
| Effect size (Cohen's d) | >0.5 | <0.2 | Practical significance |

**If Mahalanobis < 1.0**: Your features don't separate real/fake. Try:
- Different feature extractor
- Localized instead of global
- More data

**If condition number > 1e6**: Covariance is ill-conditioned. Use:
- More PCA (reduce dimensions)
- Higher shrinkage regularization

---

## Project Structure

```
├── main.py                    # CLI entry point
├── analyze_data.py            # Data analysis & diagnostics
├── test_localized.py          # Classifier comparison
├── core/
│   ├── preprocess/
│   │   ├── dct_extractor.py       # Global DCT features
│   │   ├── cnn_extractor.py       # EfficientNet features
│   │   └── localized_dct.py       # Patch-based DCT features
│   ├── models/
│   │   └── schrodinger_bridge.py  # Gaussian SB model
│   ├── train/
│   │   └── trainer.py             # Training pipeline
│   └── evaluate/
│       └── evaluator.py           # Metrics & visualization
├── data/
│   ├── cnn/                   # Extracted CNN features
│   ├── localized/             # Extracted localized DCT features
│   └── dct/                   # Extracted global DCT features
├── experiments/
│   └── checkpoints/           # Saved models
└── artifacts/                 # Evaluation outputs
```

---

## Evaluation Metrics

### Detection Metrics
- **Balanced Accuracy**: (TPR + TNR) / 2 — handles class imbalance
- **ROC-AUC**: Area under ROC curve
- **EER**: Equal Error Rate (where FPR = FNR)
- **F1 Score**: Harmonic mean of precision and recall

### Diagnostic Metrics
- **Mahalanobis distance**: Separation between class centroids
- **Train-Val gap**: Overfitting indicator (should be <10%)
- **Per-class accuracy**: Reveals bias toward one class

---

## Common Issues & Solutions

### "Balanced accuracy ~50%" (random chance)
**Cause**: Features don't separate classes.
**Fix**: 
- Try localized DCT instead of global
- Check `analyze_data.py` output for low Mahalanobis
- Use different feature extractor

### "High fake accuracy, low real accuracy"
**Cause**: Class imbalance (more fake than real samples).
**Fix**:
- Undersample fake to match real
- Tune classification threshold
- Use anomaly detection (train only on real)

### "Train accuracy >> Val accuracy"
**Cause**: Overfitting, too few samples for dimensionality.
**Fix**:
- Reduce PCA dimensions (rule: samples > 10× dimensions)
- Increase shrinkage regularization
- Get more training data

### QDA fails with "covariance not full rank"
**Cause**: More features than samples in one class.
**Fix**:
- Use LDA instead
- Reduce PCA dimensions below min class size

---

## Docker

```bash
# Build CPU image
docker-compose build deepfake-cpu

# Extract features 
docker-compose run deepfake-cpu python main.py preprocess --features localized --output data/local

# Run analysis
docker-compose run deepfake-cpu python analyze_data.py data/local --save_plots

# Train
docker-compose run deepfake-cpu python main.py train --data_dir data/local --pca_dim 64
```

---

## References

- **Schrödinger Bridges**: De Bortoli et al., "Diffusion Schrödinger Bridge" (2021)
- **DCT Forensics**: Fridrich & Kodovsky, "Rich Models for Steganalysis" (2012)
- **Optimal Transport**: Peyré & Cuturi, "Computational Optimal Transport" (2019)
- **FaceForensics++**: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images" (2019)
