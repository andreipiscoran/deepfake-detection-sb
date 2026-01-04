"""
DCT + Schrödinger Bridge Deepfake Detection
===========================================

Usage:
    python main.py download                                                    # Download FF-C23 dataset
    python main.py preprocess --output data/processed                          # Preprocess all FF-C23 data
    python main.py preprocess --fake_types Deepfakes,Face2Face --max_samples 100  # Subset for testing
    python main.py train --data_dir data/processed --epochs 100
    python main.py evaluate --data_dir data/processed --checkpoint experiments/checkpoints/checkpoint_100.pt --n_samples 100
    python main.py demo --video path/to/video.mp4 --checkpoint experiments/checkpoints/checkpoint_100.pt

FF-C23 Dataset Structure:
    original/          -> Real videos (1000)
    DeepFakeDetection/ -> Fake (1000)
    Deepfakes/         -> Fake (1000)
    Face2Face/         -> Fake (1000)
    FaceShifter/       -> Fake (1000)
    FaceSwap/          -> Fake (1000)
    NeuralTextures/    -> Fake (1000)
"""

import argparse
import numpy as np
import torch
import kagglehub
from pathlib import Path
from datetime import datetime

from core.preprocess import DCTExtractor, CNNExtractor, DeepfakeDataset, LocalizedDCTExtractor
from core.models import SchrodingerBridge
from core.train import Trainer
from core.evaluate import Evaluator

# Cache for downloaded dataset path
_dataset_path = None


def download(args):
    """Download the FF-C23 dataset from Kaggle."""
    global _dataset_path
    print("Downloading FF-C23 dataset from Kaggle...")
    base_path = kagglehub.dataset_download("xdxd003/ff-c23")
    _dataset_path = str(Path(base_path) / "FaceForensics++_C23")
    print(f"Dataset downloaded to: {_dataset_path}")
    return _dataset_path


def get_dataset_path():
    """Get the FF-C23 dataset path, downloading if necessary."""
    global _dataset_path
    if _dataset_path is None:
        print("Dataset not downloaded yet, downloading...")
        base_path = kagglehub.dataset_download("xdxd003/ff-c23")
        # Dataset is nested inside FaceForensics++_C23 folder
        _dataset_path = str(Path(base_path) / "FaceForensics++_C23")
        print(f"Dataset path: {_dataset_path}")
    return _dataset_path


def preprocess(args):
    """Extract features from video dataset."""
    data_dir = args.data_dir if args.data_dir else get_dataset_path()
    
    # Choose feature extractor
    if args.features == "cnn":
        print(f"Extracting CNN (EfficientNet) features from {data_dir}")
        extractor = CNNExtractor()
    elif args.features == "localized":
        print(f"Extracting Localized DCT features from {data_dir}")
        extractor = LocalizedDCTExtractor(
            patch_size=args.patch_size,
            patch_stride=args.patch_stride
        )
    else:
        print(f"Extracting DCT features from {data_dir}")
        extractor = DCTExtractor()
    
    # Use FF-C23 loader for Kaggle dataset (has original + fake method folders)
    # Use simple loader for custom data_dir with real/fake structure
    data_path = Path(data_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    
    if (data_path / "original").exists():
        # FF-C23 structure - process sequentially for CNN (uses GPU)
        fake_types = args.fake_types.split(",") if args.fake_types else None
        
        if args.features == "cnn":
            # CNN extraction returns (features, labels, class_names)
            features, labels, class_names = _extract_ffc23_sequential(data_dir, extractor, fake_types, args.max_samples)
            np.save(output / "features.npy", features)
            np.save(output / "labels.npy", labels)
            np.save(output / "class_names.npy", np.array(class_names))
            print(f"Saved {len(labels)} samples to {output}")
            return
        else:
            dataset = DeepfakeDataset.from_ffc23(
                data_dir, extractor, 
                fake_types=fake_types,
                max_per_class=args.max_samples,
                n_workers=args.workers
            )
    else:
        # Simple real/fake structure
        dataset = DeepfakeDataset.from_directory(data_dir, extractor)
    
    np.save(output / "features.npy", dataset.features.numpy())
    np.save(output / "labels.npy", dataset.labels.numpy())
    print(f"Saved {len(dataset)} samples to {output}")


def _extract_ffc23_sequential(data_dir: str, extractor, fake_types, max_per_class):
    """Parallel extraction for CNN features using threading. Returns features, labels, and class names."""
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    data_dir = Path(data_dir)
    all_fake_types = ["DeepFakeDetection", "Deepfakes", "Face2Face", 
                      "FaceShifter", "FaceSwap", "NeuralTextures"]
    fake_types = fake_types or all_fake_types
    
    # Track (video_path, label, class_name)
    tasks = []
    
    # Real videos
    real_path = data_dir / "original"
    if real_path.exists():
        videos = list(real_path.glob("*.mp4"))
        if max_per_class:
            videos = videos[:max_per_class]
        tasks.extend([(str(v), 0, "real") for v in videos])
        print(f"Found {len(videos)} real videos")
    
    # Fake videos - track which type each comes from
    for fake_type in fake_types:
        fake_path = data_dir / fake_type
        if fake_path.exists():
            videos = list(fake_path.glob("*.mp4"))
            if max_per_class:
                videos = videos[:max_per_class]
            tasks.extend([(str(v), 1, fake_type) for v in videos])
            print(f"Found {len(videos)} fake videos from '{fake_type}'")
    
    # Thread-safe results storage
    results = []
    results_lock = threading.Lock()
    progress = [0]
    
    def process_video(task):
        video_path, label, class_name = task
        try:
            feat = extractor.extract_from_video(video_path)
            with results_lock:
                results.append((feat, label, class_name, None))
                progress[0] += 1
                print(f"  [{progress[0]}/{len(tasks)}] {Path(video_path).name}", end="\r")
        except Exception as e:
            with results_lock:
                progress[0] += 1
                print(f"  [{progress[0]}/{len(tasks)}] Error: {Path(video_path).name}: {e}")
    
    n_workers = min(8, len(tasks))  # Limit threads to avoid memory issues
    print(f"\nProcessing {len(tasks)} videos with CNN ({n_workers} threads)...")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_video, task) for task in tasks]
        for future in as_completed(futures):
            pass  # Results collected in process_video
    
    # Unpack results
    features = [r[0] for r in results]
    labels = [r[1] for r in results]
    class_names = [r[2] for r in results]
    
    print(f"\nTotal: {labels.count(0)} real, {labels.count(1)} fake")
    return np.array(features), np.array(labels), class_names


def train(args):
    """Fit Gaussian Schrödinger Bridge."""
    print(f"Loading data from {args.data_dir}")
    features = np.load(Path(args.data_dir) / "features.npy")
    labels = np.load(Path(args.data_dir) / "labels.npy")
    dataset = DeepfakeDataset(features, labels)
    
    print(f"Dataset: {dataset.num_real} real, {dataset.num_fake} fake")
    print(f"Feature dim: {dataset.features.shape[1]}")
    
    # Fit (no training loop - just statistics)
    trainer = Trainer(dataset, pca_dim=args.pca_dim, val_split=args.val_split, 
                      shrinkage=args.shrinkage)
    trainer.train(save_dir=args.save_dir)
    
    print("\nDone.")


def evaluate(args):
    """Evaluate the trained Schrödinger Bridge model."""
    
    print(f"Loading data from {args.data_dir}")
    features = np.load(Path(args.data_dir) / "features.npy")
    labels = np.load(Path(args.data_dir) / "labels.npy")
    
    # Load class names if available (for per-class metrics)
    class_names_path = Path(args.data_dir) / "class_names.npy"
    class_names = np.load(class_names_path, allow_pickle=True) if class_names_path.exists() else None
    
    # Track sample indices for reporting
    all_indices = np.arange(len(features))
    selected_indices = all_indices
    
    # Subsample if requested
    n_samples = args.n_samples
    n_real, n_fake = None, None
    if n_samples and n_samples < len(features):
        # Stratified sampling: keep ratio of real/fake
        real_idx = np.where(labels == 0)[0]
        fake_idx = np.where(labels == 1)[0]
        n_real = min(len(real_idx), n_samples // 2)
        n_fake = min(len(fake_idx), n_samples - n_real)
        
        np.random.seed(42)
        selected_real = np.random.choice(real_idx, n_real, replace=False)
        selected_fake = np.random.choice(fake_idx, n_fake, replace=False)
        selected_indices = np.concatenate([selected_real, selected_fake])
        
        features = features[selected_indices]
        labels = labels[selected_indices]
        if class_names is not None:
            class_names = class_names[selected_indices]
        print(f"Using {n_samples} samples ({n_real} real, {n_fake} fake)")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model and load state
    input_dim = ckpt.get('input_dim', 64)
    model = SchrodingerBridge(input_dim=input_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Apply preprocessing: normalize → PCA
    features = (features - ckpt["mean"]) / ckpt["std"]
    features = ckpt["pca"].transform(features)
    print(f"Applied preprocessing: PCA ({features.shape[1]}D)")
    
    n_real = n_real or (labels == 0).sum()
    n_fake = n_fake or (labels == 1).sum()
    print(f"Test set: {n_real} real, {n_fake} fake")
    
    # Get samples for evaluation
    real_mask = labels == 0
    fake_mask = labels == 1
    real_samples = torch.tensor(features[real_mask], dtype=torch.float32)
    fake_samples = torch.tensor(features[fake_mask], dtype=torch.float32)
    
    # Get fake class names for per-class metrics
    fake_class_names = None
    if class_names is not None:
        fake_class_names = [class_names[i] for i in range(len(labels)) if labels[i] == 1]
    
    # Evaluate with centroid-based detection
    print("\nEvaluating with Schrödinger Bridge (centroid distance)...")
    model.evaluate_accuracy(real_samples, fake_samples, fake_class_names)
    
    # Create dataset for evaluator  
    dataset = DeepfakeDataset(features, labels)
    evaluator = Evaluator(model)
    
    # Compute metrics (this uses bridge detection internally)
    metrics = evaluator.compute_metrics(dataset)
    
    # Compute complexity if verbose
    complexity = None
    if args.verbose:
        complexity = evaluator.compute_bridge_complexity(dataset)
    
    # Create timestamp and artifacts folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path("artifacts") / args.category
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Print results
    print("\n" + "="*40)
    print("         DETECTION METRICS")
    print("="*40)
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"  EER:         {metrics['eer']:.4f}")
    print(f"  Threshold:   {metrics['threshold']:.4f}")
    print("="*40)
    
    if args.verbose and complexity:
        print("\n=== Bridge Complexity ===")
        for k, v in complexity.items():
            print(f"  {k}: {v:.4f}")
    
    # Save evaluation results to text file
    eval_file = artifacts_dir / f"evaluation_{timestamp}.txt"
    with open(eval_file, "w") as f:
        f.write(f"Evaluation Results - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Data directory: {args.data_dir}\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Category: {args.category}\n")
        f.write(f"  Timestamp: {timestamp}\n\n")
        
        f.write("Samples Evaluated:\n")
        f.write(f"  Total: {len(dataset)}\n")
        f.write(f"  Real: {n_real}\n")
        f.write(f"  Fake: {n_fake}\n")
        f.write(f"  Sample indices: {selected_indices.tolist()}\n\n")
        
        f.write("Detection Metrics:\n")
        f.write(f"  F1 Score:    {metrics['f1']:.4f}\n")
        f.write(f"  Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"  ROC AUC:     {metrics['roc_auc']:.4f}\n")
        f.write(f"  EER:         {metrics['eer']:.4f}\n")
        f.write(f"  Threshold:   {metrics['threshold']:.4f}\n")
        
        if complexity:
            f.write("\nBridge Complexity:\n")
            for k, v in complexity.items():
                f.write(f"  {k}: {v:.4f}\n")
    
    print(f"\nResults saved to: {eval_file}")
    
    # Save plot
    plot_file = artifacts_dir / f"results_{timestamp}.png"
    evaluator.plot_results(dataset, save_path=str(plot_file))
    print(f"Plot saved to: {plot_file}")


def demo(args):
    """Run detection on a single video."""
    extractor = DCTExtractor()
    features = extractor.extract_from_video(args.video)
    
    # Need to know feature dim - infer from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    input_dim = features.shape[0]
    
    model = SchrodingerBridge(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0)
        score, _ = model.detect(x)
    
    print(f"Video: {args.video}")
    print(f"Anomaly score: {score.item():.4f}")
    print(f"Prediction: {'FAKE' if score.item() > 0.5 else 'REAL'}")


def main():
    parser = argparse.ArgumentParser(description="DCT + Schrödinger Bridge Deepfake Detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Download
    p = subparsers.add_parser("download", help="Download FF-C23 dataset from Kaggle")
    
    # Preprocess
    p = subparsers.add_parser("preprocess", help="Extract features from videos")
    p.add_argument("--data_dir", default=None, help="Input directory (defaults to FF-C23 from Kaggle)")
    p.add_argument("--output", default="data/processed", help="Output directory")
    p.add_argument("--features", choices=["dct", "cnn", "localized"], default="cnn",
                   help="Feature type: 'dct' (global), 'cnn' (EfficientNet), 'localized' (patch-based DCT). Default: cnn")
    p.add_argument("--patch_size", type=int, default=32, help="Patch size for localized DCT (default: 32)")
    p.add_argument("--patch_stride", type=int, default=16, help="Patch stride for localized DCT (default: 16)")
    p.add_argument("--fake_types", default=None, 
                   help="Comma-separated fake types to include (e.g. 'Deepfakes,Face2Face'). Default: all")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max samples per folder (for faster testing)")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers (default: CPU count, DCT only)")
    
    # Train
    p = subparsers.add_parser("train", help="Fit Gaussian Schrödinger Bridge")
    p.add_argument("--data_dir", required=True, help="Directory with features.npy and labels.npy")
    p.add_argument("--pca_dim", type=int, default=32, help="PCA dimensions (default: 32)")
    p.add_argument("--shrinkage", type=float, default=0.1, help="Covariance shrinkage λ (default: 0.1)")
    p.add_argument("--save_dir", default="experiments/checkpoints")
    p.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate trained model")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    p.add_argument("--category", default="default", help="Category name for organizing artifacts (e.g. 'Deepfakes', 'Face2Face')")
    p.add_argument("--verbose", action="store_true", help="Show bridge complexity metrics")
    
    # Demo
    p = subparsers.add_parser("demo", help="Run on single video")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    
    args = parser.parse_args()
    
    if args.command == "download":
        download(args)
    elif args.command == "preprocess":
        preprocess(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "demo":
        demo(args)


if __name__ == "__main__":
    main()

