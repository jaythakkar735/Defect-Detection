"""
Unit tests for precomputed benchmark cache and cloud demo fallback.
Ensures zero-setup deployment on Streamlit Cloud or containerized environments.
"""
import pickle
from pathlib import Path
import numpy as np
import torch
from config import CATEGORIES, PROJECT_ROOT, ASSETS_DIR
from src.dataset import MVTecCategoryDataset, get_default_transforms
from src.model import ResNetAnomalyDetector

def test_precomputed_cache_integrity():
    cache_path = ASSETS_DIR / "precomputed" / "benchmark_cache.pkl"
    assert cache_path.exists(), "Benchmark cache file benchmark_cache.pkl is missing."
    
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    for cat in CATEGORIES:
        assert cat in cache, f"Category '{cat}' missing from precomputed cache."
        entry = cache[cat]
        assert "normal_centroid" in entry
        assert entry["normal_centroid"].shape == (512,)
        assert "threshold" in entry
        assert isinstance(entry["threshold"], (float, int, np.floating))
        assert "auroc" in entry
        assert 0.0 <= entry["auroc"] <= 1.0
        assert "scores" in entry
        assert len(entry["scores"]) > 0

def test_demo_samples_dataset():
    transform = get_default_transforms()
    for cat in CATEGORIES:
        demo_dir = PROJECT_ROOT / "data" / "demo_samples" / cat
        assert demo_dir.exists(), f"Demo samples directory missing for {cat}"
        
        ds = MVTecCategoryDataset(demo_dir, split="demo", transform=transform)
        assert len(ds) > 0, f"Demo dataset has no samples for {cat}"
        
        # Test loading first item
        item = ds[0]
        assert item["image"].shape == (3, 256, 256)
        assert item["label"] in [0, 1]

def test_detector_with_precomputed_weights():
    cache_path = ASSETS_DIR / "precomputed" / "benchmark_cache.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    tb_cache = cache["toothbrush"]
    detector = ResNetAnomalyDetector(device="cpu", backbone="resnet18")
    detector.load_precomputed(tb_cache["normal_centroid"], tb_cache["threshold"])
    
    # Test scoring dummy features
    dummy_feat = np.ones((1, 512), dtype=np.float32)
    score = detector.score_features(dummy_feat)
    assert isinstance(float(score[0]), float)
    assert score[0] > 0.0
