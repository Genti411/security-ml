"""Pure-numpy tests for the dataset generator and metrics (no torch/tf needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import make_dataset, standardize, train_test_split
from common.metrics import binary_metrics, roc_auc


def test_dataset_shape_and_labels():
    X, y = make_dataset(n=5000, seed=0)
    assert X.shape == (5000, 6)
    assert set(np.unique(y)).issubset({0, 1})
    # Realistic class imbalance: anomalies are a minority but present.
    assert 0.05 < y.mean() < 0.45


def test_dataset_is_deterministic():
    X1, y1 = make_dataset(n=1000, seed=7)
    X2, y2 = make_dataset(n=1000, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_standardize_uses_train_stats_only():
    X, _ = make_dataset(n=1000, seed=1)
    tr, te = X[:800], X[800:]
    trs, tes = standardize(tr, te)
    # Train columns are ~zero-mean/unit-std; test is transformed with train stats.
    assert np.allclose(trs.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(trs.std(axis=0), 1, atol=1e-2)
    assert tes.shape == te.shape


def test_metrics_perfect_and_auc():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, probs)
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert roc_auc(y, probs) == 1.0


def test_split_sizes():
    X, y = make_dataset(n=1000, seed=2)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.2, seed=3)
    assert len(Xtr) == 800 and len(Xte) == 200
    assert len(ytr) == 800 and len(yte) == 200
