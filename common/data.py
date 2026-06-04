"""Synthetic but realistic login-event dataset for anomaly classification.

No external download: the data is generated deterministically from a seed, so
results are reproducible and the repo is self-contained. Each row is one login
attempt; the label is 1 for anomalous (account-takeover-like) and 0 for normal.

Features (the kind of signals a real detection pipeline uses):
  hour              hour of day (0-23), normalized to [0,1]
  failed_attempts   recent failed logins for the account (0-8)
  is_new_device     1 if the device fingerprint is unseen
  country_risk      geo/IP reputation risk in [0,1]
  ip_velocity       "impossible travel" speed proxy in [0,1]
  off_hours         1 if outside the user's usual active window

The label is a noisy function of these, so a model has to learn a real (not
trivially separable) decision boundary.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["hour", "failed_attempts", "is_new_device",
            "country_risk", "ip_velocity", "off_hours"]


def make_dataset(n: int = 12000, seed: int = 42):
    """Return (X, y): X is (n, 6) float32, y is (n,) int64 in {0,1}."""
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, size=n)
    failed = rng.poisson(0.4, size=n).clip(0, 8)
    new_device = (rng.random(n) < 0.18).astype(int)
    country_risk = rng.beta(1.5, 6.0, size=n)          # mostly low, some high
    ip_velocity = rng.beta(1.3, 8.0, size=n)           # mostly low, occasional spikes
    off_hours = ((hour < 6) | (hour > 22)).astype(int)

    # Risk score is a deterministic function of the (observable) features, so a
    # model CAN learn the boundary. We threshold it to get a realistic minority
    # of anomalies, then flip a small fraction of labels as irreducible noise —
    # enough to keep the problem non-trivial without making it unlearnable.
    score = (
        2.5 * country_risk
        + 2.2 * ip_velocity
        + 1.0 * new_device
        + 0.8 * (failed / 8.0)
        + 0.7 * off_hours
    )
    threshold = np.quantile(score, 0.72)        # ~28% anomalous
    y = (score > threshold).astype(np.int64)
    flip = rng.random(n) < 0.04                  # 4% label noise
    y[flip] = 1 - y[flip]

    X = np.column_stack([
        hour / 23.0,
        failed / 8.0,
        new_device,
        country_risk,
        ip_velocity,
        off_hours,
    ]).astype(np.float32)
    return X, y


def train_test_split(X, y, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def standardize(train, test):
    """Z-score using TRAIN statistics only (no leakage from the test set)."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std
