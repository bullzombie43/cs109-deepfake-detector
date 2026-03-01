"""
train.py — MLE training for Gaussian Naive Bayes deepfake detector.

Reads data/features.csv, fits per-class Gaussian parameters via MLE,
and saves the model + train/test split to models/params.json.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
PARAMS_JSON  = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
EPSILON      = 1e-6   # Laplace smoothing floor on sigma
TEST_SIZE    = 0.20
RANDOM_STATE = 42


def load_features(path):
    """Load features.csv → lists of (video_id, f1, f2, f3, f4, f5, label)."""
    rows = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            video_id = parts[0]
            f1, f2, f3, f4, f5 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            label = int(parts[6])
            rows.append((video_id, f1, f2, f3, f4, f5, label))
    return rows


def split_data(rows):
    """Stratified 80/20 video-level split."""
    labels = [r[6] for r in rows]
    train_rows, test_rows = train_test_split(
        rows, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    return train_rows, test_rows


# Features fitted in log-space (chi-squared statistics are right-skewed;
# log-transform makes them approximately Gaussian and reduces variance asymmetry).
LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}


def fit_class(rows):
    """
    Compute MLE Gaussian parameters for a single class.
    f1 and f2 are log-transformed before fitting to correct right-skew.
    Returns dict: { 'f1': {'mu': ..., 'sigma': ..., 'log_transform': bool}, ... }
    """
    f1_vals = np.array([r[1] for r in rows])
    f2_vals = np.array([r[2] for r in rows])
    f3_vals = np.array([r[3] for r in rows])
    f4_vals = np.array([r[4] for r in rows])
    f5_vals = np.array([r[5] for r in rows])

    params = {}
    for name, vals in [('f1', f1_vals), ('f2', f2_vals), ('f3', f3_vals), ('f4', f4_vals), ('f5', f5_vals)]:
        if name in LOG_TRANSFORM_FEATURES:
            vals = np.log1p(vals)
        params[name] = {
            'mu':            float(np.mean(vals)),
            'sigma':         float(np.std(vals) + EPSILON),
            'log_transform': name in LOG_TRANSFORM_FEATURES,
        }
    return params


def train(features_path=FEATURES_CSV, params_path=PARAMS_JSON):
    rows = load_features(features_path)
    print(f"Loaded {len(rows)} videos from {features_path}")

    train_rows, test_rows = split_data(rows)
    print(f"Train: {len(train_rows)} videos  |  Test: {len(test_rows)} videos")

    real_rows = [r for r in train_rows if r[6] == 0]
    fake_rows = [r for r in train_rows if r[6] == 1]
    print(f"Train class counts — real: {len(real_rows)}, fake: {len(fake_rows)}")

    real_params = fit_class(real_rows)
    fake_params = fit_class(fake_rows)

    prior_real = len(real_rows) / len(train_rows)
    prior_fake = len(fake_rows) / len(train_rows)

    output = {
        "classes": {
            "0": {"prior": prior_real, **real_params},
            "1": {"prior": prior_fake, **fake_params},
        },
        "train_ids": [r[0] for r in train_rows],
        "test_ids":  [r[0] for r in test_rows],
    }

    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved model parameters to {params_path}")

    # Summary
    for cls, name in [("0", "Real"), ("1", "Fake")]:
        p = output["classes"][cls]
        print(f"\n  {name} (prior={p['prior']:.3f})")
        for feat in ['f1', 'f2', 'f3', 'f4', 'f5']:
            print(f"    {feat}: mu={p[feat]['mu']:.4f},  sigma={p[feat]['sigma']:.4f}")


if __name__ == '__main__':
    train()
