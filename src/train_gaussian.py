"""
train_gaussian.py — MLE Gaussian Naive Bayes training.

Fits per-class Gaussian parameters (mu, sigma) via MLE on the current
features CSV, reusing the same train/test split as params.json so results
are directly comparable to KDE-NB.

Saves model to models/params_gaussian.json.
"""

import json
import os
import numpy as np

FEATURES_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
KDE_PARAMS_JSON = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
PARAMS_JSON     = os.path.join(os.path.dirname(__file__), '..', 'models', 'params_gaussian.json')
EPSILON         = 1e-6   # floor on sigma (Laplace smoothing)

LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}


def load_features(path):
    rows = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            vid = parts[0]
            f1, f2, f3, f4, f5 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            label = int(parts[6])
            rows[vid] = (f1, f2, f3, f4, f5, label)
    return rows


def fit_class(rows_for_class):
    """MLE Gaussian params for one class. Returns dict of {feat: {mu, sigma, log_transform}}."""
    feat_names = ['f1', 'f2', 'f3', 'f4', 'f5']
    idx_map    = {'f1': 0, 'f2': 1, 'f3': 2, 'f4': 3, 'f5': 4}
    params = {}
    for feat in feat_names:
        vals = np.array([r[idx_map[feat]] for r in rows_for_class])
        if feat in LOG_TRANSFORM_FEATURES:
            vals = np.log1p(vals)
        params[feat] = {
            'mu':            float(np.mean(vals)),
            'sigma':         float(np.std(vals) + EPSILON),
            'log_transform': feat in LOG_TRANSFORM_FEATURES,
        }
    return params


def train(features_path=FEATURES_CSV, kde_params_path=KDE_PARAMS_JSON, params_path=PARAMS_JSON):
    features = load_features(features_path)
    print(f"Loaded {len(features)} videos from {features_path}")

    # Reuse exact same split as KDE classifier for a fair comparison
    with open(kde_params_path) as f:
        kde_params = json.load(f)
    train_ids = set(kde_params["train_ids"])
    test_ids  = set(kde_params["test_ids"])

    train_rows = [features[vid]      for vid in train_ids if vid in features]
    real_rows  = [r for r in train_rows if r[5] == 0]
    fake_rows  = [r for r in train_rows if r[5] == 1]
    print(f"Train: {len(train_rows)} videos  |  Test: {len(test_ids)} videos")
    print(f"Train class counts — real: {len(real_rows)}, fake: {len(fake_rows)}")

    prior_real = len(real_rows) / len(train_rows)
    prior_fake = len(fake_rows) / len(train_rows)

    real_params = fit_class(real_rows)
    fake_params = fit_class(fake_rows)

    output = {
        "classes": {
            "0": {"prior": prior_real, **real_params},
            "1": {"prior": prior_fake, **fake_params},
        },
        "log_transform_features": list(LOG_TRANSFORM_FEATURES),
        "train_ids": list(train_ids),
        "test_ids":  list(test_ids),
    }

    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved Gaussian MLE parameters to {params_path}")

    for cls, name in [("0", "Real"), ("1", "Fake")]:
        p = output["classes"][cls]
        print(f"\n  {name} (prior={p['prior']:.3f})")
        for feat in ['f1', 'f2', 'f3', 'f4', 'f5']:
            print(f"    {feat}: mu={p[feat]['mu']:.4f},  sigma={p[feat]['sigma']:.4f}")


if __name__ == '__main__':
    train()
