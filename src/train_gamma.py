"""
train_gamma.py — MLE Gamma Naive Bayes training.

Fits per-class Gamma parameters (alpha, beta) via MLE on the current
features CSV, reusing the same train/test split as params.json so results
are directly comparable to KDE-NB and Gaussian-NB.

MLE estimates:
    alpha = (mean / std)^2        (method of moments, equivalent to MLE for Gamma)
    beta  = std^2 / mean

Saves model to models/params_gamma.json.
"""

import json
import os
import numpy as np

FEATURES_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
KDE_PARAMS_JSON = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
PARAMS_JSON     = os.path.join(os.path.dirname(__file__), '..', 'models', 'params_gamma.json')
EPSILON         = 1e-6   # floor on alpha/beta to avoid log(0)

# Gamma distribution requires positive inputs — log-transformed features
# are always positive (log1p >= 0), so apply the same transforms as elsewhere.
# f4 (temporal MAD) is non-negative but NOT log-transformed.
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
    """MLE Gamma params for one class. Returns dict of {feat: {alpha, beta, log_transform}}."""
    feat_names = ['f1', 'f2', 'f3', 'f4', 'f5']
    idx_map    = {'f1': 0, 'f2': 1, 'f3': 2, 'f4': 3, 'f5': 4}
    params = {}
    for feat in feat_names:
        vals = np.array([r[idx_map[feat]] for r in rows_for_class], dtype=float)
        if feat in LOG_TRANSFORM_FEATURES:
            vals = np.log1p(vals)
        # Shift to ensure all values strictly positive before fitting
        vals = vals + EPSILON
        mean = np.mean(vals)
        var  = np.var(vals) + EPSILON
        alpha = float((mean ** 2) / var)
        beta  = float(var / mean)          # scale parameter: E[X] = alpha * beta
        params[feat] = {
            'alpha':         alpha,
            'beta':          beta,
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
    print(f"Saved Gamma MLE parameters to {params_path}")

    for cls, name in [("0", "Real"), ("1", "Fake")]:
        p = output["classes"][cls]
        print(f"\n  {name} (prior={p['prior']:.3f})")
        for feat in ['f1', 'f2', 'f3', 'f4', 'f5']:
            print(f"    {feat}: alpha={p[feat]['alpha']:.4f},  beta={p[feat]['beta']:.4f}")


if __name__ == '__main__':
    train()
