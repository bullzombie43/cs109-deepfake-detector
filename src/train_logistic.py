"""
train_logistic.py — Logistic Regression training.

Fits a logistic regression model on the current features CSV,
reusing the same train/test split as params.json for fair comparison.

Saves model to models/params_logistic_<features>.json.

Usage:
    python src/train_logistic.py             # all 5 features
    python src/train_logistic.py f5          # single feature
    python src/train_logistic.py f1 f3 f5    # feature subset
"""

import json
import math
import os
import sys
import numpy as np

FEATURES_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
KDE_PARAMS_JSON = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
MODELS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'models')

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


def build_X(rows_list, features):
    feat_idx = {'f1': 0, 'f2': 1, 'f3': 2, 'f4': 3, 'f5': 4}
    X = []
    for r in rows_list:
        row = []
        for feat in features:
            val = r[feat_idx[feat]]
            if feat in LOG_TRANSFORM_FEATURES:
                val = float(np.log1p(val))
            row.append(val)
        X.append(row)
    return np.array(X)

def calc_sigmoid(theta, x):
    return 1 / (1 + math.exp(-np.dot(theta, x)))


def grad_descent(X_train, y_train, num_steps=1000, step_size=0.01):
    n, num_features = X_train.shape
    # Prepend bias column of 1s (equivalent to x0 = 1)
    X = np.hstack([np.ones((n, 1)), X_train])
    thetas = [0.0] * (num_features + 1)

    for step in range(num_steps):
        gradients = [0.0] * (num_features + 1)
        for i in range(n):
            features = X[i]
            label = y_train[i]
            for j in range(num_features + 1):
                gradients[j] += features[j] * (label - calc_sigmoid(thetas, features))
        for j in range(num_features + 1):
            thetas[j] += step_size * gradients[j]

    return thetas



def train(features_path=FEATURES_CSV, kde_params_path=KDE_PARAMS_JSON,
          models_dir=MODELS_DIR, features_subset=('f1', 'f2', 'f3', 'f4', 'f5')):
    features = load_features(features_path)
    print(f"Loaded {len(features)} videos from {features_path}")

    with open(kde_params_path) as f:
        kde_params = json.load(f)
    train_ids = set(kde_params["train_ids"])
    test_ids  = set(kde_params["test_ids"])

    train_rows = [features[vid] for vid in train_ids if vid in features]
    test_rows  = [features[vid] for vid in test_ids  if vid in features]
    print(f"Train: {len(train_rows)} videos  |  Test: {len(test_rows)} videos")

    X_train = build_X(train_rows, features_subset)
    y_train = np.array([r[5] for r in train_rows])

    real_count = int((y_train == 0).sum())
    fake_count = int((y_train == 1).sum())
    print(f"Train class counts — real: {real_count}, fake: {fake_count}")

    thetas = grad_descent(X_train, y_train, num_steps=1000, step_size=0.01)
    print(f"Fitted logistic regression on {len(X_train)} training samples")

    tag = '_'.join(features_subset)
    out_path = os.path.join(models_dir, f'params_logistic_{tag}.json')
    os.makedirs(models_dir, exist_ok=True)

    # thetas[0] is the bias/intercept; thetas[1:] are feature coefficients
    output = {
        "features":               list(features_subset),
        "coef":                   thetas[1:],
        "intercept":              thetas[0],
        "log_transform_features": list(LOG_TRANSFORM_FEATURES),
        "train_ids":              list(train_ids),
        "test_ids":               list(test_ids),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved logistic regression parameters to {out_path}")

    print(f"\n  Features: {list(features_subset)}")
    for feat, coef in zip(features_subset, thetas[1:]):
        print(f"    {feat}: coef={coef:.4f}")
    print(f"  Intercept: {thetas[0]:.4f}")


if __name__ == '__main__':
    subset = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ('f1', 'f2', 'f3', 'f4', 'f5')
    print(f"Features used: {subset}")
    train(features_subset=subset)
