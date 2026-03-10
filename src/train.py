"""
train.py — KDE Naive Bayes training (Proposal 3).

Saves per-class training feature vectors (log-transformed) to params.json
so classify.py can fit KDE likelihoods at test time instead of Gaussian.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
PARAMS_JSON  = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
TEST_SIZE    = 0.20
RANDOM_STATE = 42

LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}


def load_features(path):
    rows = []
    with open(path) as f:
        f.readline() #skip the labels
        for line in f:
            parts = line.strip().split(',')
            video_id = parts[0]
            f1, f2, f3, f4, f5 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            label = int(parts[6])
            rows.append((video_id, f1, f2, f3, f4, f5, label))
    return rows


def split_data(rows):
    labels = [r[6] for r in rows] #label of data: 1 deepfake or 0 real
    train_rows, test_rows = train_test_split(
        rows, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    return train_rows, test_rows


#this just extracts the features and saves the log_transformed values doens't actually calculate weights
def train(features_path=FEATURES_CSV, params_path=PARAMS_JSON):
    rows = load_features(features_path)
    print(f"Loaded {len(rows)} videos from {features_path}")

    train_rows, test_rows = split_data(rows)
    print(f"Train: {len(train_rows)} videos  |  Test: {len(test_rows)} videos")

    feat_names = ['f1', 'f2', 'f3', 'f4', 'f5']
    idx_map    = {'f1': 1, 'f2': 2, 'f3': 3, 'f4': 4, 'f5': 5} #idx 0 is the video id

    real_rows = [r for r in train_rows if r[6] == 0]
    fake_rows = [r for r in train_rows if r[6] == 1]
    print(f"Train class counts — real: {len(real_rows)}, fake: {len(fake_rows)}")

    prior = {
        "0": len(real_rows) / len(train_rows),
        "1": len(fake_rows) / len(train_rows),
    }

    # Save raw (log-transformed) training values per class per feature
    kde_data = {"0": {}, "1": {}}
    for cls_str, cls_rows in [("0", real_rows), ("1", fake_rows)]:
        for feat in feat_names:
            vals = [r[idx_map[feat]] for r in cls_rows]
            if feat in LOG_TRANSFORM_FEATURES:
                vals = [float(np.log1p(v)) for v in vals]
            kde_data[cls_str][feat] = vals

    output = {
        "kde_training_data":      kde_data,
        "log_transform_features": list(LOG_TRANSFORM_FEATURES),
        "log_priors": {
            "0": float(np.log(prior["0"])),
            "1": float(np.log(prior["1"])),
        },
        "train_ids": [r[0] for r in train_rows],
        "test_ids":  [r[0] for r in test_rows],
    }

    os.makedirs(os.path.dirname(params_path), exist_ok=True) #cache/save everything in json file for wuick testing
    with open(params_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved KDE training data to {params_path}")

    print(f"\nClass priors — real: {prior['0']:.3f}, fake: {prior['1']:.3f}")
    for cls_str, name in [("0", "Real"), ("1", "Fake")]:
        print(f"\n  {name}:")
        for feat in feat_names:
            vals = kde_data[cls_str][feat]
            print(f"    {feat}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == '__main__':
    train()
