"""
classify_logistic.py — Logistic Regression classifier.

Loads model from models/params_logistic_<features>.json, scores the test set,
optimises threshold via Youden's J on the training set, and evaluates.

Usage:
    python src/classify_logistic.py             # all 5 features
    python src/classify_logistic.py f5          # single feature
    python src/classify_logistic.py f1 f3 f5    # feature subset
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

FEATURES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_one(raw: dict, params: dict) -> float:
    """Returns P(fake) for one video."""
    log_tf    = set(params["log_transform_features"])
    features  = params["features"]
    coef      = params["coef"]
    intercept = params["intercept"]

    logit = intercept
    for feat, c in zip(features, coef):
        val = raw[feat]
        if feat in log_tf:
            val = math.log1p(val)
        logit += c * val
    return sigmoid(logit)


def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def load_params(models_dir, features_subset):
    tag  = '_'.join(features_subset)
    path = os.path.join(models_dir, f'params_logistic_{tag}.json')
    with open(path) as f:
        return json.load(f)


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


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Real', 'Deepfake']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix (Logistic Regression)')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def plot_roc(y_true, y_scores, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve (Logistic Regression)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve       -> {out_path}")


def plot_score_distribution(y_true, y_scores, out_path, threshold):
    real_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_scores, y_true) if l == 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 25)
    ax.hist(real_scores, bins=bins, alpha=0.6, label='Real',     color='steelblue')
    ax.hist(fake_scores, bins=bins, alpha=0.6, label='Deepfake', color='tomato')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=0.9, label=f'Threshold={threshold:.3f}')
    ax.set(xlabel='P(Deepfake)', ylabel='Count', title='Score Distribution (Logistic Regression)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved score dist.     -> {out_path}")


def evaluate(features_path=FEATURES_CSV, models_dir=MODELS_DIR, results_dir=RESULTS_DIR,
             features_subset=('f1', 'f2', 'f3', 'f4', 'f5')):
    params   = load_params(models_dir, features_subset)
    features = load_features(features_path)
    train_ids = set(params["train_ids"])
    test_ids  = set(params["test_ids"])

    # Score training set to find optimal threshold
    train_true, train_scores = [], []
    for vid in sorted(train_ids):
        if vid not in features:
            continue
        f1, f2, f3, f4, f5, true_label = features[vid]
        raw = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}
        train_scores.append(score_one(raw, params))
        train_true.append(true_label)

    threshold = find_optimal_threshold(train_true, train_scores)
    print(f"Optimal threshold (Youden's J on train): {threshold:.3f}")

    # Score test set
    y_true, y_scores = [], []
    missing = []
    for vid in sorted(test_ids):
        if vid not in features:
            missing.append(vid)
            continue
        f1, f2, f3, f4, f5, true_label = features[vid]
        raw = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}
        y_scores.append(score_one(raw, params))
        y_true.append(true_label)

    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    if missing:
        print(f"WARNING: {len(missing)} test videos not found in CSV: {missing}")

    n = len(y_true)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1s  = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_scores)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f"  [Logistic Regression -- features: {features_subset}]")
    print(f"  Threshold: {threshold:.3f}  (optimised on train)")
    print(f"  Test set: {n} videos")
    print(f"  Accuracy:  {acc:.3f}  ({int(acc*n)}/{n} correct)")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1s:.3f}")
    print(f"  ROC AUC:   {auc:.3f}")
    print(f"{'='*40}\n")
    print(f"Confusion matrix (rows=true, cols=predicted):\n{cm}")

    os.makedirs(results_dir, exist_ok=True)
    tag = 'logistic_' + '_'.join(features_subset)
    plot_confusion_matrix(cm, os.path.join(results_dir, f'confusion_matrix_{tag}.png'))
    plot_roc(y_true, y_scores, auc, os.path.join(results_dir, f'roc_curve_{tag}.png'))
    plot_score_distribution(y_true, y_scores, os.path.join(results_dir, f'score_distribution_{tag}.png'),
                            threshold=threshold)

    metrics = {
        "experiment": f"Logistic-Regression features={list(features_subset)}",
        "threshold": threshold,
        "n_test": n, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1s, "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(results_dir, f'metrics_{tag}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {os.path.join(results_dir, f'metrics_{tag}.json')}")
    return acc, auc


if __name__ == '__main__':
    subset = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ('f1', 'f2', 'f3', 'f4', 'f5')
    print(f"Features used: {subset}")
    evaluate(features_subset=subset)
