"""
classify.py — Phase 5: Classification and evaluation.

Loads models/params.json and data/features.csv, runs Gaussian Naive Bayes
on the held-out test set, and writes metrics + plots to results/.
"""

import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

PARAMS_JSON   = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
FEATURES_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
THRESHOLD     = 0.5


# ---------------------------------------------------------------------------
# Gaussian log-likelihood
# ---------------------------------------------------------------------------

def log_gaussian(x: float, mu: float, sigma: float) -> float:
    """Log of N(x; mu, sigma) — computed in log space for numerical stability."""
    return -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


# ---------------------------------------------------------------------------
# Softmax (two classes)
# ---------------------------------------------------------------------------

def softmax2(a: float, b: float):
    """Softmax over two log-scores. Returns (p_a, p_b)."""
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


# ---------------------------------------------------------------------------
# Classify a single video
# ---------------------------------------------------------------------------

def classify_one(f1: float, f2: float, f3: float, params: dict, features=('f1', 'f2', 'f3', 'f4', 'f5'), f4: float = 0.0, f5: float = 0.0):
    """
    Run log-space Naive Bayes for one video.

    Features marked with log_transform=True in params are log1p-transformed
    before likelihood computation, matching the transformation applied at training.

    Args:
        features: tuple of feature names to include (default: all)
    Returns:
        label (int):       predicted class (0=real, 1=deepfake)
        p_fake (float):    P(Deepfake | features) after softmax
    """
    raw = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}
    # Apply log transform where the model was trained in log-space
    ref_class = params["classes"]["0"]
    vals = {
        feat: math.log1p(raw[feat]) if ref_class[feat].get('log_transform') else raw[feat]
        for feat in ('f1', 'f2', 'f3', 'f4', 'f5')
    }

    log_scores = {}
    for cls in ("0", "1"):
        p = params["classes"][cls]
        log_score = math.log(p["prior"])
        for feat in features:
            log_score += log_gaussian(vals[feat], p[feat]["mu"], p[feat]["sigma"])
        log_scores[cls] = log_score

    p_real, p_fake = softmax2(log_scores["0"], log_scores["1"])
    label = 1 if p_fake >= THRESHOLD else 0
    return label, p_fake


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_params(path):
    with open(path) as f:
        return json.load(f)


def load_features(path):
    rows = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(',')
            vid   = parts[0]
            f1, f2, f3, f4, f5 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            label = int(parts[6])
            rows[vid] = (f1, f2, f3, f4, f5, label)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Real', 'Deepfake']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")


def plot_roc(y_true, y_scores, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve       → {out_path}")


def plot_score_distribution(y_true, y_scores, out_path):
    real_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_scores, y_true) if l == 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 25)
    ax.hist(real_scores, bins=bins, alpha=0.6, label='Real',     color='steelblue')
    ax.hist(fake_scores, bins=bins, alpha=0.6, label='Deepfake', color='tomato')
    ax.axvline(THRESHOLD, color='black', linestyle='--', linewidth=0.9, label=f'Threshold={THRESHOLD}')
    ax.set(xlabel='P(Deepfake)', ylabel='Count', title='Classifier Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved score dist.     → {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(params_path=PARAMS_JSON, features_path=FEATURES_CSV, results_dir=RESULTS_DIR,
             features_subset=('f1', 'f2', 'f3')):
    params   = load_params(params_path)
    features = load_features(features_path)
    test_ids = set(params["test_ids"])

    y_true, y_pred, y_scores = [], [], []
    missing = []

    for vid in sorted(test_ids):
        if vid not in features:
            missing.append(vid)
            continue
        f1, f2, f3, f4, f5, true_label = features[vid]
        pred_label, p_fake = classify_one(f1, f2, f3, params, features=features_subset, f4=f4, f5=f5)
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(p_fake)

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
    print(f"  Test set: {n} videos")
    print(f"  Accuracy:  {acc:.3f}  ({int(acc*n)}/{n} correct)")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1s:.3f}")
    print(f"  ROC AUC:   {auc:.3f}")
    print(f"{'='*40}\n")
    print(f"Confusion matrix (rows=true, cols=predicted):\n{cm}")

    os.makedirs(results_dir, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(results_dir, 'confusion_matrix.png'))
    plot_roc(y_true, y_scores, auc, os.path.join(results_dir, 'roc_curve.png'))
    plot_score_distribution(y_true, y_scores, os.path.join(results_dir, 'score_distribution.png'))

    metrics = {
        "n_test": n, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1s, "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics         → {os.path.join(results_dir, 'metrics.json')}")


if __name__ == '__main__':
    import sys
    subset = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ('f1', 'f2', 'f3', 'f4', 'f5')
    print(f"Features used: {subset}")
    evaluate(features_subset=subset)
