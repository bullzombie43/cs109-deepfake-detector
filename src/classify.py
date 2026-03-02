"""
classify.py — KDE Naive Bayes classifier (Proposal 3).

Loads per-class training vectors from params.json, fits KDE likelihoods
(Gaussian kernel, Scott's bandwidth), and runs log-space NB.
Supports feature subset via CLI args.
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
from sklearn.neighbors import KernelDensity

PARAMS_JSON   = os.path.join(os.path.dirname(__file__), '..', 'models', 'params.json')
FEATURES_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_p5_tighter_crop.csv')
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
THRESHOLD     = 0.5

LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}


def softmax2(a: float, b: float):
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


def build_kdes(params: dict):
    """Pre-fit one KDE per class per feature from saved training vectors."""
    kde_data = params["kde_training_data"]
    kdes = {}
    for cls in ("0", "1"):
        kdes[cls] = {}
        for feat, vals in kde_data[cls].items():
            arr = np.array(vals).reshape(-1, 1)
            kde = KernelDensity(bandwidth='scott', kernel='gaussian')
            kde.fit(arr)
            kdes[cls][feat] = kde
    return kdes


def classify_one(raw: dict, params: dict, kdes: dict, features: tuple):
    """KDE-NB classification for one video. Returns (label, p_fake) at fixed 0.5 threshold."""
    log_tf = set(params["log_transform_features"])
    vals = {
        feat: float(np.log1p(raw[feat])) if feat in log_tf else raw[feat]
        for feat in raw
    }

    log_scores = {}
    for cls in ("0", "1"):
        log_score = params["log_priors"][cls]
        for feat in features:
            x = np.array([[vals[feat]]])
            log_score += kdes[cls][feat].score_samples(x)[0]
        log_scores[cls] = log_score

    p_real, p_fake = softmax2(log_scores["0"], log_scores["1"])
    return p_fake


def find_optimal_threshold(y_true, y_scores):
    """Find P(fake) cutoff maximising Youden's J = TPR - FPR on training scores."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def load_params(path):
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
           title='Confusion Matrix (KDE-NB)')
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
           title='ROC Curve (KDE-NB)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve       -> {out_path}")


def plot_score_distribution(y_true, y_scores, out_path, threshold=THRESHOLD):
    real_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_scores, y_true) if l == 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 25)
    ax.hist(real_scores, bins=bins, alpha=0.6, label='Real',     color='steelblue')
    ax.hist(fake_scores, bins=bins, alpha=0.6, label='Deepfake', color='tomato')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=0.9, label=f'Threshold={threshold:.3f}')
    ax.set(xlabel='P(Deepfake)', ylabel='Count', title='Score Distribution (KDE-NB)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved score dist.     -> {out_path}")


def evaluate(params_path=PARAMS_JSON, features_path=FEATURES_CSV, results_dir=RESULTS_DIR,
             features_subset=('f1', 'f2', 'f3', 'f4', 'f5')):
    params   = load_params(params_path)
    features = load_features(features_path)
    test_ids  = set(params["test_ids"])
    train_ids = set(params["train_ids"])

    print(f"Fitting KDEs on training data...")
    kdes = build_kdes(params)

    # Score training set to find the optimal threshold
    train_true, train_scores = [], []
    for vid in sorted(train_ids):
        if vid not in features:
            continue
        f1, f2, f3, f4, f5, true_label = features[vid]
        raw = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}
        train_scores.append(classify_one(raw, params, kdes, features_subset))
        train_true.append(true_label)

    threshold = find_optimal_threshold(train_true, train_scores)
    print(f"Optimal threshold (Youden's J on train): {threshold:.3f}  (was 0.500)")

    # Score test set, applying the optimised threshold
    y_true, y_scores = [], []
    missing = []

    for vid in sorted(test_ids):
        if vid not in features:
            missing.append(vid)
            continue
        f1, f2, f3, f4, f5, true_label = features[vid]
        raw = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5}
        y_scores.append(classify_one(raw, params, kdes, features_subset))
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
    print(f"  [KDE-NB -- features: {features_subset}]")
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
    tag = '_'.join(features_subset)
    plot_confusion_matrix(cm, os.path.join(results_dir, f'confusion_matrix_{tag}.png'))
    plot_roc(y_true, y_scores, auc, os.path.join(results_dir, f'roc_curve_{tag}.png'))
    plot_score_distribution(y_true, y_scores, os.path.join(results_dir, f'score_distribution_{tag}.png'),
                            threshold=threshold)

    metrics = {
        "experiment": f"KDE-NB features={list(features_subset)}",
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
    import sys
    subset = tuple(sys.argv[1:]) if len(sys.argv) > 1 else ('f1', 'f2', 'f3', 'f4', 'f5')
    print(f"Features used: {subset}")
    evaluate(features_subset=subset)
