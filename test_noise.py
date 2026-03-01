"""
test_noise.py — Candidate feature: noise level ratio (face vs background).

Feature: per-video mean of per-frame ratios:
    noise_var(face_gray) / (noise_var(bg_gray) + 1e-8)

where noise_var is the variance of the high-frequency residual after
Gaussian smoothing (5x5, sigma=0).

Hypothesis: FaceSwap synthesis alters the sensor noise pattern in the face
region. Real faces carry camera noise that roughly matches their local
background; deepfake faces may show a different noise level.
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/Users/justin/VSCODE PROJECTS/cs109_project"
REAL_DIR  = os.path.join(PROJECT_ROOT, "data", "real")
FAKE_DIR  = os.path.join(PROJECT_ROOT, "data", "deepfake")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from segmentation import process_video


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def noise_var(gray_float: np.ndarray) -> float:
    """Variance of the high-frequency residual after Gaussian smoothing."""
    blurred = cv2.GaussianBlur(gray_float, (5, 5), 0)
    residual = gray_float - blurred
    return float(np.var(residual))


def compute_noise_ratio(video_path: str, n: int = 10) -> float | None:
    """
    Return per-video mean noise level ratio: noise_var(face) / noise_var(bg).
    Returns None if no usable frames are found.
    """
    tuples = process_video(video_path, n=n)
    if not tuples:
        return None

    ratios = []
    for face, bg, _frame, _bbox in tuples:
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
        bg_gray   = cv2.cvtColor(bg,   cv2.COLOR_BGR2GRAY).astype(float)
        ratio = noise_var(face_gray) / (noise_var(bg_gray) + 1e-8)
        ratios.append(ratio)

    return float(np.mean(ratios)) if ratios else None


# ---------------------------------------------------------------------------
# Gaussian NB helpers
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log N(x; mu, sigma) for each sample in x."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def softmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(class=1) via two-class softmax of log-scores a (class 0) and b (class 1)."""
    # Numerically stable: subtract max
    m = np.maximum(a, b)
    exp_a = np.exp(a - m)
    exp_b = np.exp(b - m)
    return exp_b / (exp_a + exp_b)


def fit_nb(X_train: np.ndarray, y_train: np.ndarray, eps: float = 1e-6):
    """MLE Gaussian NB. Returns (mu0, sigma0, mu1, sigma1, log_prior0, log_prior1)."""
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    mu0, sigma0 = float(np.mean(X0)), float(np.std(X0)) + eps
    mu1, sigma1 = float(np.mean(X1)), float(np.std(X1)) + eps
    log_prior0 = np.log(len(X0) / len(X_train))
    log_prior1 = np.log(len(X1) / len(X_train))
    return mu0, sigma0, mu1, sigma1, log_prior0, log_prior1


def predict_nb(X_test: np.ndarray, mu0, sigma0, mu1, sigma1, lp0, lp1):
    """Return (y_pred, p_fake) arrays."""
    log0 = gaussian_log_likelihood(X_test, mu0, sigma0) + lp0
    log1 = gaussian_log_likelihood(X_test, mu1, sigma1) + lp1
    p_fake = softmax(log0, log1)
    y_pred = (p_fake >= 0.5).astype(int)
    return y_pred, p_fake


def evaluate(y_true, y_pred, p_fake, label: str):
    """Print and return metrics dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, p_fake)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n--- {label} ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_videos(directory: str, label: int):
    paths, labels = [], []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".mp4"):
            paths.append(os.path.join(directory, fname))
            labels.append(label)
    return paths, labels


def main():
    # -- Collect video paths -------------------------------------------------
    real_paths, real_labels = collect_videos(REAL_DIR, 0)
    fake_paths, fake_labels = collect_videos(FAKE_DIR, 1)

    all_paths  = real_paths + fake_paths
    all_labels = real_labels + fake_labels

    print(f"Found {len(real_paths)} real videos and {len(fake_paths)} deepfake videos.")
    print(f"Total: {len(all_paths)} videos.\n")

    # -- Extract features with tqdm ------------------------------------------
    features = []
    valid_labels = []
    skipped = 0

    for vpath, vlabel in tqdm(zip(all_paths, all_labels),
                               total=len(all_paths),
                               desc="Extracting noise ratio"):
        val = compute_noise_ratio(vpath, n=10)
        if val is None:
            skipped += 1
            continue
        features.append(val)
        valid_labels.append(vlabel)

    features = np.array(features, dtype=float)
    valid_labels = np.array(valid_labels, dtype=int)

    print(f"\nFeature extraction complete.")
    print(f"  Usable videos : {len(features)}")
    print(f"  Skipped (no face detected) : {skipped}")

    # -- Train/test split (video-level, stratified) --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features, valid_labels,
        test_size=0.20,
        random_state=42,
        stratify=valid_labels
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # -- Raw fit & stats -----------------------------------------------------
    X0_tr = X_train[y_train == 0]
    X1_tr = X_train[y_train == 1]
    mu0_raw = float(np.mean(X0_tr))
    mu1_raw = float(np.mean(X1_tr))
    sd0_raw = float(np.std(X0_tr))
    sd1_raw = float(np.std(X1_tr))

    print(f"\n=== Training set class statistics (RAW) ===")
    print(f"  Real  (0): mean={mu0_raw:.6f}, std={sd0_raw:.6f}")
    print(f"  Fake  (1): mean={mu1_raw:.6f}, std={sd1_raw:.6f}")

    sep_raw = abs(mu1_raw - mu0_raw) / (0.5 * (sd0_raw + sd1_raw) + 1e-10)
    print(f"  Separation ratio (|fake_mean - real_mean| / avg_std): {sep_raw:.4f}")

    # -- Log1p fit & stats ---------------------------------------------------
    X_train_log = np.log1p(X_train)
    X_test_log  = np.log1p(X_test)

    X0_log = X_train_log[y_train == 0]
    X1_log = X_train_log[y_train == 1]
    mu0_log = float(np.mean(X0_log))
    mu1_log = float(np.mean(X1_log))
    sd0_log = float(np.std(X0_log))
    sd1_log = float(np.std(X1_log))

    print(f"\n=== Training set class statistics (LOG1P) ===")
    print(f"  Real  (0): mean={mu0_log:.6f}, std={sd0_log:.6f}")
    print(f"  Fake  (1): mean={mu1_log:.6f}, std={mu1_log:.6f}")

    sep_log = abs(mu1_log - mu0_log) / (0.5 * (sd0_log + sd1_log) + 1e-10)
    print(f"  Separation ratio (|fake_mean - real_mean| / avg_std): {sep_log:.4f}")

    # -- Fit and evaluate — RAW ----------------------------------------------
    params_raw = fit_nb(X_train, y_train)
    y_pred_raw, p_fake_raw = predict_nb(X_test, *params_raw)
    metrics_raw = evaluate(y_test, y_pred_raw, p_fake_raw, "Gaussian NB — RAW feature")

    # -- Fit and evaluate — LOG1P --------------------------------------------
    params_log = fit_nb(X_train_log, y_train)
    y_pred_log, p_fake_log = predict_nb(X_test_log, *params_log)
    metrics_log = evaluate(y_test, y_pred_log, p_fake_log, "Gaussian NB — LOG1P feature")

    # -- Compare and summarize -----------------------------------------------
    print("\n=== Summary ===")
    print(f"  RAW   — Accuracy: {metrics_raw['accuracy']:.4f}, AUC: {metrics_raw['auc']:.4f}")
    print(f"  LOG1P — Accuracy: {metrics_log['accuracy']:.4f}, AUC: {metrics_log['auc']:.4f}")

    if metrics_log['auc'] > metrics_raw['auc']:
        winner = "LOG1P"
        delta  = metrics_log['auc'] - metrics_raw['auc']
    else:
        winner = "RAW"
        delta  = metrics_raw['auc'] - metrics_log['auc']

    print(f"\n  Best by AUC: {winner} (delta={delta:.4f})")
    print(f"  Log1p {'helped' if winner == 'LOG1P' else 'did NOT help'} "
          f"for this feature.")

    # -- Additional distribution info ----------------------------------------
    print(f"\n=== Feature distribution (all {len(features)} videos) ===")
    all_real = features[valid_labels == 0]
    all_fake = features[valid_labels == 1]
    print(f"  Real  — min={all_real.min():.4f}, median={np.median(all_real):.4f}, "
          f"max={all_real.max():.4f}")
    print(f"  Fake  — min={all_fake.min():.4f}, median={np.median(all_fake):.4f}, "
          f"max={all_fake.max():.4f}")


if __name__ == "__main__":
    main()
