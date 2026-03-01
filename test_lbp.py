"""
test_lbp.py — Self-contained test of LBP chi-squared feature for deepfake detection.

Feature: LBP (Local Binary Pattern) histogram chi-squared divergence, face vs background.
Hypothesis: FaceSwap synthesizes the face from a different source domain, altering
            micro-texture patterns, so the LBP distribution of the face diverges more
            from the background in deepfakes.

Run:
    .venv/bin/python test_lbp.py
"""

import os
import sys
import glob
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Add src/ to path so we can import process_video
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from segmentation import process_video  # noqa: E402

# ---------------------------------------------------------------------------
# LBP helpers — numpy only, no skimage
# ---------------------------------------------------------------------------

def compute_lbp_hist(gray_uint8: np.ndarray) -> np.ndarray:
    """Return normalized 256-bin LBP histogram for a grayscale uint8 image."""
    h, w = gray_uint8.shape
    if h < 3 or w < 3:
        # Image too small to compute LBP — return uniform histogram
        return np.ones(256, dtype=float) / 256.0

    center = gray_uint8[1:h-1, 1:w-1].astype(np.int32)
    neighbors = [
        gray_uint8[0:h-2, 0:w-2],  # top-left
        gray_uint8[0:h-2, 1:w-1],  # top
        gray_uint8[0:h-2, 2:w  ],  # top-right
        gray_uint8[1:h-1, 2:w  ],  # right
        gray_uint8[2:h,   2:w  ],  # bottom-right
        gray_uint8[2:h,   1:w-1],  # bottom
        gray_uint8[2:h,   0:w-2],  # bottom-left
        gray_uint8[1:h-1, 0:w-2],  # left
    ]
    lbp_img = np.zeros(center.shape, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        lbp_img |= ((n.astype(np.int32) >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp_img, bins=np.arange(257))
    return hist.astype(float) / (hist.sum() + 1e-12)


def chi_squared(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric chi-squared divergence between two normalized histograms."""
    denom = p + q
    mask = denom > 1e-12
    return float(0.5 * np.sum(((p[mask] - q[mask]) ** 2) / denom[mask]))


def compute_lbp_feature(video_path: str) -> float | None:
    """
    Process one video: sample 10 frames, compute per-frame LBP chi-sq (face vs bg),
    return the mean across frames. Returns None if no faces detected.
    """
    try:
        tuples = process_video(video_path, n=10)
    except Exception as e:
        print(f"\n  [WARN] process_video failed for {os.path.basename(video_path)}: {e}")
        return None

    if not tuples:
        return None

    frame_scores = []
    for face_bgr, bg_bgr, _frame, _bbox in tuples:
        import cv2
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        bg_gray   = cv2.cvtColor(bg_bgr,  cv2.COLOR_BGR2GRAY)
        score = chi_squared(
            compute_lbp_hist(face_gray),
            compute_lbp_hist(bg_gray),
        )
        frame_scores.append(score)

    return float(np.mean(frame_scores))


# ---------------------------------------------------------------------------
# Gaussian NB helpers (log-space)
# ---------------------------------------------------------------------------

def fit_gaussian_nb(X_train: np.ndarray, y_train: np.ndarray, eps: float = 1e-6):
    """Return (mu0, sigma0, mu1, sigma1, log_prior0, log_prior1)."""
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    mu0, sigma0 = float(np.mean(X0)), float(np.std(X0)) + eps
    mu1, sigma1 = float(np.mean(X1)), float(np.std(X1)) + eps
    n  = len(y_train)
    log_prior0 = np.log(len(X0) / n)
    log_prior1 = np.log(len(X1) / n)
    return mu0, sigma0, mu1, sigma1, log_prior0, log_prior1


def gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def predict_proba(X: np.ndarray, params: tuple) -> np.ndarray:
    """Return P(fake | x) for each sample using softmax of log-posteriors."""
    mu0, sigma0, mu1, sigma1, log_prior0, log_prior1 = params
    log_p0 = np.array([gaussian_log_likelihood(x, mu0, sigma0) for x in X]) + log_prior0
    log_p1 = np.array([gaussian_log_likelihood(x, mu1, sigma1) for x in X]) + log_prior1
    # Numerically stable softmax for the two-class case
    log_diff = log_p1 - log_p0
    prob_fake = 1.0 / (1.0 + np.exp(-log_diff))
    return prob_fake


def evaluate(X_test: np.ndarray, y_test: np.ndarray, params: tuple, label: str):
    """Print full evaluation metrics and return AUC."""
    prob_fake = predict_proba(X_test, params)
    y_pred = (prob_fake >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, prob_fake)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  [{label}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC AUC  : {auc:.4f}")
    print(f"    Confusion matrix (rows=actual, cols=predicted):")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")

    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    real_dir = os.path.join(ROOT, "data", "real")
    fake_dir = os.path.join(ROOT, "data", "deepfake")

    real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))

    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} deepfake videos.")
    assert len(real_videos) > 0, f"No .mp4 files found in {real_dir}"
    assert len(fake_videos) > 0, f"No .mp4 files found in {fake_dir}"

    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]

    # --- Feature extraction ---
    print("\nExtracting LBP chi-squared features (10 frames/video)...")
    video_ids, features, labels = [], [], []

    for video_path, label in tqdm(all_videos, desc="Processing videos", unit="video"):
        score = compute_lbp_feature(video_path)
        if score is None:
            print(f"\n  [SKIP] No faces detected: {os.path.basename(video_path)}")
            continue
        vid_id = os.path.splitext(os.path.basename(video_path))[0]
        video_ids.append(vid_id)
        features.append(score)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    print(f"\nSuccessfully processed {len(X)} / {len(all_videos)} videos.")
    print(f"  Real   : {int((y == 0).sum())}  |  Fake: {int((y == 1).sum())}")

    # --- Raw feature stats ---
    real_vals = X[y == 0]
    fake_vals = X[y == 1]
    print("\n--- Raw LBP chi-squared statistics (training set approximation) ---")
    print(f"  Real  — mean: {np.mean(real_vals):.6f},  std: {np.std(real_vals):.6f}")
    print(f"  Fake  — mean: {np.mean(fake_vals):.6f},  std: {np.std(fake_vals):.6f}")
    raw_sep = abs(np.mean(fake_vals) - np.mean(real_vals)) / (
        0.5 * (np.std(real_vals) + np.std(fake_vals)) + 1e-12
    )
    print(f"  Separation ratio (all data): {raw_sep:.4f}")

    # --- Train / test split (stratified 80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # --- Fit on training set, print class stats ---
    real_tr = X_train[y_train == 0]
    fake_tr = X_train[y_train == 1]
    print("\n--- Training set class statistics (raw) ---")
    print(f"  Real  — mean: {np.mean(real_tr):.6f},  std: {np.std(real_tr):.6f}")
    print(f"  Fake  — mean: {np.mean(fake_tr):.6f},  std: {np.std(fake_tr):.6f}")
    sep = abs(np.mean(fake_tr) - np.mean(real_tr)) / (
        0.5 * (np.std(real_tr) + np.std(fake_tr)) + 1e-12
    )
    print(f"  Separation ratio: {sep:.4f}")

    # --- Evaluate: RAW ---
    print("\n=== Evaluation: RAW feature ===")
    params_raw = fit_gaussian_nb(X_train, y_train)
    auc_raw = evaluate(X_test, y_test, params_raw, "RAW")

    # --- Evaluate: log1p-transformed ---
    print("\n=== Evaluation: log1p-transformed feature ===")
    X_train_log = np.log1p(X_train)
    X_test_log  = np.log1p(X_test)

    real_tr_log = X_train_log[y_train == 0]
    fake_tr_log = X_train_log[y_train == 1]
    print(f"\n--- Training set class statistics (log1p) ---")
    print(f"  Real  — mean: {np.mean(real_tr_log):.6f},  std: {np.std(real_tr_log):.6f}")
    print(f"  Fake  — mean: {np.mean(fake_tr_log):.6f},  std: {np.std(fake_tr_log):.6f}")
    sep_log = abs(np.mean(fake_tr_log) - np.mean(real_tr_log)) / (
        0.5 * (np.std(real_tr_log) + np.std(fake_tr_log)) + 1e-12
    )
    print(f"  Separation ratio (log1p): {sep_log:.4f}")

    params_log = fit_gaussian_nb(X_train_log, y_train)
    auc_log = evaluate(X_test_log, y_test, params_log, "LOG1P")

    # --- Summary ---
    print("\n=== Summary ===")
    better = "log1p" if auc_log > auc_raw else "raw"
    print(f"  AUC (raw)   : {auc_raw:.4f}")
    print(f"  AUC (log1p) : {auc_log:.4f}")
    print(f"  Better transform: {better}")
    if auc_log > auc_raw:
        print("  => log1p transform HELPED.")
    elif auc_log < auc_raw:
        print("  => log1p transform did NOT help (raw is better).")
    else:
        print("  => log1p transform made no difference.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
