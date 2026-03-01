"""
test_laplacian.py — Standalone evaluation of the Laplacian variance ratio feature.

Feature hypothesis:
  FaceSwap synthesis over-smooths the face region → lower Laplacian variance →
  lap_var_ratio = var(Lap(face_gray)) / (var(Lap(bg_gray)) + 1e-8) < 1 for deepfakes.

Usage:
  .venv/bin/python test_laplacian.py
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/Users/justin/VSCODE PROJECTS/cs109_project"
REAL_DIR     = os.path.join(PROJECT_ROOT, "data", "real")
FAKE_DIR     = os.path.join(PROJECT_ROOT, "data", "deepfake")

# Add src/ to path so we can import segmentation
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from segmentation import process_video  # noqa: E402


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_lap_var_ratio(face: np.ndarray, bg: np.ndarray) -> float:
    """
    Compute the per-frame Laplacian variance ratio.

    Returns var(Laplacian(face_gray)) / (var(Laplacian(bg_gray)) + 1e-8).
    """
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(bg,   cv2.COLOR_BGR2GRAY).astype(float)

    lap_face = cv2.Laplacian(face_gray, cv2.CV_64F)
    lap_bg   = cv2.Laplacian(bg_gray,   cv2.CV_64F)

    var_face = float(np.var(lap_face))
    var_bg   = float(np.var(lap_bg))

    return var_face / (var_bg + 1e-8)


def extract_feature(video_path: str, n: int = 10) -> float | None:
    """
    Process a video and return the per-video mean Laplacian variance ratio.
    Returns None if no face frames were found.
    """
    try:
        tuples = process_video(video_path, n=n)
    except Exception as e:
        print(f"  [WARN] {os.path.basename(video_path)}: {e}")
        return None

    if not tuples:
        return None

    ratios = []
    for face, bg, _frame, _bbox in tuples:
        ratios.append(compute_lap_var_ratio(face, bg))

    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# Gaussian NB helpers
# ---------------------------------------------------------------------------

def softmax2(a: float, b: float) -> tuple[float, float]:
    """Numerically stable softmax for two logits."""
    m = max(a, b)
    ea, eb = np.exp(a - m), np.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


def gaussian_log_likelihood(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """log N(x; mu, sigma) — element-wise."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def fit_gaussian_nb(X_train: np.ndarray, y_train: np.ndarray):
    """
    MLE fit: per-class mean and std.
    Returns dict with keys 'mu_real', 'sigma_real', 'mu_fake', 'sigma_fake',
    'log_prior_real', 'log_prior_fake'.
    """
    X_real = X_train[y_train == 0]
    X_fake = X_train[y_train == 1]

    mu_real    = float(np.mean(X_real))
    sigma_real = float(np.std(X_real, ddof=1)) + 1e-6
    mu_fake    = float(np.mean(X_fake))
    sigma_fake = float(np.std(X_fake, ddof=1)) + 1e-6

    n = len(y_train)
    log_prior_real = np.log(len(X_real) / n)
    log_prior_fake = np.log(len(X_fake) / n)

    return {
        "mu_real": mu_real, "sigma_real": sigma_real,
        "mu_fake": mu_fake, "sigma_fake": sigma_fake,
        "log_prior_real": log_prior_real,
        "log_prior_fake": log_prior_fake,
    }


def predict_proba(X: np.ndarray, params: dict) -> np.ndarray:
    """Return P(fake | x) for each sample using log-space Gaussian NB + softmax."""
    log_real = (
        gaussian_log_likelihood(X, params["mu_real"], params["sigma_real"])
        + params["log_prior_real"]
    )
    log_fake = (
        gaussian_log_likelihood(X, params["mu_fake"], params["sigma_fake"])
        + params["log_prior_fake"]
    )

    p_fake = np.array([softmax2(lr, lf)[1] for lr, lf in zip(log_real, log_fake)])
    return p_fake


def evaluate_model(X_train, y_train, X_test, y_test, label: str):
    """
    Fit a Gaussian NB on train, report full metrics on test.
    Returns dict of metrics plus the params.
    """
    params = fit_gaussian_nb(X_train, y_train)

    # Training set statistics
    mu_r, sig_r = params["mu_real"], params["sigma_real"]
    mu_f, sig_f = params["mu_fake"], params["sigma_fake"]

    separation = abs(mu_f - mu_r) / (0.5 * (sig_r + sig_f))

    print(f"\n--- [{label}] Training set class statistics ---")
    print(f"  Real : mean={mu_r:.4f}  std={sig_r:.4f}")
    print(f"  Fake : mean={mu_f:.4f}  std={sig_f:.4f}")
    print(f"  Separation ratio: {separation:.4f}")

    # Test set predictions
    p_fake = predict_proba(X_test, params)
    y_pred = (p_fake >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, p_fake)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n--- [{label}] Test set metrics ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "label": label,
        "mu_real": mu_r, "sigma_real": sig_r,
        "mu_fake": mu_f, "sigma_fake": sig_f,
        "separation": separation,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Laplacian Variance Ratio — Feature Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Collect all video paths + labels
    # ------------------------------------------------------------------
    real_videos = sorted([
        os.path.join(REAL_DIR, f)
        for f in os.listdir(REAL_DIR)
        if f.lower().endswith(".mp4")
    ])
    fake_videos = sorted([
        os.path.join(FAKE_DIR, f)
        for f in os.listdir(FAKE_DIR)
        if f.lower().endswith(".mp4")
    ])

    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)

    print(f"\nReal videos : {len(real_videos)}")
    print(f"Fake videos : {len(fake_videos)}")
    print(f"Total       : {len(all_videos)}")

    # ------------------------------------------------------------------
    # 2. Extract features with progress bar
    # ------------------------------------------------------------------
    print("\nExtracting Laplacian variance ratio from all videos...")
    features_raw = []
    valid_labels = []
    skipped = 0

    for video_path, label in tqdm(
        zip(all_videos, all_labels),
        total=len(all_videos),
        desc="Processing videos",
        unit="video",
    ):
        val = extract_feature(video_path, n=10)
        if val is None:
            skipped += 1
            continue
        features_raw.append(val)
        valid_labels.append(label)

    print(f"\nSuccessfully extracted: {len(features_raw)} videos")
    print(f"Skipped (no face / error): {skipped} videos")

    X_raw = np.array(features_raw)
    y     = np.array(valid_labels)

    # Quick distribution summary
    real_vals = X_raw[y == 0]
    fake_vals = X_raw[y == 1]
    print(f"\nRaw feature distribution:")
    print(f"  Real  — min={real_vals.min():.3f}  max={real_vals.max():.3f}  "
          f"mean={real_vals.mean():.3f}  median={np.median(real_vals):.3f}  std={real_vals.std():.3f}")
    print(f"  Fake  — min={fake_vals.min():.3f}  max={fake_vals.max():.3f}  "
          f"mean={fake_vals.mean():.3f}  median={np.median(fake_vals):.3f}  std={fake_vals.std():.3f}")

    # Check right-skew: compare mean vs median; positive skew → mean > median
    real_skew = real_vals.mean() - np.median(real_vals)
    fake_skew = fake_vals.mean() - np.median(fake_vals)
    print(f"\n  Right-skew indicator (mean - median):")
    print(f"    Real: {real_skew:+.3f}{'  [right-skewed]' if real_skew > 0 else ''}")
    print(f"    Fake: {fake_skew:+.3f}{'  [right-skewed]' if fake_skew > 0 else ''}")

    # log1p transformation
    X_log = np.log1p(X_raw)
    real_log = X_log[y == 0]
    fake_log = X_log[y == 1]

    # Check variance ratio before/after (lower ratio → more balanced → log1p helps)
    raw_var_ratio = max(real_vals.var(), fake_vals.var()) / (
        min(real_vals.var(), fake_vals.var()) + 1e-8
    )
    log_var_ratio = max(real_log.var(), fake_log.var()) / (
        min(real_log.var(), fake_log.var()) + 1e-8
    )
    print(f"\n  Variance imbalance (max_var / min_var) — lower is better for Gaussian NB:")
    print(f"    Raw   : {raw_var_ratio:.3f}")
    print(f"    Log1p : {log_var_ratio:.3f}")
    log1p_helps = log_var_ratio < raw_var_ratio
    print(f"  log1p {'REDUCES' if log1p_helps else 'INCREASES'} variance imbalance "
          f"→ log1p {'helps' if log1p_helps else 'does NOT help'}.")

    # ------------------------------------------------------------------
    # 3. Stratified 80/20 train/test split
    # ------------------------------------------------------------------
    indices = np.arange(len(X_raw))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )

    X_raw_train, X_raw_test = X_raw[idx_train], X_raw[idx_test]
    X_log_train, X_log_test = X_log[idx_train], X_log[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"\nTrain set: {len(y_train)} samples  "
          f"(real={int((y_train==0).sum())}  fake={int((y_train==1).sum())})")
    print(f"Test  set: {len(y_test)} samples  "
          f"(real={int((y_test==0).sum())}  fake={int((y_test==1).sum())})")

    # ------------------------------------------------------------------
    # 4. Fit and evaluate — RAW feature
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating RAW Laplacian variance ratio")
    print("=" * 60)
    results_raw = evaluate_model(X_raw_train, y_train, X_raw_test, y_test, label="RAW")

    # ------------------------------------------------------------------
    # 5. Fit and evaluate — LOG1P feature
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating LOG1P(Laplacian variance ratio)")
    print("=" * 60)
    results_log = evaluate_model(X_log_train, y_train, X_log_test, y_test, label="LOG1P")

    # ------------------------------------------------------------------
    # 6. Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    winner = "LOG1P" if results_log["auc"] >= results_raw["auc"] else "RAW"
    print(f"\n{'Transform':<10}  {'Accuracy':>9}  {'AUC':>7}  {'F1':>7}  {'Sep.':>7}")
    print("-" * 50)
    for r in [results_raw, results_log]:
        print(f"{r['label']:<10}  {r['accuracy']:>9.4f}  {r['auc']:>7.4f}  "
              f"{r['f1']:>7.4f}  {r['separation']:>7.4f}")
    print(f"\nBetter AUC: {winner}")
    print(f"log1p {'HELPS' if results_log['auc'] > results_raw['auc'] else 'does NOT help'} "
          f"(AUC raw={results_raw['auc']:.4f} vs log1p={results_log['auc']:.4f})")

    # Hypothesis check
    mu_diff = results_raw["mu_real"] - results_raw["mu_fake"]
    print(f"\nHypothesis check (real > fake means synthesis over-smooths):")
    print(f"  mu_real={results_raw['mu_real']:.4f}  mu_fake={results_raw['mu_fake']:.4f}  "
          f"diff(real-fake)={mu_diff:+.4f}")
    if mu_diff > 0:
        print("  CONFIRMED: Real faces have higher Laplacian variance than fake — "
              "synthesis over-smoothing detected.")
    else:
        print("  NOT CONFIRMED: Fake faces show equal or higher Laplacian variance. "
              "Hypothesis does not hold on this dataset.")

    print("\nDone.")


if __name__ == "__main__":
    main()
