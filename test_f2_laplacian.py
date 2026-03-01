"""
test_f2_laplacian.py — Combined test: f2 (block-DCT β) + Laplacian variance ratio.

Loads f2 from data/features.csv (already computed), recomputes Laplacian
variance ratio for all 200 videos, then evaluates:
  1. f2 alone
  2. Laplacian alone
  3. f2 + Laplacian (2-feature NB)

Usage:
  .venv/bin/python test_f2_laplacian.py
"""

import os
import sys
import csv
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

PROJECT_ROOT  = "/Users/justin/VSCODE PROJECTS/cs109_project"
FEATURES_CSV  = os.path.join(PROJECT_ROOT, "data", "features.csv")
CACHE_CSV     = os.path.join(PROJECT_ROOT, "data", "features_f2_lap.csv")
REAL_DIR      = os.path.join(PROJECT_ROOT, "data", "real")
FAKE_DIR      = os.path.join(PROJECT_ROOT, "data", "deepfake")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from segmentation import process_video  # noqa: E402


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_lap_var_ratio(face: np.ndarray, bg: np.ndarray) -> float:
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(bg,   cv2.COLOR_BGR2GRAY).astype(float)
    lap_face  = cv2.Laplacian(face_gray, cv2.CV_64F)
    lap_bg    = cv2.Laplacian(bg_gray,   cv2.CV_64F)
    return float(np.var(lap_face)) / (float(np.var(lap_bg)) + 1e-8)


# ---------------------------------------------------------------------------
# Gaussian NB helpers (multi-feature, log-space)
# ---------------------------------------------------------------------------

EPSILON = 1e-6


def fit_nb(X_train: np.ndarray, y_train: np.ndarray):
    """MLE per-class Gaussian params. X_train shape: (n, n_features)."""
    X_real = X_train[y_train == 0]
    X_fake = X_train[y_train == 1]
    n = len(y_train)
    return {
        "mu_real":    np.mean(X_real, axis=0),
        "sigma_real": np.std(X_real, axis=0, ddof=1) + EPSILON,
        "mu_fake":    np.mean(X_fake, axis=0),
        "sigma_fake": np.std(X_fake, axis=0, ddof=1) + EPSILON,
        "log_prior_real": np.log(len(X_real) / n),
        "log_prior_fake": np.log(len(X_fake) / n),
    }


def predict_proba(X: np.ndarray, params: dict) -> np.ndarray:
    """Return P(fake) for each row using log-space NB + softmax."""
    def log_like(X, mu, sigma):
        return np.sum(-0.5 * ((X - mu) / sigma) ** 2 - np.log(sigma), axis=1)

    ll_real = log_like(X, params["mu_real"], params["sigma_real"]) + params["log_prior_real"]
    ll_fake = log_like(X, params["mu_fake"], params["sigma_fake"]) + params["log_prior_fake"]

    # Numerically stable softmax
    stacked = np.column_stack([ll_real, ll_fake])
    stacked -= stacked.max(axis=1, keepdims=True)
    exp_s = np.exp(stacked)
    return exp_s[:, 1] / exp_s.sum(axis=1)


def evaluate(X_train, y_train, X_test, y_test, feature_names, log_transform=None):
    """
    Fit NB on train, evaluate on test.
    log_transform: list of bool per feature column, or None (no transform).
    """
    if log_transform is None:
        log_transform = [False] * X_train.shape[1]

    def apply_log(X):
        X = X.copy()
        for i, do_log in enumerate(log_transform):
            if do_log:
                X[:, i] = np.log1p(X[:, i])
        return X

    X_tr = apply_log(X_train)
    X_te = apply_log(X_test)

    params = fit_nb(X_tr, y_train)

    # Class separation per feature
    for i, name in enumerate(feature_names):
        mu_r, sig_r = params["mu_real"][i], params["sigma_real"][i]
        mu_f, sig_f = params["mu_fake"][i], params["sigma_fake"][i]
        sep = abs(mu_f - mu_r) / (0.5 * (sig_r + sig_f))
        print(f"  {name}: real μ={mu_r:.4f} σ={sig_r:.4f} | fake μ={mu_f:.4f} σ={sig_f:.4f} | sep={sep:.3f}")

    p_fake = predict_proba(X_te, params)
    y_pred = (p_fake >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, p_fake)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return {"accuracy": acc, "auc": auc, "f1": f1, "precision": prec, "recall": rec}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Load f2 from features.csv
    # ------------------------------------------------------------------
    print("Loading f2 from features.csv...")
    f2_by_id = {}
    label_by_id = {}
    with open(FEATURES_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = row["video_id"]
            f2_by_id[vid]    = float(row["f2"])
            label_by_id[vid] = int(row["label"])

    print(f"  Loaded {len(f2_by_id)} entries from features.csv")

    # ------------------------------------------------------------------
    # 2. Compute Laplacian variance ratio (or load from cache)
    # ------------------------------------------------------------------
    if os.path.exists(CACHE_CSV):
        print(f"\nLoading cached features from {CACHE_CSV}...")
        records = []
        with open(CACHE_CSV) as fh:
            for row in csv.DictReader(fh):
                records.append((row["video_id"], float(row["f2"]),
                                 float(row["lap"]), int(row["label"])))
        print(f"  Loaded {len(records)} records from cache.")
    else:
        print("\nComputing Laplacian variance ratio for all 200 videos...")
        records = []  # (video_id, f2, lap_ratio, label)

        for subdir, label in [("real", 0), ("deepfake", 1)]:
            folder = os.path.join(PROJECT_ROOT, "data", subdir)
            videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))
            for fname in tqdm(videos, desc=subdir):
                vid = os.path.splitext(fname)[0]
                if vid not in f2_by_id:
                    print(f"  [SKIP] {vid} not in features.csv")
                    continue

                video_path = os.path.join(folder, fname)
                tuples = process_video(video_path, n=10)
                if not tuples:
                    print(f"  [SKIP] {vid} — no faces detected")
                    continue

                lap_ratios = [compute_lap_var_ratio(face, bg) for face, bg, _, _ in tuples]
                lap_ratio = float(np.mean(lap_ratios))
                records.append((vid, f2_by_id[vid], lap_ratio, label_by_id[vid]))

        # Save cache so future runs skip video processing
        with open(CACHE_CSV, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video_id", "f2", "lap", "label"])
            writer.writerows(records)
        print(f"  Saved cache → {CACHE_CSV}")

    print(f"\nProcessed: {len(records)} videos")
    real_count = sum(1 for r in records if r[3] == 0)
    fake_count = sum(1 for r in records if r[3] == 1)
    print(f"  Real: {real_count}  Fake: {fake_count}")

    # ------------------------------------------------------------------
    # 3. Build arrays and split
    # ------------------------------------------------------------------
    vids   = [r[0] for r in records]
    F2     = np.array([r[1] for r in records])
    LAP    = np.array([r[2] for r in records])
    labels = np.array([r[3] for r in records])

    idx = np.arange(len(records))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.20, random_state=42, stratify=labels
    )

    F2_tr,  F2_te  = F2[idx_train],  F2[idx_test]
    LAP_tr, LAP_te = LAP[idx_train], LAP[idx_test]
    y_tr,   y_te   = labels[idx_train], labels[idx_test]
    print(f"\nTrain: {len(y_tr)}  Test: {len(y_te)}")

    # ------------------------------------------------------------------
    # 4. Evaluate three combinations
    # ------------------------------------------------------------------
    results = {}

    print("\n" + "="*60)
    print("1. f2 alone (block-DCT β, log1p)")
    print("="*60)
    X_f2 = F2.reshape(-1, 1)
    results["f2"] = evaluate(
        X_f2[idx_train], y_tr, X_f2[idx_test], y_te,
        feature_names=["f2"],
        log_transform=[True],
    )

    print("\n" + "="*60)
    print("2. Laplacian variance ratio alone (log1p)")
    print("="*60)
    X_lap = LAP.reshape(-1, 1)
    results["lap"] = evaluate(
        X_lap[idx_train], y_tr, X_lap[idx_test], y_te,
        feature_names=["lap"],
        log_transform=[True],
    )

    print("\n" + "="*60)
    print("3. f2 + Laplacian NB product (both log1p)")
    print("="*60)
    X_both = np.column_stack([F2, LAP])
    results["f2+lap (NB)"] = evaluate(
        X_both[idx_train], y_tr, X_both[idx_test], y_te,
        feature_names=["f2", "lap"],
        log_transform=[True, True],
    )

    print("\n" + "="*60)
    print("4. f2 + Laplacian probability average (soft ensemble)")
    print("="*60)
    # Train each single-feature NB independently, then average P(fake)
    params_f2  = fit_nb(np.log1p(F2[idx_train].reshape(-1, 1)), y_tr)
    params_lap = fit_nb(np.log1p(LAP[idx_train].reshape(-1, 1)), y_tr)

    p_f2  = predict_proba(np.log1p(F2[idx_test].reshape(-1, 1)),  params_f2)
    p_lap = predict_proba(np.log1p(LAP[idx_test].reshape(-1, 1)), params_lap)
    p_avg = (p_f2 + p_lap) / 2.0

    y_pred_avg = (p_avg >= 0.5).astype(int)
    acc  = accuracy_score(y_te, y_pred_avg)
    prec = precision_score(y_te, y_pred_avg, zero_division=0)
    rec  = recall_score(y_te, y_pred_avg, zero_division=0)
    f1   = f1_score(y_te, y_pred_avg, zero_division=0)
    auc  = roc_auc_score(y_te, p_avg)
    cm   = confusion_matrix(y_te, y_pred_avg)
    print(f"  (P(fake) = average of individual NB probabilities)")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    results["f2+lap (avg)"] = {"accuracy": acc, "auc": auc, "f1": f1,
                                "precision": prec, "recall": rec}

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<12}  {'Acc':>6}  {'AUC':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<12}  {r['accuracy']:>6.4f}  {r['auc']:>6.4f}  {r['f1']:>6.4f}  "
              f"{r['precision']:>6.4f}  {r['recall']:>6.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
