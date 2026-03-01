"""
features.py — Phase 3: Feature extraction.

Three features per video (PRD §3.2):
  f1: DCT chi-squared divergence (compression artifact detector)
  f2: Color histogram chi-squared, summed across R, G, B channels
  f3: Absolute difference of mean absolute DCT coefficients

Public API:
  compute_f1(face, background)            -> float
  compute_f2(face, background)            -> float
  compute_f3(face, background)            -> float
  extract_features(face_bg_pairs)         -> list of (f1, f2, f3) tuples
  aggregate_features(frame_features)      -> (f1, f2, f3) mean across frames
  process_all_videos(data_dir, out_csv)   -> writes features CSV

Run directly to process all videos and write data/features.csv:
  python src/features.py
"""

import os
import sys
import csv

import cv2
import numpy as np

# Allow running as `python src/features.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import dct2, chi_squared
from src.segmentation import process_video, process_video_pairs


# ---------------------------------------------------------------------------
# Per-frame feature computation
# ---------------------------------------------------------------------------

def compute_f1(face: np.ndarray, background: np.ndarray) -> float:
    """
    DCT chi-squared divergence (PRD §3.2.1).

    Measures mismatch in frequency-domain coefficient distributions between
    the face and background. High values indicate double-compression artifacts
    typical of deepfakes.
    """
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(float)

    face_coeffs = dct2(face_gray).flatten()
    bg_coeffs   = dct2(bg_gray).flatten()

    # Shared bin edges spanning the combined coefficient range (50 bins)
    min_val = min(face_coeffs.min(), bg_coeffs.min())
    max_val = max(face_coeffs.max(), bg_coeffs.max())
    bin_edges = np.linspace(min_val, max_val, 51)  # 51 edges → 50 bins

    face_hist, _ = np.histogram(face_coeffs, bins=bin_edges)
    bg_hist,   _ = np.histogram(bg_coeffs,   bins=bin_edges)

    # Normalize to probability distributions so pixel-count imbalance
    # (large background vs small face crop) doesn't dominate chi-squared.
    face_hist = face_hist / (face_hist.sum() + 1e-12)
    bg_hist   = bg_hist   / (bg_hist.sum()   + 1e-12)

    return chi_squared(face_hist, bg_hist)


def _block_dct_beta(gray: np.ndarray) -> np.ndarray:
    """
    Compute β = std/√2 for each of the 63 AC positions across all 8×8 DCT blocks.

    Crops the image to the nearest multiple of 8, divides into non-overlapping
    8×8 blocks, applies 2D DCT to each, then collects coefficient values at
    each AC position (positions 1–63 in row-major order, skipping DC at 0).
    Returns a 63-element array of β values, or None if too small.
    """
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return None

    cropped = gray[:h8, :w8]
    ac_values = [[] for _ in range(63)]

    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            coeffs = dct2(cropped[i:i + 8, j:j + 8]).flatten()
            for k in range(63):
                ac_values[k].append(coeffs[k + 1])  # skip DC at index 0

    return np.array([np.std(v) / np.sqrt(2) for v in ac_values])


def compute_f2(face: np.ndarray, background: np.ndarray) -> float:
    """
    Block-DCT β-statistic divergence: face vs. background.

    Divides face and background into 8×8 blocks, applies 2D DCT per block,
    and computes β = std/√2 for each of the 63 AC coefficient positions across
    all blocks. Returns the mean absolute difference of β values across all
    63 positions.

    FaceSwap puts the face through two encode/decode cycles (source→manipulate
    →output), changing the AC coefficient scale parameters relative to the
    background (encoded once). On c0 (lossless) data this double-processing
    artifact is preserved and measurable.

    Based on: Giudice et al. "Fighting Deepfakes by Detecting GAN DCT
    Anomalies", MDPI Journal of Imaging, 2021.
    """
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(float)

    face_beta = _block_dct_beta(face_gray)
    bg_beta   = _block_dct_beta(bg_gray)

    if face_beta is None or bg_beta is None:
        return 0.0

    return float(np.mean(np.abs(face_beta - bg_beta)))


def compute_f5(face: np.ndarray, background: np.ndarray) -> float:
    """
    Laplacian variance ratio: var(Laplacian(face)) / var(Laplacian(bg)).

    Measures sharpness of the face region relative to the background.
    FaceSwap introduces sharpening/ringing artifacts in the face region,
    raising Laplacian variance; real faces match their local background.
    Log1p-transformed at training time.
    """
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(float)
    lap_face  = cv2.Laplacian(face_gray, cv2.CV_64F)
    lap_bg    = cv2.Laplacian(bg_gray,   cv2.CV_64F)
    return float(np.var(lap_face)) / (float(np.var(lap_bg)) + 1e-8)


def compute_f3(face: np.ndarray, background: np.ndarray) -> float:
    """
    Absolute difference of mean absolute DCT coefficients (PRD §3.2.3).

    Captures smoothing/texture-loss effects from AI processing. Deepfake faces
    often appear slightly blurred relative to the background.
    """
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)
    bg_gray   = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(float)

    face_mean = np.mean(np.abs(dct2(face_gray)))
    bg_mean   = np.mean(np.abs(dct2(bg_gray)))

    return abs(face_mean - bg_mean)


# ---------------------------------------------------------------------------
# f4: Temporal face inconsistency
# ---------------------------------------------------------------------------

_FACE_SIZE = (64, 64)  # common resize target for cross-frame comparison


def compute_f4(face_pairs: list) -> float:
    """
    Temporal inconsistency of the face crop across consecutive frame pairs.

    For each (face_t, face_t1) pair:
      1. Resize both crops to 64x64
      2. Convert to grayscale
      3. Normalize each frame by its mean intensity (removes lighting drift)
      4. Compute mean absolute pixel difference

    f4 = mean of these differences across all anchor pairs.

    Expected: fake > real, because per-frame warp estimation introduces
    jitter on top of natural head motion.
    """
    if not face_pairs:
        return 0.0

    diffs = []
    for face_t, face_t1 in face_pairs:
        ft_gray  = cv2.cvtColor(cv2.resize(face_t,  _FACE_SIZE), cv2.COLOR_BGR2GRAY).astype(float)
        ft1_gray = cv2.cvtColor(cv2.resize(face_t1, _FACE_SIZE), cv2.COLOR_BGR2GRAY).astype(float)

        # Normalize by mean intensity so lighting changes don't dominate
        ft_norm  = ft_gray  / (ft_gray.mean()  + 1e-6)
        ft1_norm = ft1_gray / (ft1_gray.mean() + 1e-6)

        diffs.append(float(np.mean(np.abs(ft_norm - ft1_norm))))

    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Aggregation (Approach B — per-video)
# ---------------------------------------------------------------------------

def extract_features(tuples: list) -> list:
    """
    Compute (f1, f2, f3, f5) for each (face, background, frame, bbox) tuple.
    Returns a list of tuples, one per frame.
    """
    return [
        (compute_f1(face, bg), compute_f2(face, bg), compute_f3(face, bg), compute_f5(face, bg))
        for face, bg, frame, bbox in tuples
    ]


def aggregate_features(frame_features: list) -> tuple:
    """
    Aggregate per-frame features into a single video-level feature vector
    by taking the mean across frames (PRD §3.3, Approach B).
    """
    arr = np.array(frame_features)  # shape (n_frames, 4)
    return tuple(np.mean(arr, axis=0))


# ---------------------------------------------------------------------------
# Full pipeline: all videos → CSV
# ---------------------------------------------------------------------------

def process_all_videos(data_dir: str, out_csv: str) -> None:
    """
    Extract features for every video in data_dir/real/ and data_dir/deepfake/,
    then write a CSV to out_csv with columns: video_id, f1, f2, f3, f4, label.

    label: 0 = real, 1 = deepfake
    """
    from tqdm import tqdm

    subdirs = [("real", 0), ("deepfake", 1)]
    rows = []
    skipped = 0

    for subdir, label in subdirs:
        folder = os.path.join(data_dir, subdir)
        videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))
        print(f"\nProcessing {len(videos)} videos from {folder} (label={label})")

        for fname in tqdm(videos, desc=subdir):
            video_path = os.path.join(folder, fname)

            # f1, f2, f3 — from single-frame (face, background, frame, bbox) tuples
            tuples = process_video(video_path)
            if not tuples:
                print(f"  WARNING: no faces found in {fname}, skipping")
                skipped += 1
                continue
            frame_features = extract_features(tuples)
            f1, f2, f3, f5 = aggregate_features(frame_features)

            # f4 — from consecutive frame pairs (temporal inconsistency)
            face_pairs = process_video_pairs(video_path)
            f4 = compute_f4(face_pairs)

            video_id = os.path.splitext(fname)[0]
            rows.append((video_id, f1, f2, f3, f4, f5, label))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "f1", "f2", "f3", "f4", "f5", "label"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}  ({skipped} videos skipped)")


# ---------------------------------------------------------------------------
# Sanity check — run with: python src/features.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = "data"
    OUT_CSV  = "data/features.csv"

    process_all_videos(DATA_DIR, OUT_CSV)

    # Print feature range summary
    import csv as _csv
    with open(OUT_CSV) as f:
        reader = _csv.DictReader(f)
        data = list(reader)

    if not data:
        print("ERROR: features.csv is empty")
        sys.exit(1)

    for feat in ("f1", "f2", "f3", "f4", "f5"):
        vals = [float(row[feat]) for row in data]
        real_vals = [float(row[feat]) for row in data if row["label"] == "0"]
        fake_vals = [float(row[feat]) for row in data if row["label"] == "1"]
        print(f"\n{feat}:")
        print(f"  overall  min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}")
        if real_vals:
            print(f"  real     mean={sum(real_vals)/len(real_vals):.2f}")
        if fake_vals:
            print(f"  deepfake mean={sum(fake_vals)/len(fake_vals):.2f}")
