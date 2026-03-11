"""
demo/generate_cache.py — Run once locally to pre-bake all frames + predictions.

Outputs:
  demo/cache/frames/<video_id>.jpg   — one representative frame per video
  demo/cache/predictions.json        — {video_id: {label, model_guess, p_fake}}

Run from project root:
  .venv/bin/python demo/generate_cache.py
"""

import io
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_curve

ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
CACHE_DIR    = os.path.join(ROOT, 'cache', 'frames')
OUT_JSON     = os.path.join(ROOT, 'cache', 'predictions.json')

PARAMS_JSON  = os.path.join(PROJECT_ROOT, 'models', 'params.json')
FEATURES_CSV = os.path.join(PROJECT_ROOT, 'data', 'features_p5_tighter_crop.csv')
REAL_DIR     = os.path.join(PROJECT_ROOT, 'data', 'real')
FAKE_DIR     = os.path.join(PROJECT_ROOT, 'data', 'deepfake')

FEATURE_SUBSET         = ('f5',)
LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Model helpers (same as app.py) ─────────────────────────────────────────

def load_features_csv(path):
    rows = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            vid = parts[0]
            f1, f2, f3, f4, f5 = (float(parts[i]) for i in range(1, 6))
            label = int(parts[6])
            rows[vid] = {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, 'label': label}
    return rows

def calc_scotts(data):
    n = len(data)
    h = math.pow(n, -1/5) * np.std(data)
    return h, data

def build_kdes(params):
    kdes = {}
    for cls in ('0', '1'):
        kdes[cls] = {}
        for feat, vals in params['kde_training_data'][cls].items():
            kdes[cls][feat] = calc_scotts(np.array(vals, dtype=float))
    return kdes

def softmax2(a, b):
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

def compute_log_density(kde, x):
    h, samples = kde
    terms = -0.5 * ((x - samples) / h) ** 2
    return logsumexp(terms) - math.log(len(samples) * h * math.sqrt(2 * math.pi))

def classify_one(raw_feats, params, kdes, features):
    log_tf = set(params['log_transform_features'])
    vals = {
        feat: float(np.log1p(raw_feats[feat])) if feat in log_tf else raw_feats[feat]
        for feat in raw_feats
    }
    log_scores = {}
    for cls in ('0', '1'):
        s = params['log_priors'][cls]
        for feat in features:
            s += compute_log_density(kdes[cls][feat], vals[feat])
        log_scores[cls] = s
    _, p_fake = softmax2(log_scores['0'], log_scores['1'])
    return p_fake

def find_threshold(params, all_features, kdes, features):
    train_ids = set(params['train_ids'])
    y_true, y_scores = [], []
    for vid in train_ids:
        if vid not in all_features:
            continue
        row = all_features[vid]
        raw = {k: row[k] for k in ('f1', 'f2', 'f3', 'f4', 'f5')}
        y_scores.append(classify_one(raw, params, kdes, features))
        y_true.append(row['label'])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j = np.array(tpr) - np.array(fpr)
    return float(thresholds[np.argmax(j)])

# ── Frame extraction ────────────────────────────────────────────────────────

def get_video_path(video_id, label):
    folder = FAKE_DIR if label == 1 else REAL_DIR
    path = os.path.join(folder, f'{video_id}.mp4')
    if os.path.exists(path):
        return path
    for fname in os.listdir(folder):
        if fname.startswith(video_id) and fname.endswith('.mp4'):
            return os.path.join(folder, fname)
    return None

def extract_frame_jpeg(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 30
    lo, hi = total // 3, 2 * total // 3
    frame_index = random.randint(lo, max(lo, hi))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_index, total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        frame = np.full((360, 480, 3), 128, dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = 640 / w
    frame = cv2.resize(frame, (640, int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

# ── Main ────────────────────────────────────────────────────────────────────

print("Loading model...", end=' ', flush=True)
params       = json.load(open(PARAMS_JSON))
all_features = load_features_csv(FEATURES_CSV)
kdes         = build_kdes(params)
threshold    = find_threshold(params, all_features, kdes, FEATURE_SUBSET)
print(f"done. Threshold = {threshold:.3f}")

predictions = {}
video_ids   = sorted(all_features.keys())
total       = len(video_ids)

for i, vid in enumerate(video_ids, 1):
    row   = all_features[vid]
    label = row['label']
    raw   = {k: row[k] for k in ('f1', 'f2', 'f3', 'f4', 'f5')}
    p_fake      = classify_one(raw, params, kdes, FEATURE_SUBSET)
    model_guess = 1 if p_fake >= threshold else 0

    predictions[vid] = {
        'label':       label,
        'model_guess': model_guess,
        'p_fake':      round(p_fake, 4),
    }

    # Extract and save frame
    frame_path = os.path.join(CACHE_DIR, f'{vid}.jpg')
    if not os.path.exists(frame_path):
        path = get_video_path(vid, label)
        if path:
            jpeg = extract_frame_jpeg(path)
            with open(frame_path, 'wb') as f:
                f.write(jpeg)
        else:
            print(f"  [WARN] video not found: {vid}")

    print(f"\r  {i}/{total} videos processed", end='', flush=True)

print()

with open(OUT_JSON, 'w') as f:
    json.dump({'threshold': threshold, 'videos': predictions}, f)

print(f"Done. Cache written to demo/cache/  ({total} videos)")
