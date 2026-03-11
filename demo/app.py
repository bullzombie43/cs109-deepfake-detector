"""
demo/app.py — Deepfake Detection Game

Human vs. KDE-NB model. Both guess each video; first wrong loses.
Run: ../../.venv/bin/python app.py  (from demo/) or
     .venv/bin/python demo/app.py   (from project root)
"""

import io
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, send_file, session
from scipy.special import logsumexp
from sklearn.metrics import roc_curve

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

PARAMS_JSON  = os.path.join(PROJECT_ROOT, 'models', 'params.json')
FEATURES_CSV = os.path.join(PROJECT_ROOT, 'data', 'features_p5_tighter_crop.csv')
REAL_DIR     = os.path.join(PROJECT_ROOT, 'data', 'real')
FAKE_DIR     = os.path.join(PROJECT_ROOT, 'data', 'deepfake')

FEATURE_SUBSET = ('f5',)
LOG_TRANSFORM_FEATURES = {'f1', 'f2', 'f3', 'f5'}

app = Flask(__name__)
app.secret_key = 'deepfake-demo-2026'

# ── Model loading (done once at startup) ───────────────────────────────────

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
    """Youden's J on training set."""
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


print("Loading model...", end=' ', flush=True)
_params = json.load(open(PARAMS_JSON))
_all_features = load_features_csv(FEATURES_CSV)
_kdes = build_kdes(_params)
_threshold = find_threshold(_params, _all_features, _kdes, FEATURE_SUBSET)
print(f"done. Threshold = {_threshold:.3f}")

# Sorted list of all video IDs for the game
ALL_VIDEO_IDS = sorted(_all_features.keys())
random.shuffle(ALL_VIDEO_IDS)  # randomise once at startup


# ── Frame extraction ────────────────────────────────────────────────────────

def get_video_path(video_id, label):
    folder = FAKE_DIR if label == 1 else REAL_DIR
    # Try exact match then glob
    path = os.path.join(folder, f'{video_id}.mp4')
    if os.path.exists(path):
        return path
    # Fallback: find any file starting with video_id
    for fname in os.listdir(folder):
        if fname.startswith(video_id) and fname.endswith('.mp4'):
            return os.path.join(folder, fname)
    return None


def extract_frame_jpeg(video_path, frame_index=None):
    """Extract a single frame from video, return JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 30
    if frame_index is None:
        # Pick a frame from the middle third (avoid very first/last frames)
        lo = total // 3
        hi = 2 * total // 3
        frame_index = random.randint(lo, max(lo, hi))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_index, total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        # Return a grey placeholder
        frame = np.full((360, 480, 3), 128, dtype=np.uint8)
    # Resize to consistent width
    h, w = frame.shape[:2]
    target_w = 640
    scale = target_w / w
    frame = cv2.resize(frame, (target_w, int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    session.clear()
    session['queue'] = ALL_VIDEO_IDS[:]
    random.shuffle(session['queue'])
    session['human_score'] = 0
    session['model_score'] = 0
    session['round'] = 0
    return render_template('index.html')


@app.route('/api/next')
def api_next():
    queue = session.get('queue', [])
    if not queue:
        return jsonify({'done': True, 'reason': 'No more videos.'})

    video_id = queue.pop()
    session['queue'] = queue
    session['current_video'] = video_id
    session['round'] = session.get('round', 0) + 1

    row = _all_features[video_id]
    label = row['label']
    raw = {k: row[k] for k in ('f1', 'f2', 'f3', 'f4', 'f5')}
    p_fake = classify_one(raw, _params, _kdes, FEATURE_SUBSET)
    model_guess = 1 if p_fake >= _threshold else 0

    # Store for reveal
    session['truth'] = label
    session['model_guess'] = model_guess
    session['p_fake'] = round(p_fake, 4)

    return jsonify({
        'done': False,
        'video_id': video_id,
        'round': session['round'],
        'human_score': session['human_score'],
        'model_score': session['model_score'],
    })


@app.route('/api/frame/<video_id>')
def api_frame(video_id):
    if video_id not in _all_features:
        return 'Not found', 404
    label = _all_features[video_id]['label']
    path = get_video_path(video_id, label)
    if path is None:
        placeholder = np.full((360, 480, 3), 60, dtype=np.uint8)
        cv2.putText(placeholder, 'Video not found', (80, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')
    jpeg = extract_frame_jpeg(path)
    return send_file(io.BytesIO(jpeg), mimetype='image/jpeg')


@app.route('/api/guess/<int:human_guess>')
def api_guess(human_guess):
    truth      = session.get('truth')
    model_guess = session.get('model_guess')
    p_fake     = session.get('p_fake', 0.0)
    video_id   = session.get('current_video', '')

    if truth is None:
        return jsonify({'error': 'No active round'}), 400

    human_correct = int(human_guess == truth)
    model_correct = int(model_guess == truth)

    if human_correct:
        session['human_score'] = session.get('human_score', 0) + 1
    if model_correct:
        session['model_score'] = session.get('model_score', 0) + 1

    # Clear truth so it can't be resubmitted
    session.pop('truth', None)

    return jsonify({
        'truth': truth,
        'model_guess': model_guess,
        'p_fake': p_fake,
        'human_correct': human_correct,
        'model_correct': model_correct,
        'human_score': session['human_score'],
        'model_score': session['model_score'],
        'video_id': video_id,
    })


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"Starting demo server on http://127.0.0.1:{port}")
    app.run(debug=False, port=port)
