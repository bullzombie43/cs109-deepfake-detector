# Deepfake Detection — Project Phases & Todos

## Context
The project is fully spec'd (PRD + CLAUDE.md) but has zero implementation. We need to build the complete pipeline from scratch: dataset → segmentation → feature extraction → MLE training → Naive Bayes classification → evaluation. Goal: 70–85% accuracy on low-quality face-swap deepfakes for CS 109.

---

## Phase 1: Project Setup & Dataset Collection
*Get the environment ready and gather enough video data to train and test.*

**Todos:**
- [x] Create directory structure: `src/`, `data/real/`, `data/deepfake/`, `models/`, `notebooks/`, `results/`
- [x] Install dependencies: `pip install opencv-python numpy scipy scikit-learn matplotlib`
- [x] Collect **real videos** (≥50, target 100–200): 100 real videos via FaceForensics++ original YouTube sequences (c23)
- [x] Collect **deepfake videos** (≥50, target 100–200): 100 FaceSwap deepfakes via FaceForensics++ (c23)
- [x] Verify videos: FF++ is a curated research dataset — face visibility guaranteed by design
- [x] Document video sources in a `data/README.md` for reproducibility

**Done when:** ≥100 total videos (≥50 per class) are in `data/real/` and `data/deepfake/`.
---

## Phase 2: Segmentation (`src/segmentation.py`)
*Extract face and background regions from each video.*

**Todos:**
- [x] Implement `sample_frames(video_path, n=10)` — uniform frame sampling across video duration
- [x] Implement face detection using OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`)
- [x] Handle edge cases:
  - No face detected → skip frame
  - Multiple faces → use largest bounding box
  - Face too small (<20×20 px) → skip frame
- [x] Implement `extract_regions(frame, bbox)` → returns `(face_array, background_array)`
  - Background = full frame with face region masked/excluded
- [x] Implement `process_video(video_path)` → returns list of `(face, background)` pairs (up to 10)
- [x] Write a quick test: run on 2–3 videos, visualize extracted faces with `cv2.imshow` or matplotlib

**Key files:** `src/segmentation.py`
**Done when:** Can run `python src/segmentation.py` on a real and a deepfake video and see valid face crops.

---

## Phase 3: Feature Extraction (`src/features.py`)
*Compute f1, f2, f3 per frame, then aggregate per video.*

**Todos:**
- [x] Implement `compute_f1(face, background)` — DCT chi-squared divergence
- [x] Implement `compute_f2(face, background)` — color histogram chi-squared
- [x] Implement `compute_f3(face, background)` — mean DCT difference
- [x] Implement `extract_features(face_bg_pairs)` → per-frame feature list `[(f1, f2, f3), ...]`
- [x] Implement `aggregate_features(frame_features)` → single `(f1, f2, f3)` via mean across frames
- [x] Implement `process_all_videos(data_dir)` → writes `data/features.csv`
- [x] Sanity-check feature ranges after collecting data

**Key files:** `src/features.py`, `src/utils.py` (shared DCT/histogram helpers)
**Done when:** `data/features.csv` exists with one row per video, values in expected ranges.

---

## Phase 4: Training (`src/train.py`)
*Fit the Naive Bayes model via MLE.*

**Todos:**
- [ ] Load `data/features.csv`
- [ ] Implement video-level 80/20 train/test split (stratified)
- [ ] Save train/test split to `data/split.json`
- [ ] Implement MLE: per-class μ, σ + Laplace smoothing ε=1e-6
- [ ] Save model to `models/params.json`
- [ ] Print training summary

**Key files:** `src/train.py`
**Done when:** `models/params.json` exists with plausible parameters.

---

## Phase 5: Classification & Evaluation (`src/classify.py`)
*Run Naive Bayes on the test set and measure performance.*

**Todos:**
- [ ] Implement `load_model(path)` → loads `models/params.json`
- [ ] Implement `predict_one(features, model)` — log-space Gaussian NB + softmax
- [ ] Implement `evaluate(X_test, y_test, model)` → accuracy, precision, recall, F1
- [ ] Generate and save confusion matrix to `results/confusion_matrix.png`
- [ ] Baseline comparisons: random, majority-class, f1-only

**Key files:** `src/classify.py`
**Done when:** Test accuracy is reported, confusion matrix saved, baselines compared.

---

## Phase 6: Analysis & Reporting (`notebooks/`)
*Visualize results and write the CS 109 report.*

**Todos:**
- [ ] Create `notebooks/analysis.ipynb` with feature distributions, scatter plots, confusion matrix, ROC curve
- [ ] Write 3-page CS 109 report

**Done when:** Analysis notebook runs end-to-end without errors.

---

## Critical Files
| File | Purpose |
|------|---------|
| `deepfake_detection_prd.md` | Source of truth for all formulas and design decisions |
| `src/segmentation.py` | Frame sampling + face/background extraction |
| `src/features.py` | f1, f2, f3 computation + CSV output |
| `src/utils.py` | Shared DCT, histogram helpers |
| `src/train.py` | MLE fitting + model persistence |
| `src/classify.py` | Naive Bayes inference + evaluation |
| `data/features.csv` | Feature matrix (one row per video) |
| `models/params.json` | Trained model parameters |

## End-to-End Verification
```bash
# Run full pipeline
python src/segmentation.py        # verify face crops look correct
python src/features.py            # check features.csv + value ranges
python src/train.py               # check models/params.json
python src/classify.py            # check accuracy > 60% (MVP)
jupyter notebook notebooks/analysis.ipynb  # verify all plots render
```
