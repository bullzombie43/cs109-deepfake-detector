# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Reference Document

**When confused or unsure about any design decision, consult `deepfake_detection_prd.md`.** It defines the complete system architecture, feature formulas, mathematical methods, dataset requirements, and success criteria for this project.

## Project Overview

A Naive Bayes deepfake detection system for CS 109. The pipeline: video → face segmentation → feature extraction → MLE training → classification. Target: 70–85% accuracy on low-quality face-swap deepfakes (Viggle AI, Reface, FaceSwap).

## Technology Stack

- Python 3.8+
- OpenCV — face detection, video I/O
- NumPy — numerical computation
- SciPy — 2D DCT (`scipy.fftpack.dct`)
- Scikit-learn — train/test split, metrics
- Matplotlib — visualization

## Code Structure

```
deepfake_detector/
├── data/                  # Raw videos and extracted feature CSVs
├── models/                # Saved model parameters (JSON)
├── src/
│   ├── segmentation.py    # Face detection + frame sampling
│   ├── features.py        # f1 (DCT chi-sq), f2 (color chi-sq), f3 (mean DCT diff)
│   ├── train.py           # MLE: per-class μ, σ estimation + prior estimation
│   ├── classify.py        # Naive Bayes: log-likelihood + softmax → label + confidence
│   └── utils.py           # Shared helpers
├── notebooks/             # Jupyter analysis notebooks
└── results/               # Plots, confusion matrices, metrics
```

## Key Architecture Decisions (per PRD)

**Segmentation:** OpenCV Haar Cascades (start here; upgrade to dlib if accuracy insufficient). Sample **10 frames per video** uniformly distributed. For multi-face frames, use the largest bounding box. Skip frames with no detected face.

**Features** — three features per video (see PRD §3.2 for full formulas):
- `f1`: DCT chi-squared divergence between face and background (50 histogram bins)
- `f2`: Color histogram chi-squared summed across R, G, B channels (256 bins each)
- `f3`: Absolute difference of mean absolute DCT coefficients

**Aggregation:** Per-video (Approach B) — compute features per frame, then aggregate with mean/median across the sampled frames before classifying. This preserves the i.i.d. assumption.

**Training:** MLE — compute per-class (Deepfake / Real) mean and std for each feature. Apply Laplace smoothing `ε = 1e-6` to σ. Estimate priors from class frequencies.

**Model persistence:** Save parameters as JSON to `models/`. Schema defined in PRD §4.5.

**Classification:** Log-space Naive Bayes with Gaussian likelihoods, then softmax to produce `P(Deepfake | f)`. Threshold at 0.5.

**Data split:** Video-level 80/20 train/test split (never split frames from the same video across train and test).

## Feature CSV Format

```
video_id, f1, f2, f3, label   # label: 0=real, 1=deepfake
```

## Running the Pipeline

Install dependencies:
```bash
pip install opencv-python numpy scipy scikit-learn matplotlib
```

Run individual stages (adapt paths as implemented):
```bash
python src/segmentation.py        # Extract frames + face regions
python src/features.py            # Extract features → data/features.csv
python src/train.py               # Fit model → models/params.json
python src/classify.py            # Evaluate on test set → results/
```

Run a notebook for analysis:
```bash
jupyter notebook notebooks/
```

## Git & Commit Standards

- Do **not** add `Co-Authored-By: Claude` or any Claude authorship to commit messages.

## Writeup Submodule

The writeup lives in `writeup/` as a git submodule (remote: `https://github.com/bullzombie43/CS-109-Challenge-Writeup`).

After making any changes to `writeup/main.tex` or other writeup files, always:
1. Commit inside the submodule: `cd writeup && git add <files> && git commit -m "..."`
2. Push the submodule: `git push origin main` (from inside `writeup/`)
3. Update the parent repo's submodule pointer: `cd .. && git add writeup && git commit -m "Update writeup submodule" && git push origin master`

## Evaluation Targets

- Primary: Accuracy, Precision, Recall, F1
- Secondary: Confusion matrix; ROC/AUC if time permits
- Baselines to compare against: random (≈50%), majority-class, single-feature (f1 only)
- MVP threshold: >60% accuracy; target: 70–85%
