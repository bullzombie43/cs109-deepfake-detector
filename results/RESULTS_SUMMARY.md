# Deepfake Detector — Results Summary

All experiments use 80/20 stratified video-level train/test split, `RANDOM_STATE=42`.
Threshold optimised via Youden's J on train set (from V9 onward). Earlier versions used fixed threshold=0.5.

---

## 1. Version Progression (development arc)

### Dataset: c23 compressed (100 videos, test n=20 → 40)

| Version | Key Change | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|
| V1 | Baseline: f1+f2+f3, raw histograms, Gaussian NB | 0.550 | 0.625 | 0.250 | 0.357 | 0.535 |
| V2 | Normalized histograms (f1, f2) | 0.475 | 0.444 | 0.200 | 0.276 | 0.580 |
| V3 | Same-sized background patch | 0.525 | 1.000 | 0.050 | 0.095 | 0.365 |
| V3.5 | Feature isolation diagnostic — f3 only | 0.500 | 0.500 | 0.600 | 0.545 | 0.525 |
| V4 | log1p transform on f1, f2 | 0.500 | 0.500 | 0.050 | 0.091 | 0.353 |
| V5 | + f4 temporal MAD | 0.550 | 0.545 | 0.600 | 0.571 | 0.485 |

**Conclusion on c23:** Codec compression suppresses all manipulation artifacts. Switched to c0.

---

### Dataset: c0 uncompressed (200 videos, test n=40)

| Version | Key Change | Acc | AUC | Notes |
|---|---|---|---|---|
| V6 | Switch to c0 data | 0.525 | 0.565 | Immediate AUC gain |
| V7 | Replace f2 with block-DCT β-statistic | 0.625 | 0.600 | Double-encoding artifact |
| V8 | Add f5: Laplacian variance ratio (face/bg) | 0.625 | 0.607 | f5 targets sharpening artifacts |
| V9 | P5 tighter bbox crop + KDE likelihoods + Youden's J threshold | 0.675 | 0.678 | Biggest single jump |

---

### Dataset: c0 uncompressed (400 videos, test n=80) — current

| Version | Key Change | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|
| **V10** | **2× dataset (200 real + 200 fake)** | **0.688** | **0.741** | **0.575** | **0.648** | **0.683** |

> V10 uses KDE-NB (Scott bandwidth), f5 only, threshold=0.513.

---

## 2. V10 Feature Subset Comparison (KDE-NB, 400 videos, test n=80)

| Features | Acc | Prec | Rec | F1 | AUC | Threshold |
|---|---|---|---|---|---|---|
| **f5 only** | **0.688** | **0.741** | **0.575** | **0.648** | **0.683** | 0.513 |
| f2 + f5 | 0.650 | 0.643 | 0.675 | 0.659 | 0.675 | 0.506 |
| f2 + f3 + f5 | 0.638 | 0.628 | 0.675 | 0.651 | 0.670 | 0.496 |
| f1+f2+f3+f4+f5 | 0.638 | 0.628 | 0.675 | 0.651 | 0.670 | 0.495 |

**Finding:** Adding features degrades performance. NB independence assumption violated at scale — f1/f3/f4 add noise.

---

## 3. Alternative Method Experiments (400 videos, test n=80)

All compared against V10 baseline: **68.8% acc / 0.683 AUC (f5 only, KDE-NB)**.

| Experiment | Method | Features | Acc | AUC | Verdict |
|---|---|---|---|---|---|
| exp/map-nb | MAP-NB (NIG prior, Gaussian) | f5 only | 0.588 | 0.680 | Did not beat KDE |
| exp/map-nb | MAP-NB (NIG prior, Gaussian) | all-5 | 0.600 | 0.666 | Did not beat KDE |
| exp/gamma-nb | Gamma-NB (MLE, scipy.stats.gamma) | f5 only | 0.588 | 0.680 | Did not beat KDE |
| exp/gamma-nb | Gamma-NB (MLE, scipy.stats.gamma) | all-5 | 0.588 | 0.668 | Did not beat KDE |

**Finding:** All parametric families (Gaussian, Gamma) produce identical AUC (~0.680) for f5-only.
KDE's non-parametric form captures distribution shape that no parametric family models well.
The accuracy gap (10pp) reflects worse probability calibration in parametric approaches.

---

## 4. Failed Feature / Architecture Experiments (200 videos, test n=40)

| Experiment | Approach | Acc | AUC | Notes |
|---|---|---|---|---|
| exp/p1-pca | PCA-whitened Gaussian NB | 0.625 | 0.623 | Below KDE |
| exp/p2-frame-vote | Frame-level majority vote NB | 0.525 | 0.548 | Worse than per-video aggregation |
| exp/p4-block-boundary | Block boundary ratio feature (f6) | ~0.525 | ~0.330 | Inverted signal; hurts accuracy |
| — | LBP chi-squared feature | — | 0.410 | Tested inline; no dedicated branch |
| — | Noise level ratio feature | — | 0.540 | Tested inline; no dedicated branch |

---

## 5. Baselines

| Baseline | Acc | AUC |
|---|---|---|
| Random (50/50) | ~0.500 | ~0.500 |
| Majority class | 0.500 | 0.500 |
| Single feature f1 only (V3.5, c23) | 0.525 | 0.380 |

---

## Summary: Best Result per Method Family

| Method Family | Best Acc | Best AUC | Config |
|---|---|---|---|
| **KDE-NB (current best)** | **0.688** | **0.683** | f5 only, Scott BW, 400 videos |
| MAP-NB (Gaussian + NIG prior) | 0.600 | 0.680 | all-5, 400 videos |
| Gamma-NB | 0.588 | 0.680 | f5 only, 400 videos |
| PCA-whitened Gaussian NB | 0.625 | 0.623 | all features, 200 videos |
| Frame-vote NB | 0.525 | 0.548 | all features, 200 videos |
