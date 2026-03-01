# Deepfake Detector — Progress Log

---

## V1 — Baseline Pipeline (2026-02-28)

### What We Do
- **Segmentation:** OpenCV Haar Cascades, 10 frames sampled uniformly per video
- **Face/background:** Face = largest bounding box crop; background = full frame (all pixels)
- **Features:**
  - `f1`: DCT chi-squared divergence between face and background (50 bins, raw counts)
  - `f2`: Color histogram chi-squared summed across R, G, B (256 bins, raw counts)
  - `f3`: Absolute difference of mean absolute DCT coefficients
- **Aggregation:** Per-video mean across sampled frames (Approach B)
- **Training:** MLE Gaussian Naive Bayes, 80/20 stratified video-level split, ε=1e-6
- **Classification:** Log-space Naive Bayes + softmax, threshold=0.5

### Test Results (40 videos)
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.550 (22/40) |
| Precision | 0.625 |
| Recall    | 0.250 |
| F1        | 0.357 |
| ROC AUC   | 0.535 |

**Confusion Matrix (rows=true, cols=predicted):**
```
              Pred Real   Pred Fake
True Real         17          3
True Fake         15          5
```

### Issues Identified
- **Root cause of near-random performance:** `f1` and `f2` use raw pixel counts in histograms.
  The background is the full frame (~1M pixels) while the face crop is much smaller (~10K pixels).
  This pixel-count imbalance inflates the chi-squared statistic regardless of actual content,
  drowning out any real signal between real and fake classes.
- Class means for f1/f2 are nearly identical across real and fake (~565K vs ~572K for f1),
  confirming these features carry almost no discriminative information in their current form.
- f3 is unaffected (operates on means, not histograms) but alone is insufficient.

### Planned Fix → V2
Normalize `f1` and `f2` histograms to sum to 1 (probability distributions) before computing
chi-squared. This puts face and background on the same probability scale regardless of their
respective pixel counts, isolating shape/frequency differences rather than size differences.

---

## V2 — Normalized Histograms (2026-02-28)

### Changes from V1
- **f1 and f2:** Histograms now normalized to sum to 1 (probability distributions)
  before computing chi-squared, eliminating the pixel-count imbalance between
  the large background frame (~1M pixels) and small face crop (~10K pixels).

### Test Results (40 videos)
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.475 (19/40) |
| Precision | 0.444 |
| Recall    | 0.200 |
| F1        | 0.276 |
| ROC AUC   | 0.580 |

**Confusion Matrix (rows=true, cols=predicted):**
```
              Pred Real   Pred Fake
True Real         15          5
True Fake         16          4
```

### Analysis
Normalization fixed the scale issue but did not improve accuracy — it actually dropped
slightly below V1. The AUC improved (0.535 → 0.580), suggesting the scores are better
calibrated, but the classifier is still barely above chance.

The feature means are still nearly identical between classes:
- f1: real=0.0013, fake=0.0015 (both near zero; very little frequency-domain divergence)
- f2: real=12.41, fake=12.26 (overlapping distributions)
- f3: real=1.80, fake=1.81 (same as before)

**Root cause revisited:** The problem may not be the histogram scale — it may be that
comparing the face to the *full frame* (which includes the face itself) dilutes the
background signal. A cleaner comparison would use a background patch that *excludes*
the face region.

### Planned Fix → V3
Extract the background as the full frame with the face region masked/excluded,
so the chi-squared measures face vs. true background rather than face vs. (face + background).

---

## V3 — Same-Sized Background Patch (2026-02-28)

### Changes from V2
- **Background extraction:** Replaced the full masked frame with a same-sized patch
  cropped from the nearest non-overlapping region (right → left → below → above → top-left fallback).
  This eliminates DCT ringing from zero-filled rectangles and histogram skew toward black pixels.

### Test Results (40 videos)
| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.525 (21/40) |
| Precision | 1.000 |
| Recall    | 0.050 |
| F1        | 0.095 |
| ROC AUC   | 0.365 |

**Confusion Matrix (rows=true, cols=predicted):**
```
              Pred Real   Pred Fake
True Real         20          0
True Fake         19          1
```

### Analysis
The background patch approach revealed a real signal in f1: fake class mean (0.039)
is now ~2x the real class mean (0.020), suggesting DCT artifacts are present.
However, the model almost always predicts "real" (catches only 1/20 fakes).

Key issues:
- **f1 fake sigma is 3x higher than real sigma** (0.155 vs 0.048). The Gaussian NB
  likelihood formula penalizes high-variance distributions via the -log(σ) term,
  making the wide fake distribution score lower than the tight real distribution for
  most values — even though fake mean is higher.
- **f2 and f3 are essentially identical across classes** (real=41.31 vs fake=41.24
  for f2; real=2.97 vs fake=3.00 for f3), contributing noise that drowns out f1's signal.
- **AUC of 0.365 < 0.5** means the classifier is effectively inverting predictions —
  the model's P(fake) scores are negatively correlated with true labels.

The feature separation is finally present in f1, but Gaussian NB is the wrong model
for a distribution with such unequal variances between classes.

### Planned Fix → V3.5 (diagnostic)
Investigate each feature in isolation to identify which carries signal.

---

## V3.5 — Feature Isolation Diagnostic (2026-02-28)

### Changes from V3
- No code changes. Added a `features_subset` parameter to `evaluate()` in `classify.py`
  and re-ran with each feature independently.

### Test Results — Feature Isolation (40 videos)
| Features | Accuracy | Precision | Recall | F1    | AUC   | Behavior |
|----------|----------|-----------|--------|-------|-------|----------|
| f1 only  | 0.525    | 1.000     | 0.050  | 0.095 | 0.380 | Almost always predicts real |
| f2 only  | 0.500    | 0.500     | 0.900  | 0.643 | 0.417 | Almost always predicts fake |
| f3 only  | 0.500    | 0.500     | 0.600  | 0.545 | 0.525 | Most balanced, closest to chance |
| All      | 0.525    | 1.000     | 0.050  | 0.095 | 0.365 | f1 dominates, drags result down |

### Analysis
Every feature produces AUC ≤ 0.5, meaning the model scores *backwards* for all of them.
This is not a feature quality problem — it is a **variance asymmetry** issue in Gaussian NB.

The log-likelihood formula includes a `-log(σ)` term. Because fake distributions have
significantly higher variance than real distributions (e.g., f1: fake σ=0.155, real σ=0.048),
the real class receives a constant likelihood bonus just for being tighter, regardless of
where the observed value falls. The model is penalizing the fake class for having higher variance.

**The signal is present but inverted.** Flipping the AUCs gives: f1=0.620, f2=0.583,
f3=0.525 — all above chance. The classifier is correctly ordering the evidence, just labeling
it backwards due to the σ asymmetry.

### Planned Fix → V4
Log-transform f1 and f2 before fitting and classifying. Chi-squared statistics follow a
right-skewed distribution that becomes approximately Gaussian after log-transformation,
collapsing the variance asymmetry and allowing the model to score in the correct direction.

---

## V4 — Log-Transform f1 and f2 (2026-02-28)

### Changes from V3.5
- f1 and f2 are now log1p-transformed before MLE fitting in `train.py`.
- `classify.py` applies the same log1p transform at inference time (flag stored in params.json).

### Test Results (40 videos)
| Features | Accuracy | Precision | Recall | F1    | AUC   |
|----------|----------|-----------|--------|-------|-------|
| All      | 0.500    | 0.500     | 0.050  | 0.091 | 0.353 |
| f1 only  | 0.500    | 0.500     | 0.050  | 0.091 | 0.410 |
| f2 only  | 0.425    | 0.440     | 0.550  | 0.489 | 0.445 |
| f3 only  | 0.500    | 0.500     | 0.600  | 0.545 | 0.525 |

### Analysis
The log transform did not improve results. The root cause is that f1 values after the
background patch fix are already very small (0.00–1.35, mean ~0.03), so log1p(x) ≈ x
for small x — the transform has no meaningful effect at this scale.

More critically, **all three features show nearly identical class distributions**:
- f2 (log): real μ=3.236 σ=0.993 vs fake μ=3.279 σ=0.971 — almost no separation
- f3: real μ=2.97 σ=1.84 vs fake μ=3.00 σ=1.80 — essentially the same distribution

**Root cause reassessment:** FaceSwap pastes a face that has been geometrically warped
and color-corrected to match the target. The result naturally looks similar to the
background in color and texture — which is exactly what our features measure. The face
is *designed* to blend in, so face-vs-background chi-squared and DCT comparisons
have little discriminative power for this manipulation type.

The remaining signal (f3 AUC=0.525) is marginal and consistent with chance.

### Planned Fix → V5
Add f4: temporal face inconsistency measured from consecutive frame pairs.

---

## V5 — f4: Temporal Inconsistency Feature (2026-02-28)

### Changes from V4
- **New feature f4:** Mean frame-to-frame absolute pixel difference of normalized
  grayscale face crops, computed from N consecutive frame pairs sampled across the video.
- `segmentation.py`: added `sample_frame_pairs()` and `process_video_pairs()` — each
  sample an anchor frame and its immediate successor (1 frame apart, ~33ms).
- `features.py`: added `compute_f4(face_pairs)` — resizes face crops to 64×64, converts
  to grayscale, normalizes by mean intensity, computes MAD across consecutive pairs.
- `train.py` and `classify.py`: updated to include f4 column throughout.

### Trained Feature Values
| Feature | Real μ | Real σ | Fake μ | Fake σ |
|---------|--------|--------|--------|--------|
| f4      | 0.0348 | 0.0143 | 0.0351 | 0.0135 |

### Test Results (40 videos)
| Features   | Accuracy | Precision | Recall | F1    | AUC   |
|------------|----------|-----------|--------|-------|-------|
| All (f1–f4)| 0.500    | 0.500     | 0.050  | 0.091 | 0.380 |
| f4 only    | 0.550    | 0.545     | 0.600  | 0.571 | 0.485 |
| f1 + f4    | 0.500    | 0.500     | 0.050  | 0.091 | 0.415 |

### Analysis
f4 alone is our best single-feature result so far (accuracy 55%, F1 0.571, most balanced
confusion matrix yet), but the class means are nearly identical: real μ=0.0348 vs fake
μ=0.0351 — a difference of only 0.9%. The feature has marginal signal, not enough to push
AUC above 0.5 reliably.

Adding f4 to other features doesn't help — the variance asymmetry problem in f1 dominates
and drags everything down.

**Root cause:** The frame-to-frame difference is dominated by natural head motion and
lighting flicker, not warp jitter. At c23 compression, inter-frame codec motion
compensation may also suppress the per-frame warp error we were hoping to detect.

### Conclusion
We have now exhausted improvements on c23 data with the current feature set. All features
show near-zero class separation. The compression codec appears to be suppressing the
manipulation artifacts that the features are designed to detect.

### Planned Fix → V6
Switch to c0 (uncompressed) videos and re-run the full pipeline. Expected improvements:
- f2: color boundary artifacts at face edges no longer flattened by codec
- f4: per-frame warp jitter not suppressed by inter-frame motion compensation
- f1: double-compression artifacts no longer masked by uniform re-encoding

---
