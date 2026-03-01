# Deepfake Detection System

Product Requirements Document

## Executive Summary

This document outlines the requirements for building a deepfake detection system using probabilistic methods from CS 109. The system will classify videos as either authentic or deepfaked by analyzing statistical differences between face regions and background regions.

**Primary Goal:** Detect low-quality deepfake videos (e.g., face-swapped viral videos created with tools like Viggle AI, Reface, FaceSwap) using Naive Bayes classification.

**Success Criteria:** Achieve 70-85% accuracy on test set, with clear understanding and documentation of the probabilistic methods used.

## 1. System Overview

### 1.1 High-Level Architecture

The system consists of four main stages:

- Region Segmentation: Divide video frames into face region and background region
- Feature Extraction: Extract statistical features from both regions
- Training (MLE): Estimate probability distribution parameters from training data
- Classification (Naive Bayes): Compute posterior probabilities and make classification decision

### 1.2 System Pipeline Diagram

Input Video → Region Segmentation → Feature Extraction → Naive Bayes Classifier → Output (Deepfake/Real + Confidence)

## 2. Stage 1: Region Segmentation

### 2.1 Objective

Divide each video frame into two regions: face region and background region. This segmentation is critical since all features measure the statistical differences between these regions.

### 2.2 Input

- Video file (MP4, AVI, or similar format)
- Individual frames extracted from video

### 2.3 Output

- Face bounding box coordinates (x, y, width, height)
- Face region pixel array
- Background region pixel array (remaining frame excluding face)

### 2.4 Implementation Approach

Use existing face detection library (not building from scratch):

- Option 1: OpenCV Haar Cascades - Fast, lightweight, pre-trained
- Option 2: dlib - More accurate, includes facial landmarks
- Option 3: MTCNN - Deep learning based, most accurate but slower

Recommended: Start with OpenCV (easiest), upgrade to dlib if needed.

### 2.5 Frame Sampling Strategy

Challenge: Videos contain many frames (30 fps = 1800 frames per minute)

Solution: Sample frames rather than processing every single frame

- Approach A: Uniform sampling - Extract every Nth frame (e.g., N=5 for 6 fps)
- Approach B: Fixed count sampling - Extract exactly K frames evenly distributed (e.g., K=10)
- Approach C: Temporal middle - Extract frames from middle portion of video (skip intro/outro)

Recommendation: Start with Approach B (10 frames per video), adjust if needed

### 2.6 Edge Cases

- No face detected: Skip frame or mark video as unclassifiable
- Multiple faces: Use largest face (by bounding box area)
- Face too small/large: Set minimum and maximum size thresholds

## 3. Stage 2: Feature Extraction

### 3.1 Objective

Extract numerical features that capture statistical differences between face and background regions. These features form the basis for classification.

### 3.2 Feature Set

We will use the following three features for classification:

- f1: DCT Chi-Squared Divergence - Detects compression artifacts from double compression
- f2: Color Histogram Chi-Squared - Detects color and lighting inconsistencies
- f3: Mean DCT Difference - Detects smoothing effects and texture loss from AI processing

This combination provides complementary detection capabilities: f1 captures frequency-domain compression mismatches, f2 captures color-space artifacts independent of compression, and f3 provides a simple baseline measure of texture differences. Together, these three features offer robust detection of low-quality deepfakes while remaining computationally feasible to implement.

### 3.2.1 Feature 1: DCT Chi-Squared Divergence

**Purpose:** Detects compression artifacts caused by double compression when a face is extracted, processed, and reinserted into a video.

**Why it indicates deepfakes:** Real videos undergo uniform compression across the entire frame. Deepfakes have faces that went through a different compression history than the background, creating detectable statistical mismatches in the frequency domain.

**Calculation Steps:**

Step 1: Convert face and background regions to grayscale

```python
face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
background_gray = cv2.cvtColor(background_region, cv2.COLOR_BGR2GRAY)
```

Step 2: Apply 2D Discrete Cosine Transform to both regions

```python
face_dct = scipy.fftpack.dct(scipy.fftpack.dct(face_gray.T, norm='ortho').T, norm='ortho')
background_dct = scipy.fftpack.dct(scipy.fftpack.dct(background_gray.T, norm='ortho').T, norm='ortho')
```

Step 3: Flatten DCT coefficient matrices to 1D arrays

```python
face_coeffs = face_dct.flatten()
background_coeffs = background_dct.flatten()
```

Step 4: Create histograms with 50 bins spanning the range of coefficient values

```python
min_val = min(face_coeffs.min(), background_coeffs.min())
max_val = max(face_coeffs.max(), background_coeffs.max())
bins = np.linspace(min_val, max_val, 50)
face_hist, _ = np.histogram(face_coeffs, bins=bins)
background_hist, _ = np.histogram(background_coeffs, bins=bins)
```

Step 5: Compute chi-squared statistic

```python
chi_squared = 0
for i in range(len(face_hist)):
    if background_hist[i] > 0:
        chi_squared += (face_hist[i] - background_hist[i])**2 / background_hist[i]
f1 = chi_squared
```

**Interpretation:** Higher values indicate greater distributional mismatch. Real videos typically have chi-squared values under 1000, while deepfakes often exceed 2000-5000.

### 3.2.2 Feature 2: Color Histogram Chi-Squared

**Purpose:** Detects color and lighting inconsistencies between the face and background regions.

**Why it indicates deepfakes:** In real videos, lighting and color grading are consistent across the frame. Deepfakes often have faces from different sources with different lighting conditions (warm classroom lighting vs. cool arena lighting), different cameras (phone vs. professional camera), or imperfect color correction applied by the deepfake tool.

**Calculation Steps:**

Step 1: Split face and background regions into RGB color channels

```python
face_b, face_g, face_r = cv2.split(face_region)
back_b, back_g, back_r = cv2.split(background_region)
```

Step 2: Create histograms for each channel (256 bins for 0-255 intensity values)

```python
bins = 256
range_vals = (0, 256)
face_hist_r, _ = np.histogram(face_r.flatten(), bins=bins, range=range_vals)
back_hist_r, _ = np.histogram(back_r.flatten(), bins=bins, range=range_vals)
# Repeat for green and blue channels
```

Step 3: Compute chi-squared for each channel

```python
def chi_squared(hist1, hist2):
    chi_sq = 0
    for i in range(len(hist1)):
        if hist2[i] > 0:
            chi_sq += (hist1[i] - hist2[i])**2 / hist2[i]
    return chi_sq
chi_r = chi_squared(face_hist_r, back_hist_r)
chi_g = chi_squared(face_hist_g, back_hist_g)
chi_b = chi_squared(face_hist_b, back_hist_b)
```

Step 4: Sum across all three channels

```python
f2 = chi_r + chi_g + chi_b
```

**Interpretation:** Higher values indicate color/lighting mismatch. Real videos typically have values under 5000-10000 per channel. Deepfakes with obvious color mismatches can exceed 20000-50000.

### 3.2.3 Feature 3: Mean DCT Difference

**Purpose:** Detects differences in average frequency content between face and background, indicating smoothing or detail loss from AI processing.

**Why it indicates deepfakes:** Low-quality deepfake tools often produce faces that appear slightly "smoothed" or "plastic-y" because the AI averages out fine details. Additionally, double compression tends to reduce high-frequency content (fine textures) more in the face region. This shows up as a lower mean DCT coefficient in the face compared to the background.

**Calculation Steps:**

Step 1: Apply DCT to face and background (same as Feature 1, Steps 1-2)

```python
face_dct = scipy.fftpack.dct(scipy.fftpack.dct(face_gray.T, norm='ortho').T, norm='ortho')
background_dct = scipy.fftpack.dct(scipy.fftpack.dct(background_gray.T, norm='ortho').T, norm='ortho')
```

Step 2: Compute mean of absolute DCT coefficients for each region

```python
face_mean = np.mean(np.abs(face_dct))
background_mean = np.mean(np.abs(background_dct))
```

Step 3: Take absolute difference

```python
f3 = abs(face_mean - background_mean)
```

**Interpretation:** Real videos typically have differences under 2.0. Deepfakes with noticeable smoothing or compression artifacts often show differences of 3.0-10.0 or higher. This is the simplest feature to compute and interpret, serving as a good baseline.

### 3.2.4 Why Use All Three Features Together?

Each feature captures a different type of artifact that deepfakes introduce:

- DCT Chi-Squared: Catches compression mismatches (most powerful for detecting double compression)
- Color Chi-Squared: Catches lighting and color inconsistencies (independent of compression)
- Mean DCT Difference: Catches AI smoothing effects and texture loss (simple baseline)

By using all three together in Naive Bayes, we maximize detection capability. A high-quality deepfake might successfully match colors (low f2) but still fail on compression artifacts (high f1). Conversely, a carefully compressed deepfake might pass the compression test but show color mismatches. Using multiple independent features increases robustness.

### 3.3 Per-Frame vs. Per-Video Features

Question: Do we extract features from each frame individually, or aggregate across frames?

**Approach A: Per-frame classification**

- Extract features from each frame independently
- Classify each frame separately
- Aggregate frame-level predictions (majority vote or average probability)
- Pros: More training data (each frame is a sample), can detect temporal inconsistencies
- Cons: Frames from same video are not independent (violates i.i.d. assumption)

**Approach B: Per-video aggregation**

- Extract features from each frame
- Aggregate features across frames (mean, median, or max)
- Classify video based on aggregated features
- Pros: Respects independence assumption, cleaner train/test split
- Cons: Fewer training samples, loses temporal information

Recommendation: Start with Approach B (per-video), which is more statistically sound

### 3.4 Output Format

For each video, produce a feature vector:

```
f = [f1, f2, f3]
```

Store in CSV or similar format:

```
video_id, f1, f2, f3, label (0=real, 1=deepfake)
```

## 4. Stage 3: Training (Parameter Estimation via MLE)

### 4.1 Objective

Estimate the probability distribution parameters for each feature in each class using Maximum Likelihood Estimation.

### 4.2 Input

- Training dataset: N videos with extracted features and ground truth labels
- Feature matrix X of shape (N, 3) and label vector y of shape (N,)

### 4.3 Output

Trained model parameters:

- For each feature i (i=1 to 3):
  - μiD, σiD (mean and std dev for Deepfake class)
  - μiR, σiR (mean and std dev for Real class)
- Prior probabilities: P(Deepfake), P(Real)

### 4.4 Implementation

For each feature fi and each class c ∈ {Deepfake, Real}:

- Filter training samples by class: X_c = {x : y = c}
- Compute sample mean: μic = mean(X_c[:, i])
- Compute sample standard deviation: σic = std(X_c[:, i])
- Apply Laplace smoothing: σic = σic + ε (where ε = 1e-6)

Estimate priors:

- P(Deepfake) = (# deepfake samples) / N
- P(Real) = (# real samples) / N

### 4.5 Model Persistence

Save trained parameters to disk (JSON or pickle) to avoid retraining:

```json
{
  "deepfake": {"mu": [μ1D, μ2D, μ3D], "sigma": [σ1D, σ2D, σ3D]},
  "real": {"mu": [μ1R, μ2R, μ3R], "sigma": [σ1R, σ2R, σ3R]},
  "priors": {"deepfake": 0.5, "real": 0.5}
}
```

## 5. Stage 4: Classification (Naive Bayes)

### 5.1 Objective

Given a new video's feature vector, compute the posterior probability of each class and make a classification decision.

### 5.2 Input

- Test video feature vector: f = [f1, f2, f3]
- Trained model parameters (from Stage 3)

### 5.3 Output

- Classification label: "Deepfake" or "Real"
- Confidence score: P(Deepfake | f) ∈ [0, 1]

### 5.4 Implementation

Step 1: Compute log-likelihood for each class

For each class c ∈ {Deepfake, Real}:

```
log P(f | c) = Σi log P(fi | c)
             = Σi [-½log(2πσic²) - (fi - μic)² / (2σic²)]
```

Step 2: Add log prior

```
log P(c | f) ∝ log P(c) + log P(f | c)
```

Step 3: Convert to probabilities (softmax)

```
P(Deepfake | f) = exp(log P(Deepfake | f)) / [exp(log P(Deepfake | f)) + exp(log P(Real | f))]
```

Step 4: Make decision

```
If P(Deepfake | f) > 0.5: classify as Deepfake
Else: classify as Real
```

## 6. Dataset Requirements

### 6.1 Training and Test Split Strategy

**Standard Approach: Train/Test Split**

- Split data at the video level (not frame level)
- 80% training, 20% test
- Ensure balanced classes in both splits

**Enhanced Approach: K-Fold Cross-Validation**

- Divide dataset into K folds (e.g., K=5)
- Train on K-1 folds, test on remaining fold
- Rotate K times so each fold serves as test set once
- Report average accuracy and standard deviation
- Benefit: Better estimates model generalization, uses all data for both training and testing

Recommendation: Start with simple train/test split. Add cross-validation if time permits.

### 6.2 Dataset Size Requirements

- Minimum viable: 50 videos per class (100 total)
- Comfortable: 100-200 videos per class (200-400 total)
- Ideal: 500+ videos per class (1000+ total)

Note: Naive Bayes works reasonably well with smaller datasets compared to deep learning, making it suitable for a class project with limited data collection resources.

### 6.3 Dataset Sources

**Real Videos:**

- NBA game highlights from YouTube
- Sports clips (basketball, football, etc.)
- News broadcasts, interviews

**Deepfake Videos:**

- Create our own using Viggle AI, Reface, or similar tools
- Scrape viral deepfake videos from TikTok/Instagram (with proper attribution)
- Use existing deepfake datasets (e.g., FaceForensics++, Celeb-DF) if available

### 6.4 Data Quality Considerations

- Video quality: Consistent resolution (e.g., 720p or 1080p)
- Video length: At least 2-3 seconds (minimum 60 frames at 30fps)
- Face visibility: Face must be clearly visible (not occluded, good lighting)
- Diversity: Include different individuals, lighting conditions, angles

## 7. Testing and Evaluation

### 7.1 Performance Metrics

**Primary Metrics:**

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP) - How many predicted deepfakes are actually deepfakes?
- Recall: TP / (TP + FN) - How many actual deepfakes did we catch?
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)

**Secondary Metrics:**

- Confusion matrix (TP, TN, FP, FN)
- ROC curve and AUC (if time permits)
- Confidence score distribution for correct vs. incorrect predictions

### 7.2 Baseline Comparisons

Compare against simple baselines to validate approach:

- Random classifier: Should get ~50% accuracy
- Single-feature classifier: Use only f1 (DCT chi-squared) to see if multiple features help
- Majority class classifier: Always predict most common class

### 7.3 Error Analysis

Investigate failure cases:

- Manually inspect videos that were misclassified
- Identify patterns: Does the model fail on certain types of videos?
- Examine feature values for failures vs. successes
- Determine if failures are due to face detection errors, feature extraction issues, or model limitations

### 7.4 Feature Importance Analysis

Determine which features contribute most to classification:

- Train models with different feature subsets
- Compare accuracy: All features vs. leave-one-out
- Plot feature distributions for real vs. deepfake classes

## 8. Implementation Plan

### 8.1 Technology Stack

- Language: Python 3.8+
- Libraries:
  - OpenCV (face detection, video processing)
  - NumPy (numerical computation)
  - SciPy (DCT, statistical functions)
  - Scikit-learn (train/test split, metrics)
  - Matplotlib (visualization)

### 8.2 Project Milestones

- Dataset Collection (Week 1): Collect/create 100+ videos (50+ per class)
- Region Segmentation (Week 1-2): Implement face detection pipeline
- Feature Extraction (Week 2): Implement 3 features, validate on sample videos
- Training (Week 3): Implement MLE parameter estimation
- Classification (Week 3): Implement Naive Bayes classifier
- Evaluation (Week 4): Run experiments, compute metrics, analyze results
- Documentation (Week 4): Write 3-page report with mathematical framework and results

### 8.3 Code Structure

Suggested file organization:

```
deepfake_detector/
├── data/                  # Raw videos and processed features
├── models/                # Saved model parameters
├── src/
│   ├── segmentation.py    # Face detection
│   ├── features.py        # Feature extraction
│   ├── train.py           # MLE training
│   ├── classify.py        # Naive Bayes classifier
│   └── utils.py           # Helper functions
├── notebooks/             # Jupyter notebooks for analysis
├── results/               # Plots, metrics, confusion matrices
└── README.md
```

## 9. Success Criteria

### 9.1 Minimum Viable Product (MVP)

- System can classify videos as real or deepfake
- Achieves >60% accuracy (better than random)
- Complete mathematical documentation of approach
- Working code that can be demonstrated

### 9.2 Target Performance

- 70-85% accuracy on test set
- Balanced precision and recall (no severe class bias)
- Clear understanding of when/why the model fails
- Comprehensive writeup connecting CS 109 concepts to implementation

### 9.3 Stretch Goals

- Implement cross-validation for robust evaluation
- Add bootstrapping for confidence intervals on parameters
- Compare Naive Bayes to simpler baselines (single feature, logistic regression)
- Build interactive demo/web interface
- Test on different types of deepfakes (not just NBA face swaps)

## 10. Open Questions and Decisions Needed

- Frame Aggregation: Per-frame or per-video classification? How many frames to sample?
- Face Detection Library: OpenCV vs. dlib vs. MTCNN? Trade-off between speed and accuracy
- Dataset Size: How many videos can we realistically collect/create?
- Evaluation Strategy: Simple train/test split or k-fold cross-validation?
- Feature Transformations: Should we apply log transforms or other preprocessing to make features more Gaussian?

## 11. Immediate Next Steps

- Begin dataset collection (real videos + create some deepfakes)
- Set up development environment (install libraries)
- Implement face detection pipeline as proof-of-concept
- Test feature extraction on 2-3 sample videos to validate approach
