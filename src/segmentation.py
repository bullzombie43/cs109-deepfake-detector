"""
segmentation.py — Phase 2: Frame sampling + face/background extraction.

Public API:
  sample_frames(video_path, n=10)       -> list of frames (numpy arrays)
  extract_regions(frame, bbox)          -> (face, background) tuple
  process_video(video_path, n=10)       -> list of (face, background) pairs

Run directly to visually test on one real and one deepfake video:
  python src/segmentation.py
"""

import cv2
import numpy as np

# Minimum face size to accept (width x height in pixels)
MIN_FACE_SIZE = (20, 20)

# Load Haar cascade once at module level
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def sample_frames(video_path: str, n: int = 10) -> list:
    """
    Uniformly sample up to n frames from a video.

    Returns a list of BGR numpy arrays. May return fewer than n frames
    if the video is shorter than n frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Compute uniformly spaced indices across the full video duration
    sample_count = min(n, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def _detect_face(frame: np.ndarray):
    """
    Detect faces in a BGR frame using Haar Cascades.

    Returns the bounding box (x, y, w, h) of the largest detected face,
    or None if no acceptable face is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    if len(faces) == 0:
        return None

    # Use the largest face by area
    largest = max(faces, key=lambda b: b[2] * b[3])
    x, y, w, h = largest

    # Skip if too small
    if w < MIN_FACE_SIZE[0] or h < MIN_FACE_SIZE[1]:
        return None

    return (x, y, w, h)


def extract_regions(frame: np.ndarray, bbox: tuple) -> tuple:
    """
    Given a frame and a face bounding box (x, y, w, h), return:
      face        — the cropped face region (BGR array)
      background  — a same-sized patch cropped from the nearest non-overlapping
                    region of the frame (right → left → below → above)

    Using a same-sized background patch avoids DCT ringing artifacts from
    zero-filled regions and histogram skew from artificially dark pixels.
    Returns None for background if no valid patch fits in the frame.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    face = frame[y:y + h, x:x + w].copy()

    # Try candidate patch origins in priority order: right, left, below, above
    candidates = [
        (x + w, y),           # right of face
        (x - w, y),           # left of face
        (x, y + h),           # below face
        (x, y - h),           # above face
    ]

    for px, py in candidates:
        if px >= 0 and py >= 0 and px + w <= fw and py + h <= fh:
            background = frame[py:py + h, px:px + w].copy()
            return face, background

    # Fallback: top-left corner patch if no candidate avoids the face
    background = frame[0:h, 0:w].copy()
    return face, background




def sample_frame_pairs(video_path: str, n: int = 10) -> list:
    """
    Sample n anchor positions and return (frame_t, frame_t+1) pairs of truly
    consecutive frames. Anchors are spread uniformly across the video, but each
    pair is only 1 frame apart (~33ms), isolating warp jitter from natural motion.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []

    # Anchor indices spaced across the video; t+1 always valid since max = total-2
    sample_count = min(n, total_frames - 1)
    indices = np.linspace(0, total_frames - 2, sample_count, dtype=int)

    pairs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret1, frame_t  = cap.read()
        ret2, frame_t1 = cap.read()   # immediately next frame
        if ret1 and ret2:
            pairs.append((frame_t, frame_t1))

    cap.release()
    return pairs


def process_video_pairs(video_path: str, n: int = 10) -> list:
    """
    Return a list of (face_t, face_t1) consecutive face crop pairs.

    Face bbox is detected in frame_t and applied to frame_t1 — this ensures
    we compare the exact same spatial region across both frames, so any
    difference is purely temporal (warp jitter, flicker) rather than spatial
    bbox drift.
    """
    frame_pairs = sample_frame_pairs(video_path, n=n)
    face_pairs = []

    for frame_t, frame_t1 in frame_pairs:
        bbox = _detect_face(frame_t)
        if bbox is None:
            continue
        x, y, w, h = bbox
        face_t  = frame_t [y:y + h, x:x + w].copy()
        face_t1 = frame_t1[y:y + h, x:x + w].copy()
        face_pairs.append((face_t, face_t1))

    return face_pairs


def process_video(video_path: str, n: int = 10) -> list:
    """
    Extract up to n (face, background, frame, bbox) tuples from a video.

    frame and bbox are passed through so features.py can compute region-based
    features (e.g. boundary gradient) that require the full frame context.

    Frames with no detectable face are silently skipped.
    Returns a (possibly empty) list of tuples.
    """
    frames = sample_frames(video_path, n=n)
    result = []

    for frame in frames:
        bbox = _detect_face(frame)
        if bbox is None:
            continue
        face, background = extract_regions(frame, bbox)
        result.append((face, background, frame, bbox))

    return result


# ---------------------------------------------------------------------------
# Visual test — run with: python src/segmentation.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")  # headless — saves to file instead of displaying
    import matplotlib.pyplot as plt

    REAL_DIR = "data/real"
    FAKE_DIR = "data/deepfake"
    OUT_PATH = "results/segmentation_test.png"

    def _first_video(directory: str) -> str:
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".mp4"):
                return os.path.join(directory, fname)
        return None

    real_path = _first_video(REAL_DIR)
    fake_path = _first_video(FAKE_DIR)

    if real_path is None or fake_path is None:
        print("ERROR: Could not find videos in data/real/ or data/deepfake/")
        sys.exit(1)

    print(f"Real video : {real_path}")
    print(f"Fake video : {fake_path}")

    real_pairs = process_video(real_path)
    fake_pairs = process_video(fake_path)

    print(f"Real  → {len(real_pairs)} face/background pairs extracted")
    print(f"Fake  → {len(fake_pairs)} face/background pairs extracted")

    if not real_pairs and not fake_pairs:
        print("ERROR: No faces detected in either video. Check cascade path.")
        sys.exit(1)

    # Build a grid: row 0 = real faces, row 1 = fake faces (up to 5 each)
    max_show = 5
    real_faces = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f, _ in real_pairs[:max_show]]
    fake_faces = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f, _ in fake_pairs[:max_show]]

    fig, axes = plt.subplots(2, max_show, figsize=(15, 6))
    fig.suptitle("Segmentation Test — Face Crops", fontsize=14)

    labels = ["Real", "Fake"]
    for row, (faces, label) in enumerate([(real_faces, "Real"), (fake_faces, "Fake")]):
        for col in range(max_show):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(faces):
                ax.imshow(faces[col])
                if col == 0:
                    ax.set_title(label, fontsize=10, loc="left")
            else:
                ax.set_facecolor("#eeeeee")

    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=100)
    print(f"Saved face crop grid → {OUT_PATH}")
