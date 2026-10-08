"""
phase2_data_collection.py — Frame Extractor for Liquid Classifier
==================================================================
Reads MP4 videos from dataset/my_dataset/<class>/ folders and
extracts sampled JPEG frames into frames/<class>/ ready for
torchvision.datasets.ImageFolder in Phase 3 training.

Duplicate videos (same MD5 hash) are automatically skipped.

Usage:
    python phase2_data_collection.py
"""

import os
import cv2
import hashlib
from tqdm import tqdm

# ──────────────────────────── Config ────────────────────────────────────────
DATASET_DIR  = r'dataset\my_dataset'   # root containing one folder per class
OUTPUT_DIR   = r'frames'               # where extracted JPEGs are saved
FRAME_STEP   = 5                       # sample 1 frame every FRAME_STEP frames
RESIZE       = (224, 224)              # MobileNetV3 input resolution
JPEG_QUALITY = 95                      # JPEG compression quality (0-100)
# ─────────────────────────────────────────────────────────────────────────────


def md5_of_file(path: str) -> str:
    """Return MD5 hex-digest of a file — used to detect exact duplicates."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_frames(video_path: str, out_folder: str, vid_idx: int) -> int:
    """
    Extract every FRAME_STEP-th frame from a video, resize to RESIZE,
    and save as JPEG.  Returns number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            resized = cv2.resize(frame, RESIZE, interpolation=cv2.INTER_AREA)
            filename = f"vid_{vid_idx:03d}_frame_{frame_idx:05d}.jpg"
            out_path = os.path.join(out_folder, filename)
            cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    # Discover class folders
    class_folders = sorted([
        f for f in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, f))
    ])

    if not class_folders:
        print(f"[ERROR] No class folders found in '{DATASET_DIR}'.")
        return

    print(f"\nFound {len(class_folders)} classes: {class_folders}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Frame step       : every {FRAME_STEP} frames")
    print(f"Resize           : {RESIZE[0]}×{RESIZE[1]}\n")

    total_frames = 0

    for cls in class_folders:
        cls_input  = os.path.join(DATASET_DIR, cls)
        # Use sanitised folder name (replace spaces) for the output directory
        cls_safe   = cls.replace(' ', '_').lower()
        cls_output = os.path.join(OUTPUT_DIR, cls_safe)
        os.makedirs(cls_output, exist_ok=True)

        # Collect all MP4 / AVI files
        videos = [
            os.path.join(cls_input, f)
            for f in sorted(os.listdir(cls_input))
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]

        if not videos:
            print(f"  [{cls}] No video files found — skipping.")
            continue

        # Deduplicate by MD5
        seen_hashes = set()
        unique_videos = []
        for vp in videos:
            h = md5_of_file(vp)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_videos.append(vp)

        skipped = len(videos) - len(unique_videos)
        print(f"  [{cls}]  {len(videos)} videos found  ->  "
              f"{len(unique_videos)} unique  "
              f"({skipped} duplicate(s) skipped)")

        cls_frames = 0
        for vid_idx, vp in enumerate(tqdm(unique_videos, desc=f"  Extracting [{cls}]", unit="video")):
            n = extract_frames(vp, cls_output, vid_idx)
            cls_frames += n

        print(f"           -> {cls_frames} frames saved to '{cls_output}'\n")
        total_frames += cls_frames

    print(f"{'='*55}")
    print(f"  Frame extraction complete!")
    print(f"  Total frames saved : {total_frames}")
    print(f"  Output root        : {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*55}\n")
    print("Next step -> run:  python phase3_model_training.py")


if __name__ == '__main__':
    main()
