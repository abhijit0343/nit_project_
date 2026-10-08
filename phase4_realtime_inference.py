"""
phase4_realtime_inference.py — Liquid Classifier Inference
===========================================================
Classifies video frames as 'mustard oil' or 'water' using the
trained MobileNetV3-Small model.

Modes:
  1. Webcam (default)  — live per-frame classification with on-screen overlay
  2. Video file        — majority vote over all sampled frames, prints result

Usage:
    # Webcam mode (default)
    python phase4_realtime_inference.py

    # Video file mode
    python phase4_realtime_inference.py --video "path/to/video.mp4"
"""

import argparse
import os
import sys
import time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import deque
from torchvision import transforms

from model import LiquidClassifier

# ──────────────────────────── Config ─────────────────────────────────────────
MODEL_PATH   = 'liquid_classifier.pth'
LABELS_PATH  = 'liquid_classes.npy'
FRAME_STEP   = 5      # sample every Nth frame in video-file mode
BUFFER_SIZE  = 10     # rolling majority-vote window (webcam mode)
CONF_THRESH  = 0.60   # minimum avg confidence to display label
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation — must match training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Display colours per class (BGR)
CLASS_COLORS = {
    'mustard_oil': (0, 180, 255),   # orange
    'mustard oil': (0, 180, 255),
    'water':       (255, 180, 0),   # blue
}
DEFAULT_COLOR = (200, 200, 200)


def load_model(device):
    """Load class names and model weights."""
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"'{LABELS_PATH}' not found. Run phase3_model_training.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Run phase3_model_training.py first.")

    classes = np.load(LABELS_PATH, allow_pickle=True).tolist()
    model   = LiquidClassifier(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded -> classes: {classes}")
    return model, classes


@torch.no_grad()
def predict_frame(model, frame_bgr, device):
    """
    Run a single BGR frame through the model.
    Returns (class_index, confidence_float).
    """
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(rgb).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return idx.item(), conf.item()


def draw_overlay(frame, label: str, confidence: float, fps: float, color):
    """Draw a stylish label + confidence bar + FPS counter on the frame."""
    h, w = frame.shape[:2]

    # ── Background panel ────────────────────────────────────────────────────
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Label text ───────────────────────────────────────────────────────────
    label_upper = label.upper()
    cv2.putText(frame, label_upper,
                (15, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label_upper,
                (15, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, color,    2, cv2.LINE_AA)

    # ── Confidence bar ────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 15, 55, 260, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (bar_x + bar_w + 8, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────── Mode A: Video File ──────────────────────────────

def classify_video(video_path: str, model, classes, device):
    """Majority-vote classification of a video file."""
    print(f"\nClassifying: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_preds      = []
    all_confs      = []
    frame_idx      = 0
    processed      = 0

    pbar = __import__('tqdm').tqdm(total=total_frames, desc="Processing frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            idx, conf = predict_frame(model, frame, device)
            all_preds.append(idx)
            all_confs.append(conf)
            processed += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not all_preds:
        print("[ERROR] No frames could be processed.")
        return

    # Majority vote
    vote_idx   = max(set(all_preds), key=all_preds.count)
    avg_conf   = float(np.mean(all_confs))
    label      = classes[vote_idx]

    print(f"\n{'='*50}")
    print(f"  Result      : {label.upper()}")
    print(f"  Confidence  : {avg_conf*100:.1f}%  (avg over {processed} frames)")
    print(f"  Vote counts : ", end="")
    for i, cls in enumerate(classes):
        print(f"{cls}={all_preds.count(i)}", end="  ")
    print(f"\n{'='*50}\n")


# ──────────────────────────── Mode B: Webcam ─────────────────────────────────

def run_webcam(model, classes, device):
    """Real-time webcam inference with on-screen overlay."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\nWebcam live!  Press 'q' to quit.\n")

    pred_buffer = deque(maxlen=BUFFER_SIZE)
    conf_buffer = deque(maxlen=BUFFER_SIZE)
    current_label = "Warming up..."
    current_conf  = 0.0
    current_color = DEFAULT_COLOR

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # selfie / mirror view

        # Inference on every frame
        idx, conf = predict_frame(model, frame, device)
        pred_buffer.append(idx)
        conf_buffer.append(conf)

        # Majority vote + average confidence over buffer
        vote_idx     = max(set(pred_buffer), key=pred_buffer.count)
        avg_conf     = float(np.mean(conf_buffer))

        if avg_conf >= CONF_THRESH:
            label        = classes[vote_idx]
            current_label = label
            current_conf  = avg_conf
            current_color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        else:
            current_label = "Uncertain"
            current_conf  = avg_conf
            current_color = DEFAULT_COLOR

        # FPS calculation
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Draw overlay
        frame = draw_overlay(frame, current_label, current_conf, fps, current_color)

        cv2.imshow('Liquid Classifier - Press Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ──────────────────────────── Entry Point ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Liquid Classifier Inference")
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a video file. Omit for webcam mode.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on: {device}")
    if device.type == 'cuda':
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    try:
        model, classes = load_model(device)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if args.video:
        classify_video(args.video, model, classes, device)
    else:
        run_webcam(model, classes, device)


if __name__ == '__main__':
    main()
