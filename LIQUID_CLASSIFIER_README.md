# 🧪 Liquid Classifier: Mustard Oil vs Water
### CNN Transfer Learning — Changes to Existing Pipeline Files

---

## 📌 Overview

This document describes **exact changes to existing files** to repurpose the pipeline
for classifying two liquid types: **mustard oil** and **water**, using video recordings
stored in `dataset/my_dataset/`.

| Class | Visual Cue |
|-------|-----------|
| 🟡 `mustard oil` | Golden/yellow, opaque, viscous |
| 💧 `water` | Transparent, colorless, thin |

> ⚠️ **Why NOT the existing LSTM/MediaPipe approach?**
> `ActionLSTM` + MediaPipe reads 33 **body skeleton joints** — useless for liquid color.
> We switch `phase3_model_training.py` and `phase4_realtime_inference.py` to use
> a **MobileNetV3-Small CNN** (visual appearance classifier) instead.

---

## 🗂️ Your Dataset (Ready to Use)

```
dataset/my_dataset/
├── mustard oil/     ← 14 MP4 videos  (~2-4 sec each)
└── water/           ← 25 MP4 videos  (some duplicates — handled automatically)
```

> ⚠️ Some water videos share identical file sizes = duplicates.
> `phase2_data_collection.py` will skip them via MD5 hash check.

---

## 📋 Files Changed & How

| File | Change Type | What Changes |
|------|------------|--------------|
| `phase2_data_collection.py` | **Major rewrite** | Replace MediaPipe keypoint extraction with frame extraction from MP4s → save as images |
| `model.py` | **Major rewrite** | Replace `ActionLSTM` with `LiquidClassifier` (MobileNetV3-Small) |
| `phase3_model_training.py` | **Major rewrite** | Replace LSTM training with CNN fine-tuning on image frames |
| `phase4_realtime_inference.py` | **Major rewrite** | Replace pose-based inference with CNN frame inference on webcam/video |
| `feature_extraction.py` | **No change needed** | Not used in new pipeline |
| `phase1_pose_detection.py` | **No change needed** | Standalone pose viewer, unaffected |
| `phase2b_data_augmentation.py` | **No change needed** | Augmentation now handled inside training |
| `run_pipeline.py` | **Minor update** | Update phase names/descriptions |
| `requirements.txt` | **No change needed** | `torch`, `torchvision`, `opencv-python` already listed |

---

## 🏗️ Architecture: What Replaces What

### OLD (HAR Pipeline)
```
MP4 → MediaPipe skeleton → 231 features/frame → LSTM sequences → ActionLSTM → action label
```

### NEW (Liquid Classifier)
```
MP4 → Sampled frames (224×224 JPEGs) → MobileNetV3-Small CNN → liquid label
```

### Model Architecture (replaces ActionLSTM in model.py)
```
Input: 224×224 RGB frame
        │
        ▼
MobileNetV3-Small (pretrained ImageNet backbone, frozen)
        │
        ▼
Custom head:
   Linear(576 → 256) → Hardswish → Dropout(0.3) → Linear(256 → 2)
        │
        ▼
Softmax → [mustard_oil, water]
```

---

## ✏️ Detailed Changes Per File

---

### 1. `phase2_data_collection.py` — Frame Extractor

**Remove:** All MediaPipe, ProcessPoolExecutor, `extract_keypoints`, SEQUENCE_LENGTH logic.

**Replace with:**
```python
# New constants
DATASET_DIR    = r'dataset\my_dataset'
OUTPUT_DIR     = r'frames'           # extracted JPEGs go here
FRAME_STEP     = 5                   # sample 1 frame every 5
RESIZE         = (224, 224)

# New logic (per class folder):
# 1. Compute MD5 hash of each .mp4 → skip duplicates
# 2. Open with cv2.VideoCapture
# 3. Read every FRAME_STEP-th frame
# 4. Resize to 224×224
# 5. Save to frames/<class_name>/vid_XX_frame_YYYY.jpg
# 6. Print: "Extracted N frames from M unique videos per class"

# Output:
# frames/
# ├── mustard oil/
# │   └── vid_00_frame_000.jpg, vid_00_frame_005.jpg, ...
# └── water/
#     └── vid_00_frame_000.jpg, ...
```

**Expected output:**
```
[mustard oil] 14 videos found → 14 unique → ~1,400 frames extracted
[water]       25 videos found → 15 unique → ~1,500 frames extracted
Saved → frames/
```

---

### 2. `model.py` — LiquidClassifier (replaces ActionLSTM)

**Remove:** `SelfAttention`, `ActionLSTM` classes entirely.

**Replace with:**
```python
import torch
import torch.nn as nn
import torchvision.models as models

class LiquidClassifier(nn.Module):
    """
    MobileNetV3-Small fine-tuned for binary liquid classification.
    Backbone frozen; only the custom head is trained by default.

    Args:
        num_classes     : number of output classes (default 2)
        freeze_backbone : freeze feature extractor layers (default True)
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Replace the built-in classifier head
        backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)
```

---

### 3. `phase3_model_training.py` — CNN Training (replaces LSTM training)

**Remove:** `PoseDataset`, all `X.npy`/`y.npy` loading, `ActionLSTM` usage.

**Replace with:**

```python
# New constants
FRAMES_DIR    = r'frames'
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-3
PATIENCE      = 5             # early stopping
MODEL_SAVE    = 'liquid_classifier.pth'
LABELS_SAVE   = 'liquid_classes.npy'

# New training logic:
# 1. Load dataset with torchvision.datasets.ImageFolder(FRAMES_DIR)
#    → auto-detects class names from folder names: ['mustard oil', 'water']
#
# 2. Train/Val split: 80/20 stratified
#
# 3. Transforms:
#    Train: RandomHorizontalFlip, ColorJitter(brightness=0.3, hue=0.1),
#           RandomRotation(15), ToTensor, Normalize(ImageNet mean/std)
#    Val:   Resize(224), CenterCrop(224), ToTensor, Normalize
#
# 4. WeightedRandomSampler → handles class imbalance automatically
#
# 5. Model: LiquidClassifier(num_classes=2, freeze_backbone=True)
#    Optimizer: AdamW(lr=1e-3)
#    Scheduler: CosineAnnealingLR(T_max=EPOCHS)
#    Loss: CrossEntropyLoss
#
# 6. EarlyStopping on val_loss (patience=5)
# 7. Save best model → liquid_classifier.pth
# 8. Save class names → liquid_classes.npy
# 9. Print: Confusion Matrix + Classification Report
```

**Expected training output:**
```
Classes: ['mustard oil', 'water']
Train: 2304 frames | Val: 576 frames

Epoch  1/30 | Train Loss: 0.612 | Val Acc: 81.3%
Epoch  5/30 | Train Loss: 0.289 | Val Acc: 93.7%
Epoch 12/30 | Train Loss: 0.174 | Val Acc: 96.2%  ← best saved
Early stopping triggered.

              precision  recall  f1-score
mustard oil      0.97     0.95      0.96
water            0.96     0.98      0.97
```

---

### 4. `phase4_realtime_inference.py` — CNN Inference (replaces pose inference)

**Remove:** All MediaPipe, `extract_keypoints`, `ActionLSTM`, `sequence deque` logic.

**Replace with:**

```python
# New constants
MODEL_PATH  = 'liquid_classifier.pth'
LABELS_PATH = 'liquid_classes.npy'
BUFFER_SIZE = 10   # majority vote over last 10 frame predictions
CONF_THRESH = 0.6  # minimum confidence to display label

# New inference logic:
# Mode A — Video file (via CLI arg --video path/to/file.mp4):
#   1. Open video with cv2.VideoCapture
#   2. Sample every 5th frame
#   3. Preprocess: resize 224×224, normalize
#   4. Run through LiquidClassifier → softmax probabilities
#   5. Collect all frame predictions → majority vote
#   6. Print: "Result: MUSTARD OIL  (avg confidence: 96.3%)"
#
# Mode B — Webcam (default, no args):
#   1. Open webcam (index 0)
#   2. Per frame: preprocess → model inference
#   3. Rolling deque(maxlen=10) of predictions → majority vote
#   4. Overlay on frame:
#      - Label text: "MUSTARD OIL" or "WATER"
#      - Confidence bar (colored rectangle)
#      - FPS counter
#   5. Press 'q' to quit

# On-screen overlay:
#  ┌──────────────────────────────┐
#  │  🟡 MUSTARD OIL             │
#  │  Confidence: 96.3% ████░░  │
#  │  FPS: 28                    │
#  └──────────────────────────────┘
```

**Usage after training:**
```bash
# Webcam mode
python phase4_realtime_inference.py

# Video file mode
python phase4_realtime_inference.py --video "dataset/my_dataset/mustard oil/WhatsApp Video 2026-10-08 at 9.58.12 AM (1).mp4"
```

---

### 5. `run_pipeline.py` — Minor Update

**Change only** the phase descriptions and the final save message:

```python
# Before:
phases = {
    '2':  ('Phase 2 — Keypoint Extraction (Data Collection)', 'phase2_data_collection.py'),
    '2b': ('Phase 2b — Data Augmentation',                    'phase2b_data_augmentation.py'),
    '3':  ('Phase 3 — Model Training (GPU)',                  'phase3_model_training.py'),
}

# After:
phases = {
    '2':  ('Phase 2 — Frame Extraction from Videos',          'phase2_data_collection.py'),
    '3':  ('Phase 3 — CNN Model Training (MobileNetV3)',       'phase3_model_training.py'),
}
# Remove '2b' from order (augmentation is now inside phase3)
```

---

## 📋 Execution Order

```bash
# Step 1: Extract frames from all MP4 videos
python phase2_data_collection.py

# Step 2: Train the CNN classifier
python phase3_model_training.py

# Step 3a: Live webcam classification
python phase4_realtime_inference.py

# Step 3b: Classify a specific video file
python phase4_realtime_inference.py --video "dataset/my_dataset/water/WhatsApp Video 2026-10-08 at 10.00.07 AM.mp4"

# OR run steps 1+2 together:
python run_pipeline.py
```

---

## 📊 Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Validation Accuracy | > 92% |
| Inference Speed (CPU) | 25–35 FPS |
| Inference Speed (GPU) | 60+ FPS |
| Training Time (CPU) | ~5–10 minutes |
| Training Time (GPU) | < 2 minutes |
| Saved model size | ~10 MB (`liquid_classifier.pth`) |

---

## ⚠️ Key Notes

1. **`feature_extraction.py`** is no longer called anywhere — safe to keep as-is.
2. **`phase2b_data_augmentation.py`** is no longer needed — augmentation is built into training.
3. **Old `.npy` files** (`X.npy`, `y.npy`, `X_aug.npy`, `y_aug.npy`, `actions.npy`) are ignored.
4. **Old model weights** (`action_model.pth`, `action_model_best.pth`) are ignored.
5. New outputs: `frames/` folder + `liquid_classifier.pth` + `liquid_classes.npy`

---

## ✅ Confirm Before I Start Coding

- [ ] Okay to overwrite `model.py`, `phase2`, `phase3`, `phase4`?
- [ ] Webcam inference needed, or just video file mode?
- [ ] GPU available for training?

*Awaiting your go-ahead!* 🚀
