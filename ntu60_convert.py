"""
ntu60_convert.py — NTU60 HRNet → Training Data Converter
==========================================================
Reads dataset/ntu60_hrnet.pkl.zip directly (no manual extraction needed).
Converts skeleton sequences to X.npy / y.npy / actions.npy compatible with
the existing phase3_model_training.py pipeline.

Output shapes:
  X.npy       : (N, 30, 51)   — N sequences × 30 frames × 51 features
  y.npy       : (N,)          — integer class labels 0–59
  actions.npy : (60,)         — NTU60 class name strings

Feature vector per frame (51 features):
  For each of 17 COCO keypoints: [x_norm, y_norm, confidence]

Split: xsub_train + xsub_val  (standard cross-subject benchmark)
"""

import os
import zipfile
import pickle
import numpy as np
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
ZIP_PATH       = os.path.join('dataset', 'ntu60_hrnet.pkl.zip')
PKL_NAME       = 'ntu60_hrnet.pkl'
SEQUENCE_LEN   = 30        # fixed output sequence length (frames)
NUM_KEYPOINTS  = 17        # COCO skeleton keypoints
FEATURES_PER_KP = 3        # x, y, confidence
FEATURE_SIZE   = NUM_KEYPOINTS * FEATURES_PER_KP   # 51

# NTU-RGB+D 60 action class names (label index 0–59)
NTU60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_sequence(keypoints, scores, target_len):
    """
    Resample a variable-length skeleton sequence to exactly `target_len` frames.

    keypoints : (T, 17, 2)  — x, y coordinates (float16 → float32)
    scores    : (T, 17)     — confidence scores
    Returns   : (target_len, 51) float32
    """
    T = keypoints.shape[0]
    kp  = keypoints.astype(np.float32)   # (T, 17, 2)
    sc  = scores.astype(np.float32)      # (T, 17)

    # Normalize x,y to [0,1] using per-sequence min/max
    xy_min = kp.min(axis=(0, 1), keepdims=True)   # (1,1,2)
    xy_max = kp.max(axis=(0, 1), keepdims=True)
    denom  = (xy_max - xy_min)
    denom[denom == 0] = 1.0
    kp_norm = (kp - xy_min) / denom               # (T, 17, 2)

    # Resample to target_len via linear interpolation
    if T == target_len:
        kp_rs = kp_norm
        sc_rs = sc
    else:
        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, target_len)
        kp_rs = np.stack([
            np.interp(new_idx, old_idx, kp_norm[:, j, c])
            for j in range(NUM_KEYPOINTS)
            for c in range(2)
        ], axis=1).reshape(target_len, NUM_KEYPOINTS, 2)   # (target_len, 17, 2)
        sc_rs = np.stack([
            np.interp(new_idx, old_idx, sc[:, j])
            for j in range(NUM_KEYPOINTS)
        ], axis=1)                                          # (target_len, 17)

    # Concatenate x, y, score → (target_len, 17, 3) → (target_len, 51)
    feat = np.concatenate([kp_rs, sc_rs[:, :, np.newaxis]], axis=2)  # (T,17,3)
    return feat.reshape(target_len, FEATURE_SIZE).astype(np.float32)


def main():
    print("=" * 60)
    print("  NTU60 HRNet -> Training Data Converter")
    print("=" * 60)

    # ── Load pkl from zip ──────────────────────────────────────────────────────
    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] Dataset not found: {ZIP_PATH}")
        return

    print(f"\nLoading {ZIP_PATH}  (~700 MB, please wait)...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(PKL_NAME) as f:
            data = pickle.load(f)

    annotations = data['annotations']
    split       = data['split']

    # Build a lookup: frame_dir → annotation
    print("Building frame_dir index...")
    ann_by_id = {ann['frame_dir']: ann for ann in annotations}

    # xsub split: combine train + val (we'll re-split in phase3)
    train_ids = set(split['xsub_train'])
    val_ids   = set(split['xsub_val'])
    all_ids   = train_ids | val_ids
    print(f"  xsub_train : {len(train_ids):,} sequences")
    print(f"  xsub_val   : {len(val_ids):,} sequences")
    print(f"  Total      : {len(all_ids):,} sequences")

    # ── Convert sequences ─────────────────────────────────────────────────────
    X_list, y_list = [], []
    skipped = 0

    ids_ordered = sorted(all_ids)   # deterministic order
    for fdir in tqdm(ids_ordered, desc="Converting sequences", unit="seq"):
        if fdir not in ann_by_id:
            skipped += 1
            continue
        ann = ann_by_id[fdir]

        label       = ann['label']            # 0–59
        kp_raw      = ann['keypoint']         # (persons, T, 17, 2)
        sc_raw      = ann['keypoint_score']   # (persons, T, 17)

        # Use only the first person (NTU60 is mostly single-person actions)
        kp = kp_raw[0]   # (T, 17, 2)
        sc = sc_raw[0]   # (T, 17)

        T = kp.shape[0]
        if T < 2:
            skipped += 1
            continue

        feat = normalize_sequence(kp, sc, SEQUENCE_LEN)   # (30, 51)
        X_list.append(feat)
        y_list.append(label)

    print(f"\nSkipped {skipped} sequences (missing/too short).")

    # ── Save ──────────────────────────────────────────────────────────────────
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    actions = np.array(NTU60_CLASSES[:60])

    print(f"\n{'─'*60}")
    print(f"  X shape   : {X.shape}  (sequences, frames, features)")
    print(f"  y shape   : {y.shape}")
    print(f"  Classes   : {len(actions)}  ({actions[0]} … {actions[-1]})")
    class_counts = {int(i): int((y == i).sum()) for i in range(60)}
    print(f"  Min seqs per class : {min(class_counts.values())}")
    print(f"  Max seqs per class : {max(class_counts.values())}")
    print(f"{'─'*60}\n")

    np.save('X.npy',       X)
    np.save('y.npy',       y)
    np.save('actions.npy', actions)

    # Remove augmented data from previous run so training loads fresh X.npy
    for f in ['X_aug.npy', 'y_aug.npy']:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed old {f}")

    print("Saved  →  X.npy, y.npy, actions.npy")
    print("\nNext step:  python phase3_model_training.py")
    print("       or:  python run_pipeline.py --from 3\n")


if __name__ == '__main__':
    main()
