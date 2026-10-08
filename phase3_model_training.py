"""
phase3_model_training.py — CNN Fine-Tuning for Liquid Classifier
=================================================================
Trains a MobileNetV3-Small model on the JPEG frames extracted by
phase2_data_collection.py, then saves the best weights.

Usage:
    python phase3_model_training.py
"""

import os
import sys
import copy
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from model import LiquidClassifier

# ──────────────────────────── Hyperparameters ────────────────────────────────
FRAMES_DIR    = 'frames'                # root of extracted JPEG frames
MODEL_SAVE    = 'liquid_classifier.pth' # best model weights
LABELS_SAVE   = 'liquid_classes.npy'   # class-name array
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5                       # early-stop patience (val-loss)
VAL_SPLIT     = 0.20                    # 80/20 train-val split
NUM_WORKERS   = 0                       # keep 0 on Windows
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation — required because backbone was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def make_weighted_sampler(labels):
    """Returns a WeightedRandomSampler that balances class frequencies."""
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                 num_samples=len(weights), replacement=True)


def evaluate(model, loader, criterion, device):
    """Returns (avg_loss, accuracy%, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


def main():
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    print(f"{'='*60}\n")

    # ── Sanity check ─────────────────────────────────────────────────────────
    if not os.path.isdir(FRAMES_DIR):
        print(f"[ERROR] Frames directory '{FRAMES_DIR}' not found.")
        print("Please run phase2_data_collection.py first.")
        return

    # ── Load full dataset (val transform first to get labels) ────────────────
    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(FRAMES_DIR, transform=val_tf)
    classes      = full_dataset.classes
    num_classes  = len(classes)
    all_labels   = np.array([s[1] for s in full_dataset.samples])

    print(f"  Classes   : {classes}")
    print(f"  Total frames : {len(full_dataset)}")
    for i, cls in enumerate(classes):
        print(f"    [{i}] {cls:20s} -> {(all_labels == i).sum()} frames")
    print()

    # ── Stratified 80/20 split ───────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    # Apply correct transforms to each split
    train_dataset = datasets.ImageFolder(FRAMES_DIR, transform=train_tf)
    val_dataset   = datasets.ImageFolder(FRAMES_DIR, transform=val_tf)

    train_subset  = Subset(train_dataset, train_idx)
    val_subset    = Subset(val_dataset,   val_idx)

    print(f"  Train frames : {len(train_subset)}")
    print(f"  Val frames   : {len(val_subset)}\n")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_labels  = all_labels[train_idx]
    sampler       = make_weighted_sampler(train_labels)
    pin           = (device.type == 'cuda')

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=pin)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=pin)

    # ── Model ────────────────────────────────────────────────────────────────
    model = LiquidClassifier(num_classes=num_classes, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters : {trainable:,} / {total:,} (backbone frozen)\n")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"Starting training - up to {EPOCHS} epochs (early-stop patience={PATIENCE})...\n")
    best_val_loss     = float('inf')
    best_model_wts    = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{EPOCHS} [train]",
                    leave=False, unit="batch")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device, non_blocking=True), \
                           labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # --- Validate ---
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # --- Best model checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, MODEL_SAVE)
            print(f"           [*] Best model saved  (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # --- Early stopping ---
        if epochs_no_improve >= PATIENCE:
            print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Stopping.")
            break

    # ── Final evaluation ─────────────────────────────────────────────────────
    print(f"\nLoading best model from '{MODEL_SAVE}'...")
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    _, final_acc, preds, true_labels = evaluate(model, val_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  Final Val Accuracy : {final_acc:.2f}%")
    print(f"{'='*60}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=classes))

    # Save class names for inference
    np.save(LABELS_SAVE, np.array(classes))
    print(f"Saved -> {MODEL_SAVE}")
    print(f"Saved -> {LABELS_SAVE}")
    print("\nNext step -> run:  python phase4_realtime_inference.py")


if __name__ == '__main__':
    main()
