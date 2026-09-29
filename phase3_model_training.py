import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import ActionLSTM

# ─────────────────────────── Hyperparameters ────────────────────────────────
HIDDEN_SIZE    = 128      # increased from 64 for better capacity
NUM_LAYERS     = 2
BATCH_SIZE     = 64       # larger batch -> better GPU utilisation
LEARNING_RATE  = 1e-3
EPOCHS         = 300
WEIGHT_DECAY   = 1e-4
PATIENCE       = 30       # early-stopping patience (epochs without val-loss improvement)
NUM_WORKERS    = 0        # set >0 only on Linux; Windows multiprocessing in DataLoader is tricky
# ─────────────────────────────────────────────────────────────────────────────


class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, criterion, device):
    """Returns (avg_loss, accuracy%) over the given loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # ── GPU setup ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True   # faster cuDNN kernels after warm-up
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        if os.path.exists('X_aug.npy') and os.path.exists('y_aug.npy'):
            X = np.load('X_aug.npy')
            y = np.load('y_aug.npy')
            print("Loaded augmented dataset  (X_aug.npy / y_aug.npy)")
        else:
            X = np.load('X.npy')
            y = np.load('y.npy')
            print("Loaded raw dataset  (X.npy / y.npy)")
        actions = np.load('actions.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}")
        print("Please run Phase 2 (data collection) first.")
        return

    if X.ndim != 3 or X.shape[0] == 0:
        print("[ERROR] X.npy is empty or has wrong shape. Run Phase 2 first.")
        return

    num_classes = len(actions)
    input_size  = X.shape[2]   # 231 features per frame
    print(f"  Dataset : {X.shape[0]} sequences  |  {num_classes} classes  |  {input_size} features/frame\n")

    # ── Train / Val / Test split  (70 / 15 / 15) ────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"  Train  : {X_train.shape[0]} sequences")
    print(f"  Val    : {X_val.shape[0]} sequences")
    print(f"  Test   : {X_test.shape[0]} sequences\n")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    pin = (device.type == 'cuda')
    train_loader = DataLoader(PoseDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(PoseDataset(X_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(PoseDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ActionLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {total_params:,}\n")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"Starting training for up to {EPOCHS} epochs  (early-stop patience={PATIENCE})...\n")
    best_val_loss  = float('inf')
    epochs_no_improve = 0
    best_model_path = 'action_model_best.pth'

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{EPOCHS} [train]",
                    leave=False, unit="batch")
        for batch_X, batch_y in loop:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), \
                               batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validate ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # --- Log ---
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # --- Best model checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        # --- Early stopping ---
        if epochs_no_improve >= PATIENCE:
            print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Stopping.")
            break

    # ── Final evaluation on test set ─────────────────────────────────────────
    print(f"\nLoading best model from '{best_model_path}'...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.2f}%")
    print(f"{'='*60}\n")

    # Save final model (same weights as best)
    torch.save(model.state_dict(), 'action_model.pth')
    print("Saved  ->  action_model.pth  (best checkpoint)")
    print("Saved  ->  action_model_best.pth  (same best checkpoint, kept separately)")


if __name__ == '__main__':
    main()
