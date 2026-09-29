"""
run_pipeline.py — End-to-end pipeline runner
=============================================
Runs all phases in order:
  1. Data collection  (phase2_data_collection.py)
  2. Data augmentation(phase2b_data_augmentation.py)
  3. Model training   (phase3_model_training.py)

Usage:
    python run_pipeline.py              # run all phases
    python run_pipeline.py --from 2b   # start from augmentation
    python run_pipeline.py --from 3    # start from training
"""

import argparse
import subprocess
import sys
import os
import time


def run_phase(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  >  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)
    print(f"\n[OK] {name} completed in {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Run the full action-recognition pipeline.")
    parser.add_argument(
        '--from', dest='start_phase', default='2',
        choices=['2', '2b', '3'],
        help="Which phase to start from (default: 2 — full pipeline)")
    args = parser.parse_args()

    # Verify CUDA is available before starting
    import torch
    print(f"\n{'='*60}")
    print(f"  GPU check")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available — {torch.cuda.get_device_name(0)}")
    else:
        print("  [WARN] CUDA not available — training will run on CPU (slower).")
        print("     Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    phases = {
        '2':  ('Phase 2 — Keypoint Extraction (Data Collection)', 'phase2_data_collection.py'),
        '2b': ('Phase 2b — Data Augmentation',                    'phase2b_data_augmentation.py'),
        '3':  ('Phase 3 — Model Training (GPU)',                  'phase3_model_training.py'),
    }

    order = ['2', '2b', '3']
    start_idx = order.index(args.start_phase)
    selected = order[start_idx:]

    pipeline_start = time.time()
    for phase_key in selected:
        name, script = phases[phase_key]
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)
        run_phase(name, script)

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  [DONE] Full pipeline finished in {total/60:.1f} min")
    print(f"  Model saved ->  action_model.pth")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
