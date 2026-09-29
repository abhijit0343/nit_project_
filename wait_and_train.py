"""
wait_and_train.py — Polls until PyTorch is available, then runs the full pipeline.

Run this in a separate terminal:
    python wait_and_train.py

It will keep checking every 30 seconds until `torch` is importable with CUDA,
then automatically launch the full pipeline.
"""
import subprocess
import sys
import time


def check_torch_cuda():
    """Returns (torch_available, cuda_available, gpu_name)."""
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; print(torch.__version__); print(torch.cuda.is_available()); '
         'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU")'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, False, None
    lines = result.stdout.strip().split('\n')
    if len(lines) < 3:
        return False, False, None
    version = lines[0]
    cuda_ok = lines[1].strip() == 'True'
    gpu_name = lines[2].strip()
    return True, cuda_ok, gpu_name


def main():
    print("="*60)
    print("  AUTO-LAUNCH: Waiting for PyTorch CUDA to be ready...")
    print("="*60)

    attempt = 0
    while True:
        attempt += 1
        torch_ok, cuda_ok, gpu_name = check_torch_cuda()

        if torch_ok and cuda_ok:
            print(f"\n✔  PyTorch is ready! GPU: {gpu_name}")
            break
        elif torch_ok and not cuda_ok:
            print(f"[{attempt}] PyTorch installed but CUDA not available. "
                  "Make sure CUDA-enabled torch was installed.")
            print("  Retrying in 30s... (or press Ctrl+C to abort)")
        else:
            print(f"[{attempt}] PyTorch not yet installed. Checking again in 30s...")

        time.sleep(30)

    print("\nLaunching full pipeline...\n")
    result = subprocess.run([sys.executable, 'run_pipeline.py'], check=False)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
