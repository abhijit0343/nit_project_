"""
install_torch_cuda.py — Robust PyTorch CUDA installer with resume support.

Downloads the PyTorch CUDA wheel with a progress bar and resume capability,
then installs it. Much more resilient than raw pip for large files on slow connections.

Usage:
    python install_torch_cuda.py
"""
import os
import sys
import urllib.request
import subprocess
import time

WHEEL_URL = "https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl"
WHEEL_FILE = "torch-2.5.1+cu124-cp311-cp311-win_amd64.whl"
TOTAL_SIZE_GB = 2.51

TORCHVISION_URL = "https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp311-cp311-win_amd64.whl"
TORCHVISION_FILE = "torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl"

TORCHAUDIO_URL = "https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl"
TORCHAUDIO_FILE = "torchaudio-2.5.1+cu124-cp311-cp311-win_amd64.whl"


def download_with_resume(url: str, dest: str, description: str = ""):
    """Download a file with resume support and a progress bar."""
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0

    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header('Range', f'bytes={existing}-')
        print(f"Resuming '{description}' from {existing/1e6:.1f} MB...")
    else:
        print(f"Starting download: '{description}'")

    try:
        response = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        print(f"[ERROR] Could not connect: {e}")
        return False

    total = int(response.headers.get('Content-Length', 0)) + existing
    mode = 'ab' if existing > 0 else 'wb'

    downloaded = existing
    start = time.time()
    last_print = time.time()

    try:
        with open(dest, mode) as f:
            while True:
                chunk = response.read(1024 * 512)   # 512 KB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 5:   # print every 5 seconds
                    pct = downloaded / total * 100 if total else 0
                    speed = (downloaded - existing) / (now - start) / 1e6
                    eta_s = (total - downloaded) / (speed * 1e6) if speed > 0 else 0
                    bar_len = 30
                    filled = int(bar_len * downloaded / total) if total else 0
                    bar = '█' * filled + '░' * (bar_len - filled)
                    print(f"  [{bar}] {pct:.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB  "
                          f"{speed:.2f} MB/s  ETA: {int(eta_s//60)}m {int(eta_s%60)}s",
                          flush=True)
                    last_print = now
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Partial file saved to '{dest}'. Re-run to resume.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Download interrupted: {e}")
        print(f"Partial file saved. Re-run script to resume.")
        return False

    print(f"\n✔  Download complete: {dest}  ({downloaded/1e9:.2f} GB)")
    return True


def install_wheel(path: str):
    print(f"\nInstalling {path}...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', path, '--no-deps'],
        check=False
    )
    if result.returncode != 0:
        print(f"[ERROR] pip install failed for {path}")
        return False
    return True


def verify_cuda():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; print(torch.__version__); '
         'print("CUDA:", torch.cuda.is_available()); '
         'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return 'True' in result.stdout


def main():
    print("="*60)
    print("  PyTorch CUDA Installer (with resume support)")
    print("="*60)

    # Check if already installed
    check = subprocess.run(
        [sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'],
        capture_output=True, text=True
    )
    if check.returncode == 0 and 'True' in check.stdout:
        print("✔  PyTorch CUDA already installed!")
        verify_cuda()
        return

    wheels = [
        (WHEEL_URL, WHEEL_FILE, "torch 2.5.1+cu124 (2.51 GB)"),
        (TORCHVISION_URL, TORCHVISION_FILE, "torchvision 0.20.1+cu124"),
        (TORCHAUDIO_URL, TORCHAUDIO_FILE, "torchaudio 2.5.1+cu124"),
    ]

    for url, filename, desc in wheels:
        print(f"\n{'─'*60}")
        print(f"  {desc}")
        print(f"{'─'*60}")
        while True:
            ok = download_with_resume(url, filename, desc)
            if ok:
                break
            print("Retrying in 10 seconds...")
            time.sleep(10)

        if not install_wheel(filename):
            print(f"[FATAL] Could not install {filename}. Exiting.")
            sys.exit(1)
        # Clean up wheel after install to save disk space
        os.remove(filename)
        print(f"  Cleaned up {filename}")

    # Install remaining deps that pip normally resolves
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'filelock', 'typing-extensions', 'networkx', 'jinja2',
                    'fsspec', 'sympy', 'mpmath'], check=False)

    print("\n" + "="*60)
    print("  Verifying CUDA installation...")
    print("="*60)
    if verify_cuda():
        print("\n🎉  PyTorch CUDA installed successfully!")
        print("    Run:  python run_pipeline.py")
    else:
        print("\n[ERROR] CUDA verification failed.")


if __name__ == '__main__':
    main()
