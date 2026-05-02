#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
#  setup.sh — install dependencies for helmet-detection inference on Pi5
#
#  Tested on:  Raspberry Pi 5 (8 GB) running Raspberry Pi OS Bookworm 64-bit
#
#  What this does:
#    1. Installs system libraries needed by OpenCV at runtime
#    2. Creates a Python virtual environment at ./venv
#    3. Installs minimal Python deps (no PyTorch — we use ONNX Runtime)
#    4. Optionally exports best.pt → best.onnx if you have best.pt here
#
#  After this finishes:
#    source venv/bin/activate
#    python inference.py --mode server --model best.onnx
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "════════════════════════════════════════════════════════════════"
echo "  Helmet-detection Pi5 setup"
echo "════════════════════════════════════════════════════════════════"

# ── 0. sanity check ──────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] python3 not found. Install it first: sudo apt install python3"
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[*] python3 = $PY_VER"

# ── 1. system packages (need sudo) ───────────────────────────────────────
echo ""
echo "[1/4] Installing system libraries (sudo required)…"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    libopenblas0

# ── 2. Python virtualenv ────────────────────────────────────────────────
echo ""
echo "[2/4] Creating virtualenv at ./venv …"
if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ── 3. Python packages ──────────────────────────────────────────────────
echo ""
echo "[3/4] Installing Python packages (this may take a few minutes)…"
# Pin versions known to ship ARM64 wheels on PyPI / piwheels.
pip install --no-cache-dir \
    "numpy>=1.24,<2.2" \
    "opencv-python==4.10.0.84" \
    "onnxruntime>=1.17"

echo ""
echo "[*] Installed:"
python -c "import numpy, cv2, onnxruntime as ort; \
print(f'    numpy        {numpy.__version__}'); \
print(f'    opencv       {cv2.__version__}'); \
print(f'    onnxruntime  {ort.__version__}')"

# ── 4. ONNX export helper ───────────────────────────────────────────────
echo ""
echo "[4/4] Checking for model file…"
if [ -f best.onnx ]; then
    echo "[*] best.onnx already present — ready to run."
elif [ -f best.pt ]; then
    echo "[*] Found best.pt but no best.onnx."
    echo ""
    echo "    The Pi5 setup intentionally avoids PyTorch (it is large and slow"
    echo "    to install). Export the ONNX model on your laptop instead:"
    echo ""
    echo "        pip install ultralytics"
    echo "        yolo export model=best.pt format=onnx imgsz=640 opset=12"
    echo ""
    echo "    …or, using the original yolov5 repo:"
    echo ""
    echo "        git clone https://github.com/ultralytics/yolov5"
    echo "        cd yolov5"
    echo "        pip install -r requirements.txt onnx"
    echo "        python export.py --weights ../best.pt --include onnx --imgsz 640"
    echo ""
    echo "    Then copy best.onnx to this Pi5 (e.g. via scp)."
else
    echo "[!] No best.pt or best.onnx found in this directory."
    echo "    Copy your trained model here before running inference.py."
fi

# ── done ────────────────────────────────────────────────────────────────
PI_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  This Pi5 IP appears to be:  ${PI_IP:-<unknown>}"
echo ""
echo "  To run the inference server:"
echo "      source venv/bin/activate"
echo "      python inference.py --mode server --model best.onnx --port 5555"
echo ""
echo "  Then on the laptop (same network):"
echo "      python inference.py --mode client --host ${PI_IP:-<PI5_IP>} --port 5555 --source 0"
echo "════════════════════════════════════════════════════════════════"
