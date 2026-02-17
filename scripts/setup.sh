#!/bin/bash
# Setup script for RenderFormer CUDA Engine
# Downloads model weights and configures build

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== RenderFormer CUDA Engine Setup ==="

# 1. Check dependencies
echo ""
echo "[1/4] Checking dependencies..."
missing=()
command -v cmake >/dev/null 2>&1 || missing+=("cmake (>= 3.24)")
command -v nvcc >/dev/null 2>&1 || missing+=("CUDA Toolkit (>= 12.0)")
pkg-config --exists hdf5 2>/dev/null || dpkg -l libhdf5-dev >/dev/null 2>&1 || missing+=("HDF5 C library")
pkg-config --exists zlib 2>/dev/null || missing+=("")  # usually present

if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing dependencies:"
    for dep in "${missing[@]}"; do
        [ -n "$dep" ] && echo "  - $dep"
    done
    echo ""
    echo "Install on Ubuntu/Debian:"
    echo "  sudo apt install cmake libhdf5-dev zlib1g-dev"
    echo "  # CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  # cuDNN: https://developer.nvidia.com/cudnn"
    echo ""
    echo "Install on Arch Linux:"
    echo "  sudo pacman -S cmake hdf5 zlib"
    echo ""
fi

# 2. Initialize submodules
echo "[2/4] Initializing submodules..."
cd "$ROOT_DIR"
git submodule update --init --recursive

# 3. Download model weights
MODEL_DIR="$ROOT_DIR/models"
SAFETENSORS="$MODEL_DIR/model.safetensors"
CONFIG="$MODEL_DIR/config.json"

echo "[3/4] Downloading model weights..."
if [ -f "$SAFETENSORS" ]; then
    echo "  Model already exists at $SAFETENSORS"
else
    mkdir -p "$MODEL_DIR"
    HF_REPO="microsoft/renderformer-v1.1-swin-large"

    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "  Using huggingface-cli to download $HF_REPO..."
        huggingface-cli download "$HF_REPO" model.safetensors config.json --local-dir "$MODEL_DIR"
    elif command -v wget >/dev/null 2>&1; then
        echo "  Downloading from HuggingFace Hub..."
        wget -q --show-progress -O "$SAFETENSORS" \
            "https://huggingface.co/$HF_REPO/resolve/main/model.safetensors"
        wget -q --show-progress -O "$CONFIG" \
            "https://huggingface.co/$HF_REPO/resolve/main/config.json"
    elif command -v curl >/dev/null 2>&1; then
        echo "  Downloading from HuggingFace Hub..."
        curl -L --progress-bar -o "$SAFETENSORS" \
            "https://huggingface.co/$HF_REPO/resolve/main/model.safetensors"
        curl -L --progress-bar -o "$CONFIG" \
            "https://huggingface.co/$HF_REPO/resolve/main/config.json"
    else
        echo "  ERROR: No download tool found (huggingface-cli, wget, or curl)"
        exit 1
    fi
    echo "  Model downloaded to $MODEL_DIR/"
fi

# 4. Configure and build
echo "[4/4] Building..."

# Detect GPU architecture
ARCH=86  # default: Ampere (RTX 3080 Ti, A100)
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$GPU_NAME" in
        *4090*|*4080*|*4070*|*4060*) ARCH=89 ;;  # Ada Lovelace
        *3090*|*3080*|*3070*|*3060*|*A100*) ARCH=86 ;;  # Ampere
        *A6000*|*A5000*) ARCH=86 ;;
        *H100*|*H200*) ARCH=90 ;;  # Hopper
        *5090*|*5080*|*5070*) ARCH=100 ;;  # Blackwell
    esac
    echo "  Detected GPU: $GPU_NAME (sm_$ARCH)"
fi

cmake -B "$ROOT_DIR/build" \
    -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
    -DCMAKE_BUILD_TYPE=Release \
    "$ROOT_DIR"

cmake --build "$ROOT_DIR/build" -j$(nproc)

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  ./build/renderformer --model models/model.safetensors --scene <scene.h5> --output <dir>"
echo ""
echo "Interactive viewer (if GLFW installed):"
echo "  ./build/renderformer_app --model models/model.safetensors --scene <scene.h5>"
