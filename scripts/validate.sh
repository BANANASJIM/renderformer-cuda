#!/bin/bash
# Build CUDA engine and validate render MSE against Python reference
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$CUDA_DIR")"

MODEL="$HOME/.cache/huggingface/hub/models--microsoft--renderformer-v1.1-swin-large/snapshots/d45d4531548a809ab8e5eb474cc7d89be3f90b34/model.safetensors"
SCENE="$REPO_DIR/tmp/cbox/cbox.h5"
REF="$REPO_DIR/output/ref/cbox_view_0.exr"
OUT="/tmp/cuda_validate_$$"
PSNR_THRESHOLD=${PSNR_THRESHOLD:-15}

if [ ! -f "$MODEL" ]; then echo "Model not found: $MODEL"; exit 1; fi
if [ ! -f "$SCENE" ]; then echo "Scene not found: $SCENE"; exit 1; fi
if [ ! -f "$REF" ]; then echo "Reference not found: $REF"; exit 1; fi

echo "Building..."
cmake --build "$CUDA_DIR/build" -j$(nproc) 2>&1 | tail -1

echo "Rendering..."
"$CUDA_DIR/build/renderformer" --model "$MODEL" --scene "$SCENE" --output "$OUT" 2>&1 | grep -E "(Timing|TOTAL|Done)"

echo "Validating..."
python3 -c "
import numpy as np, imageio.v3 as iio, sys
ref = iio.imread('$REF')
cuda = iio.imread('$OUT/view_0.exr')
mse = np.mean((ref - cuda)**2)
psnr = 10 * np.log10(ref.max()**2 / mse) if mse > 0 else float('inf')
print(f'MSE: {mse:.6f}  PSNR: {psnr:.2f} dB  (threshold: $PSNR_THRESHOLD dB)')
if psnr < $PSNR_THRESHOLD:
    print('FAIL: PSNR below threshold')
    sys.exit(1)
print('PASS')
"

rm -rf "$OUT"
