#!/bin/bash
# Serverless worker startup:
#  1. Adapt the network volume (/runpod-volume) to the /workspace paths the template expects
#  2. Start ComfyUI in the background
#  3. Start the RunPod handler in the foreground
set -e

echo "[start] === Serverless worker starting ==="

# --- 1. Volume adaptation ---
# Template expects models at /workspace/ComfyUI/models/...
# Serverless mounts the network volume at /runpod-volume (fixed path).
if [ -d "/runpod-volume/ComfyUI/models" ]; then
    echo "[start] Found models on /runpod-volume, linking to /workspace"
    mkdir -p /workspace
    if [ ! -e "/workspace/ComfyUI" ]; then
        ln -s /runpod-volume/ComfyUI /workspace/ComfyUI
        echo "[start] Linked /workspace/ComfyUI -> /runpod-volume/ComfyUI"
    else
        echo "[start] /workspace/ComfyUI already exists, skipping link"
    fi
    # Sanity check: list model dirs (no downloads here — never overwrite existing models)
    echo "[start] Model dirs on volume:"
    ls /runpod-volume/ComfyUI/models/ 2>/dev/null || echo "[start] WARNING: models dir empty or unreadable"
else
    echo "[start] WARNING: /runpod-volume/ComfyUI/models not found."
    echo "[start] Expected the network volume with pre-staged models. Continuing anyway;"
    echo "[start] ComfyUI will fail on missing models if the volume is not attached."
fi

# ComfyUI input/output dirs must exist and be writable
mkdir -p /ComfyUI/input /ComfyUI/output 2>/dev/null || true

# --- 2. Start ComfyUI in background ---
# The template's default command starts ComfyUI; we replicate the essential part.
# If the image has its own start script, prefer it.
COMFYUI_CMD=""
if [ -f "/ComfyUI/main.py" ]; then
    COMFYUI_CMD="python3 /ComfyUI/main.py --listen 0.0.0.0 --port 8188 --enable-cors-header '*'"
elif [ -f "/workspace/ComfyUI/main.py" ]; then
    COMFYUI_CMD="python3 /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --enable-cors-header '*'"
fi

if [ -z "$COMFYUI_CMD" ]; then
    echo "[start] ERROR: ComfyUI main.py not found at /ComfyUI or /workspace/ComfyUI"
    exit 1
fi

echo "[start] Starting ComfyUI: $COMFYUI_CMD"
nohup $COMFYUI_CMD > /tmp/comfyui.log 2>&1 &
echo "[start] ComfyUI starting in background (log: /tmp/comfyui.log)"

# --- 3. Start the RunPod handler (foreground, keeps container alive) ---
# The handler waits for ComfyUI itself (wait_for_comfyui) before taking jobs.
echo "[start] Starting RunPod handler"
exec python3 /rp_handler.py
