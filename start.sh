#!/bin/bash
# Serverless worker startup:
#  1. Adapt the network volume (/runpod-volume) to the paths ComfyUI expects
#  2. Start ComfyUI in the background
#  3. Start the RunPod handler in the foreground
set -e

echo "[start] === Serverless worker starting ==="

# --- 1. Volume adaptation ---
# The template runs ComfyUI from /ComfyUI, which reads models at /ComfyUI/models.
# Serverless mounts the network volume at /runpod-volume (fixed path).
# We symlink /ComfyUI/models -> /runpod-volume/ComfyUI/models (never copy, never overwrite).
if [ -d "/runpod-volume/ComfyUI/models" ]; then
    echo "[start] Found models on /runpod-volume, linking into /ComfyUI"
    if [ -d "/ComfyUI/models" ] && [ ! -L "/ComfyUI/models" ]; then
        rmdir /ComfyUI/models 2>/dev/null || {
            echo "[start] WARNING: /ComfyUI/models not empty, cannot replace. Listing:"
            ls /ComfyUI/models/ | head -5
        }
    fi
    if [ ! -e "/ComfyUI/models" ]; then
        ln -s /runpod-volume/ComfyUI/models /ComfyUI/models
        echo "[start] Linked /ComfyUI/models -> /runpod-volume/ComfyUI/models"
    else
        echo "[start] /ComfyUI/models already exists, skipping link"
    fi
    for d in input output; do
        if [ ! -e "/ComfyUI/$d" ]; then
            mkdir -p /runpod-volume/ComfyUI/$d
            ln -s /runpod-volume/ComfyUI/$d /ComfyUI/$d
            echo "[start] Linked /ComfyUI/$d -> /runpod-volume/ComfyUI/$d"
        fi
    done
    echo "[start] Model dirs on volume:"
    ls /runpod-volume/ComfyUI/models/ 2>/dev/null || echo "[start] WARNING: models dir empty or unreadable"
else
    echo "[start] WARNING: /runpod-volume/ComfyUI/models not found."
    echo "[start] Expected the network volume with pre-staged models. Continuing anyway;"
    echo "[start] ComfyUI will fail on missing models if the volume is not attached."
fi

mkdir -p /ComfyUI/input /ComfyUI/output 2>/dev/null || true

# --- 2. Start ComfyUI in background ---
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
echo "[start] Starting RunPod handler"
exec python3 /rp_handler.py
