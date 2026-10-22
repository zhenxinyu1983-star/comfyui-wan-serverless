#!/bin/bash
# Serverless worker startup:
#  1. Adapt the network volume (/runpod-volume) to the /workspace paths the template expects
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
    # Move the template's built-in models dir aside (if not already a link), then symlink to volume
    if [ -d "/ComfyUI/models" ] && [ ! -L "/ComfyUI/models" ]; then
        mv /ComfyUI/models /ComfyUI/models.template_backup
        echo "[start] Moved template models to /ComfyUI/models.template_backup"
    fi
    if [ ! -e "/ComfyUI/models" ]; then
        ln -s /runpod-volume/ComfyUI/models /ComfyUI/models
        echo "[start] Linked /ComfyUI/models -> /runpod-volume/ComfyUI/models"
    else
        echo "[start] /ComfyUI/models already exists, skipping link"
    fi
    # Also link input/output so uploaded files and results persist on the volume
    for d in input output; do
        if [ ! -e "/ComfyUI/$d" ]; then
            mkdir -p /runpod-volume/ComfyUI/$d
            ln -s /runpod-volume/ComfyUI/$d /ComfyUI/$d
            echo "[start] Linked /ComfyUI/$d -> /runpod-volume/ComfyUI/$d"
        fi
    done
    # Sanity check: list model dirs (no downloads here — never overwrite existing models)
    echo "[start] Model dirs on volume:"
    ls /runpod-volume/ComfyUI/models/ 2>/dev/null || echo "[start] WARNING: models dir empty or unreadable"
else
    echo "[start] WARNING: /runpod-volume/ComfyUI/models not found."
    echo "[start] Expected the network volume with pre-staged models. Continuing anyway;"
    echo "[start] ComfyUI will fail on missing models if the volume is not attached."
fi

# --- RIFE weights for ComfyUI-Frame-Interpolation (1080P project, prepared 2026-10-21) ---
# The extension loads weights from its own ckpts/rife/ dir (its config.yaml: ckpts_path:./ckpts),
# NOT from ComfyUI/models. The weight lives on the volume; link it in before ComfyUI starts.
RIFE_SRC=""
for cand in /runpod-volume/rife_ckpts/rife49.pth /runpod-volume/ComfyUI/models/rife_ckpts/rife49.pth; do
    if [ -f "$cand" ]; then RIFE_SRC="$cand"; break; fi
done
RIFE_DIR=/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife
if [ -n "$RIFE_SRC" ] && [ -d /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation ]; then
    mkdir -p "$RIFE_DIR" || echo "[start] NOTE: could not create RIFE ckpts dir (continuing)"
    if [ ! -e "$RIFE_DIR/rife49.pth" ]; then
        ln -s "$RIFE_SRC" "$RIFE_DIR/rife49.pth" || echo "[start] NOTE: RIFE link failed (continuing)"
        echo "[start] Linked RIFE weight $RIFE_SRC -> $RIFE_DIR/rife49.pth"
    fi
else
    echo "[start] NOTE: RIFE weight not found on volume (1080P interpolation unavailable, rest unaffected)"
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
