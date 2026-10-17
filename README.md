# ComfyUI Wan Serverless Worker

RunPod Serverless worker for the AI video service. Reuses the tested
`hearmeman/comfyui-wan-template:v29` environment; models come from the
existing 300GB network volume (mounted at `/runpod-volume` on serverless).

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | `FROM hearmeman/comfyui-wan-template:v29` + `runpod` SDK + handler |
| `rp_handler.py` | RunPod handler: job → ComfyUI → video (base64) or error |
| `start.sh` | Symlinks `/runpod-volume/ComfyUI` → `/workspace/ComfyUI`, starts ComfyUI + handler |
| `build.sh` | Build + push to Docker Hub |

## Job API (unchanged from the pod)

Input:
```json
{
  "input": {
    "workflow": { "...": "ComfyUI API-format workflow JSON (same as POST /prompt body)" },
    "images": [
      {"filename": "ref.png", "data": "<base64>"}
    ]
  }
}
```

Output (success):
```json
{"status": "success", "video": "<base64 mp4>", "filename": "...", "prompt_id": "..."}
```

Output (error):
```json
{"status": "error", "error_type": "...", "error_message": "...", "prompt_id": "..."}
```

## Build & push (on a machine with Docker)

```bash
cd serverless-worker
chmod +x build.sh start.sh
./build.sh <your-dockerhub-username> v1
```

## Endpoint setup (RunPod console)

1. Serverless → New Endpoint
2. Image: `docker.io/<username>/comfyui-wan-serverless:v1`
3. GPU: 1× RTX 4090 (or 24GB+ class)
4. Network volume: attach `k0oyzqob81` (video-models, 300GB) — same datacenter as the endpoint
5. Workers: min 0 (scale-to-zero), max 3
6. Container disk: 30GB is enough (models are on the volume, not in the image)

## Notes

- Models are **never** baked into the image and never re-downloaded:
  `start.sh` only symlinks the volume. If `/runpod-volume/ComfyUI/models`
  is missing, the worker logs a warning and jobs will fail on missing models.
- Cold start ≈ image pull + ComfyUI init (a few minutes). First job per worker pays it.
- One model at a time in VRAM (55GB pod RAM lesson applies to workers too).
