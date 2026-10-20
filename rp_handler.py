"""
RunPod Serverless handler for ComfyUI video generation.

Job input format (keeps the website's existing API contract):
{
  "workflow": { ... },        # ComfyUI API-format workflow JSON (same body as POST /prompt)
  "images": [                 # optional: files to stage into ComfyUI input/ before submit
    {"filename": "ref.png", "data": "<base64>"},
    ...
  ]
}

Job output on success:
{
  "status": "success",
  "video": "<base64 mp4>",
  "filename": "output_00001.mp4",
  "prompt_id": "..."
}

Job output on failure:
{
  "status": "error",
  "error_type": "...",
  "error_message": "...",
  "prompt_id": "..." | None
}
"""

import base64
import json
import os
import time
import urllib.request
import urllib.error

import runpod

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/ComfyUI/input")
COMFYUI_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", "/ComfyUI/output")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))
MAX_WAIT_SECONDS = int(os.environ.get("MAX_WAIT_SECONDS", "3600"))  # 60 min cap


def _http(method, path, data=None, timeout=60):
    url = COMFYUI_URL + path
    headers = {}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return json.loads(raw)
            return raw
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI HTTP {e.code} on {method} {path}: {e.read(300).decode(errors='replace')}")


def wait_for_comfyui(timeout=600):
    """Block until ComfyUI answers /system_stats (cold start warmup)."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            _http("GET", "/system_stats", timeout=10)
            return True
        except Exception as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"ComfyUI did not come up within {timeout}s: {last_err}")


def stage_input_files(images):
    """Write input files into ComfyUI's input directory.

    Items are {filename, data} (base64) or {filename, url} (presigned URL,
    used for files too large for the /run payload — downloads directly).
    """
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    staged = []
    for item in images or []:
        filename = os.path.basename(item["filename"])  # prevent path traversal
        dest = os.path.join(COMFYUI_INPUT_DIR, filename)
        if item.get("url"):
            _download_to_file(item["url"], dest)
        else:
            data = base64.b64decode(item["data"])
            with open(dest, "wb") as f:
                f.write(data)
        staged.append(filename)
    return staged


def _download_to_file(url, dest, timeout=600):
    """Stream-download url to dest (chunked, no full-file memory load)."""
    req = urllib.request.Request(url, headers={"User-Agent": "rp-handler/1.0"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def normalize_driving_video(workflow, filename, target_frames, fps=16):
    """Rewrite a staged driving video to exactly target_frames @ fps (CFR).

    Wan Animate's pose-conditioning latent length comes from however many
    frames VHS manages to load from the user's clip -- NOT from the
    workflow's num_frames. A real "exactly 10.0s" clip yields only 160
    frames at 16fps (then 157 after the Wan 4k+1 rule -> 40 latents) while
    the sampler builds 41 latents from num_frames=161, and predict_with_cfg
    dies on the size mismatch. Normalizing the file so VHS is guaranteed to
    deliver exactly target_frames makes both sides identical by
    construction, for any source fps / duration / frame count:
      - fps=<fps> resamples any source (incl. VFR phone clips) to CFR
      - tpad clones the last frame to pad short sources
      - -frames:v N hard-truncates long sources
    The workflow's references to the original filename are rewritten to
    the normalized file. Returns the new filename.
    """
    import subprocess

    src = os.path.join(COMFYUI_INPUT_DIR, os.path.basename(filename))
    if not os.path.exists(src):
        raise RuntimeError(f"normalize target not staged: {filename}")
    stem = os.path.splitext(os.path.basename(filename))[0]
    new_name = f"{stem}_norm.mp4"
    dest = os.path.join(COMFYUI_INPUT_DIR, new_name)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"fps={fps},tpad=stop_mode=clone:stop_duration=40",
        "-frames:v", str(int(target_frames)),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", dest,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(dest):
        tail = (r.stderr or "")[-800:]
        raise RuntimeError(f"ffmpeg normalize failed for {filename}: {tail}")
    # Point every workflow input that referenced the old file at the new one
    for node in (workflow or {}).values():
        inputs = node.get("inputs") or {}
        for key, val in inputs.items():
            if val == filename:
                inputs[key] = new_name
    return new_name


def submit_workflow(workflow):
    resp = _http("POST", "/prompt", {"prompt": workflow}, timeout=120)
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in /prompt response: {resp}")
    return prompt_id


def _prompt_in_queue(prompt_id):
    """True if ComfyUI's queue currently holds this prompt."""
    try:
        q = _http("GET", "/queue", timeout=15)
    except Exception:
        return False
    for key in ("queue_running", "queue_pending"):
        for item in q.get(key, []) or []:
            if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                return True
    return False


def wait_for_result(prompt_id):
    """Poll /history until the prompt finishes or fails.

    Liveness guards (a dead/restarted ComfyUI must not burn the full
    MAX_WAIT_SECONDS of billed worker time):
      - 6 consecutive /history failures -> probe /system_stats; if that
        also fails, ComfyUI is gone: raise immediately.
      - prompt absent from BOTH history and queue for >120s -> it was
        lost in a ComfyUI restart: raise.
    """
    deadline = time.time() + MAX_WAIT_SECONDS
    consec_errors = 0
    missing_since = None
    while time.time() < deadline:
        try:
            hist = _http("GET", f"/history/{prompt_id}", timeout=30)
            consec_errors = 0
        except Exception as e:
            consec_errors += 1
            if consec_errors >= 6:
                try:
                    _http("GET", "/system_stats", timeout=15)
                    consec_errors = 0  # ComfyUI alive; history endpoint flaky
                except Exception:
                    raise RuntimeError(
                        f"ComfyUI unreachable after {consec_errors} poll failures: {e}"
                    )
            time.sleep(POLL_INTERVAL)
            continue
        entry = hist.get(prompt_id)
        if not entry:
            if _prompt_in_queue(prompt_id):
                missing_since = None
            else:
                if missing_since is None:
                    missing_since = time.time()
                elif time.time() - missing_since > 120:
                    raise RuntimeError(
                        "Prompt vanished from history and queue (ComfyUI restarted?)"
                    )
            time.sleep(POLL_INTERVAL)
            continue
        missing_since = None
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                outputs = entry.get("outputs", {})
                # Find the first video output
                for node_id, node_out in outputs.items():
                    for media in node_out.get("gifs", []):
                        return media  # {filename, subfolder, type}
                raise RuntimeError(f"Prompt completed but no video output found: {list(outputs.keys())}")
            # Not completed yet — check for node errors surfaced in status
            msgs = status.get("messages", [])
            for m in msgs:
                if isinstance(m, (list, tuple)) and len(m) > 1 and "error" in str(m[0]).lower():
                    raise RuntimeError(f"ComfyUI execution error: {m[1]}")
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"Timed out waiting {MAX_WAIT_SECONDS}s for prompt {prompt_id}")


def fetch_video(media):
    """Download the rendered video via /view and base64-encode it."""
    filename = media["filename"]
    subfolder = media.get("subfolder", "")
    mtype = media.get("type", "output")
    qs = f"filename={urllib.parse.quote(filename)}&subfolder={urllib.parse.quote(subfolder)}&type={urllib.parse.quote(mtype)}"
    raw = _http("GET", f"/view?{qs}", timeout=300)
    if not isinstance(raw, bytes) or len(raw) == 0:
        raise RuntimeError(f"Empty video data for {filename}")
    return base64.b64encode(raw).decode(), filename


def run_debug(command):
    """Run a shell command for debugging and return its output."""
    import subprocess
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return {
            "status": "debug",
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
        }
    except Exception as e:
        return {
            "status": "debug",
            "command": command,
            "error": str(e)[:1000],
        }


def _cleanup_job_files(staged_names, media):
    """Delete this job's staged inputs and rendered output from local disk.

    Serverless workers are reused across jobs and the endpoint disk is
    small (10GB); without cleanup, inputs/outputs accumulate until the
    disk fills and every job starts failing. Best-effort: never raises.
    """
    try:
        for name in staged_names or []:
            for cand in (name, os.path.splitext(name)[0] + "_norm.mp4"):
                try:
                    os.remove(os.path.join(COMFYUI_INPUT_DIR, cand))
                except OSError:
                    pass
        if media:
            out_path = os.path.join(
                COMFYUI_OUTPUT_DIR, media.get("subfolder", ""), media.get("filename", "")
            )
            try:
                os.remove(out_path)
            except OSError:
                pass
    except Exception:
        pass


def _sweep_stale_files(max_age_sec=6 * 3600):
    """Drop input/output files older than max_age_sec (leftovers of crashed jobs)."""
    import time as _time

    for d in (COMFYUI_INPUT_DIR, COMFYUI_OUTPUT_DIR):
        try:
            for name in os.listdir(d):
                fp = os.path.join(d, name)
                try:
                    if os.path.isfile(fp) and _time.time() - os.path.getmtime(fp) > max_age_sec:
                        os.remove(fp)
                except OSError:
                    pass
        except OSError:
            pass


def handler(job):
    job_input = job.get("input", {})

    # Debug mode: run a shell command and return output
    if "debug" in job_input:
        return run_debug(job_input["debug"])

    workflow = job_input.get("workflow")
    if not workflow:
        return {"status": "error", "error_type": "bad_input",
                "error_message": "Missing 'workflow' in job input", "prompt_id": None}

    prompt_id = None
    staged = []
    media = None
    try:
        _sweep_stale_files()
        wait_for_comfyui()
        staged = stage_input_files(job_input.get("images"))
        nv = job_input.get("normalize_video")
        if nv:
            normalize_driving_video(workflow, nv["filename"], nv["frames"], nv.get("fps", 16))
        prompt_id = submit_workflow(workflow)
        media = wait_for_result(prompt_id)
        video_b64, filename = fetch_video(media)
        return {"status": "success", "video": video_b64,
                "filename": filename, "prompt_id": prompt_id}
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__,
                "error_message": str(e)[:2000], "prompt_id": prompt_id}
    finally:
        _cleanup_job_files(staged, media)


import urllib.parse  # noqa: E402  (kept at bottom to avoid shadowing in _http)

runpod.serverless.start({"handler": handler})
