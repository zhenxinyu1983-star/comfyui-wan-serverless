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
    """Write base64 input files into ComfyUI's input directory."""
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    staged = []
    for item in images or []:
        filename = os.path.basename(item["filename"])  # prevent path traversal
        data = base64.b64decode(item["data"])
        dest = os.path.join(COMFYUI_INPUT_DIR, filename)
        with open(dest, "wb") as f:
            f.write(data)
        staged.append(filename)
    return staged


def submit_workflow(workflow):
    resp = _http("POST", "/prompt", {"prompt": workflow}, timeout=120)
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in /prompt response: {resp}")
    return prompt_id


def wait_for_result(prompt_id):
    """Poll /history until the prompt finishes or fails."""
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        # Check queue first for execution errors
        try:
            hist = _http("GET", f"/history/{prompt_id}", timeout=30)
        except Exception as e:
            time.sleep(POLL_INTERVAL)
            continue
        entry = hist.get(prompt_id)
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
    try:
        wait_for_comfyui()
        stage_input_files(job_input.get("images"))
        prompt_id = submit_workflow(workflow)
        media = wait_for_result(prompt_id)
        video_b64, filename = fetch_video(media)
        return {"status": "success", "video": video_b64,
                "filename": filename, "prompt_id": prompt_id}
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__,
                "error_message": str(e)[:2000], "prompt_id": prompt_id}


import urllib.parse  # noqa: E402  (kept at bottom to avoid shadowing in _http)

runpod.serverless.start({"handler": handler})
