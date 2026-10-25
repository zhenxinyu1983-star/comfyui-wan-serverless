"""WanCameraOrbitEmbedding — Orbit (arc/orbit) camera trajectory for Wan 2.2
Fun Camera control, as a drop-in replacement for the native WanCameraEmbedding
node (same WAN_CAMERA_EMBEDDING output type and packing).

Why this exists: the native node's 9 presets contain no orbit — every preset is
a pure translation (+roll); there is no yaw anywhere. This node generates a
true per-frame orbit trajectory (camera travels on a circular arc around the
subject point while looking at it) and feeds it through the SAME downstream
function the native node uses (comfy_extras.nodes_camera_trajectory.
process_pose_params), so relativization and Plücker ray computation are the
official ones. No ComfyUI source is modified and nothing is monkey-patched.

Entry layout follows the NATIVE side convention exactly (23 values per frame):
    [0, fx, fy, cx, cy, 0, 0, <4x4 c2w row-major, 16 values>]
WARNING: the kijai wrapper's fun_camera entries use a different layout
(entry[7:] = 3x4 w2c). Mixing the two mirrors the motion. Native layout only.

Geometry (OpenCV convention: x right, y down, z forward; world frame = the
frame-0 camera frame, matching the native relativization):
  - Orbit center O = (center_x * 1.5, -center_y * 1.5, r0) with
    r0 = orbit_radius * 1.5 (1.5 = CAMERA_DICT base_T_norm, same unit as the
    native presets' translations). O is FIXED across frames: the subject
    point must not swim while the radius varies. center_x/center_y default
    0, so O = (0, 0, r0) exactly as in v1.
  - Frame i of n, u = i/(n-1) (u = 0 for n == 1; u reaches 1 exactly on the
    last frame, unlike the native presets' i/n convention):
      theta_i = DIR_SIGN[direction] * (sweep_start + intensity
                * radians(orbit_degrees) * easing(u))
    sweep_start is the start angle in degrees (default 0); the end angle is
    derived, not a second input: ComfyUI FLOAT defaults cannot follow another
    input, so an angle_end default would silently break orbit_degrees. Any
    start->end sweep is reachable: end >= start uses ccw with
    (sweep_start=start, degrees=end-start); end < start uses cw with
    (sweep_start=-start, degrees=start-end).
  - Radius/height vary LINEARLY in u (not eased — the path stays
    predictable): r_i = (orbit_radius + radius_delta * u) * 1.5,
    h_i = (orbit_height + height_delta * u) * 1.5. Deltas default 0, so
    constant radius/height is the default. radius_delta/height_delta are
    deltas (not end values) for the same default-following reason as above.
  - Camera center C_i = (O_x - r_i*sin(theta_i), O_y - h_i,
    O_z - r_i*cos(theta_i)).
  - Orientation is lookAt(O): z = normalize(O - C_i), x = normalize(z x up),
    y = z x x, with up = (0, -1, 0). For h = 0 and zero offsets this reduces
    exactly to R = Ry(theta_i) with the translation above.
  - speed multiplies the whole sweep (legacy tempo knob); intensity is a
    separate CALIBRATION multiplier on the sweep only (the start angle is
    never scaled). intensity does NOT promise degree precision: the model
    executes poses approximately, so the calibrated value must come from
    measured singles (future batch), never from this node.

Backward compatibility: all new parameters are optional with defaults that
reproduce the v1 formula EXACTLY (linear easing, sweep_start 0, offsets 0,
deltas 0, intensity 1). The local selftest asserts default-output identity
against the v1 formula. Static-equivalence (degrees=0 == native Static after
relativization) additionally requires sweep_start 0 and deltas 0; a center
offset cancels in relativization, a radius/height delta does not.

v1 caps (manager ruling 2026-10-20): orbit_degrees <= 90, orbit_radius <= 1.5.
New-parameter ranges are provisional for manager review: sweep_start -90..90,
center_x -1..1, center_y -0.5..0.5, radius_delta/height_delta -0.5..0.5,
intensity 0.1..5.0. Effective sweep cap (manager ruling 2026-10-25):
intensity x speed x degrees <= 180, else fail-fast (ValueError); the cap
governs sweep magnitude, not start/end absolute angles, so a sweep_start of
±90 is allowed regardless of the sweep.
DIR_SIGN is PROVISIONAL: which sign reads as "cw" from the viewer's side is
settled by the paired-direction 5s GPU test; if reversed, flip ONLY this map.

The same package also carries TSTSaveLastFramePNG (Bot suggestion #7, PNG
half only — RIFE interpolation is deferred by manager review): it saves the
last frame of an image batch as a PNG at save time, for chaining. It is a
generic save node, not orbit-specific, but hangs on this package per the
batch order (one package change, not four).
"""

import math

import numpy as np

# Provisional direction signs — see module docstring. Flip ONLY this mapping
# if the GPU test shows the parallax running opposite to the chosen label.
DIR_SIGN = {"ccw": 1.0, "cw": -1.0}

BASE_T_NORM = 1.5  # CAMERA_DICT["base_T_norm"] in nodes_camera_trajectory.py


def _ease(u, easing):
    """Easing curves on u in [0, 1]. All pass through (0, 0) and (1, 1)."""
    if easing == "linear":
        return u
    if easing == "ease_in":
        return u * u
    if easing == "ease_out":
        return 1.0 - (1.0 - u) ** 2
    if easing == "ease_in_out":
        return u * u * (3.0 - 2.0 * u)
    raise ValueError(
        "unknown easing %r (expected one of: linear, ease_in, ease_out, "
        "ease_in_out)" % (easing,))


def orbit_entries(n, orbit_degrees, orbit_direction, orbit_radius,
                  orbit_height, speed=1.0, fx=0.5, fy=0.5, cx=0.5, cy=0.5,
                  easing="linear", sweep_start=0.0,
                  center_x=0.0, center_y=0.0,
                  radius_delta=0.0, height_delta=0.0, intensity=1.0):
    """Build the (n, 23) native-layout per-frame pose entries for an orbit.

    New parameters are all optional and default to the v1 behavior; see the
    module docstring for the geometry and the backward-compatibility proof.
    """
    if n < 1:
        raise ValueError("length must be >= 1")
    if float(orbit_radius) <= 0:
        raise ValueError("orbit_radius must be > 0")
    if float(orbit_radius) + float(radius_delta) <= 0:
        raise ValueError("orbit_radius + radius_delta must be > 0")
    if float(intensity) <= 0:
        raise ValueError("intensity must be > 0")
    # Effective sweep cap (manager ruling 2026-10-25): the cap governs
    # sweep magnitude, not start/end absolute angles, so sweep_start does
    # not count toward it.
    sweep_deg = float(intensity) * float(speed) * float(orbit_degrees)
    if sweep_deg > 180.0:
        raise ValueError(
            "effective sweep (intensity x speed x degrees) is %.1f deg, "
            "exceeding the 180 deg cap" % sweep_deg)
    sign = DIR_SIGN[orbit_direction]
    total = math.radians(orbit_degrees) * float(speed)
    start = math.radians(sweep_start)
    r0 = float(orbit_radius) * BASE_T_NORM
    # Fixed orbit center (see module docstring: never swims with radius).
    center = np.array([float(center_x) * BASE_T_NORM,
                       -float(center_y) * BASE_T_NORM,
                       r0], dtype=np.float64)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    entries = []
    for i in range(n):
        u = (i / (n - 1)) if n > 1 else 0.0
        theta = sign * (start + float(intensity) * total * _ease(u, easing))
        r = (float(orbit_radius) + float(radius_delta) * u) * BASE_T_NORM
        h = (float(orbit_height) + float(height_delta) * u) * BASE_T_NORM
        cam_c = np.array([center[0] - r * math.sin(theta),
                          center[1] - h,
                          center[2] - r * math.cos(theta)],
                         dtype=np.float64)
        z_axis = center - cam_c
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, up)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        c2w = np.zeros((4, 4), dtype=np.float64)
        c2w[0:3, 0] = x_axis
        c2w[0:3, 1] = y_axis
        c2w[0:3, 2] = z_axis
        c2w[0:3, 3] = cam_c
        c2w[3, 3] = 1.0
        entry = [0.0, fx, fy, cx, cy, 0.0, 0.0]
        entry.extend(float(v) for v in c2w.reshape(-1))
        entries.append(entry)
    return np.array(entries, dtype=np.float64)


def _pack_embedding(cam_params, width, height):
    """process_pose_params + the native node's exact packing steps.

    Imports process_pose_params from the host ComfyUI so the relativization
    and Plücker math are byte-identical to the native node's. The packing
    below mirrors WanCameraEmbedding.execute verbatim (permute, first-frame
    repeat x4, temporal grouping by 4 -> 24 channels).
    """
    import torch
    from comfy_extras.nodes_camera_trajectory import process_pose_params
    import comfy.model_management

    control_camera_video = process_pose_params(
        cam_params, width=width, height=height)
    control_camera_video = control_camera_video.permute([3, 0, 1, 2])
    control_camera_video = control_camera_video.unsqueeze(0).to(
        device=comfy.model_management.intermediate_device())
    control_camera_video = torch.concat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1],
                                    repeats=4, dim=2),
            control_camera_video[:, :, 1:],
        ], dim=2
    ).transpose(1, 2)
    b, f, c, h, w = control_camera_video.shape
    control_camera_video = control_camera_video.contiguous().view(
        b, f // 4, 4, c, h, w).transpose(2, 3)
    control_camera_video = control_camera_video.contiguous().view(
        b, f // 4, c * 4, h, w).transpose(1, 2)
    return control_camera_video


class WanCameraOrbitEmbedding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "step": 4}),
                "orbit_degrees": ("FLOAT", {"default": 90.0, "min": 0.0,
                                            "max": 90.0, "step": 1.0}),
                "orbit_direction": (["ccw", "cw"], {"default": "ccw"}),
                "orbit_radius": ("FLOAT", {"default": 1.0, "min": 0.5,
                                           "max": 1.5, "step": 0.05}),
                "orbit_height": ("FLOAT", {"default": 0.0, "min": -0.5,
                                           "max": 0.5, "step": 0.05}),
            },
            "optional": {
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0,
                                    "step": 0.1}),
                "fx": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "fy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "cx": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "cy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                # Batch extensions (Bot suggestions #3/#4/#5/#2): all default
                # to v1 behavior, so old workflows load unchanged.
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out"],
                           {"default": "linear"}),
                "sweep_start": ("FLOAT", {"default": 0.0, "min": -90.0,
                                          "max": 90.0, "step": 1.0}),
                "center_x": ("FLOAT", {"default": 0.0, "min": -1.0,
                                       "max": 1.0, "step": 0.05}),
                "center_y": ("FLOAT", {"default": 0.0, "min": -0.5,
                                       "max": 0.5, "step": 0.05}),
                "radius_delta": ("FLOAT", {"default": 0.0, "min": -0.5,
                                           "max": 0.5, "step": 0.05}),
                "height_delta": ("FLOAT", {"default": 0.0, "min": -0.5,
                                           "max": 0.5, "step": 0.05}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.1,
                                        "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("WAN_CAMERA_EMBEDDING", "INT", "INT", "INT")
    RETURN_NAMES = ("camera_embedding", "width", "height", "length")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/wan/camera"

    def execute(self, width, height, length, orbit_degrees, orbit_direction,
                orbit_radius, orbit_height, speed=1.0, fx=0.5, fy=0.5,
                cx=0.5, cy=0.5, easing="linear", sweep_start=0.0,
                center_x=0.0, center_y=0.0, radius_delta=0.0,
                height_delta=0.0, intensity=1.0):
        cam_params = orbit_entries(
            length, orbit_degrees, orbit_direction, orbit_radius,
            orbit_height, speed=speed, fx=fx, fy=fy, cx=cx, cy=cy,
            easing=easing, sweep_start=sweep_start,
            center_x=center_x, center_y=center_y,
            radius_delta=radius_delta, height_delta=height_delta,
            intensity=intensity)
        cam_params = np.array([[float(x) for x in pose] for pose in cam_params])
        embedding = _pack_embedding(cam_params, width, height)
        return (embedding, width, height, length)


def last_frame_array(images):
    """Last frame of an image batch as a uint8 HWC numpy array (0-255).

    Accepts a ComfyUI IMAGE tensor (B, H, W, C float 0-1) or a numpy array
    (float 0-1 or uint8). Pure numpy/PIL: locally testable without ComfyUI.
    """
    if hasattr(images, "cpu"):  # torch tensor
        arr = images.cpu().numpy()
    else:
        arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError("images must be a 4D batch (B, H, W, C)")
    frame = arr[-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def save_last_frame_png(images, directory, filename_prefix):
    """Save the last frame as a PNG, SaveImage-style counter suffix.

    Returns the full path. Only the file writing needs the output
    directory; the frame selection/encoding above is locally testable.
    """
    import os

    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    counter = 1
    while True:
        path = os.path.join(
            directory, "%s_%05d_.png" % (filename_prefix, counter))
        if not os.path.exists(path):
            break
        counter += 1
    Image.fromarray(last_frame_array(images)).save(path)
    return path


class TSTSaveLastFramePNG:
    """Save the last frame of a video/image batch as a PNG (Bot #7, PNG half).

    For chaining: the saved PNG is the next segment's start frame without a
    separate extraction step. OUTPUT_NODE so the save runs even when the
    outputs are not consumed downstream. The last frame is also returned as
    a 1-frame IMAGE batch plus the saved path string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING",
                                   {"default": "orbit_lastframe"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("last_frame", "saved_path")
    FUNCTION = "execute"
    CATEGORY = "image/save"
    OUTPUT_NODE = True

    def execute(self, images, filename_prefix):
        import os

        import folder_paths

        path = save_last_frame_png(
            images, folder_paths.get_output_directory(), filename_prefix)
        if hasattr(images, "cpu"):  # torch tensor: keep tensor type
            last = images[-1:]
        else:
            last = np.asarray(images)[-1:]
        ui = {"images": [{"filename": os.path.basename(path),
                          "subfolder": "", "type": "output"}]}
        return {"ui": ui, "result": (last, path)}


NODE_CLASS_MAPPINGS = {
    "WanCameraOrbitEmbedding": WanCameraOrbitEmbedding,
    "TSTSaveLastFramePNG": TSTSaveLastFramePNG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanCameraOrbitEmbedding": "Wan Camera Orbit Embedding (TST)",
    "TSTSaveLastFramePNG": "Save Last Frame PNG (TST)",
}
