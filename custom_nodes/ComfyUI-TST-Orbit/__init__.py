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
  - Orbit center O = (0, 0, r) on frame 0's optical axis, with
    r = orbit_radius * 1.5 (1.5 = CAMERA_DICT base_T_norm, same unit as the
    native presets' translations).
  - Frame i of n: azimuth theta_i = DIR_SIGN[direction] * radians(degrees)
    * speed * i/(n-1)  (i/(n-1) is DELIBERATE: the last frame reaches the
    nominal angle exactly, unlike the native presets' i/n convention).
  - Camera center C_i = (-r*sin(theta_i), -h, r - r*cos(theta_i)),
    h = orbit_height * 1.5 (y is down, so +height raises the camera: -h).
  - Orientation is lookAt(O): z = normalize(O - C_i), x = normalize(z x up),
    y = z x x, with up = (0, -1, 0). For h = 0 this reduces exactly to
    R = Ry(theta_i) with the translation above.

v1 caps (manager ruling 2026-10-20): orbit_degrees <= 90, orbit_radius <= 1.5.
DIR_SIGN is PROVISIONAL: which sign reads as "cw" from the viewer's side is
settled by the paired-direction 5s GPU test; if reversed, flip ONLY this map.
"""

import math

import numpy as np
import torch

# Provisional direction signs — see module docstring. Flip ONLY this mapping
# if the GPU test shows the parallax running opposite to the chosen label.
DIR_SIGN = {"ccw": 1.0, "cw": -1.0}

BASE_T_NORM = 1.5  # CAMERA_DICT["base_T_norm"] in nodes_camera_trajectory.py


def orbit_entries(n, orbit_degrees, orbit_direction, orbit_radius,
                  orbit_height, speed=1.0, fx=0.5, fy=0.5, cx=0.5, cy=0.5):
    """Build the (n, 23) native-layout per-frame pose entries for an orbit."""
    if n < 1:
        raise ValueError("length must be >= 1")
    sign = DIR_SIGN[orbit_direction]
    total = math.radians(orbit_degrees) * float(speed)
    r = float(orbit_radius) * BASE_T_NORM
    h = float(orbit_height) * BASE_T_NORM
    center = np.array([0.0, 0.0, r], dtype=np.float64)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    entries = []
    for i in range(n):
        theta = sign * total * (i / (n - 1)) if n > 1 else 0.0
        cam_c = np.array([-r * math.sin(theta), -h, r - r * math.cos(theta)],
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
            },
        }

    RETURN_TYPES = ("WAN_CAMERA_EMBEDDING", "INT", "INT", "INT")
    RETURN_NAMES = ("camera_embedding", "width", "height", "length")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/wan/camera"

    def execute(self, width, height, length, orbit_degrees, orbit_direction,
                orbit_radius, orbit_height, speed=1.0, fx=0.5, fy=0.5,
                cx=0.5, cy=0.5):
        cam_params = orbit_entries(
            length, orbit_degrees, orbit_direction, orbit_radius,
            orbit_height, speed=speed, fx=fx, fy=fy, cx=cx, cy=cy)
        cam_params = np.array([[float(x) for x in pose] for pose in cam_params])
        embedding = _pack_embedding(cam_params, width, height)
        return (embedding, width, height, length)


NODE_CLASS_MAPPINGS = {
    "WanCameraOrbitEmbedding": WanCameraOrbitEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanCameraOrbitEmbedding": "Wan Camera Orbit Embedding (TST)",
}
