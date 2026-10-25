"""Offline dual-assertion self-test for WanCameraOrbitEmbedding.

Run INSIDE the worker container (needs the host ComfyUI + torch):
    cd /ComfyUI && python3 custom_nodes/ComfyUI-TST-Orbit/selftest_orbit.py

Run LOCALLY (no ComfyUI, no torch — numpy + PIL only):
    python3 selftest_orbit.py --local

Assertions (manager ruling 2026-10-20 — BOTH must pass before any GPU run):
  (a) orbit_degrees=0 output == native WanCameraEmbedding "Static" output,
      value-for-value. Validates the entry layout + relativization path.
  (b) Native "Pan Right" pose entries (built with the native
      get_camera_motion, assembled exactly like the native execute) fed
      through this node's packing == native "Pan Right" output (allclose).
      Validates the entry->tensor path end to end and rules out the
      c2w/w2c mirror trap on CPU, for $0.
Small geometry (64x64, 21 frames) keeps it far inside the debug channel's
60s budget; the layout/math under test is size-independent.

Local-only groups (run with --local, no container needed):
  (c) Pure-math assertions on orbit_entries itself (numpy only) — regression
      guard for the trajectory generator. Mirrors main's local test set.
  (c2) Batch-extension assertions: default-output identity vs the v1
      formula, easing curves, sweep_start, center offset, radius/height
      deltas, intensity, and validation errors.
  (c3) Save-last-frame helper assertions: last-frame selection, float->uint8
      conversion, PNG round-trip, counter suffix (no overwrite).
"""

import importlib.util
import os
import sys

import numpy as np

W, H, N = 64, 64, 21


def _load_orbit_module():
    # Sibling __init__.py: in-container this IS the /ComfyUI path, locally
    # it is the workspace copy — same file, no hardcoded container path.
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "tst_orbit", os.path.join(here, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unwrap(out):
    """Native execute may return io.NodeOutput (new API) or a tuple."""
    if hasattr(out, "args"):
        return out.args
    return tuple(out)


def _native_execute(*args, **kwargs):
    from comfy_extras.nodes_camera_trajectory import WanCameraEmbedding
    try:
        return _unwrap(WanCameraEmbedding.execute(*args, **kwargs))
    except TypeError:
        return _unwrap(WanCameraEmbedding().execute(*args, **kwargs))


def _assertion_c(orbit):
    """(c) Pure-math assertions on orbit_entries itself (numpy only, no
    native dependency) — regression guard for the trajectory generator.
    Mirrors main's local test set (2026-10-20)."""
    import math

    def c2w(e):
        return np.array(e[7:]).reshape(4, 4)

    # deg=0, h=0 -> every frame identity (Static-equivalent matrices)
    E = orbit.orbit_entries(81, 0.0, "ccw", 1.0, 0.0)
    assert E.shape == (81, 23)
    assert all(np.allclose(c2w(e), np.eye(4)) for e in E), "(c) deg0 identity"
    # entry layout: [0]=0, [1:5]=fx fy cx cy, [5:7]=0, bottom row [0,0,0,1]
    e0 = E[0]
    assert e0[0] == 0 and list(e0[1:5]) == [0.5] * 4 and list(e0[5:7]) == [0, 0]
    assert list(c2w(e0)[3]) == [0, 0, 0, 1], "(c) layout"
    # last frame reaches the nominal angle exactly (i/(n-1)); mid frame half
    for direction, sgn in (("ccw", 1), ("cw", -1)):
        E = orbit.orbit_entries(81, 90.0, direction, 1.0, 0.0)
        for idx, want in ((-1, 90.0), (40, 45.0)):
            R = c2w(E[idx])[:3, :3]
            theta = math.degrees(math.atan2(R[0, 2], R[0, 0]))
            assert abs(theta - sgn * want) < 1e-9, f"(c) angle {direction}"
    # camera centers on the circle around O=(0,0,r); C_0 = (0,-h,0); lookAt
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.3, 0.2)
    r, h = 1.3 * 1.5, 0.2 * 1.5
    for e in E:
        M = c2w(e)
        R, C = M[:3, :3], M[:3, 3]
        assert abs((C[0] ** 2 + (C[2] - r) ** 2) - r ** 2) < 1e-9, "(c) circle"
        assert abs(C[1] + h) < 1e-12, "(c) height"
        z = np.array([0.0, 0.0, r]) - C
        z = z / np.linalg.norm(z)
        assert np.allclose(R[:, 2], z, atol=1e-12), "(c) lookAt"
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "(c) orthonormal"
        assert abs(np.linalg.det(R) - 1) < 1e-12, "(c) det+1"
    assert np.allclose(c2w(E[0])[:3, 3], [0, -h, 0]), "(c) C0"
    # speed scales the total sweep
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, speed=0.5)
    R = c2w(E[-1])[:3, :3]
    assert abs(math.degrees(math.atan2(R[0, 2], R[0, 0])) - 45.0) < 1e-9, \
        "(c) speed scale"
    print("(c) orbit_entries pure-math assertions: PASS")


def _assertion_c2(orbit):
    """(c2) Batch-extension assertions (numpy only, locally runnable)."""
    import math

    def c2w(e):
        return np.array(e[7:]).reshape(4, 4)

    def angle_deg(E, idx):
        R = c2w(E[idx])[:3, :3]
        return math.degrees(math.atan2(R[0, 2], R[0, 0]))

    def v1_entries(n, degrees, direction, radius, height, speed=1.0):
        """The pre-batch formula, reimplemented inline for identity proof."""
        sign = orbit.DIR_SIGN[direction]
        total = math.radians(degrees) * float(speed)
        r = float(radius) * orbit.BASE_T_NORM
        h = float(height) * orbit.BASE_T_NORM
        center = np.array([0.0, 0.0, r])
        up = np.array([0.0, -1.0, 0.0])
        out = []
        for i in range(n):
            theta = sign * total * (i / (n - 1)) if n > 1 else 0.0
            cam_c = np.array([-r * math.sin(theta), -h,
                              r - r * math.cos(theta)])
            z = center - cam_c
            z = z / np.linalg.norm(z)
            x = np.cross(z, up)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            m = np.zeros((4, 4))
            m[0:3, 0] = x
            m[0:3, 1] = y
            m[0:3, 2] = z
            m[0:3, 3] = cam_c
            m[3, 3] = 1.0
            e = [0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
            e.extend(float(v) for v in m.reshape(-1))
            out.append(e)
        return np.array(out)

    # default-output identity vs v1 (exact: same op order, defaults collapse)
    for args in ((81, 90.0, "ccw", 1.0, 0.0),
                 (81, 90.0, "cw", 1.3, 0.2),
                 (21, 45.0, "ccw", 0.7, -0.3)):
        assert np.array_equal(orbit.orbit_entries(*args),
                              v1_entries(*args)), f"(c2) default != v1 {args}"
    assert np.array_equal(
        orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, speed=0.5),
        v1_entries(81, 90.0, "ccw", 1.0, 0.0, speed=0.5)), "(c2) speed identity"

    # easing: endpoints fixed, mid-frame follows the curve
    for easing, mid in (("linear", 45.0), ("ease_in", 22.5),
                        ("ease_out", 67.5), ("ease_in_out", 45.0)):
        E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, easing=easing)
        assert abs(angle_deg(E, 0)) < 1e-9, f"(c2) {easing} start"
        assert abs(angle_deg(E, -1) - 90.0) < 1e-9, f"(c2) {easing} end"
        assert abs(angle_deg(E, 40) - mid) < 1e-9, f"(c2) {easing} mid"

    # sweep_start: first frame at start, last at start+degrees (ccw)
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, sweep_start=-30.0)
    assert abs(angle_deg(E, 0) + 30.0) < 1e-9, "(c2) sweep first"
    assert abs(angle_deg(E, -1) - 60.0) < 1e-9, "(c2) sweep last"
    # any sweep reachable: 30 -> -30 via cw with sweep_start=-30, degrees=60
    E = orbit.orbit_entries(81, 60.0, "cw", 1.0, 0.0, sweep_start=-30.0)
    assert abs(angle_deg(E, 0) - 30.0) < 1e-9, "(c2) sweep map first"
    assert abs(angle_deg(E, -1) + 30.0) < 1e-9, "(c2) sweep map last"

    # center offset: O fixed, C0 shifted, lookAt + per-frame circle hold
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0,
                            center_x=0.2, center_y=0.1)
    O = np.array([0.3, -0.15, 1.5])
    assert np.allclose(c2w(E[0])[:3, 3], [0.3, -0.15, 0.0]), "(c2) center C0"
    for e in E:
        M = c2w(e)
        R, C = M[:3, :3], M[:3, 3]
        assert abs((C[0] - O[0]) ** 2 + (C[2] - O[2]) ** 2 - 1.5 ** 2) < 1e-9, \
            "(c2) center circle"
        z = O - C
        z = z / np.linalg.norm(z)
        assert np.allclose(R[:, 2], z, atol=1e-12), "(c2) center lookAt"

    # radius/height deltas: linear in u, frame 0 identical to no-delta
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.2,
                            radius_delta=-0.4, height_delta=0.2)
    E0 = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.2)
    assert np.allclose(c2w(E[0])[:3, 3], c2w(E0[0])[:3, 3]), "(c2) delta C0"
    for idx, r_want, h_want in ((0, 1.5, 0.3), (40, 1.2, 0.45), (-1, 0.9, 0.6)):
        C = c2w(E[idx])[:3, 3]
        assert abs((C[0] ** 2 + (C[2] - 1.5) ** 2) - r_want ** 2) < 1e-9, \
            "(c2) delta radius"
        assert abs(C[1] + h_want) < 1e-9, "(c2) delta height"

    # intensity: scales the sweep only, never the start angle
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, intensity=2.0)
    assert abs(angle_deg(E, 0)) < 1e-9, "(c2) intensity first"
    assert abs(abs(angle_deg(E, -1)) - 180.0) < 1e-9, "(c2) intensity last"
    E = orbit.orbit_entries(81, 45.0, "ccw", 1.0, 0.0,
                            sweep_start=10.0, intensity=2.0)
    assert abs(angle_deg(E, 0) - 10.0) < 1e-9, "(c2) intensity start"
    assert abs(angle_deg(E, -1) - 100.0) < 1e-9, "(c2) intensity end"
    assert np.array_equal(
        orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, intensity=1.0),
        orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0)), \
        "(c2) intensity=1 identity"

    # effective sweep cap (manager ruling 2026-10-25): intensity x speed x
    # degrees <= 180, fail-fast above; the cap governs sweep magnitude,
    # not start/end absolute angles
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, intensity=2.0)
    assert abs(abs(angle_deg(E, -1)) - 180.0) < 1e-9, "(c2) sweep cap boundary"
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, speed=2.0)
    assert abs(abs(angle_deg(E, -1)) - 180.0) < 1e-9, "(c2) sweep cap speed"
    # start angle does not count toward the cap: sweep 180 passes even
    # though the end angle sits at 270 absolute
    E = orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0,
                            sweep_start=90.0, intensity=2.0)
    assert abs(angle_deg(E, 0) - 90.0) < 1e-9, "(c2) sweep cap start"
    for kwargs in ({"intensity": 2.1},
                   {"speed": 2.1},
                   {"intensity": 5.0}):
        try:
            orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"(c2) sweep cap not raised for {kwargs}")

    # validation errors fail fast
    for kwargs, exc in (({"easing": "bounce"}, ValueError),
                        ({"radius_delta": -1.0}, ValueError),
                        ({"intensity": 0.0}, ValueError)):
        try:
            orbit.orbit_entries(81, 90.0, "ccw", 1.0, 0.0, **kwargs)
        except exc:
            pass
        else:
            raise AssertionError(f"(c2) no error for {kwargs}")
    try:
        orbit.orbit_entries(81, 90.0, "sideways", 1.0, 0.0)
    except KeyError:
        pass
    else:
        raise AssertionError("(c2) no error for bad direction")

    # Static precondition: degrees=0 + sweep_start=0 + deltas=0 -> all frames
    # identical (relativization then cancels to Static); a radius delta
    # breaks it, a center offset does not
    E = orbit.orbit_entries(81, 0.0, "ccw", 1.0, 0.0,
                            center_x=0.2, center_y=0.1)
    assert all(np.array_equal(c2w(e), c2w(E[0])) for e in E), \
        "(c2) static precondition"
    E = orbit.orbit_entries(81, 0.0, "ccw", 1.0, 0.0, radius_delta=0.2)
    assert not all(np.array_equal(c2w(e), c2w(E[0])) for e in E), \
        "(c2) delta breaks static"

    # n == 1: single frame sits at the start angle
    E = orbit.orbit_entries(1, 90.0, "ccw", 1.0, 0.0, sweep_start=30.0)
    assert E.shape == (1, 23) and abs(angle_deg(E, 0) - 30.0) < 1e-9, "(c2) n==1"
    print("(c2) batch-extension assertions: PASS")


def _assertion_c3(orbit):
    """(c3) Save-last-frame helper assertions (numpy + PIL only)."""
    import tempfile

    from PIL import Image

    # float batch: last frame selected, 0-1 -> uint8 (truncation documented)
    batch = np.zeros((3, 4, 5, 3), dtype=np.float64)
    batch[2] = 0.5
    frame = orbit.last_frame_array(batch)
    assert frame.shape == (4, 5, 3) and frame.dtype == np.uint8, \
        "(c3) shape/dtype"
    assert int(frame[0, 0, 0]) == 127, "(c3) 0.5 -> 127"
    # uint8 batch passes through unchanged
    batch_u8 = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3)
    assert np.array_equal(orbit.last_frame_array(batch_u8), batch_u8[1]), \
        "(c3) uint8 passthrough"
    # torch-tensor duck type (no torch installed locally)
    class _FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    assert np.array_equal(orbit.last_frame_array(_FakeTensor(batch)), frame), \
        "(c3) tensor path"
    # PNG round-trip + counter suffix (no overwrite)
    with tempfile.TemporaryDirectory() as d:
        p1 = orbit.save_last_frame_png(batch, d, "test")
        p2 = orbit.save_last_frame_png(batch, d, "test")
        assert p1.endswith("test_00001_.png") and p2.endswith(
            "test_00002_.png"), "(c3) counter"
        reloaded = np.array(Image.open(p1))
        assert np.array_equal(reloaded, frame), "(c3) PNG round-trip"
    # node registration (plain classes: no ComfyUI import needed)
    assert "TSTSaveLastFramePNG" in orbit.NODE_CLASS_MAPPINGS, "(c3) mapping"
    assert orbit.NODE_CLASS_MAPPINGS["TSTSaveLastFramePNG"].OUTPUT_NODE \
        is True, "(c3) output node"
    print("(c3) save-last-frame assertions: PASS")


def main():
    orbit = _load_orbit_module()
    if "--local" in sys.argv:
        _assertion_c(orbit)
        _assertion_c2(orbit)
        _assertion_c3(orbit)
        print("LOCAL SELFTEST PASS: (c), (c2), (c3) — no ComfyUI/torch needed.")
        return 0
    import torch

    node = orbit.WanCameraOrbitEmbedding()
    _assertion_c(orbit)

    # ---- assertion (a): degrees=0 == Static, exact ----
    native_static = _native_execute("Static", W, H, N)[0].cpu().float()
    mine_static = node.execute(W, H, N, 0.0, "ccw", 1.0, 0.0)[0].cpu().float()
    assert native_static.shape == mine_static.shape, (
        f"(a) shape mismatch: native {native_static.shape} "
        f"vs orbit {mine_static.shape}")
    max_diff_a = (native_static - mine_static).abs().max().item()
    print(f"(a) Static vs orbit(0deg): shape {tuple(mine_static.shape)}, "
          f"max|diff| = {max_diff_a:.3e}")
    assert torch.equal(native_static, mine_static), "(a) FAILED: not identical"

    # ---- assertion (b): Pan Right entries through our packing ----
    from comfy_extras.nodes_camera_trajectory import (
        CAMERA_DICT, get_camera_motion)
    from comfy_extras.nodes_camera_trajectory import process_pose_params  # noqa: F401  (import sanity)

    spec = CAMERA_DICT["Pan Right"]
    rt = get_camera_motion(np.array(spec["angle"]), np.array(spec["T"]),
                           1.0, N)
    trajs = []
    for cp in rt.tolist():
        traj = [0.5, 0.5, 0.5, 0.5, 0, 0]
        traj.extend(cp[0])
        traj.extend(cp[1])
        traj.extend(cp[2])
        traj.extend([0, 0, 0, 1])
        trajs.append(traj)
    cam_params = np.array([[float(x) for x in pose] for pose in trajs])
    cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]),
                                 cam_params], 1)
    mine_pan = orbit._pack_embedding(cam_params, W, H).cpu().float()
    native_pan = _native_execute("Pan Right", W, H, N)[0].cpu().float()
    max_diff_b = (native_pan - mine_pan).abs().max().item()
    print(f"(b) Pan Right entries->our packing vs native: "
          f"max|diff| = {max_diff_b:.3e}")
    assert torch.allclose(native_pan, mine_pan, atol=1e-5), "(b) FAILED"

    print("SELFTEST PASS: (a) exact, (b) allclose — entry layout and packing "
          "match the native node.")


if __name__ == "__main__":
    sys.exit(main())
