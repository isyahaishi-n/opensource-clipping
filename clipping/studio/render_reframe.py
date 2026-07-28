"""
render_reframe.py — Custom "Pan & Zoom Reframe" template

This is a custom render template integrated from the standalone
`main123.py` (reframe_16x9_to_9x16.py) script. It converts a landscape
(16:9) source into a vertical clip using an animated crop that moves/zooms
from a wide "letterbox" framing toward a tight framing on the subject —
WITHOUT stretching/distortion (uses ffmpeg's `crop`/`zoompan` correctly).

It exposes `buat_video_reframe(...)` with the SAME call signature the rest
of the pipeline expects from `buat_video_hybrid`, so it can be dropped in
as an alternative render mode selected by a config flag (`cfg.use_reframe`).

Because this template renders directly with a single ffmpeg pass (no
per-frame OpenCV writer), it returns a `get_x(t)` callable that simply maps
to the horizontal center of the crop, which is enough for subtitle
positioning in the downstream ASS generation step.
"""

import importlib.util
import json
import math
import os
import subprocess


def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
detect_video_encoder = _ffmpeg_utils.detect_video_encoder

utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_get_render_dims = utils._get_render_dims
_is_vertical_ratio = utils._is_vertical_ratio


# ─────────────────────────────────────────────────────────────────────────
# Core reframe logic (ported from main123.py, adapted to the pipeline)
# ─────────────────────────────────────────────────────────────────────────

def _probe_video(path):
    """Return (width, height, fps, duration_seconds) of the first video stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])
    fps = 24.0
    afr = stream.get("avg_frame_rate", "0/0")
    try:
        num, den = afr.split("/")
        num, den = float(num), float(den)
        if den:
            fps = num / den
    except (ValueError, ZeroDivisionError):
        pass
    try:
        dur = float(data["format"]["duration"])
    except (KeyError, ValueError):
        dur = None
    return w, h, fps, dur


def _build_filter(src_w, src_h, out_w, out_h, fps, move_dur,
                  start_scale, end_scale,
                  start_cx, end_cx,
                  start_cy, end_cy,
                  letterbox=True):
    """
    Build the ffmpeg -vf filter string that animates from a START framing
    to an END framing (letterbox wide -> tight crop), never stretching.

    Ported directly from main123.py::build_filter.
    """
    if move_dur <= 0:
        p = "1"
    else:
        move_frames = max(1.0, move_dur * fps)
        p = f"min(1,on/{move_frames})"
    cx = f"({start_cx}+({end_cx}-{start_cx})*{p})"

    if not letterbox:
        # ---- Full-bleed (no black bars, always fills the frame) ----
        inter_w = round(src_w / src_h * out_h)
        if inter_w % 2:
            inter_w += 1
        sc = f"({start_scale}+({end_scale}-{start_scale})*{p})"
        z = f"(1/{sc})"
        cy = f"({start_cy}+({end_cy}-{start_cy})*{p})"
        vw = f"(iw*{sc})"
        vh = f"(ih*{sc})"
        x = f"max(0,min({cx}*iw-({vw})/2,iw-({vw})))"
        y = f"max(0,min({cy}*ih-({vh})/2,ih-({vh})))"
        zoompan = (f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={inter_w}x{out_h}:fps={fps}")
        crop_x = (inter_w - out_w) // 2
        crop = f"crop={out_w}:{out_h}:{crop_x}:0"
        return f"{zoompan},{crop},setsar=1"

    # ---- Letterbox start -> full-bleed end ----
    pad_h = round(src_w * out_h / out_w)
    if pad_h % 2:
        pad_h += 1
    pad_y = (pad_h - src_h) // 2  # black bar height on top (and bottom)

    end_scale_pad = end_scale * src_h / pad_h
    sc = f"({start_scale}+({end_scale_pad}-{start_scale})*{p})"
    z = f"(1/{sc})"

    start_cy_pad = (start_cy * src_h + pad_y) / pad_h
    end_cy_pad = (end_cy * src_h + pad_y) / pad_h
    cy = f"({start_cy_pad}+({end_cy_pad}-{start_cy_pad})*{p})"

    vw = f"(iw*{sc})"
    vh = f"(ih*{sc})"
    x = f"max(0,min({cx}*iw-({vw})/2,iw-({vw})))"
    y = f"max(0,min({cy}*ih-({vh})/2,ih-({vh})))"

    pad = f"pad={src_w}:{pad_h}:0:{pad_y}:black"
    zoompan = f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={out_w}x{out_h}:fps={fps}"
    return f"{pad},{zoompan},setsar=1"


def buat_video_reframe(
    input_video,
    output_video,
    start_clip,
    end_clip,
    rasio,
    cfg,
    broll_data=None,
    label="Reframe",
):
    """
    Render a clip using the custom animated pan/zoom reframe template.

    Signature mirrors `buat_video_hybrid` so it can be selected by the
    pipeline as an alternative render mode.

    Args:
        input_video (str): Source video file path.
        output_video (str): Output (silent) video file path.
        start_clip (float): Start timestamp in seconds.
        end_clip (float): End timestamp in seconds.
        rasio (str): Output ratio string (e.g. '9:16').
        cfg: Runtime config object. Reads optional reframe.* tuning attrs.
        broll_data (list, optional): Ignored by this template (kept for API
            compatibility with the pipeline).
        label (str, optional): Progress label.

    Returns:
        callable: get_x(t) mapping any timestamp to the horizontal crop
        center in SOURCE pixels (used for subtitle positioning).
    """
    if broll_data is None:
        broll_data = []

    src_w, src_h, src_fps, _src_dur = _probe_video(input_video)
    out_w, out_h = _get_render_dims(cfg, rasio, source_h=src_h)
    duration = max(0.0, float(end_clip) - float(start_clip))

    # ── Tunable parameters (fall back to main123.py defaults) ──
    move_dur    = float(getattr(cfg, "reframe_move_duration", 3.0))
    start_scale = float(getattr(cfg, "reframe_start_scale", 1.0))
    end_scale   = float(getattr(cfg, "reframe_end_scale", 0.72))
    start_cx    = float(getattr(cfg, "reframe_start_cx", 0.50))
    start_cy    = float(getattr(cfg, "reframe_start_cy", 0.50))
    end_cx      = float(getattr(cfg, "reframe_end_cx", 0.47))
    end_cy      = float(getattr(cfg, "reframe_end_cy", 0.48))
    letterbox   = not bool(getattr(cfg, "reframe_no_letterbox", False))

    vf = _build_filter(
        src_w, src_h, out_w, out_h, src_fps, move_dur,
        start_scale, end_scale,
        start_cx, end_cx,
        start_cy, end_cy,
        letterbox=letterbox,
    )

    video_encoder = detect_video_encoder(cfg, target_h=out_h)

    print(f"🎬 {label} - Reframe (pan/zoom) render dimulai...", flush=True)
    print(f"   ↪ Source {src_w}x{src_h}@{src_fps:.2f}fps → {out_w}x{out_h} ({rasio})", flush=True)
    print(f"   ↪ Filter: {vf}", flush=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_clip),
    ]
    cmd += ["-i", input_video]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", vf]
    cmd += video_encoder["args"]
    cmd += ["-pix_fmt", "yuv420p", "-an", output_video]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg reframe gagal (label={label}):\n{result.stderr[-1500:]}"
        )

    print(f"✅ {label} selesai.", flush=True)

    # get_x(t): interpolate the horizontal crop center in SOURCE pixels so the
    # downstream subtitle placement can follow the animated framing.
    def get_x(t):
        if duration <= 0 or move_dur <= 0:
            frac = 1.0
        else:
            frac = min(1.0, max(0.0, float(t) / move_dur))
        cx_frac = start_cx + (end_cx - start_cx) * frac
        return cx_frac * src_w

    return get_x
