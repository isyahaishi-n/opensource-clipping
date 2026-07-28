"""
render_reframe.py — Custom "Pan & Zoom Reframe" template (FACE-TRACKING)

Originally ported from the standalone `main123.py` reframe script (a single
FFmpeg pan/zoom pass), this template has been UPGRADED to have full feature
parity with the default `render_hybrid.py` renderer:

  • Real face detection (Mediapipe or YOLO) — same detectors as the default.
  • Frame-by-frame OpenCV render loop with a smoothed virtual camera that
    FOLLOWS the subject (deadzone / smoothing / snap — identical tuning knobs
    to the default via cfg.track_*).
  • B-roll overlay compositing (same behavior as the default).
  • `--static-crop` support for non-face ratios.

On TOP of that tracking, it applies YOUR signature reframe motion: an animated
ZOOM curve that eases the crop window from `reframe_start_scale` to
`reframe_end_scale` over `reframe_move_duration` seconds, plus an optional
vertical framing bias (`reframe_end_cy`) for headroom.

So instead of a fixed, content-unaware pan (the old behavior), the pan/zoom
now tracks whoever is on screen — you keep the cinematic zoom feel AND gain
the default's face tracking.

Exposes `buat_video_reframe(...)` with the SAME signature the pipeline expects
from `buat_video_hybrid`, and returns a `get_x(t)` callable for subtitle
positioning (tracked horizontal crop center in SOURCE pixels).
"""

import importlib.util
import math
import os
import statistics as _st
import subprocess

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - mediapipe optional at import time
    mp = None


def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer

utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_resize_frame = utils._resize_frame
_get_render_dims = utils._get_render_dims
_is_vertical_ratio = utils._is_vertical_ratio
RATIO_MAP = utils.RATIO_MAP

broll_mod = _load_studio_internal_module("broll.py", "clipping_studio_broll")
crop_center_broll = broll_mod.crop_center_broll

face_detection = _load_studio_internal_module("face_detection.py", "clipping_studio_face_detection")
get_face_detector = face_detection.get_face_detector


def _fmt(s):
    mins = int(s) // 60
    secs = int(s % 60)
    return f"{mins:02d}:{secs:02d}"


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
    Render a clip using the face-tracking Reframe (pan/zoom) template.

    Combines the default hybrid renderer's face tracking with an animated zoom
    curve. Signature mirrors `buat_video_hybrid`.

    Args:
        input_video (str): Source video file path.
        output_video (str): Output (silent) video file path.
        start_clip (float): Start timestamp in seconds.
        end_clip (float): End timestamp in seconds.
        rasio (str): Output ratio string (e.g. '9:16').
        cfg: Runtime config object (reads track_* and reframe_* attrs).
        broll_data (list, optional): B-roll timing/filepath dicts to overlay.
        label (str, optional): Progress label.

    Returns:
        callable: get_x(t) -> tracked horizontal crop center in SOURCE pixels.
    """
    if broll_data is None:
        broll_data = []

    # ── Camera tuning knobs (identical to the default hybrid renderer) ──
    STEP_DETEKSI     = cfg.track_step if getattr(cfg, "track_step", None) is not None else 0.25
    DEADZONE_RATIO   = cfg.track_deadzone if getattr(cfg, "track_deadzone", None) is not None else 0.15
    SMOOTH_FACTOR    = cfg.track_smooth if getattr(cfg, "track_smooth", None) is not None else 0.30
    JITTER_THRESHOLD = cfg.track_jitter if getattr(cfg, "track_jitter", None) is not None else 5
    SNAP_THRESHOLD   = cfg.track_snap if getattr(cfg, "track_snap", None) is not None else 0.25

    # ── Reframe zoom curve knobs (your signature motion) ──
    move_dur    = float(getattr(cfg, "reframe_move_duration", 3.0))
    start_scale = float(getattr(cfg, "reframe_start_scale", 1.0))
    end_scale   = float(getattr(cfg, "reframe_end_scale", 0.72))
    end_cy      = float(getattr(cfg, "reframe_end_cy", 0.48))

    video_encoder = detect_video_encoder(cfg)

    # ── Face detector setup (same as default) ──
    yolo_model = None
    detector = None
    if getattr(cfg, "face_detector", "mediapipe") == "yolo":
        if not os.path.exists(cfg.file_yolo_model):
            print(f"   📥 Mendownload YOLOv8 Face Model ({cfg.yolo_size})...")
            import urllib.request
            urllib.request.urlretrieve(cfg.url_yolo_model, cfg.file_yolo_model)
        from ultralytics import YOLO
        yolo_model = YOLO(cfg.file_yolo_model)
    else:
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(orig_fps) or orig_fps == 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(end_clip) - float(start_clip)

    # Base (un-zoomed) crop window dimensions for the target ratio.
    w_part, h_part = RATIO_MAP.get(rasio, (9, 16))
    if _is_vertical_ratio(rasio):
        base_crop_w = int(height * w_part / h_part)
        base_crop_h = height
    else:
        base_crop_w = width
        base_crop_h = height
    base_crop_w = min(base_crop_w, width)
    base_crop_h = min(base_crop_h, height)

    default_cx = width // 2
    default_cy = height // 2

    base_out_w, base_out_h = _get_render_dims(cfg, rasio, source_h=height)

    # ── B-roll captures ──
    broll_caps = []
    for br in broll_data:
        if "filepath" in br and os.path.exists(br["filepath"]):
            broll_caps.append({
                "start": br["start_time"],
                "end": br["end_time"],
                "cap": cv2.VideoCapture(br["filepath"]),
            })

    skip_tracking = getattr(cfg, "static_crop", False) and rasio in ["1:1", "3:4", "4:5"]

    # ─────────────────────────────────────────────────────────────
    # PHASE 1 — FACE DETECTION (same as default hybrid)
    # ─────────────────────────────────────────────────────────────
    raw_data = []
    current_time = 0.0
    last_detect_percent = -1

    if skip_tracking:
        print(f"🧠 {label} - Static Crop aktif (tanpa face tracking)...", flush=True)
    else:
        print(f"🧠 {label} - Analisa wajah dimulai...", flush=True)

    while current_time <= duration and not skip_tracking:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        face_box = None
        center_x = default_cx
        center_y = default_cy

        if yolo_model is not None:
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                x1, y1, x2, y2 = boxes[largest_idx]
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2
                face_box = (x1, y1, x2, y2)
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )
            if results.detections:
                largest_face = max(
                    results.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                ).bounding_box
                center_x = largest_face.origin_x + (largest_face.width / 2)
                center_y = largest_face.origin_y + (largest_face.height / 2)
                face_box = (
                    largest_face.origin_x,
                    largest_face.origin_y,
                    largest_face.origin_x + largest_face.width,
                    largest_face.origin_y + largest_face.height,
                )

        raw_data.append({
            "time": current_time,
            "cx": center_x if face_box else default_cx,
            "cy": center_y if face_box else default_cy,
            "box": face_box,
        })

        detect_percent = min(100, int((current_time / duration) * 100)) if duration > 0 else 100
        if detect_percent != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_percent:3d}%", flush=True)
            last_detect_percent = detect_percent

        current_time += STEP_DETEKSI

    # ─────────────────────────────────────────────────────────────
    # PHASE 2 — SMOOTH VIRTUAL CAMERA (same as default hybrid)
    # ─────────────────────────────────────────────────────────────
    smooth_data = []
    if raw_data:
        initial_cxs = [d["cx"] for d in raw_data[:5]]
        initial_cys = [d["cy"] for d in raw_data[:5]]
        cam_cx = _st.median(initial_cxs) if initial_cxs else raw_data[0]["cx"]
        cam_cy = _st.median(initial_cys) if initial_cys else raw_data[0]["cy"]

        deadzone_px = base_crop_w * DEADZONE_RATIO
        temp_snap = SNAP_THRESHOLD if SNAP_THRESHOLD < 0.1 else 0.08
        snap_px = width * temp_snap

        for d in raw_data:
            face_cx = d["cx"]
            face_cy = d["cy"]

            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR

            cam_cy += (face_cy - cam_cy) * SMOOTH_FACTOR

            smooth_data.append({"time": d["time"], "cx": cam_cx, "cy": cam_cy})

    def _get_pos(t):
        if not smooth_data:
            return default_cx, default_cy
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["cx"], smooth_data[0]["cy"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["cx"], smooth_data[-1]["cy"]
        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                cx1, cx2 = smooth_data[i]["cx"], smooth_data[i + 1]["cx"]
                cy1, cy2 = smooth_data[i]["cy"], smooth_data[i + 1]["cy"]
                if t1 == t2:
                    return cx1, cy1
                frac = (t - t1) / (t2 - t1)
                return cx1 + (cx2 - cx1) * frac, cy1 + (cy2 - cy1) * frac
        return default_cx, default_cy

    def get_x(t):
        cx, _ = _get_pos(t)
        return cx

    def _zoom_scale(t):
        """Reframe zoom curve: eases crop scale start_scale -> end_scale."""
        if move_dur <= 0:
            p = 1.0
        else:
            p = min(1.0, max(0.0, float(t) / move_dur))
        # smoothstep easing for a cinematic feel
        p = p * p * (3 - 2 * p)
        return start_scale + (end_scale - start_scale) * p

    # ─────────────────────────────────────────────────────────────
    # PHASE 3 — RENDER FRAMES (tracked crop + reframe zoom curve)
    # ─────────────────────────────────────────────────────────────
    writer = open_ffmpeg_video_writer(
        output_video, base_out_w, base_out_h, orig_fps, video_encoder
    )

    TRANSITION_DUR = 0.3
    BROLL_MAX_ZOOM = 1.10

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render frame (face-track + zoom) dimulai...", flush=True)

        while True:
            ret, frame_utama = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break
            waktu_absolut = start_clip + t

            if _is_vertical_ratio(rasio):
                cx_base, cy_base = _get_pos(t)

                # Apply the reframe zoom curve to the crop window size.
                sc = _zoom_scale(t)
                cw = int(round(base_crop_w * sc))
                ch = int(round(base_crop_h * sc))
                cw = max(16, min(cw, width))
                ch = max(16, min(ch, height))

                # Optional vertical framing bias for headroom (reframe_end_cy).
                cy_target = cy_base + (end_cy - 0.5) * ch

                x1_crop = int(max(0, min(cx_base - cw // 2, width - cw)))
                y1_crop = int(max(0, min(cy_target - ch // 2, height - ch)))
                cropped = frame_utama[y1_crop:y1_crop + ch, x1_crop:x1_crop + cw]
                frame_out = _resize_frame(cropped, (base_out_w, base_out_h), cfg)
            else:
                # Non-vertical: simple fit-to-canvas (reframe zoom not applied).
                frame_out = _resize_frame(frame_utama, (base_out_w, base_out_h), cfg)

            # ── B-roll overlay (same as default) ──
            for bc in broll_caps:
                if bc["start"] <= waktu_absolut <= bc["end"]:
                    elapsed_broll = waktu_absolut - bc["start"]
                    bc["cap"].set(cv2.CAP_PROP_POS_MSEC, elapsed_broll * 1000)
                    ret_b, frame_b = bc["cap"].read()
                    if ret_b:
                        durasi_total = bc["end"] - bc["start"]
                        progress = elapsed_broll / durasi_total if durasi_total > 0 else 0
                        zoom_factor = 1.0 + ((BROLL_MAX_ZOOM - 1.0) * progress)
                        frame_b_crop = crop_center_broll(frame_b, base_out_w, base_out_h)
                        M = cv2.getRotationMatrix2D((base_out_w / 2, base_out_h / 2), 0, zoom_factor)
                        frame_b_zoomed = cv2.warpAffine(frame_b_crop, M, (base_out_w, base_out_h))
                        alpha = 1.0
                        if elapsed_broll < TRANSITION_DUR:
                            alpha = elapsed_broll / TRANSITION_DUR
                        elif (bc["end"] - waktu_absolut) < TRANSITION_DUR:
                            alpha = (bc["end"] - waktu_absolut) / TRANSITION_DUR
                        if alpha >= 1.0:
                            frame_out = frame_b_zoomed
                        else:
                            frame_out = cv2.addWeighted(frame_b_zoomed, alpha, frame_out, 1.0 - alpha, 0)
                    break

            writer.stdin.write(frame_out.tobytes())
            frame_count += 1

            render_percent = min(100, int((t / duration) * 100)) if duration > 0 else 100
            if render_percent != last_render_percent:
                print(
                    f"⏳ {label} - Render frame: {render_percent:3d}% | "
                    f"{_fmt(t)} / {_fmt(duration)}",
                    flush=True,
                )
                last_render_percent = render_percent

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()
        for bc in broll_caps:
            bc["cap"].release()

    return get_x
