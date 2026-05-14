"""
track.py — F1 Track Loader
Loads real circuit coordinates from FastF1 telemetry and normalizes
them into screen space for Arcade rendering.

Usage:
    from src.visualization.track import load_track
    path = load_track(session)   # list of (x, y) screen coordinates
"""


import numpy as np

# ── Screen constants (must match arcade_view.py) ──────────────────────────────
SCREEN_W     = 1280
SCREEN_H     = 780
TRACK_MARGIN = 80


def _normalize_track(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """
    Scale raw FastF1 X/Y coordinates into screen pixel space.
    """

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # FIX: safe width/height (avoid division by zero)
    track_w = max(x_max - x_min, 1.0)
    track_h = max(y_max - y_min, 1.0)

    avail_w = SCREEN_W - 2 * TRACK_MARGIN
    avail_h = SCREEN_H - 2 * TRACK_MARGIN

    scale = min(avail_w / track_w, avail_h / track_h)

    offset_x = TRACK_MARGIN + (avail_w - track_w * scale) / 2
    offset_y = TRACK_MARGIN + (avail_h - track_h * scale) / 2

    return [
        (offset_x + (x - x_min) * scale,
         offset_y + (y - y_min) * scale)
        for x, y in zip(xs, ys)
    ]


def load_track(session) -> list[tuple[float, float]]:
    """
    Extract and normalize the circuit layout from a FastF1 session.
    """

    # FIX: ensure fastest lap exists
    lap = session.laps.pick_fastest()
    if lap is None:
        raise ValueError("No fastest lap available in session.")

    pos = lap.get_pos_data()

    # FIX: validate telemetry
    if pos is None or pos.empty:
        raise ValueError("No positional telemetry found in the fastest lap.")

    if "X" not in pos.columns or "Y" not in pos.columns:
        raise ValueError("Telemetry is missing 'X' or 'Y' columns.")

    xs = pos["X"].values[::3].astype(float)
    ys = pos["Y"].values[::3].astype(float)

    # FIX: ensure enough points
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Not enough telemetry points to render a track.")

    path = _normalize_track(xs, ys)

    print(f"[track] Loaded {len(path)} track points.")
    return path