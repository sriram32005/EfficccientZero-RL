"""Draw the per-frame "what is the agent thinking" panel next to the Atari screen.

For EfficientZero this shows the MCTS visit-count distribution (the searched, improved
policy), the raw network prior, the per-action Q values and the root value -- i.e. the
actual decision the planner made. For REINFORCE there is no search, so it shows the policy distribution.
"""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (18, 18, 22)
FG = (235, 235, 240)
MUTED = (140, 140, 150)
ACCENT = (90, 200, 250)
ACCENT2 = (250, 170, 90)
GOOD = (110, 220, 140)


def _text(img, txt, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _bar_chart(panel, y0, values, labels, highlight, title, color, fmt="{:.2f}", width=None):
    h, w = panel.shape[:2]
    width = width or (w - 150)
    _text(panel, title, (14, y0), 0.48, MUTED)
    y = y0 + 8
    vmax = float(np.max(np.abs(values))) if len(values) else 1.0
    vmax = max(vmax, 1e-6)
    row_h = 18
    for i, (v, lab) in enumerate(zip(values, labels)):
        yy = y + i * row_h
        c = GOOD if i == highlight else color
        _text(panel, lab[:9], (14, yy + 12), 0.42, FG if i == highlight else MUTED)
        length = int(width * (abs(float(v)) / vmax))
        cv2.rectangle(panel, (96, yy + 2), (96 + max(2, length), yy + 13), c, -1)
        _text(panel, fmt.format(float(v)), (96 + width + 6, yy + 12), 0.40, MUTED)
    return y + len(values) * row_h + 22


def _sparkline(panel, y0, series, title, color=ACCENT, height=52):
    h, w = panel.shape[:2]
    _text(panel, title, (14, y0), 0.48, MUTED)
    x0, x1 = 14, w - 18
    top, bot = y0 + 8, y0 + 8 + height
    cv2.rectangle(panel, (x0, top), (x1, bot), (38, 38, 46), -1)
    if len(series) >= 2:
        arr = np.asarray(series, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        rng = max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(arr):
            x = x0 + int((x1 - x0) * i / (len(arr) - 1))
            y = bot - int((bot - top - 4) * (v - lo) / rng) - 2
            pts.append((x, y))
        cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        _text(panel, f"{arr[-1]:+.2f}", (x1 - 58, top + 14), 0.42, color)
    return bot + 14


def render_frame(
    rgb,
    info,
    agent_name,
    action_meanings,
    score,
    step,
    checkpoint_label="",
    env_steps=None,
    value_history=None,
    panel_width=430,
    game_scale=3,
    extra_lines=(),
):
    """Compose one demo frame: [game screen | decision panel]. Returns an RGB uint8 array."""
    if rgb is None:
        rgb = np.zeros((210, 160, 3), dtype=np.uint8)
    game = cv2.resize(rgb, (rgb.shape[1] * game_scale, rgb.shape[0] * game_scale), interpolation=cv2.INTER_NEAREST)
    gh, gw = game.shape[:2]
    panel = np.zeros((gh, panel_width, 3), dtype=np.uint8)
    panel[:] = BG

    _text(panel, agent_name, (14, 28), 0.72, ACCENT if agent_name.startswith("Efficient") else ACCENT2, 2)
    if checkpoint_label:
        _text(panel, checkpoint_label, (14, 50), 0.46, MUTED)
    if env_steps is not None:
        _text(panel, f"trained on {env_steps:,} env steps", (14, 68), 0.44, MUTED)
    cv2.line(panel, (14, 78), (panel_width - 14, 78), (55, 55, 65), 1)

    _text(panel, f"score {score:g}", (14, 102), 0.62, FG, 2)
    _text(panel, f"step {step}", (panel_width - 120, 102), 0.5, MUTED)

    y = 134
    labels = list(action_meanings)
    chosen = int(info.get("action", -1))
    if "visit_counts" in info:
        vc = np.asarray(info["visit_counts"], dtype=np.float32)
        total = max(1.0, vc.sum())
        y = _bar_chart(panel, y, vc / total, labels, chosen, "MCTS visit distribution (searched policy)", ACCENT, "{:.0%}")
        y = _bar_chart(panel, y, np.asarray(info["q_values"], dtype=np.float32), labels, chosen, "Q(s,a) from search", (150, 130, 220), "{:+.2f}")
        _text(panel, f"root value V(s) = {info['root_value']:+.3f}", (14, y + 6), 0.50, FG)
        _text(panel, f"{info.get('num_simulations', 0)} simulations / move", (14, y + 24), 0.44, MUTED)
        y += 40
    elif "policy" in info and "q_values" not in info:
        # REINFORCE: show the learned policy distribution
        policy = np.asarray(info["policy"], dtype=np.float32)
        y = _bar_chart(panel, y, policy, labels, chosen, "policy pi(a|s) -- single forward pass", ACCENT2, "{:.0%}")
        _text(panel, f"V(s) = {info['root_value']:+.3f}", (14, y + 6), 0.50, FG)
        _text(panel, "no search: greedy over policy", (14, y + 24), 0.44, MUTED)
        y += 40
    else:
        q = np.asarray(info["q_values"], dtype=np.float32)
        y = _bar_chart(panel, y, q, labels, chosen, "Q(s,a) -- single forward pass, no search", ACCENT2, "{:+.2f}")
        _text(panel, f"max Q = {q.max():+.3f}", (14, y + 6), 0.50, FG)
        _text(panel, "no planning: argmax over Q", (14, y + 24), 0.44, MUTED)
        y += 40

    if chosen >= 0 and chosen < len(labels):
        _text(panel, f"action -> {labels[chosen]}", (14, y + 10), 0.55, GOOD, 2)
    y += 26

    if value_history is not None and y + 80 < gh:
        y = _sparkline(panel, y, value_history[-120:], "value estimate over time")

    for i, line in enumerate(extra_lines):
        yy = gh - 14 - 16 * (len(extra_lines) - 1 - i)
        _text(panel, line, (14, yy), 0.42, MUTED)

    return np.concatenate([game, panel], axis=1)


def label_strip(width, text, height=42, color=FG, bg=(28, 28, 34), scale=0.62):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = bg
    size = cv2.getTextSize(text, FONT, scale, 2)[0]
    _text(strip, text, ((width - size[0]) // 2, (height + size[1]) // 2), scale, color, 2)
    return strip
