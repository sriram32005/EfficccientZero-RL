"""Video composition for the demo: side-by-side clips, title cards and one final reel.

Everything streams frame by frame, so memory stays flat even for long episodes.
"""
from __future__ import annotations

import os

import cv2
import imageio
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pad_to(frame, height, width):
    h, w = frame.shape[:2]
    if h == height and w == width:
        return frame
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:] = (18, 18, 22)
    y = (height - h) // 2
    x = (width - w) // 2
    out[y: y + h, x: x + w] = frame[: min(h, height), : min(w, width)]
    return out


def _header(width, text, subtitle="", height=56):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (26, 26, 32)
    size = cv2.getTextSize(text, FONT, 0.8, 2)[0]
    cv2.putText(strip, text, ((width - size[0]) // 2, 30), FONT, 0.8, (240, 240, 245), 2, cv2.LINE_AA)
    if subtitle:
        s2 = cv2.getTextSize(subtitle, FONT, 0.5, 1)[0]
        cv2.putText(strip, subtitle, ((width - s2[0]) // 2, 48), FONT, 0.5, (150, 150, 160), 1, cv2.LINE_AA)
    return strip


def side_by_side(paths, labels, out_path, header_text="", subtitle="", fps=30, max_frames=None):
    """Play several clips simultaneously in one row. Shorter clips freeze on their last frame."""
    readers = [imageio.get_reader(p) for p in paths]
    firsts = []
    for r in readers:
        firsts.append(r.get_next_data())
    h = max(f.shape[0] for f in firsts)
    w = max(f.shape[1] for f in firsts)
    total_w = w * len(paths) + 6 * (len(paths) - 1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1, quality=7)
    head = _header(total_w, header_text, subtitle) if header_text else None
    label_h = 34

    current = list(firsts)
    alive = [True] * len(paths)
    n = 0
    while any(alive) and (max_frames is None or n < max_frames):
        row = []
        for i, frame in enumerate(current):
            f = _pad_to(frame, h, w)
            lab = np.zeros((label_h, w, 3), dtype=np.uint8)
            lab[:] = (34, 34, 42)
            size = cv2.getTextSize(labels[i], FONT, 0.55, 2)[0]
            cv2.putText(lab, labels[i], ((w - size[0]) // 2, 23), FONT, 0.55, (235, 235, 240), 2, cv2.LINE_AA)
            row.append(np.concatenate([lab, f], axis=0))
            if i < len(current) - 1:
                row.append(np.full((h + label_h, 6, 3), 60, dtype=np.uint8))
        combo = np.concatenate(row, axis=1)
        if head is not None:
            combo = np.concatenate([head, combo], axis=0)
        writer.append_data(combo)
        n += 1
        for i, r in enumerate(readers):
            if not alive[i]:
                continue
            try:
                current[i] = r.get_next_data()
            except (StopIteration, IndexError):
                alive[i] = False
    writer.close()
    for r in readers:
        r.close()
    return out_path


def title_card(text, subtitle, size, seconds=3.0, fps=30, out_path=None, bullets=()):
    """A full-screen slide used between sections of the reel."""
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (16, 16, 20)
    ts = cv2.getTextSize(text, FONT, 1.3, 3)[0]
    cv2.putText(frame, text, ((w - ts[0]) // 2, h // 2 - 40), FONT, 1.3, (240, 240, 245), 3, cv2.LINE_AA)
    if subtitle:
        ss = cv2.getTextSize(subtitle, FONT, 0.7, 2)[0]
        cv2.putText(frame, subtitle, ((w - ss[0]) // 2, h // 2 + 4), FONT, 0.7, (120, 200, 250), 2, cv2.LINE_AA)
    for i, b in enumerate(bullets):
        bs = cv2.getTextSize(b, FONT, 0.55, 1)[0]
        cv2.putText(frame, b, ((w - bs[0]) // 2, h // 2 + 50 + 28 * i), FONT, 0.55, (170, 170, 180), 1, cv2.LINE_AA)
    frames = [frame] * int(seconds * fps)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1, quality=7)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return out_path
    return frames


def concat(paths, out_path, fps=30):
    """Chain clips (possibly different sizes) into one reel, letter-boxing to a common canvas."""
    sizes = []
    for p in paths:
        r = imageio.get_reader(p)
        f = r.get_next_data()
        sizes.append(f.shape[:2])
        r.close()
    h = max(s[0] for s in sizes)
    w = max(s[1] for s in sizes)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1, quality=7)
    for p in paths:
        r = imageio.get_reader(p)
        for frame in r:
            writer.append_data(_pad_to(frame, h, w))
        r.close()
    writer.close()
    return out_path
