"""Plots for the demo: learning curves, sample-efficiency bar chart, metric table image."""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK = "#16161a"
EZ_C = "#5ac8fa"
REINFORCE_C = "#faaa5a"


def _read_curve(csv_path, key="eval_return"):
    xs, ys = [], []
    if not os.path.exists(csv_path):
        return np.array([]), np.array([])
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = row.get(key, "")
            if v in ("", None):
                continue
            try:
                ys.append(float(v))
                xs.append(float(row["env_steps"]))
            except ValueError:
                continue
    return np.asarray(xs), np.asarray(ys)


def learning_curves(ez_log, reinforce_log, out_path, checkpoints=None, game="Atari"):
    ez_x, ez_y = _read_curve(ez_log)
    rf_x, rf_y = _read_curve(reinforce_log)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=DARK)
    ax.set_facecolor(DARK)
    if len(ez_x):
        ax.plot(ez_x, ez_y, color=EZ_C, lw=2.2, marker="o", ms=4, label="EfficientZero")
    if len(rf_x):
        ax.plot(rf_x, rf_y, color=REINFORCE_C, lw=2.2, marker="s", ms=4, label="REINFORCE")
    for spine in ax.spines.values():
        spine.set_color("#44444c")
    ax.tick_params(colors="#c8c8d0")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("environment steps (log scale)", color="#c8c8d0")
    ax.set_ylabel("evaluation score", color="#c8c8d0")
    ax.set_title(f"Sample efficiency on {game}: EfficientZero vs REINFORCE", color="#f0f0f5", fontsize=14)
    ax.grid(alpha=0.18, color="#7a7a88")
    for cp in checkpoints or []:
        ax.axvline(cp, color="#8888aa", ls="--", lw=0.8, alpha=0.5)
    if len(ez_x) or len(rf_x):
        leg = ax.legend(facecolor="#22222a", edgecolor="#44444c")
        for t in leg.get_texts():
            t.set_color("#e8e8f0")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=DARK)
    plt.close(fig)
    return out_path


def score_bars(metrics, out_path, game="Atari"):
    """Grouped bar chart of the evaluated score at every demo checkpoint."""
    labels = [m["label"] for m in metrics]
    scores = [m["return_mean"] for m in metrics]
    errs = [m["return_std"] for m in metrics]
    colors = [EZ_C if m["agent"] == "ez" else REINFORCE_C for m in metrics]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.bar(range(len(scores)), scores, yerr=errs, color=colors, capsize=4, edgecolor="#22222a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right", color="#c8c8d0", fontsize=9)
    ax.set_ylabel("evaluation score", color="#c8c8d0")
    ax.set_title(f"Score at each demo checkpoint ({game})", color="#f0f0f5", fontsize=14)
    ax.tick_params(colors="#c8c8d0")
    for spine in ax.spines.values():
        spine.set_color("#44444c")
    ax.grid(axis="y", alpha=0.18, color="#7a7a88")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=DARK)
    plt.close(fig)
    return out_path


def metrics_table_png(metrics, out_path, game="Atari"):
    cols = ["checkpoint", "agent", "env steps", "grad steps", "score", "±std", "train time"]
    rows = []
    for m in metrics:
        rows.append([
            m["label"],
            m["agent_name"],
            f"{m['env_steps']:,}",
            f"{m['gradient_steps']:,}",
            f"{m['return_mean']:.1f}",
            f"{m['return_std']:.1f}",
            f"{m['train_wallclock_s'] / 60:.0f} min",
        ])
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.6), facecolor=DARK)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#44444c")
        if r == 0:
            cell.set_facecolor("#2a2a34")
            cell.set_text_props(color="#f0f0f5", weight="bold")
        else:
            is_ez = metrics[r - 1]["agent"] == "ez"
            cell.set_facecolor("#1c1c22")
            cell.set_text_props(color=EZ_C if is_ez else REINFORCE_C)
    ax.set_title(f"Demo metrics ({game})", color="#f0f0f5", fontsize=13, pad=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=DARK)
    plt.close(fig)
    return out_path
