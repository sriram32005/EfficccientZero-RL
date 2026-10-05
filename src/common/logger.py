"""Lightweight CSV + console logging and checkpoint bookkeeping (no external deps)."""
from __future__ import annotations

import csv
import json
import os
import time

from .utils import Timer, ensure_dir


class CSVLogger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        self.path = path
        self.fields = None
        self._fh = None
        self._writer = None

    def log(self, row: dict) -> None:
        if self._fh is None:
            self.fields = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
            self._writer.writeheader()
        clean = {k: row.get(k, "") for k in self.fields}
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class RunLogger:
    """Console + CSV logger with a wall-clock timer, used by both trainers."""

    def __init__(self, run_dir: str, name: str):
        self.run_dir = ensure_dir(run_dir)
        self.name = name
        self.csv = CSVLogger(os.path.join(run_dir, "train_log.csv"))
        self.timer = Timer()
        self._last_print = 0.0

    def log(self, row: dict, force_print: bool = False) -> None:
        row = dict(row)
        row["wallclock_s"] = round(self.timer.elapsed(), 2)
        self.csv.log(row)
        now = time.time()
        if force_print or now - self._last_print > 10:
            self._last_print = now
            parts = []
            for k, v in row.items():
                parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
            print(f"[{self.name}] " + " | ".join(parts), flush=True)

    def close(self):
        self.csv.close()


class CheckpointManager:
    """Saves checkpoints at configured env-step milestones and records a manifest."""

    def __init__(self, run_dir: str, agent: str, milestones):
        self.dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
        self.agent = agent
        self.milestones = sorted(int(m) for m in milestones)
        self._done = set()
        self.manifest_path = os.path.join(run_dir, "checkpoints.json")
        self.manifest = []

    def due(self, env_steps: int):
        """Return the milestone that has just been passed (or None)."""
        for m in self.milestones:
            if m not in self._done and env_steps >= m:
                self._done.add(m)
                return m
        return None

    def save(self, milestone: int, payload: dict, extra: dict | None = None) -> str:
        path = os.path.join(self.dir, f"{self.agent}_step{milestone:08d}.pt")
        import torch

        torch.save(payload, path)
        entry = {"agent": self.agent, "env_steps": int(milestone), "path": path}
        entry.update(extra or {})
        self.manifest.append(entry)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        print(f"[ckpt] saved {path}", flush=True)
        return path

    def save_final(self, payload: dict, extra: dict | None = None) -> str:
        path = os.path.join(self.dir, f"{self.agent}_final.pt")
        import torch

        torch.save(payload, path)
        entry = {"agent": self.agent, "env_steps": -1, "path": path, "final": True}
        entry.update(extra or {})
        self.manifest.append(entry)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return path
