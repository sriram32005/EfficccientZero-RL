"""Shared helpers: MuZero/EfficientZero value transforms, seeding, timers."""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# seeding / devices
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    @staticmethod
    def fmt(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}h{m:02d}m{s:02d}s"


# --------------------------------------------------------------------------------------
# value / reward transforms  (MuZero appendix F)
# --------------------------------------------------------------------------------------
def scalar_transform(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x  -- squashes large returns."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inverse_scalar_transform(x: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """Inverse of `scalar_transform`."""
    numer = torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1
    out = torch.sign(x) * (((numer / (2 * eps)) ** 2) - 1)
    return out


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Scalar -> two-hot categorical distribution over [-support_size, support_size]."""
    x = scalar_transform(x)
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob_upper = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1, device=x.device, dtype=torch.float32)
    idx_lower = (floor + support_size).long().unsqueeze(-1)
    logits.scatter_(2, idx_lower, (1 - prob_upper).unsqueeze(-1))
    idx_upper = idx_lower + 1
    prob_upper = prob_upper.unsqueeze(-1)
    valid = idx_upper <= 2 * support_size
    idx_upper = idx_upper.clamp(max=2 * support_size)
    logits.scatter_add_(2, idx_upper, prob_upper * valid.float())
    return logits


def support_to_scalar(logits: torch.Tensor, support_size: int, is_logits: bool = True) -> torch.Tensor:
    """Categorical distribution -> scalar (expectation, then inverse transform)."""
    probs = torch.softmax(logits, dim=-1) if is_logits else logits
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=probs.dtype)
    x = torch.sum(support * probs, dim=-1, keepdim=True)
    return inverse_scalar_transform(x)


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Pass-through forward, multiply the gradient by `scale` on the backward pass."""
    return tensor * scale + tensor.detach() * (1 - scale)


# --------------------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------------------
def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return end
    frac = min(1.0, step / duration)
    return start + frac * (end - start)


def visit_softmax_temperature(training_step: int, total_steps: int) -> float:
    """EfficientZero's temperature schedule over the visit-count distribution."""
    frac = training_step / max(1, total_steps)
    if frac < 0.5:
        return 1.0
    if frac < 0.75:
        return 0.5
    return 0.25


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def symlog_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }
