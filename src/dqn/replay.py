"""Frame-level replay buffer for DQN.

Stores single frames rather than stacked observations (4x less memory), rebuilding the
frame stack at sample time and zeroing out frames that belong to a previous episode.
Supports n-step returns.
"""
from __future__ import annotations

import numpy as np


class FrameReplayBuffer:
    def __init__(self, capacity: int, frame_shape, stack: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = int(capacity)
        self.stack = int(stack)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        c, h, w = frame_shape
        self.frames = np.zeros((self.capacity, c, h, w), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminals = np.zeros(self.capacity, dtype=bool)
        self.total = 0

    def __len__(self):
        return min(self.total, self.capacity)

    def add(self, frame: np.ndarray, action: int, reward: float, terminal: bool):
        """`frame` is the newest single frame of the state in which `action` was taken."""
        idx = self.total % self.capacity
        self.frames[idx] = frame
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.terminals[idx] = bool(terminal)
        self.total += 1

    # ----------------------------------------------------------------------------------
    def _stack_at(self, t: int) -> np.ndarray:
        """Frame stack for absolute time index `t`, zero-padded across episode boundaries."""
        idxs = [t - i for i in range(self.stack - 1, -1, -1)]
        out = []
        blocked = False
        for i in reversed(range(len(idxs))):  # walk backwards in time from t
            ti = idxs[i]
            if ti < 0 or blocked:
                out.append(np.zeros_like(self.frames[0]))
                continue
            out.append(self.frames[ti % self.capacity])
            if i > 0 and self.terminals[(ti - 1) % self.capacity]:
                blocked = True  # anything older belongs to the previous episode
        out = out[::-1]
        return np.concatenate(out, axis=0)

    def can_sample(self, batch_size: int) -> bool:
        lo, hi = self._valid_range()
        return hi - lo >= max(batch_size, 1)

    def _valid_range(self):
        lo = max(0, self.total - self.capacity) + self.stack
        hi = self.total - self.n_step  # need frame at t + n_step
        return lo, max(lo, hi)

    def sample(self, batch_size: int, rng: np.random.Generator):
        lo, hi = self._valid_range()
        ts = rng.integers(lo, hi, size=batch_size)
        states, next_states, actions, returns, dones = [], [], [], [], []
        for t in ts:
            t = int(t)
            states.append(self._stack_at(t))
            actions.append(self.actions[t % self.capacity])
            g, done, steps = 0.0, False, self.n_step
            for k in range(self.n_step):
                g += (self.gamma ** k) * float(self.rewards[(t + k) % self.capacity])
                if self.terminals[(t + k) % self.capacity]:
                    done, steps = True, k + 1
                    break
            returns.append(g)
            dones.append(done)
            next_states.append(self._stack_at(t + steps))
        return (
            np.asarray(states, dtype=np.uint8),
            np.asarray(actions, dtype=np.int64),
            np.asarray(returns, dtype=np.float32),
            np.asarray(next_states, dtype=np.uint8),
            np.asarray(dones, dtype=np.float32),
        )
