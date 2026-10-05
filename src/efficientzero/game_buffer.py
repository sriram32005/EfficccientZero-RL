"""Trajectory storage for EfficientZero.

Episodes are chopped into fixed-length *segments* so that memory stays bounded. Each
segment is padded with a short tail from the following segment (extra observations,
rewards, search statistics) so that n-step value targets and the K-step unroll never
run off the end of a segment.
"""
from __future__ import annotations

import numpy as np


class GameHistory:
    """One segment of a trajectory: observations, actions, rewards and MCTS statistics."""

    def __init__(self, stack: int, action_space_size: int, channels: int = 1):
        self.stack = stack
        self.channels = channels
        self.action_space_size = action_space_size
        self.obs_history = []       # list of (C, H, W) uint8 single frames
        self.actions = []
        self.rewards = []           # clipped env reward for taking actions[i]
        self.raw_rewards = []       # unclipped, used for reporting only
        self.child_visits = []      # (A,) improved policy from MCTS
        self.root_values = []       # MCTS root value
        self.length = 0             # number of *real* transitions (before padding)
        self.terminal = False

    # ----- construction ----------------------------------------------------------------
    def init(self, stacked_obs: np.ndarray):
        """`stacked_obs` is (stack*C, H, W) from the env reset."""
        c = self.channels
        for i in range(self.stack):
            self.obs_history.append(stacked_obs[i * c: (i + 1) * c].copy())

    def append(self, action: int, reward: float, next_stacked_obs: np.ndarray, raw_reward: float = 0.0):
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.raw_rewards.append(float(raw_reward))
        self.obs_history.append(next_stacked_obs[-self.channels:].copy())
        self.length += 1

    def store_search_stats(self, policy: np.ndarray, root_value: float):
        self.child_visits.append(np.asarray(policy, dtype=np.float32))
        self.root_values.append(float(root_value))

    # ----- access ----------------------------------------------------------------------
    def stacked_obs(self, index: int) -> np.ndarray:
        """Stacked observation *before* taking action `index`. Clamps at the end."""
        max_start = len(self.obs_history) - self.stack
        index = max(0, min(index, max_start))
        return np.concatenate(self.obs_history[index: index + self.stack], axis=0)

    def pad_from(self, nxt: "GameHistory", pad_steps: int):
        """Copy a tail from the next segment so targets at the end stay well defined."""
        n = min(pad_steps, nxt.length)
        for i in range(n):
            self.rewards.append(nxt.rewards[i])
            self.raw_rewards.append(nxt.raw_rewards[i])
            self.actions.append(nxt.actions[i])
            self.child_visits.append(nxt.child_visits[i])
            self.root_values.append(nxt.root_values[i])
        extra_obs = min(pad_steps + self.stack, len(nxt.obs_history) - self.stack)
        for i in range(max(0, extra_obs)):
            self.obs_history.append(nxt.obs_history[self.stack + i])

    def finalize(self):
        self.obs_history = np.asarray(self.obs_history, dtype=np.uint8)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.raw_rewards = np.asarray(self.raw_rewards, dtype=np.float32)
        self.child_visits = np.asarray(self.child_visits, dtype=np.float32)
        self.root_values = np.asarray(self.root_values, dtype=np.float32)
        return self

    def stacked_obs_array(self, index: int) -> np.ndarray:
        max_start = len(self.obs_history) - self.stack
        index = max(0, min(index, max_start))
        return self.obs_history[index: index + self.stack].reshape(-1, *self.obs_history.shape[2:])

    def __len__(self):
        return self.length


class PrioritizedGameBuffer:
    """Transition-level prioritised replay over segments (EfficientZero uses PER too)."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._prio = np.zeros(self.capacity, dtype=np.float64)
        self._lookup = np.full((self.capacity, 2), -1, dtype=np.int64)  # (game_id, pos)
        self._head = 0
        self._size = 0
        self.games = {}
        self._refcount = {}
        self._next_game_id = 0
        self.total_transitions = 0

    def __len__(self):
        return self._size

    def add_game(self, game: GameHistory, priorities: np.ndarray | None = None):
        game.finalize()
        gid = self._next_game_id
        self._next_game_id += 1
        self.games[gid] = game
        self._refcount[gid] = 0
        n = game.length
        if priorities is None:
            priorities = np.ones(n, dtype=np.float64) * (self._prio.max() if self._size else 1.0)
        priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)[:n]
        for pos in range(n):
            slot = self._head
            old_gid = self._lookup[slot, 0]
            if old_gid >= 0:
                self._refcount[old_gid] -= 1
                if self._refcount[old_gid] <= 0:
                    self.games.pop(old_gid, None)
                    self._refcount.pop(old_gid, None)
            self._lookup[slot] = (gid, pos)
            self._prio[slot] = max(1e-6, float(priorities[pos]))
            self._refcount[gid] += 1
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        self.total_transitions += n

    def sample(self, batch_size: int, rng: np.random.Generator):
        size = self._size
        prio = self._prio[:size] ** self.alpha
        total = prio.sum()
        if total <= 0:
            probs = np.ones(size) / size
        else:
            probs = prio / total
        slots = rng.choice(size, size=batch_size, p=probs, replace=True)
        weights = (size * probs[slots]) ** (-self.beta)
        weights = weights / weights.max()
        games = [self.games[self._lookup[s, 0]] for s in slots]
        positions = [int(self._lookup[s, 1]) for s in slots]
        return slots, games, positions, weights.astype(np.float32)

    def update_priorities(self, slots, priorities):
        priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)
        for s, p in zip(slots, priorities):
            self._prio[int(s)] = max(1e-6, float(p))
