"""Self-play data collection for EfficientZero.

Runs `num_envs` Atari environments in lock-step and searches all of their roots in a
single batched MCTS call, then writes fixed-length trajectory segments into the
prioritised replay buffer.
"""
from __future__ import annotations

import numpy as np
import torch

from ..common.atari_wrappers import make_atari, num_actions
from .game_buffer import GameHistory, PrioritizedGameBuffer
from .mcts import MCTS, select_action


class SelfPlayRunner:
    def __init__(self, cfg, model, buffer: PrioritizedGameBuffer, device, seed: int = 0):
        self.cfg = cfg
        self.model = model
        self.buffer = buffer
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.num_envs = int(cfg.selfplay.num_envs)
        self.stack = int(cfg.env.stack)
        self.channels = 1 if cfg.env.grayscale else 3
        self.segment_length = int(cfg.selfplay.segment_length)
        self.pad_steps = int(cfg.train.td_steps) + int(cfg.train.num_unroll_steps) + 1

        self.envs = [
            make_atari(
                game=cfg.env.game,
                size=cfg.env.size,
                grayscale=cfg.env.grayscale,
                stack=cfg.env.stack,
                skip=cfg.env.skip,
                episodic_life=cfg.env.episodic_life,
                clip_rewards=True,
                sticky_actions=cfg.env.sticky_actions,
                max_episode_steps=cfg.env.max_episode_steps,
                seed=seed + i,
            )
            for i in range(self.num_envs)
        ]
        self.action_space_size = num_actions(self.envs[0])
        self.mcts = MCTS(model, cfg.mcts, self.action_space_size, device)

        self.obs = []
        self.cur, self.pending = [], []
        self.ep_return, self.ep_len = [], []
        for env in self.envs:
            o, _ = env.reset()
            self.obs.append(o)
            g = GameHistory(self.stack, self.action_space_size, self.channels)
            g.init(o)
            self.cur.append(g)
            self.pending.append(None)
            self.ep_return.append(0.0)
            self.ep_len.append(0)

        self.env_steps = 0
        self.finished_returns = []
        self.finished_lengths = []

    # ----------------------------------------------------------------------------------
    def _push(self, game: GameHistory):
        if game.length > 0:
            self.buffer.add_game(game)

    def _rotate(self, i: int, terminal: bool):
        """Close the current segment for env `i`; the previous one is flushed once padded."""
        game = self.cur[i]
        game.terminal = terminal
        if self.pending[i] is not None:      # safety net: flush whatever we have
            self.pending[i].pad_from(game, self.pad_steps)
            self._push(self.pending[i])
            self.pending[i] = None
        if terminal:
            self._push(game)
            self.cur[i] = None
        else:
            self.pending[i] = game
            nxt = GameHistory(self.stack, self.action_space_size, self.channels)
            nxt.init(self.obs[i])
            self.cur[i] = nxt

    def _maybe_flush(self, i: int):
        """Push the previous segment as soon as the new one has produced enough tail."""
        if self.pending[i] is not None and self.cur[i] is not None and self.cur[i].length >= self.pad_steps:
            self.pending[i].pad_from(self.cur[i], self.pad_steps)
            self._push(self.pending[i])
            self.pending[i] = None

    @torch.no_grad()
    def collect(self, num_steps: int, temperature: float = 1.0):
        """Collect roughly `num_steps` environment steps in total (across all envs)."""
        collected = 0
        while collected < num_steps:
            obs_batch = torch.as_tensor(np.asarray(self.obs), dtype=torch.uint8, device=self.device)
            result = self.mcts.run(obs_batch, add_noise=True, rng=self.rng)

            for i, env in enumerate(self.envs):
                action = select_action(result.visit_counts[i], temperature, self.rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                raw = float(info.get("raw_reward", reward))
                self.cur[i].store_search_stats(result.policies[i], float(result.root_values[i]))
                self.cur[i].append(action, reward, next_obs, raw)
                self.obs[i] = next_obs
                self.ep_return[i] += raw
                self.ep_len[i] += 1
                collected += 1
                self.env_steps += 1

                done = terminated or truncated
                real_done = done and getattr(env.unwrapped, "ale", None) is not None
                if done:
                    self._rotate(i, terminal=True)
                    self.finished_returns.append(self.ep_return[i])
                    self.finished_lengths.append(self.ep_len[i])
                    self.ep_return[i], self.ep_len[i] = 0.0, 0
                    o, _ = env.reset()
                    self.obs[i] = o
                    g = GameHistory(self.stack, self.action_space_size, self.channels)
                    g.init(o)
                    self.cur[i] = g
                elif self.cur[i].length >= self.segment_length:
                    self._rotate(i, terminal=False)
                else:
                    self._maybe_flush(i)
                del real_done
        return {
            "env_steps": self.env_steps,
            "buffer_size": len(self.buffer),
            "recent_return": float(np.mean(self.finished_returns[-10:])) if self.finished_returns else float("nan"),
            "episodes": len(self.finished_returns),
        }

    def close(self):
        for env in self.envs:
            env.close()
