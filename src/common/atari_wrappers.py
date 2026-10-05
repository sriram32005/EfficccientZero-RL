"""Atari preprocessing shared by EfficientZero and DQN.

Both agents see exactly the same environment pipeline, which is what makes the
sample-efficiency comparison in the demo fair. The only differences are the frame
size (96x96 for EfficientZero, 84x84 for DQN -- their standard published setups).
"""
from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np

try:  # gymnasium >= 1.0 needs the ALE envs registered explicitly
    import ale_py

    gym.register_envs(ale_py)
except Exception:  # pragma: no cover
    pass

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Take a random number of no-ops on reset to de-correlate episode starts."""

    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat an action `skip` times and take the pixel-wise max of the last 2 frames."""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.last_rgb = None

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        self.last_rgb = max_frame
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[:] = 0
        self._obs_buffer[1] = obs
        self.last_rgb = obs
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat end-of-life as end-of-episode (helps value estimation); only resets on real game over."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that need it (Breakout, etc.)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


class WarpFrame(gym.ObservationWrapper):
    """Resize (and optionally grayscale) frames to `size` x `size`."""

    def __init__(self, env, size: int = 96, grayscale: bool = True):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(0, 255, (size, size, channels), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = frame[:, :, None]
        return frame


class ClipRewardEnv(gym.Wrapper):
    """Clip rewards to {-1, 0, +1} for learning, but keep the true score in `info`."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        return obs, float(np.sign(reward)), terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Stack the last `k` frames -> observation of shape (k*C, H, W), uint8, channel-first."""

    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        h, w, c = env.observation_space.shape
        self.channels = c
        self.observation_space = gym.spaces.Box(0, 255, (k * c, h, w), dtype=np.uint8)

    def _obs(self):
        # each frame is (H, W, C) -> transpose to (C, H, W), then concat on channel axis
        return np.concatenate([np.transpose(f, (2, 0, 1)) for f in self.frames], axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._obs(), reward, terminated, truncated, info


class RenderCache(gym.Wrapper):
    """Keep the most recent full-resolution RGB frame so we can record nice videos."""

    def __init__(self, env):
        super().__init__(env)
        self.rgb = None

    def _grab(self):
        try:
            self.rgb = self.env.unwrapped.render()
        except Exception:
            self.rgb = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._grab()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._grab()
        return out

    def rgb_frame(self):
        if self.rgb is None:
            self._grab()
        return self.rgb


def make_atari(
    game: str = "Pong",
    size: int = 96,
    grayscale: bool = True,
    stack: int = 4,
    skip: int = 4,
    noop_max: int = 30,
    episodic_life: bool = True,
    clip_rewards: bool = True,
    fire_reset: bool = True,
    sticky_actions: float = 0.0,
    max_episode_steps: int = 27000,
    render: bool = False,
    seed: int | None = None,
):
    """Build the shared Atari pipeline. `max_episode_steps` counts *agent* steps (post frame-skip)."""
    env_id = game if "/" in game or game.endswith("-v5") else f"ALE/{game}-v5"
    env = gym.make(
        env_id,
        frameskip=1,  # we do our own max-and-skip
        repeat_action_probability=sticky_actions,
        full_action_space=False,
        render_mode="rgb_array" if render else None,
    )
    env = RenderCache(env) if render else env
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, size=size, grayscale=grayscale)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def find_render_cache(env):
    """Walk the wrapper stack and return the RenderCache (or None)."""
    while env is not None:
        if isinstance(env, RenderCache):
            return env
        env = getattr(env, "env", None)
    return None


def num_actions(env) -> int:
    return int(env.action_space.n)
