"""Train DQN (double + dueling + n-step) with the same env pipeline as EfficientZero.

    python -m src.dqn.train --config configs/pong.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ..common.atari_wrappers import make_atari, num_actions
from ..common.config import load_config, parse_overrides, save_config
from ..common.logger import CheckpointManager, RunLogger
from ..common.utils import Timer, count_parameters, ensure_dir, get_device, linear_schedule, set_seed
from .model import DQNNet
from .replay import FrameReplayBuffer


def build_model(cfg, action_space_size, device):
    c = 1 if cfg.env.grayscale else 3
    obs_shape = (cfg.dqn.stack * c, cfg.dqn.size, cfg.dqn.size)
    model = DQNNet(obs_shape, action_space_size, dueling=bool(cfg.dqn.dueling)).to(device)
    return model, obs_shape


def make_env(cfg, seed, eval_mode=False):
    return make_atari(
        game=cfg.env.game,
        size=cfg.dqn.size,
        grayscale=cfg.env.grayscale,
        stack=cfg.dqn.stack,
        skip=cfg.env.skip,
        episodic_life=(not eval_mode) and bool(cfg.env.episodic_life),
        clip_rewards=not eval_mode,
        sticky_actions=cfg.env.sticky_actions,
        max_episode_steps=cfg.env.eval_max_episode_steps if eval_mode else cfg.env.max_episode_steps,
        seed=seed,
    )


@torch.no_grad()
def quick_eval(cfg, model, device, episodes: int = 3, seed: int = 321, epsilon: float = 0.01):
    env = make_env(cfg, seed, eval_mode=True)
    rng = np.random.default_rng(seed)
    model.eval()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                q = model(torch.as_tensor(obs[None], device=device))
                action = int(q.argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    model.train()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pong.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    if args.smoke:
        cfg.dqn.total_env_steps = 300
        cfg.dqn.learning_starts = 50
        cfg.dqn.buffer_capacity = 500
        cfg.dqn.eval_interval = 10**9
        cfg.dqn.checkpoints = [100, 300]
        cfg.dqn.batch_size = 8
        cfg.dqn.eval_episodes = 1
        cfg.dqn.exploration_steps = 100
        cfg.env.max_episode_steps = 100
        cfg.env.eval_max_episode_steps = 100

    seed = args.seed if args.seed is not None else int(cfg.seed)
    set_seed(seed)
    device = get_device(cfg.device)
    run_dir = ensure_dir(args.run_dir or os.path.join(cfg.output_dir, f"dqn_{cfg.env.game.lower()}"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    env = make_env(cfg, seed)
    action_space_size = num_actions(env)
    model, obs_shape = build_model(cfg, action_space_size, device)
    target = DQNNet(obs_shape, action_space_size, dueling=bool(cfg.dqn.dueling)).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    print(f"[dqn] device={device} actions={action_space_size} params={count_parameters(model):,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.dqn.lr), eps=1.5e-4)
    c = 1 if cfg.env.grayscale else 3
    buffer = FrameReplayBuffer(
        int(cfg.dqn.buffer_capacity), (c, cfg.dqn.size, cfg.dqn.size), int(cfg.dqn.stack),
        n_step=int(cfg.dqn.n_step), gamma=float(cfg.dqn.gamma),
    )
    logger = RunLogger(run_dir, "DQN")
    ckpt = CheckpointManager(run_dir, "dqn", cfg.dqn.checkpoints)
    timer = Timer()
    rng = np.random.default_rng(seed)

    def payload(env_steps, grad_steps, eval_score=None):
        return {
            "model": model.state_dict(),
            "cfg": json.loads(json.dumps(cfg, default=str)),
            "env_steps": int(env_steps),
            "training_steps": int(grad_steps),
            "wallclock_s": timer.elapsed(),
            "action_space_size": action_space_size,
            "obs_shape": obs_shape,
            "agent": "dqn",
            "eval_score": eval_score,
        }

    obs, _ = env.reset(seed=seed)
    ep_return, ep_len, returns_hist = 0.0, 0, []
    grad_steps, last_eval, last_loss, last_q = 0, 0, float("nan"), float("nan")

    for step in range(1, int(cfg.dqn.total_env_steps) + 1):
        eps = linear_schedule(1.0, float(cfg.dqn.final_epsilon), int(cfg.dqn.exploration_steps), step)
        if step < int(cfg.dqn.learning_starts) or rng.random() < eps:
            action = int(rng.integers(action_space_size))
        else:
            with torch.no_grad():
                q = model(torch.as_tensor(obs[None], device=device))
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs[-c:], action, reward, bool(done))
        ep_return += float(info.get("raw_reward", reward))
        ep_len += 1
        obs = next_obs
        if done:
            returns_hist.append(ep_return)
            ep_return, ep_len = 0.0, 0
            obs, _ = env.reset()

        # ----- learn -----
        if step >= int(cfg.dqn.learning_starts) and step % int(cfg.dqn.train_freq) == 0 and buffer.can_sample(int(cfg.dqn.batch_size)):
            s, a, r, s2, d = buffer.sample(int(cfg.dqn.batch_size), rng)
            s = torch.as_tensor(s, device=device)
            s2 = torch.as_tensor(s2, device=device)
            a = torch.as_tensor(a, device=device)
            r = torch.as_tensor(r, device=device)
            d = torch.as_tensor(d, device=device)

            q = model(s).gather(1, a.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                next_actions = model(s2).argmax(dim=1, keepdim=True)          # double DQN
                next_q = target(s2).gather(1, next_actions).squeeze(1)
                gamma_n = float(cfg.dqn.gamma) ** int(cfg.dqn.n_step)
                target_q = r + (1.0 - d) * gamma_n * next_q
            loss = F.smooth_l1_loss(q, target_q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.dqn.max_grad_norm))
            optimizer.step()
            grad_steps += 1
            last_loss, last_q = float(loss.item()), float(q.mean().item())

            if grad_steps % int(cfg.dqn.target_update_interval) == 0:
                target.load_state_dict(model.state_dict())

        # ----- logging / eval / checkpoints -----
        if step % int(cfg.dqn.log_interval) == 0:
            logger.log({
                "env_steps": step,
                "train_steps": grad_steps,
                "epsilon": eps,
                "loss": last_loss,
                "mean_q": last_q,
                "recent_return": float(np.mean(returns_hist[-10:])) if returns_hist else float("nan"),
                "episodes": len(returns_hist),
                "buffer_size": len(buffer),
            })

        if step - last_eval >= int(cfg.dqn.eval_interval):
            last_eval = step
            mean_r, std_r = quick_eval(cfg, model, device, episodes=int(cfg.dqn.eval_episodes), seed=seed + 5)
            logger.log({"env_steps": step, "train_steps": grad_steps, "eval_return": mean_r, "eval_std": std_r}, force_print=True)

        milestone = ckpt.due(step)
        if milestone is not None:
            score, _ = quick_eval(cfg, model, device, episodes=int(cfg.dqn.eval_episodes), seed=seed + 5)
            ckpt.save(milestone, payload(step, grad_steps, score), extra={"eval_score": score, "wallclock_s": timer.elapsed()})

    ckpt.save_final(payload(int(cfg.dqn.total_env_steps), grad_steps), extra={"wallclock_s": timer.elapsed()})
    logger.close()
    env.close()
    print(f"[dqn] done in {Timer.fmt(timer.elapsed())} -- {run_dir}", flush=True)


if __name__ == "__main__":
    main()
