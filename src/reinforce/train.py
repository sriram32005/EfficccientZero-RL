"""Train REINFORCE-with-baseline (policy gradient + learned value baseline) on Atari.

    python -m src.reinforce.train --config configs/pong.yaml

Implementation notes (why this is written the way it is -- the first version never learned):

* Pure episodic REINFORCE does one update per *episode*. On Pong that is ~1 update per 1000
  env steps, i.e. only ~3k updates in 3M steps, each from a single very noisy trajectory.
  The policy stayed at maximum entropy (ln 6 ~ 1.79) and greedy evaluation of a flat policy
  just returns one fixed action ("always LEFT").
* This version keeps the REINFORCE objective -E[log pi(a|s) * A(s,a)] with a learned
  baseline V(s), but estimates it from short n-step rollouts of several parallel envs with
  GAE advantages -- the standard "advantage actor-critic" form that does converge on Atari.
* Pseudo-code of one iteration:
      rollout T steps in N envs (no grad)  ->  GAE(gamma, lambda)  ->  one gradient step on
      policy_loss + vf_coef * value_loss - ent_coef * entropy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..common.atari_wrappers import make_atari, num_actions
from ..common.config import load_config, parse_overrides, save_config
from ..common.logger import CheckpointManager, RunLogger
from ..common.utils import Timer, count_parameters, ensure_dir, get_device, set_seed
from .model import ReinforceNet


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def build_model(cfg, action_space_size, device):
    c = 1 if cfg.env.grayscale else 3
    obs_shape = (cfg.reinforce.stack * c, cfg.reinforce.size, cfg.reinforce.size)
    model = ReinforceNet(obs_shape, action_space_size).to(device)
    return model, obs_shape


def make_env(cfg, seed, eval_mode=False):
    return make_atari(
        game=cfg.env.game,
        size=cfg.reinforce.size,
        grayscale=cfg.env.grayscale,
        stack=cfg.reinforce.stack,
        skip=cfg.env.skip,
        episodic_life=(not eval_mode) and bool(cfg.env.episodic_life),
        clip_rewards=not eval_mode,
        sticky_actions=cfg.env.sticky_actions,
        max_episode_steps=cfg.env.eval_max_episode_steps if eval_mode else cfg.env.max_episode_steps,
        seed=seed,
    )


class VecEnv:
    """Minimal synchronous vector env with explicit auto-reset (no gymnasium-version surprises).

    step() returns the *post-reset* observation for envs whose episode just ended, together
    with done=True, so `(1 - done)` masks the bootstrap exactly as the GAE code expects.
    """

    def __init__(self, envs, seed):
        self.envs = envs
        self.n = len(envs)
        self.seed = seed
        self.ep_ret = np.zeros(self.n, dtype=np.float64)
        self.finished_returns: list[float] = []

    def reset(self):
        obs = [e.reset(seed=self.seed + i)[0] for i, e in enumerate(self.envs)]
        self.ep_ret[:] = 0.0
        return np.stack(obs)

    def step(self, actions):
        obs_l, rew, done = [], np.zeros(self.n, np.float32), np.zeros(self.n, np.float32)
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = e.step(int(a))
            rew[i] = r  # clipped reward (training signal)
            self.ep_ret[i] += float(info.get("raw_reward", r))  # true score (logging)
            if term or trunc:
                done[i] = 1.0
                self.finished_returns.append(float(self.ep_ret[i]))
                self.ep_ret[i] = 0.0
                o, _ = e.reset()
            obs_l.append(o)
        return np.stack(obs_l), rew, done

    def close(self):
        for e in self.envs:
            e.close()


@torch.no_grad()
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """GAE(lambda). All inputs are (T, N) tensors, last_value is (N,).

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_v - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


@torch.no_grad()
def quick_eval(cfg, model, device, episodes: int = 3, seed: int = 321, greedy: bool = False):
    """Score the policy on full (un-clipped) games. Samples from pi by default -- that is the
    policy that was actually trained; argmax of a policy-gradient net can get stuck in loops."""
    env = make_env(cfg, seed, eval_mode=True)
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            logits, _ = model(torch.as_tensor(obs[None], device=device))
            if greedy:
                action = int(logits.argmax(dim=1).item())
            else:
                action = int(torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=gen).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    model.train(was_training)
    return float(np.mean(returns)), float(np.std(returns))


def a2c_update(model, optimizer, obs, actions, adv, returns, *, vf_coef, ent_coef, max_grad_norm,
               normalize_adv=True):
    """One gradient step on a flattened batch. Returns a dict of float diagnostics."""
    logits, values = model(obs)
    values = values.squeeze(-1)
    dist = Categorical(logits=logits)
    logp = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    if normalize_adv and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    policy_loss = -(logp * adv.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        var_y = returns.var()
        ev = float(1.0 - (returns - values).var() / (var_y + 1e-8)) if var_y > 1e-8 else float("nan")
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "grad_norm": float(grad_norm),
        "explained_var": ev,
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pong.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    rc = cfg.reinforce
    if args.smoke:
        rc.total_env_steps = 600
        rc.num_envs = 2
        rc.rollout_length = 8
        rc.eval_interval = 10**9
        rc.checkpoints = [200, 600]
        rc.eval_episodes = 1
        rc.log_interval = 100
        cfg.env.max_episode_steps = 100
        cfg.env.eval_max_episode_steps = 100

    seed = args.seed if args.seed is not None else int(cfg.seed)
    set_seed(seed)
    device = get_device(cfg.device)
    run_dir = ensure_dir(args.run_dir or os.path.join(cfg.output_dir, f"reinforce_{cfg.env.game.lower()}"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    num_envs = int(rc.num_envs)
    T = int(rc.rollout_length)
    envs = [make_env(cfg, seed + 100 * i) for i in range(num_envs)]
    vec = VecEnv(envs, seed)
    action_space_size = num_actions(envs[0])
    model, obs_shape = build_model(cfg, action_space_size, device)
    model.train()
    print(f"[reinforce] device={device} envs={num_envs} T={T} actions={action_space_size} "
          f"params={count_parameters(model):,}", flush=True)

    lr0 = float(rc.lr)
    if str(rc.optimizer).lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr0, eps=1e-5)
    else:  # classic A2C optimiser
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr0, alpha=0.99, eps=1e-5)
    logger = RunLogger(run_dir, "REINFORCE")
    ckpt = CheckpointManager(run_dir, "reinforce", rc.checkpoints)
    timer = Timer()

    gamma, lam = float(rc.gamma), float(rc.gae_lambda)
    total = int(rc.total_env_steps)
    log_interval = int(rc.log_interval)
    eval_interval = int(rc.eval_interval)
    eval_episodes = int(rc.eval_episodes)

    def payload(env_steps, grad_steps, eval_score=None):
        return {
            "model": model.state_dict(),
            "cfg": json.loads(json.dumps(cfg, default=str)),
            "env_steps": int(env_steps),
            "training_steps": int(grad_steps),
            "wallclock_s": timer.elapsed(),
            "action_space_size": action_space_size,
            "obs_shape": obs_shape,
            "agent": "reinforce",
            "eval_score": eval_score,
        }

    buf_obs = torch.zeros((T, num_envs) + tuple(obs_shape), dtype=torch.uint8, device=device)
    buf_act = torch.zeros((T, num_envs), dtype=torch.long, device=device)
    buf_rew = torch.zeros((T, num_envs), device=device)
    buf_done = torch.zeros((T, num_envs), device=device)
    buf_val = torch.zeros((T, num_envs), device=device)

    obs = vec.reset()
    step, grad_steps, last_log, last_eval = 0, 0, 0, 0
    stats = {}
    action_counts = np.zeros(action_space_size)

    while step < total:
        # linear learning-rate decay to 0 (helps the final policy settle)
        lr = lr0 * max(0.0, 1.0 - step / total) if bool(rc.lr_decay) else lr0
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ===== 1. rollout (no grad) ================================================
        for t in range(T):
            obs_t = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                logits, value = model(obs_t)
                action = Categorical(logits=logits).sample()
            next_obs, rew, done = vec.step(action.cpu().numpy())
            buf_obs[t] = obs_t
            buf_act[t] = action
            buf_val[t] = value.squeeze(-1)
            buf_rew[t] = torch.as_tensor(rew, device=device)
            buf_done[t] = torch.as_tensor(done, device=device)
            action_counts += np.bincount(action.cpu().numpy(), minlength=action_space_size)
            obs = next_obs
        step += T * num_envs

        # ===== 2. advantages / return targets ======================================
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(obs, device=device))
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_v.squeeze(-1), gamma, lam)

        # ===== 3. one gradient step on the whole (T*N) batch =======================
        stats = a2c_update(
            model, optimizer,
            buf_obs.reshape(T * num_envs, *obs_shape),
            buf_act.reshape(-1), adv.reshape(-1), ret.reshape(-1),
            vf_coef=float(rc.value_coeff), ent_coef=float(rc.entropy_coeff),
            max_grad_norm=float(rc.max_grad_norm), normalize_adv=bool(rc.normalize_advantage),
        )
        grad_steps += 1

        # ===== 4. logging / eval / checkpoints ======================================
        if step - last_log >= log_interval or step >= total:
            last_log = step
            recent = vec.finished_returns[-20:]
            row = {
                "env_steps": step,
                "train_steps": grad_steps,
                **stats,
                "lr": lr,
                "top_action_frac": float(action_counts.max() / max(1.0, action_counts.sum())),
                "recent_return": float(np.mean(recent)) if recent else float("nan"),
                "episodes": len(vec.finished_returns),
            }
            action_counts[:] = 0
            if step - last_eval >= eval_interval:
                last_eval = step
                mean_r, std_r = quick_eval(cfg, model, device, episodes=eval_episodes, seed=seed + 5)
                row["eval_return"], row["eval_std"] = mean_r, std_r
            logger.log(row, force_print=("eval_return" in row))

        milestone = ckpt.due(step)
        if milestone is not None:
            score, _ = quick_eval(cfg, model, device, episodes=eval_episodes, seed=seed + 5)
            ckpt.save(milestone, payload(step, grad_steps, score),
                      extra={"eval_score": score, "wallclock_s": timer.elapsed()})

    ckpt.save_final(payload(step, grad_steps), extra={"wallclock_s": timer.elapsed()})
    logger.close()
    vec.close()
    print(f"[reinforce] done in {Timer.fmt(timer.elapsed())} -- {run_dir}", flush=True)


if __name__ == "__main__":
    main()
