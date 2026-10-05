"""Train EfficientZero and save checkpoints at the milestones used by the demo.

    python -m src.efficientzero.train --config configs/pong.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from ..common.atari_wrappers import make_atari, num_actions
from ..common.config import load_config, parse_overrides, save_config
from ..common.logger import CheckpointManager, RunLogger
from ..common.utils import Timer, count_parameters, ensure_dir, get_device, set_seed, visit_softmax_temperature
from .game_buffer import PrioritizedGameBuffer
from .mcts import MCTS, select_action
from .model import EfficientZeroNet
from .self_play import SelfPlayRunner
from .trainer import EZTrainer


def build_model(cfg, action_space_size: int, device):
    c = 1 if cfg.env.grayscale else 3
    obs_shape = (cfg.env.stack * c, cfg.env.size, cfg.env.size)
    model = EfficientZeroNet(obs_shape, action_space_size, cfg.model).to(device)
    return model, obs_shape


@torch.no_grad()
def quick_eval(cfg, model, device, episodes: int = 2, seed: int = 123):
    """Greedy (temperature 0) MCTS evaluation, used for the learning curve."""
    env = make_atari(
        game=cfg.env.game,
        size=cfg.env.size,
        grayscale=cfg.env.grayscale,
        stack=cfg.env.stack,
        skip=cfg.env.skip,
        episodic_life=False,
        clip_rewards=False,
        sticky_actions=cfg.env.sticky_actions,
        max_episode_steps=cfg.env.eval_max_episode_steps,
        seed=seed,
    )
    mcts = MCTS(model, cfg.mcts, num_actions(env), device)
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            batch = torch.as_tensor(obs[None], dtype=torch.uint8, device=device)
            res = mcts.run(batch, add_noise=False, rng=rng)
            action = select_action(res.visit_counts[0], 0.0, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pong.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run that exercises every code path")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.batch_size=16")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    if args.smoke:
        cfg.ez.total_env_steps = 600
        cfg.ez.start_transitions = 100
        cfg.ez.collect_steps = 50
        cfg.ez.updates_per_collect = 2
        cfg.ez.eval_interval = 10**9
        cfg.ez.eval_episodes = 1
        cfg.ez.checkpoints = [300, 600]
        cfg.train.batch_size = 4
        cfg.mcts.num_simulations = 4
        cfg.selfplay.num_envs = 2
        cfg.selfplay.segment_length = 20
        cfg.model.channels = 16
        cfg.model.blocks = 1
        cfg.model.lstm_hidden = 32
        cfg.model.proj_dim = 32
        cfg.model.fc_hidden = 32
        cfg.env.max_episode_steps = 120
        cfg.env.eval_max_episode_steps = 120

    seed = args.seed if args.seed is not None else int(cfg.seed)
    set_seed(seed)
    device = get_device(cfg.device)
    run_dir = ensure_dir(args.run_dir or os.path.join(cfg.output_dir, f"ez_{cfg.env.game.lower()}"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    probe = make_atari(game=cfg.env.game, size=cfg.env.size, stack=cfg.env.stack, seed=seed)
    action_space_size = num_actions(probe)
    probe.close()

    model, obs_shape = build_model(cfg, action_space_size, device)
    target_model, _ = build_model(cfg, action_space_size, device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    print(f"[ez] device={device} actions={action_space_size} params={count_parameters(model):,}", flush=True)

    buffer = PrioritizedGameBuffer(int(cfg.ez.buffer_capacity), float(cfg.train.per_alpha), float(cfg.train.per_beta))
    runner = SelfPlayRunner(cfg, model, buffer, device, seed=seed)
    trainer = EZTrainer(cfg, model, target_model, buffer, device, seed=seed)
    logger = RunLogger(run_dir, "EfficientZero")
    ckpt = CheckpointManager(run_dir, "ez", cfg.ez.checkpoints)
    timer = Timer()

    def payload(env_steps, training_step, eval_score=None):
        return {
            "model": model.state_dict(),
            "cfg": json.loads(json.dumps(cfg, default=str)),
            "env_steps": int(env_steps),
            "training_steps": int(training_step),
            "wallclock_s": timer.elapsed(),
            "action_space_size": action_space_size,
            "obs_shape": obs_shape,
            "agent": "ez",
            "eval_score": eval_score,
        }

    print("[ez] warm-up self-play ...", flush=True)
    runner.collect(int(cfg.ez.start_transitions), temperature=1.0)
    while len(buffer) < int(cfg.train.batch_size):  # make sure the first batch is samplable
        runner.collect(int(cfg.ez.collect_steps), temperature=1.0)

    training_step = 0
    total = int(cfg.ez.total_env_steps)
    last_eval = 0
    while runner.env_steps < total:
        temperature = visit_softmax_temperature(runner.env_steps, total)
        stats = runner.collect(int(cfg.ez.collect_steps), temperature=temperature)
        losses = {}
        for _ in range(int(cfg.ez.updates_per_collect)):
            losses = trainer.update(training_step)
            training_step += 1

        row = {"env_steps": runner.env_steps, "train_steps": training_step, "temperature": temperature}
        row.update(stats)
        row.update(losses)

        if runner.env_steps - last_eval >= int(cfg.ez.eval_interval):
            last_eval = runner.env_steps
            mean_r, std_r = quick_eval(cfg, model, device, episodes=int(cfg.ez.eval_episodes), seed=seed + 5)
            row["eval_return"] = mean_r
            row["eval_std"] = std_r
        logger.log(row)

        milestone = ckpt.due(runner.env_steps)
        if milestone is not None:
            score = row.get("eval_return")
            if score is None:
                score, _ = quick_eval(cfg, model, device, episodes=int(cfg.ez.eval_episodes), seed=seed + 5)
            ckpt.save(milestone, payload(runner.env_steps, training_step, score),
                      extra={"eval_score": score, "wallclock_s": timer.elapsed()})

    ckpt.save_final(payload(runner.env_steps, training_step), extra={"wallclock_s": timer.elapsed()})
    logger.close()
    runner.close()
    print(f"[ez] done in {Timer.fmt(timer.elapsed())} -- {run_dir}", flush=True)


if __name__ == "__main__":
    main()
