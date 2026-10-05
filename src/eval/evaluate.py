"""Evaluate one checkpoint: play N episodes, write metrics JSON and (optionally) a video.

    python -m src.eval.evaluate --ckpt runs/ez_pong/checkpoints/ez_step00020000.pt \
        --episodes 5 --record demo_out/videos/ez_20k.mp4 --overlay --label "EfficientZero @ 20k"
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..common.utils import ensure_dir, get_device
from .agents import EZAgent, load_agent, load_checkpoint, make_eval_env
from .overlay import render_frame


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / max(1e-8, p.sum())
    return float(-(p * np.log(p + 1e-12)).sum())


def evaluate(
    ckpt_path,
    episodes=5,
    record=None,
    overlay=True,
    label="",
    seed=0,
    sims=None,
    fps=30,
    max_record_episodes=1,
    temperature=0.0,
    epsilon=0.01,
    device=None,
):
    device = device or get_device("auto")
    meta = load_checkpoint(ckpt_path, "cpu")
    agent_type = meta.get("agent", "ez")
    cfg = meta["cfg"]
    kwargs = {"seed": seed}
    if agent_type == "ez":
        kwargs.update({"num_simulations": sims, "temperature": temperature})
    else:
        kwargs.update({"epsilon": epsilon})
    agent = load_agent(ckpt_path, device=device, **kwargs)

    env = make_eval_env(cfg, agent_type, seed=seed, render=record is not None)
    action_meanings = env.unwrapped.get_action_meanings()

    writer = None
    if record:
        import imageio

        ensure_dir(os.path.dirname(os.path.abspath(record)))
        writer = imageio.get_writer(record, fps=fps, macro_block_size=1, quality=7)

    returns, lengths, decision_times, value_traces, entropies = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 * ep)
        done, total, steps = False, 0.0, 0
        value_history = []
        while not done:
            t0 = time.time()
            action, info = agent.act(obs)
            decision_times.append(time.time() - t0)
            value_history.append(info["root_value"])
            if "policy" in info:
                entropies.append(_entropy(info["policy"]))

            if writer is not None and ep < max_record_episodes:
                rgb = env.unwrapped.render() if hasattr(env.unwrapped, "render") else None
                if overlay:
                    frame = render_frame(
                        rgb,
                        info,
                        agent.name,
                        action_meanings,
                        total,
                        steps,
                        checkpoint_label=label,
                        env_steps=agent.env_steps,
                        value_history=value_history,
                    )
                else:
                    import cv2

                    frame = cv2.resize(rgb, (rgb.shape[1] * 3, rgb.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
                writer.append_data(frame)

            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        value_traces.append(float(np.mean(value_history)) if value_history else 0.0)
        print(f"  [{agent.name} @ {agent.env_steps:,}] episode {ep + 1}/{episodes}: score {total:g} ({steps} steps)", flush=True)

    if writer is not None:
        writer.close()
    env.close()

    metrics = {
        "checkpoint": os.path.abspath(ckpt_path),
        "agent": agent_type,
        "agent_name": agent.name,
        "label": label,
        "env_steps": agent.env_steps,
        "gradient_steps": agent.training_steps,
        "train_wallclock_s": agent.wallclock_s,
        "episodes": episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "returns": [float(r) for r in returns],
        "episode_length_mean": float(np.mean(lengths)),
        "decision_ms_mean": float(np.mean(decision_times) * 1000),
        "mean_value_estimate": float(np.mean(value_traces)) if value_traces else 0.0,
        "video": os.path.abspath(record) if record else None,
    }
    if entropies:
        metrics["policy_entropy_mean"] = float(np.mean(entropies))
    if isinstance(agent, EZAgent):
        metrics["num_simulations"] = agent.num_simulations
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--record", default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None, help="where to write the metrics json")
    args = ap.parse_args()

    metrics = evaluate(
        args.ckpt,
        episodes=args.episodes,
        record=args.record,
        overlay=args.overlay,
        label=args.label,
        seed=args.seed,
        sims=args.sims,
        fps=args.fps,
    )
    print(json.dumps(metrics, indent=2))
    if args.out:
        ensure_dir(os.path.dirname(os.path.abspath(args.out)))
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
