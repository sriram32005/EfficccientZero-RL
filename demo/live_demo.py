"""Live demo: watch an agent play with its decision process drawn next to the game.

    # converged EfficientZero, live, in a window
    python -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_final.pt

    # EfficientZero and DQN side by side, same seed
    python -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_final.pt \
                             --ckpt2 runs/dqn_pong/checkpoints/dqn_final.pt

    # no display available (projector-safe): render straight to mp4
    python -m demo.live_demo --ckpt ... --headless --out demo_out/live.mp4

Keys: SPACE pause/resume | n single step | r restart episode | q quit
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from src.common.utils import ensure_dir, get_device
from src.eval.agents import load_agent, load_checkpoint, make_eval_env
from src.eval.overlay import render_frame


class LivePlayer:
    def __init__(self, ckpt, device, seed, sims=None, label=""):
        meta = load_checkpoint(ckpt, "cpu")
        self.agent_type = meta.get("agent", "ez")
        self.cfg = meta["cfg"]
        kwargs = {"seed": seed}
        if self.agent_type == "ez":
            kwargs.update({"num_simulations": sims, "temperature": 0.0})
        self.agent = load_agent(ckpt, device=device, **kwargs)
        self.env = make_eval_env(self.cfg, self.agent_type, seed=seed, render=True)
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        self.label = label or f"{self.agent.name} @ {self.agent.env_steps:,} steps"
        self.seed = seed
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset(seed=self.seed)
        self.score = 0.0
        self.steps = 0
        self.done = False
        self.value_history = []
        self.last_info = None

    def step(self):
        if self.done:
            return
        action, info = self.agent.act(self.obs)
        self.last_info = info
        self.value_history.append(info["root_value"])
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.score += float(reward)
        self.steps += 1
        self.done = terminated or truncated

    def frame(self, panel_width=430, game_scale=3):
        rgb = self.env.unwrapped.render()
        info = self.last_info or {"q_values": np.zeros(self.agent.action_space_size), "root_value": 0.0, "action": -1}
        extra = ["EPISODE OVER -- press r to restart"] if self.done else []
        return render_frame(
            rgb,
            info,
            self.agent.name,
            self.action_meanings,
            self.score,
            self.steps,
            checkpoint_label=self.label,
            env_steps=self.agent.env_steps,
            value_history=self.value_history,
            panel_width=panel_width,
            game_scale=game_scale,
            extra_lines=extra,
        )

    def close(self):
        self.env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None, help="second agent to run side by side (usually DQN)")
    ap.add_argument("--sims", type=int, default=None, help="MCTS simulations per move (EfficientZero)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default="demo_out/live.mp4")
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()

    device = get_device("auto")
    players = [LivePlayer(args.ckpt, device, args.seed, args.sims)]
    if args.ckpt2:
        players.append(LivePlayer(args.ckpt2, device, args.seed))

    writer = None
    if args.headless:
        import imageio

        ensure_dir(os.path.dirname(os.path.abspath(args.out)))
        writer = imageio.get_writer(args.out, fps=int(args.fps), macro_block_size=1, quality=7)
    else:
        cv2.namedWindow("EfficientZero demo", cv2.WINDOW_NORMAL)

    paused, quit_now, n = False, False, 0
    period = 1.0 / max(1e-3, args.fps)
    try:
        while not quit_now and n < args.max_steps:
            t0 = time.time()
            if not paused:
                for p in players:
                    p.step()
                n += 1

            frames = [p.frame(game_scale=args.scale) for p in players]
            h = max(f.shape[0] for f in frames)
            frames = [np.pad(f, ((0, h - f.shape[0]), (0, 0), (0, 0))) for f in frames]
            canvas = frames[0] if len(frames) == 1 else np.concatenate(
                [frames[0], np.full((h, 8, 3), 60, dtype=np.uint8), frames[1]], axis=1
            )

            if writer is not None:
                writer.append_data(canvas)
                if all(p.done for p in players):
                    break
            else:
                cv2.imshow("EfficientZero demo", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                wait = max(1, int((period - (time.time() - t0)) * 1000))
                key = cv2.waitKey(wait) & 0xFF
                if key == ord("q"):
                    quit_now = True
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("n"):
                    for p in players:
                        p.step()
                    n += 1
                elif key == ord("r"):
                    for p in players:
                        p.reset()
    finally:
        if writer is not None:
            writer.close()
            print(f"[live] wrote {args.out}")
        else:
            cv2.destroyAllWindows()
        for p in players:
            p.close()
    for p in players:
        print(f"[live] {p.label}: final score {p.score:g} in {p.steps} steps")


if __name__ == "__main__":
    main()
