"""Build the whole demo from trained checkpoints, in one command.

    python -m demo.build_demo --config configs/pong.yaml

Produces, under `demo_out/`:
  videos/<label>.mp4          one clip per checkpoint (game + live decision panel)
  pairs/<label>.mp4           EfficientZero vs DQN side by side at the same env-step budget
  showcase_efficientzero.mp4  the converged EfficientZero with the full planning overlay
  demo_reel.mp4               everything above stitched together with title cards
  metrics.json / metrics.md   the numbers
  learning_curves.png, score_bars.png, metrics_table.png
"""
from __future__ import annotations

import argparse
import json
import os

from src.common.config import load_config, parse_overrides
from src.common.utils import ensure_dir, get_device
from src.eval.evaluate import evaluate

from . import compose, plots


def load_manifest(run_dir):
    path = os.path.join(run_dir, "checkpoints.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoints.json in {run_dir}. Train that agent first.")
    with open(path) as f:
        return json.load(f)


def nearest(manifest, step):
    """Pick the saved checkpoint whose env-step count is closest to `step`."""
    cands = [m for m in manifest if m.get("env_steps", -1) >= 0]
    if not cands:
        cands = manifest
    return min(cands, key=lambda m: abs(m.get("env_steps", 0) - step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pong.yaml")
    ap.add_argument("--ez-run", default=None)
    ap.add_argument("--dqn-run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes", type=int, default=None, help="episodes per checkpoint for the metrics")
    ap.add_argument("--showcase-sims", type=int, default=None, help="MCTS simulations for the final showcase clip")
    ap.add_argument("--skip-videos", action="store_true", help="metrics only (fast)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    game = cfg.env.game
    ez_run = args.ez_run or os.path.join(cfg.output_dir, f"ez_{game.lower()}")
    dqn_run = args.dqn_run or os.path.join(cfg.output_dir, f"dqn_{game.lower()}")
    out = ensure_dir(args.out or cfg.demo.out_dir)
    episodes = args.episodes or int(cfg.demo.episodes_per_checkpoint)
    device = get_device(cfg.device)

    ez_manifest = load_manifest(ez_run)
    dqn_manifest = load_manifest(dqn_run)

    # ---- resolve the 7 demo checkpoints ---------------------------------------------
    plan = []
    for pair in cfg.demo.pairs:
        ez_ck = nearest(ez_manifest, int(pair.ez_step))
        dq_ck = nearest(dqn_manifest, int(pair.dqn_step))
        plan.append({
            "stage": pair.name,
            "note": pair.note,
            "ez": {"path": ez_ck["path"], "steps": ez_ck["env_steps"]},
            "dqn": {"path": dq_ck["path"], "steps": dq_ck["env_steps"]},
        })
    dqn_final = nearest(dqn_manifest, int(cfg.demo.dqn_converged_step))

    print("=== demo plan ===")
    for p in plan:
        print(f"  {p['stage']:<22} EZ@{p['ez']['steps']:>9,}   DQN@{p['dqn']['steps']:>9,}")
    print(f"  {'dqn-converged':<22} {'':>12}   DQN@{dqn_final['env_steps']:>9,}")

    ensure_dir(os.path.join(out, "videos"))
    ensure_dir(os.path.join(out, "pairs"))

    metrics = []

    def run_one(ckpt_path, label, tag, record=True):
        video = os.path.join(out, "videos", f"{tag}.mp4") if record and not args.skip_videos else None
        m = evaluate(
            ckpt_path,
            episodes=episodes,
            record=video,
            overlay=True,
            label=label,
            seed=int(cfg.demo.seed),
            fps=int(cfg.demo.fps),
            max_record_episodes=1,
            device=device,
        )
        m["tag"] = tag
        m["label"] = label
        metrics.append(m)
        return m

    # ---- 1. the six paired checkpoints ----------------------------------------------
    pair_clips = []
    for p in plan:
        ez_label = f"EfficientZero @ {p['ez']['steps']:,}"
        dq_label = f"DQN @ {p['dqn']['steps']:,}"
        ez_m = run_one(p["ez"]["path"], ez_label, f"ez_{p['ez']['steps']}")
        dq_m = run_one(p["dqn"]["path"], dq_label, f"dqn_{p['dqn']['steps']}")
        if not args.skip_videos:
            clip = compose.side_by_side(
                [ez_m["video"], dq_m["video"]],
                [ez_label, dq_label],
                os.path.join(out, "pairs", f"{p['stage']}.mp4"),
                header_text=f"{game} -- {p['stage']}",
                subtitle=f"{p['note']}   |   EfficientZero {ez_m['return_mean']:.1f}   vs   DQN {dq_m['return_mean']:.1f}",
                fps=int(cfg.demo.fps),
            )
            pair_clips.append((p, clip))

    # ---- 2. DQN at its own convergence point ----------------------------------------
    dqn_final_label = f"DQN @ {dqn_final['env_steps']:,} (converged)"
    dqn_final_m = run_one(dqn_final["path"], dqn_final_label, f"dqn_final_{dqn_final['env_steps']}")

    # ---- 3. showcase: converged EfficientZero with deeper search ---------------------
    ez_best = nearest(ez_manifest, int(cfg.demo.pairs[-1].ez_step))
    showcase = None
    if not args.skip_videos:
        showcase_m = evaluate(
            ez_best["path"],
            episodes=1,
            record=os.path.join(out, "showcase_efficientzero.mp4"),
            overlay=True,
            label="converged EfficientZero -- live planning",
            seed=int(cfg.demo.seed) + 7,
            sims=args.showcase_sims or int(cfg.demo.showcase_simulations),
            fps=int(cfg.demo.fps),
            max_record_episodes=1,
            device=device,
        )
        showcase_m["tag"] = "showcase"
        metrics.append(showcase_m)
        showcase = showcase_m["video"]

    # ---- 4. metrics ------------------------------------------------------------------
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    ez_scores = [m for m in metrics if m["agent"] == "ez"]
    dq_scores = [m for m in metrics if m["agent"] == "dqn"]
    best_ez = max(ez_scores, key=lambda m: m["return_mean"]) if ez_scores else None
    matched = None
    if best_ez:
        reached = [m for m in dq_scores if m["return_mean"] >= best_ez["return_mean"] - 1e-6]
        matched = min(reached, key=lambda m: m["env_steps"]) if reached else None

    lines = [f"# EfficientZero vs DQN -- {game}", "",
             "| checkpoint | agent | env steps | grad steps | score (mean +/- std) | train wallclock |",
             "|---|---|---|---|---|---|"]
    for m in metrics:
        lines.append(
            f"| {m['label']} | {m['agent_name']} | {m['env_steps']:,} | {m['gradient_steps']:,} | "
            f"{m['return_mean']:.1f} +/- {m['return_std']:.1f} | {m['train_wallclock_s'] / 60:.0f} min |"
        )
    if best_ez:
        lines += ["", f"**EfficientZero best**: {best_ez['return_mean']:.1f} after {best_ez['env_steps']:,} env steps."]
        if matched and matched["env_steps"] > best_ez["env_steps"]:
            factor = matched["env_steps"] / max(1, best_ez["env_steps"])
            lines.append(f"**DQN needed {matched['env_steps']:,} env steps to match it -- {factor:.0f}x more data.**")
        elif matched:
            lines.append(f"**DQN matched it at {matched['env_steps']:,} env steps.**")
        else:
            lines.append("**DQN never matched that score within the training budget shown here.**")
    with open(os.path.join(out, "metrics.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plots.learning_curves(
        os.path.join(ez_run, "train_log.csv"),
        os.path.join(dqn_run, "train_log.csv"),
        os.path.join(out, "learning_curves.png"),
        checkpoints=[p["ez"]["steps"] for p in plan],
        game=game,
    )
    table_metrics = [m for m in metrics if m.get("tag") != "showcase"]
    plots.score_bars(table_metrics, os.path.join(out, "score_bars.png"), game=game)
    plots.metrics_table_png(table_metrics, os.path.join(out, "metrics_table.png"), game=game)

    # ---- 5. the reel -----------------------------------------------------------------
    if not args.skip_videos and pair_clips:
        import imageio

        probe = imageio.get_reader(pair_clips[0][1])
        frame = probe.get_next_data()
        probe.close()
        size = (frame.shape[1], frame.shape[0])
        fps = int(cfg.demo.fps)
        segments = []
        intro = compose.title_card(
            f"EfficientZero vs DQN -- {game}",
            "same environment, same budget, radically different sample efficiency",
            size, 4.0, fps, os.path.join(out, "_cards", "intro.mp4"),
            bullets=("EfficientZero = MuZero + self-supervised consistency + value prefix + off-policy correction",),
        )
        segments.append(intro)
        for i, (p, clip) in enumerate(pair_clips):
            card = compose.title_card(
                p["stage"].replace("-", " ").title(),
                p["note"],
                size, 3.0, fps, os.path.join(out, "_cards", f"card{i}.mp4"),
            )
            segments += [card, clip]
        card = compose.title_card(
            "DQN, finally converged",
            f"it took {dqn_final['env_steps']:,} env steps to get here",
            size, 3.0, fps, os.path.join(out, "_cards", "dqn_final.mp4"),
            bullets=(f"score {dqn_final_m['return_mean']:.1f}",),
        )
        segments += [card, metrics[-2]["video"] if showcase else dqn_final_m["video"]]
        if showcase:
            card = compose.title_card(
                "Inside the planner",
                "converged EfficientZero, one move at a time",
                size, 3.5, fps, os.path.join(out, "_cards", "showcase.mp4"),
                bullets=("left: the game", "right: MCTS visit counts, Q values and the root value"),
            )
            segments += [card, showcase]
        compose.concat(segments, os.path.join(out, "demo_reel.mp4"), fps=fps)

    print("\n=== done ===")
    print(f"  {os.path.join(out, 'demo_reel.mp4')}")
    print(f"  {os.path.join(out, 'metrics.md')}")
    print("\n" + "\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
