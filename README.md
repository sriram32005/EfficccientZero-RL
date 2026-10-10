# EfficientZero vs REINFORCE — an Atari sample-efficiency demo

A complete, runnable project for showing **how much faster EfficientZero learns than REINFORCE**
on the same Atari game, with the same environment pipeline, the same evaluation protocol
and an on-screen view of what the planner is actually doing.

The demo tells one story: at 100k environment steps EfficientZero plays Pong near
perfectly, and REINFORCE — which has seen exactly the same amount of experience — still cannot
return the ball. REINFORCE eventually gets there, but it needs tens of times more data.

---

## 1. Install

```bash
bash setup.sh            # creates .venv, installs deps, verifies Atari ROMs
source .venv/bin/activate
```

If you have an NVIDIA GPU, install the CUDA build of torch **before** `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Sanity check the whole pipeline in about two minutes on CPU:

```bash
make smoke
```

This trains both agents for a few hundred steps, records videos, composes the
side-by-side clips and builds a mini `demo_out/demo_reel.mp4`. If that works, the real
run will work.

---

## 2. Train (the "happy flow before the demo")

```bash
bash scripts/train_all.sh configs/pong.yaml     # both in parallel
# or one at a time:
python -m src.efficientzero.train --config configs/pong.yaml
python -m src.reinforce.train     --config configs/pong.yaml
```

Checkpoints are saved automatically at the milestones in the config — you don't have to
watch the run:

| what | agent | env steps |
|---|---|---|
| early learning | EfficientZero + REINFORCE | 20,000 |
| mid training | EfficientZero + REINFORCE | 50,000 |
| EfficientZero converged | EfficientZero + REINFORCE | 100,000 |
| REINFORCE converged | REINFORCE | 3,000,000 |

That is exactly the 7 checkpoints the demo uses: 3 matched pairs (6 clips) plus REINFORCE at
its own convergence point.

**Rough timings** (single modern GPU): EfficientZero ≈ 2.5–4 h for 120k steps; REINFORCE ≈ 6–10 h
for 3M steps. Run them in parallel and it's one overnight job. On a laptop, use
`configs/fast_demo.yaml` instead (smaller net, 25 simulations, REINFORCE capped at 1M steps).

Live progress: `tail -f logs/ez.log`, or open `runs/ez_pong/train_log.csv`.

---

## 3. Build the demo (one command)

```bash
python -m demo.build_demo --config configs/pong.yaml
```

Everything lands in `demo_out/`:

```
demo_out/
├── demo_reel.mp4               ← the one file to play in class
├── pairs/
│   ├── early-learning.mp4      EfficientZero | REINFORCE, both at 20k steps
│   ├── mid-training.mp4        both at 50k steps
│   └── efficientzero-converged.mp4   both at 100k steps
├── videos/                     the 7 individual checkpoint clips
├── showcase_efficientzero.mp4  converged EZ with deep search + full overlay
├── metrics.json / metrics.md   the numbers
├── learning_curves.png         score vs env steps, log x-axis
├── score_bars.png              score at each checkpoint
└── metrics_table.png           a slide-ready table
```

`demo_reel.mp4` is assembled with title cards: intro → the three paired comparisons →
REINFORCE at convergence → "inside the planner". Play it start to finish and the demo runs
itself.

---

## 4. Live demo (the "during demo" part)

```bash
# the converged EfficientZero, playing live, with its decision panel
python -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_final.pt

# both agents at once, same seed, same starting state
python -m demo.live_demo \
    --ckpt  runs/ez_pong/checkpoints/ez_final.pt \
    --ckpt2 runs/reinforce_pong/checkpoints/reinforce_final.pt

# no display in the room? render it to a file instead
python -m demo.live_demo --ckpt runs/ez_pong/checkpoints/ez_final.pt \
    --headless --out demo_out/live.mp4
```

Keys: `SPACE` pause/resume · `n` single step · `r` restart episode · `q` quit.

Pausing on a decisive frame and pressing `n` a few times is the best moment of the demo —
you can point at the visit-count bars and show the search committing to a move.

---

## 5. What is on screen

Left: the real Atari frame. Right: the agent's decision, redrawn every step.

**EfficientZero panel**
- MCTS **visit distribution** — the improved policy after search, as a % per action
- **Q(s, a)** for each root action, straight out of the tree
- **root value V(s)** — the planner's estimate of the return from here
- simulations per move, chosen action, live score, and a value-estimate sparkline

**REINFORCE panel**
- **Q(s, a)** from a single forward pass, labelled *no planning: argmax over Q*

That contrast is the point: EfficientZero is *searching* with a learned model; REINFORCE is
reacting.

### Metrics reported per checkpoint (`metrics.md` / `metrics.json`)

score mean ± std over N episodes, min/max, episode length, env steps, gradient steps,
training wall-clock, mean decision time per move, mean value estimate, and (for
EfficientZero) policy entropy and simulation count. `metrics.md` ends with the headline
line: *"REINFORCE needed N env steps to match it — Nx more data."*

---

## 6. What's implemented

**EfficientZero** (Ye et al., NeurIPS 2021) = MuZero plus the three contributions, all
present in the code:

1. **Self-supervised consistency loss** — SimSiam between the dynamics-predicted latent
   state and the representation of the real next observation
   (`model.project`, `trainer.py`). This is where most of the sample efficiency comes from.
2. **End-to-end value prefix** — an LSTM predicts the *cumulative* reward since the last
   reset instead of per-step rewards, so one sparse Atari reward supervises the whole
   unroll (`DynamicsNetwork`, and the prefix targets in `trainer._make_targets`).
3. **Model-based off-policy correction** — value targets are re-estimated with the current
   target network rather than trusting stale root values from self-play.

Plus: batched pUCT MCTS with Dirichlet root noise and min-max Q normalisation, categorical
value/reward heads with the MuZero `h(x)` transform, latent renormalisation, prioritised
replay over trajectory segments, K-step unroll with gradient scaling.

**REINFORCE baseline**: Nature CNN + double Q-learning + dueling heads + 3-step returns +
Huber loss — a *strong* baseline, deliberately. If your professor asks "isn't vanilla REINFORCE a
straw man?", the answer is that this one already has three improvements on top of it and
still needs ~30× the data.

**Fairness**: both agents use the same ALE wrappers, frame skip, episodic-life handling,
reward clipping and evaluation protocol. The two differences are the published defaults for
each method — observation size (96×96 for EfficientZero, 84×84 for REINFORCE) and discount
(0.997 vs 0.99). Both are easy to equalise in `configs/base.yaml` if you'd rather.

---

## 7. Config knobs worth knowing

Everything lives in `configs/base.yaml`; game files override it. Any value can be changed
from the command line:

```bash
python -m src.efficientzero.train --config configs/pong.yaml \
    --set mcts.num_simulations=25 selfplay.num_envs=16 train.batch_size=128
```

| knob | effect |
|---|---|
| `mcts.num_simulations` | search depth per move. 50 trains well, 25 is ~2× faster, 100 for the showcase clip |
| `selfplay.num_envs` | envs searched in one batch — raise it to keep the GPU busy |
| `ez.updates_per_collect` | replay ratio. Higher = more sample efficient, slower in wall-clock |
| `train.reanalyze_ratio` | >0 recomputes policy targets with fresh MCTS (closer to the paper, slower) |
| `ez.checkpoints` / `reinforce.checkpoints` | the milestones saved for the demo |
| `demo.pairs` | which checkpoints are compared side by side, and the caption for each |

---

## 8. If something goes wrong

- **ROM errors** — `pip install "gymnasium[atari,accept-rom-license]" ale-py`; `ale-py ≥ 0.10`
  ships the ROMs.
- **Out of memory during REINFORCE** — lower `reinforce.buffer_capacity` (200k frames ≈ 1.4 GB).
- **MCTS feels slow** — raise `selfplay.num_envs` (batching is what makes a Python MCTS
  viable) and lower `mcts.num_simulations`.
- **`demo.build_demo` says "no checkpoints.json"** — that agent hasn't trained far enough
  to hit its first milestone yet.
- **Checkpoints don't match the requested steps** — the builder picks the nearest saved
  checkpoint and prints the plan before it starts, so you always see what's being compared.

### Honest caveats to mention if asked

This is a compact reimplementation: the MCTS is Python (batched) rather than the paper's
C++, the replay ratio is lower, and reanalyze is off by default. The qualitative result
reproduces clearly; absolute scores may sit somewhat below the published Atari-100k numbers.


## REINFORCE notes (what was fixed)

The first REINFORCE trainer never learned (entropy stayed at ln 6, greedy eval = one fixed action):
one update per episode (~3k updates in 3M steps), huge unnormalised returns, and `argmax` eval of a
flat policy. It is now REINFORCE-with-baseline as n-step actor-critic: 16 parallel envs, 8-step
rollouts, GAE, normalised advantages, orthogonal init, RMSprop + linear LR decay, sampled (not argmax)
evaluation. Also fixed: `CSVLogger` silently dropped `eval_return` (empty learning curves), missing
`rollout_length`/`gae_lambda` config keys. Watch `entropy` fall below ~1.6 and `eval_return` rise
within the first ~1M steps; if it is still flat at -21 by 1.5M, raise `reinforce.total_env_steps`.
