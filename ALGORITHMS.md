# EfficientZero vs REINFORCE: how they work, and why EfficientZero wins

This document describes the two agents exactly as implemented in this repo
(`src/efficientzero/`, `src/reinforce/`) and ties each EfficientZero advantage to the
architectural feature that causes it.

Setting: Atari Pong, 6 actions, frame-skip 4, 4 stacked grayscale frames.
Reward is +1 / -1 per point, so a game is up to 21 points either way and almost every
step has reward 0.

---

## 1. REINFORCE (model-free policy gradient)

### 1.1 Idea

Learn a policy `pi(a|s)` directly. Increase the probability of actions that led to
better-than-expected outcomes, decrease it for worse-than-expected ones:

```
grad J = E[ grad log pi(a_t | s_t) * A_t ]
```

Plain REINFORCE uses the full episode return `G_t` for `A_t`. This repo uses the
standard practical form: **REINFORCE with a learned baseline** `V(s)`, estimated from
short n-step rollouts (this is the advantage actor-critic form). The baseline lowers
variance without changing the expected gradient.

### 1.2 Architecture (`src/reinforce/model.py`)

```
obs (4x84x84, uint8)  ->  /255
  Conv 8x8 s4 (32) -> ReLU
  Conv 4x4 s2 (64) -> ReLU
  Conv 3x3 s1 (64) -> ReLU -> Flatten
  Linear 512 -> ReLU                (shared trunk)
      |-> policy_out: Linear -> 6 logits   (init gain 0.01 => starts ~uniform)
      '-> value_out : Linear -> 1          (baseline V(s))
```

About 1.7M parameters. No dynamics model, no search, no memory of the world.

### 1.3 Training loop (`src/reinforce/train.py`)

```
repeat:
  1. ROLLOUT   16 envs x 8 steps, sampling a ~ pi(.|s)          (no grad)
  2. GAE       delta_t = r_t + g(1-d_t)V(s_t+1) - V(s_t)
               A_t = delta_t + g*lambda*(1-d_t)*A_t+1           (g=0.99, lambda=0.95)
  3. UPDATE    one gradient step on the 128 samples:
               L = -mean(log pi(a|s) * A_norm) + 0.5*MSE(V, A+V) - 0.01*entropy
  4. DISCARD   the 128 samples are thrown away (on-policy)
```

Supporting details: RMSprop, lr 7e-4 decayed linearly to 0, grad-norm clip 0.5,
advantages normalised per batch, evaluation by sampling from `pi`.

### 1.4 What the learning signal looks like

Every sample is used **once**. The only supervision is the scalar reward, which in Pong
is non-zero roughly once per 30 to 100 agent steps. The network must discover, from
that sparse signal alone, that "paddle position relative to the ball" matters. It has
no explicit notion of ball dynamics.

---

## 2. EfficientZero (model-based, search + self-supervision)

### 2.1 Idea

Learn a **model of the environment in a latent space** and plan with it using Monte
Carlo Tree Search (MCTS). The network is trained to be consistent with what actually
happened, so each real transition supervises many things at once. The planner then
improves on the raw policy, and the improved policy becomes the training target
(policy iteration).

### 2.2 Architecture (`src/efficientzero/model.py`)

Three networks (MuZero style) plus EfficientZero's additions:

```
                      real obs o_t (4x96x96)
                             |
                  RepresentationNetwork h         downsample conv -> 64ch x 6x6 latent s_t
                             |
        .--------------------+---------------------.
        |                                          |
 PredictionNetwork f                    DynamicsNetwork g (s_t, a_t)
   -> policy logits p_t                    -> next latent s_t+1
   -> value  logits v_t (41-bin)           -> value-prefix (via LSTM, 41-bin)
        ^                                          |
        |__________ applied to every latent <______|   (unrolled K = 5 steps)

 Projection + prediction heads (SimSiam)  <- used only in the training loss
```

Key components:

| Component | Where | What it does |
|---|---|---|
| Representation `h` | `RepresentationNetwork` | Encodes frames into a compact latent state |
| Dynamics `g` | `DynamicsNetwork` | Predicts next latent given latent + action |
| Value-prefix LSTM | inside `DynamicsNetwork` | Predicts cumulative reward since the last LSTM reset (reset every 5 levels) instead of per-step reward |
| Prediction `f` | `PredictionNetwork` | Policy and value from any latent |
| Categorical heads | support_size 20 -> 41 bins | Value/reward as a distribution, not a scalar |
| SimSiam projector | `EfficientZeroNet` | Self-supervised consistency target |
| MCTS | `mcts.py` | 50 simulations per real move, UCB selection, Dirichlet noise at root |
| Prioritised buffer | `game_buffer.py` | Samples transitions with large value error more often |
| Target network | `target_model` | Slow copy (updated every 200 steps) used for bootstrapped value targets |

### 2.3 Acting (self-play, `self_play.py` + `mcts.py`)

```
for each real step:
   s0 = h(obs)
   run 50 simulations:  select by UCB down the tree -> expand with g and f
                        -> back up value (min-max normalised)
   action ~ visit-count distribution (temperature drops during training)
   store (obs, action, reward, root value, visit-count policy) in the buffer
```

### 2.4 Learning (`trainer.py`)

Sample a transition and its next K=5 steps, then unroll the model from the real
observation and apply four losses at each unrolled step:

```
L = 1.0 * policy   CE(p_k, MCTS visit distribution)
  + 0.25* value    CE(v_k, n-step bootstrapped target, n = 5, from target net)
  + 1.0 * prefix   CE(value_prefix_k, true discounted reward sum)
  + 2.0 * consistency  -cos( predict(project(s_k_hat)), stopgrad(project(h(o_t+k))) )
```

The consistency term compares the *imagined* latent after k steps with the encoding of
the *real* observation k steps later. Gradients are scaled by 1/K across the unroll
and the model is updated 200 times after every 200 env steps (a high
update-to-data ratio, with PER, ~1 update per environment step).

---

## 3. Side-by-side

| | REINFORCE (this repo) | EfficientZero (this repo) |
|---|---|---|
| Family | Model-free, on-policy policy gradient | Model-based, off-policy, search-based |
| Learns | `pi(a|s)` and `V(s)` | latent dynamics, value-prefix, value, policy |
| Decides by | One forward pass, sample | 50-simulation tree search over a learned model |
| Supervision per transition | 1 scalar advantage | 5 unrolled steps x (policy + value + reward + consistency) |
| Data reuse | Each sample used once | Replayed many times (prioritised) |
| Training target for policy | Its own sampled actions weighted by advantage | MCTS visit counts (a better policy than the raw network) |
| Params | ~1.7M | ~0.17M in the smoke config, larger with real settings |
| Steps budget in this demo | 3,000,000 | 120,000 |
| Compute per env step | tiny | large (search every step) |

---

## 4. Why EfficientZero is better: advantage -> architectural cause

This is the core mapping. Each row names an advantage, the feature that produces it,
and the REINFORCE limitation it removes.

| # | Advantage | Architectural feature that causes it | The REINFORCE limitation it fixes |
|---|---|---|---|
| 1 | **Far fewer environment steps to reach good play** | Replay buffer with many gradient steps per collected step, plus off-policy targets (value from the target network, policy from MCTS) | REINFORCE is on-policy: after one gradient step the batch is discarded, so each costly env step is used once |
| 2 | **Dense learning signal from sparse rewards** | **Value-prefix LSTM** in the dynamics network. It predicts cumulative reward over a horizon, so a single +1/-1 supervises every step of the unroll | REINFORCE only sees the reward as one scalar advantage after the fact; most steps carry no information |
| 3 | **Learns what the ball and paddle do, not just what to press** | **Consistency loss** (SimSiam projection + prediction, stop-gradient on the real-observation branch). Forces the predicted next latent to match the encoding of the real next frame | REINFORCE has no state-prediction objective, so representation learning is driven only by reward |
| 4 | **Faster, more stable credit assignment across time** | **K=5 unroll** with gradient scaling, plus **n-step value targets** (td_steps=5) bootstrapped from the target network | REINFORCE assigns credit from a noisy short rollout and one-step-ish bootstrapping, so a point scored after a rally is attributed weakly |
| 5 | **Better actions than the raw network would pick** | **MCTS with 50 simulations** over the learned model, with UCB selection and Dirichlet exploration at the root | REINFORCE acts from a single forward pass; its action quality is capped by its current network |
| 6 | **Policy improves faster per update (policy iteration)** | **Policy target = MCTS visit distribution** (cross-entropy). Search acts as a policy-improvement operator, the network distils it | REINFORCE improves only by following a noisy advantage-weighted gradient |
| 7 | **Look-ahead handles delayed consequences** | **Learned dynamics `g` + value `f`** let search evaluate a move by imagining where the ball will be | REINFORCE must learn this implicitly in the weights from the reward alone |
| 8 | **Exploration is directed, not accidental** | Root **Dirichlet noise** + visit-count **temperature** schedule + search statistics | REINFORCE explores only through the sampling noise of its own policy and the entropy bonus; a flat policy stays flat |
| 9 | **Learns from the most informative data first** | **Prioritised replay** using value-prediction error | REINFORCE has no notion of which transitions matter and cannot revisit any |
| 10 | **Stable regression targets for values and rewards** | **Categorical (41-bin) value and reward heads** | A scalar MSE value baseline is sensitive to reward scale and outliers, which is one reason policy-gradient advantages are noisy |
| 11 | **Less target drift during training** | **Target network** refreshed every 200 steps | REINFORCE's baseline is trained on the same moving network that produces the advantages |

### The three that matter most for sample efficiency

If you have to name only three for the professor:

1. **Consistency loss (row 3).** Biggest single contributor in the EfficientZero paper's
   ablations. It turns every real frame into a supervised target for the dynamics model.
2. **Value-prefix LSTM (row 2).** Solves the "reward is almost always zero" problem.
3. **Replay + off-policy targets (row 1) with MCTS policy targets (row 6).** This is the
   difference between using each frame once and using it hundreds of times against a
   target that is better than the current policy.

### Why Pong in particular rewards these features

* The reward is sparse and delayed (the ball has to travel to score), which is exactly
  what rows 2, 4 and 7 address.
* The game is deterministic with simple physics (ball and two paddles), so a learned
  latent model (rows 3 and 7) is easy to fit and search through it is reliable.
* Early on, a flat REINFORCE policy gets almost no useful gradient at all, while MCTS
  over even a poor model still produces a sharper action distribution than the network.

---

## 5. Costs and honest caveats

EfficientZero is not better on every axis:

* **Compute per environment step** is much higher (50 simulations of network inference
  for every action, plus a larger training loop). It wins on *samples*, not wall-clock
  per step. REINFORCE can be faster in wall-clock on a cheap simulator.
* **Complexity.** Many interacting components (dynamics, LSTM, SimSiam, MCTS, PER,
  target network) and more hyperparameters.
* **Needs a simulator-like loop.** It benefits from the environment being learnable by
  a latent model; very stochastic or visually complex domains are harder.
* **The REINFORCE here is a strong-ish variant.** It uses a baseline, GAE and parallel
  envs. Textbook episodic REINFORCE would be weaker still (the first version of this
  repo's trainer was, and it never learned).
* **Results in this repo are not yet measured.** The mapping above explains *why*
  EfficientZero should be more sample efficient, and matches published results, but
  the actual curves for your runs come from `demo/build_demo.py`. Quote those numbers
  rather than the table above when presenting.

## 6. One-paragraph summary for the demo

REINFORCE learns by trial and error: it nudges its action probabilities according to how
well the last few steps turned out, then throws those experiences away, so in sparse-reward
Pong it needs millions of frames. EfficientZero learns a small internal simulator of the
game, trains it to stay consistent with real frames and to predict accumulated reward, replays
its experience many times, and uses tree search over that simulator to choose and to
teach itself better moves. Each architectural piece (consistency loss, value-prefix LSTM,
MCTS, replay with target network) removes one specific reason that model-free policy gradient
is data hungry.
