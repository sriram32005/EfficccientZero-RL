"""EfficientZero learner: target construction + the loss.

Loss = value + policy + value-prefix + consistency, unrolled K steps from a sampled
state. The consistency term is what buys most of the sample efficiency: it supervises
the latent dynamics with a SimSiam objective against the real next observation, so the
model learns a useful world model from very few environment interactions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..common.utils import scalar_to_support, scale_gradient, support_to_scalar
from .mcts import MCTS


def _cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _negative_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, p=2.0, dim=-1, eps=1e-5)
    target = F.normalize(target, p=2.0, dim=-1, eps=1e-5)
    return -(pred * target).sum(dim=-1)


class EZTrainer:
    def __init__(self, cfg, model, target_model, buffer, device, seed: int = 0):
        self.cfg = cfg
        self.model = model
        self.target_model = target_model
        self.buffer = buffer
        self.device = device
        self.rng = np.random.default_rng(seed + 777)

        t = cfg.train
        self.K = int(t.num_unroll_steps)
        self.td_steps = int(t.td_steps)
        self.discount = float(cfg.mcts.discount)
        self.batch_size = int(t.batch_size)
        self.lstm_horizon = int(cfg.mcts.lstm_horizon_len)
        self.support = int(cfg.model.support_size)
        self.reanalyze_ratio = float(t.reanalyze_ratio)

        if str(t.optimizer).lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=float(t.lr_init), weight_decay=float(t.weight_decay))
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(t.lr_init),
                momentum=float(t.momentum),
                weight_decay=float(t.weight_decay),
            )
        self.target_mcts = MCTS(target_model, cfg.mcts, model.action_space_size, device)

    # ----------------------------------------------------------------------------------
    def _lr(self, step: int) -> float:
        t = self.cfg.train
        warm = int(t.lr_warmup_steps)
        if step < warm:
            return float(t.lr_init) * (step + 1) / warm
        decays = step // max(1, int(t.lr_decay_steps))
        return float(t.lr_init) * (float(t.lr_decay_rate) ** decays)

    def _stacked(self, game, pos):
        return game.stacked_obs_array(pos)

    @torch.no_grad()
    def _make_targets(self, games, positions):
        """Returns value / value-prefix / policy targets plus the observation stacks."""
        B, K, td = len(games), self.K, self.td_steps
        obs_np = np.stack(
            [np.stack([self._stacked(g, p + j) for j in range(K + 1)], axis=0) for g, p in zip(games, positions)],
            axis=0,
        )  # (B, K+1, stack*C, H, W)
        obs = torch.as_tensor(obs_np, dtype=torch.uint8, device=self.device)

        # bootstrap values from the *target* network -- model-based off-policy correction:
        # the value target is re-estimated with the current target model rather than
        # trusting the stale root value that was stored during self-play.
        boot_np = np.stack(
            [np.stack([self._stacked(g, p + j + td) for j in range(K + 1)], axis=0) for g, p in zip(games, positions)],
            axis=0,
        )
        boot = torch.as_tensor(boot_np, dtype=torch.uint8, device=self.device).reshape(B * (K + 1), *obs.shape[2:])
        self.target_model.eval()
        boot_values = []
        chunk = 256
        for i in range(0, boot.size(0), chunk):
            out = self.target_model.initial_inference(boot[i: i + chunk])
            boot_values.append(support_to_scalar(out.value_logits, self.support).squeeze(-1))
        boot_values = torch.cat(boot_values).reshape(B, K + 1).cpu().numpy()

        A = self.model.action_space_size
        value_t = np.zeros((B, K + 1), dtype=np.float32)
        prefix_t = np.zeros((B, K), dtype=np.float32)
        policy_t = np.full((B, K + 1, A), 1.0 / A, dtype=np.float32)
        mask = np.zeros((B, K + 1), dtype=np.float32)
        actions = np.zeros((B, K), dtype=np.int64)

        for b, (game, pos) in enumerate(zip(games, positions)):
            n_real = game.length
            n_avail = len(game.rewards)          # includes the padded tail
            acc = 0.0
            for j in range(K + 1):
                t = pos + j
                mask[b, j] = 1.0 if t < n_real else 0.0
                # ----- n-step value target -----
                value = 0.0
                if t < n_avail:
                    bootstrap_index = t + td
                    if bootstrap_index < n_avail or not game.terminal:
                        value = float(boot_values[b, j]) * (self.discount ** td)
                    for i in range(td):
                        if t + i < n_avail:
                            value += float(game.rewards[t + i]) * (self.discount ** i)
                value_t[b, j] = value
                # ----- policy target -----
                if t < len(game.child_visits):
                    policy_t[b, j] = game.child_visits[t]
                # ----- value prefix target (LSTM reset every `lstm_horizon` steps) -----
                if j < K:
                    if j % self.lstm_horizon == 0:
                        acc = 0.0
                    acc += float(game.rewards[t]) if t < n_avail else 0.0
                    prefix_t[b, j] = acc
                    actions[b, j] = int(game.actions[t]) if t < len(game.actions) else 0

        out = {
            "obs": obs,
            "actions": torch.as_tensor(actions, device=self.device),
            "value": torch.as_tensor(value_t, device=self.device),
            "prefix": torch.as_tensor(prefix_t, device=self.device),
            "policy": torch.as_tensor(policy_t, device=self.device),
            "mask": torch.as_tensor(mask, device=self.device),
        }
        if self.reanalyze_ratio > 0:
            self._reanalyze(out, B)
        return out

    @torch.no_grad()
    def _reanalyze(self, targets, B):
        """Refresh policy targets for a fraction of the batch with fresh MCTS on the target net."""
        n = max(1, int(B * self.reanalyze_ratio))
        idx = self.rng.choice(B, size=n, replace=False)
        obs = targets["obs"][idx, 0]
        res = self.target_mcts.run(obs, add_noise=False, rng=self.rng)
        targets["policy"][idx, 0] = torch.as_tensor(res.policies, device=self.device)

    # ----------------------------------------------------------------------------------
    def update(self, step: int) -> dict:
        cfg = self.cfg.train
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr(step)

        slots, games, positions, weights = self.buffer.sample(self.batch_size, self.rng)
        targets = self._make_targets(games, positions)
        w = torch.as_tensor(weights, device=self.device)

        self.model.train()
        obs = targets["obs"]
        value_sup = scalar_to_support(targets["value"], self.support)
        prefix_sup = scalar_to_support(targets["prefix"], self.support)
        mask = targets["mask"]

        out = self.model.initial_inference(obs[:, 0])
        value_loss = _cross_entropy(out.value_logits, value_sup[:, 0])
        policy_loss = _cross_entropy(out.policy_logits, targets["policy"][:, 0])
        prefix_loss = torch.zeros_like(value_loss)
        consistency_loss = torch.zeros_like(value_loss)

        state = out.state
        reward_hidden = self.model.new_reward_hidden(obs.size(0), self.device)
        pred_root_value = support_to_scalar(out.value_logits, self.support).squeeze(-1).detach()

        gradient_scale = 1.0 / self.K
        for k in range(self.K):
            net = self.model.recurrent_inference(state, reward_hidden, targets["actions"][:, k])
            state, reward_hidden = net.state, net.reward_hidden
            m = mask[:, k + 1]

            # (1) consistency: predicted latent vs. representation of the true next obs
            with torch.no_grad():
                target_state = self.model.representation(obs[:, k + 1])
                proj_target = self.model.project(target_state, with_grad=False)
            proj_pred = self.model.project(state, with_grad=True)
            step_consistency = _negative_cosine(proj_pred, proj_target) * m

            step_value = _cross_entropy(net.value_logits, value_sup[:, k + 1]) * m
            step_policy = _cross_entropy(net.policy_logits, targets["policy"][:, k + 1]) * m
            step_prefix = _cross_entropy(net.value_prefix_logits, prefix_sup[:, k]) * m

            value_loss = value_loss + scale_gradient(step_value, gradient_scale)
            policy_loss = policy_loss + scale_gradient(step_policy, gradient_scale)
            prefix_loss = prefix_loss + scale_gradient(step_prefix, gradient_scale)
            consistency_loss = consistency_loss + scale_gradient(step_consistency, gradient_scale)

            state = scale_gradient(state, 0.5)  # halve the gradient flowing back through dynamics
            if (k + 1) % self.lstm_horizon == 0:
                reward_hidden = self.model.new_reward_hidden(obs.size(0), self.device)

        total = (
            float(cfg.value_loss_coeff) * value_loss
            + float(cfg.policy_loss_coeff) * policy_loss
            + float(cfg.reward_loss_coeff) * prefix_loss
            + float(cfg.consistency_loss_coeff) * consistency_loss
        )
        loss = (total * w).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
        self.optimizer.step()

        new_priority = (pred_root_value - targets["value"][:, 0]).abs().cpu().numpy() + 1e-6
        self.buffer.update_priorities(slots, new_priority)

        if step % int(cfg.target_update_interval) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return {
            "loss": float(loss.item()),
            "loss_value": float(value_loss.mean().item()),
            "loss_policy": float(policy_loss.mean().item()),
            "loss_prefix": float(prefix_loss.mean().item()),
            "loss_consistency": float(consistency_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "lr": self._lr(step),
        }
