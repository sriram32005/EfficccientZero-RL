"""Batched MCTS for EfficientZero.

All roots in the batch are searched simultaneously: each simulation does exactly one
batched forward pass through the dynamics+prediction networks, which is what makes a
pure-Python search fast enough to train with.

Value-prefix bookkeeping (the EfficientZero-specific part):
  * every node stores the LSTM hidden state and the predicted *cumulative* reward
    (value prefix) along the path from the last reset;
  * the reward of an edge is `child.value_prefix - parent_value_prefix`;
  * the LSTM state is reset every `lstm_horizon_len` levels of the tree, which stops
    the prefix from drifting on deep search paths.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..common.utils import support_to_scalar


class MinMaxStats:
    """Normalises Q values into [0, 1] so the pUCT exploration constant is scale free."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "state",
        "reward_hidden",
        "value_prefix",
        "parent_value_prefix",
        "reward",
        "depth",
    )

    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.state = None
        self.reward_hidden = None
        self.value_prefix = 0.0
        self.parent_value_prefix = 0.0
        self.reward = 0.0
        self.depth = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class SearchResult:
    """Everything the demo overlay and the trainer need out of one search."""

    __slots__ = ("visit_counts", "policies", "root_values", "prior_policies", "q_values", "roots")

    def __init__(self, visit_counts, policies, root_values, prior_policies, q_values, roots):
        self.visit_counts = visit_counts        # (B, A) int
        self.policies = policies                # (B, A) normalised visit counts
        self.root_values = root_values          # (B,) float -- MCTS-improved value estimate
        self.prior_policies = prior_policies    # (B, A) raw network policy (before search)
        self.q_values = q_values                # (B, A) Q of each root action
        self.roots = roots


class MCTS:
    def __init__(self, model, cfg, action_space_size: int, device):
        self.model = model
        self.cfg = cfg
        self.action_space_size = int(action_space_size)
        self.device = device
        self.num_simulations = int(cfg.num_simulations)
        self.discount = float(cfg.discount)
        self.pb_c_base = float(cfg.pb_c_base)
        self.pb_c_init = float(cfg.pb_c_init)
        self.root_dirichlet_alpha = float(cfg.root_dirichlet_alpha)
        self.root_exploration_fraction = float(cfg.root_exploration_fraction)
        self.lstm_horizon_len = int(cfg.lstm_horizon_len)
        self.value_delta_max = float(getattr(cfg, "value_delta_max", 0.01))

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, obs_batch: torch.Tensor, add_noise: bool = True, rng: np.random.Generator | None = None) -> SearchResult:
        was_training = self.model.training
        self.model.eval()
        rng = rng or np.random.default_rng()

        batch = obs_batch.size(0)
        out = self.model.initial_inference(obs_batch)
        root_values = support_to_scalar(out.value_logits, self.model.support_size).squeeze(-1).cpu().numpy()
        priors = torch.softmax(out.policy_logits, dim=-1).cpu().numpy()

        roots = []
        for i in range(batch):
            root = Node(0.0)
            root.state = out.state[i]
            root.reward_hidden = (out.reward_hidden[0][:, i: i + 1], out.reward_hidden[1][:, i: i + 1])
            p = priors[i]
            if add_noise:
                noise = rng.dirichlet([self.root_dirichlet_alpha] * self.action_space_size)
                frac = self.root_exploration_fraction
                p = p * (1 - frac) + noise * frac
            for a in range(self.action_space_size):
                child = Node(float(p[a]))
                child.depth = 1
                root.children[a] = child
            roots.append(root)

        min_max = [MinMaxStats() for _ in range(batch)]
        for i in range(batch):
            min_max[i].update(float(root_values[i]))

        for _ in range(self.num_simulations):
            paths, leaves, parents, actions = [], [], [], []
            for i in range(batch):
                node = roots[i]
                path = [node]
                action = None
                while node.expanded():
                    action, node = self._select_child(node, min_max[i])
                    path.append(node)
                paths.append(path)
                leaves.append(node)
                parents.append(path[-2])
                actions.append(action)

            states = torch.stack([p.state for p in parents], dim=0)
            h = torch.cat([p.reward_hidden[0] for p in parents], dim=1)
            c = torch.cat([p.reward_hidden[1] for p in parents], dim=1)
            # reset the LSTM every `lstm_horizon_len` levels of the tree
            reset_mask = torch.tensor(
                [1.0 if ((len(p) - 2) % self.lstm_horizon_len == 0) else 0.0 for p in paths],
                device=self.device,
            ).view(1, batch, 1)
            h = h * (1 - reset_mask)
            c = c * (1 - reset_mask)
            act = torch.tensor(actions, dtype=torch.long, device=self.device)

            net = self.model.recurrent_inference(states, (h, c), act)
            values = support_to_scalar(net.value_logits, self.model.support_size).squeeze(-1).cpu().numpy()
            prefixes = support_to_scalar(net.value_prefix_logits, self.model.support_size).squeeze(-1).cpu().numpy()
            child_priors = torch.softmax(net.policy_logits, dim=-1).cpu().numpy()

            for i in range(batch):
                leaf = leaves[i]
                leaf.state = net.state[i]
                leaf.reward_hidden = (net.reward_hidden[0][:, i: i + 1], net.reward_hidden[1][:, i: i + 1])
                parent_prefix = 0.0 if ((len(paths[i]) - 2) % self.lstm_horizon_len == 0) else parents[i].value_prefix
                leaf.parent_value_prefix = parent_prefix
                leaf.value_prefix = float(prefixes[i])
                leaf.reward = leaf.value_prefix - parent_prefix
                p = child_priors[i]
                for a in range(self.action_space_size):
                    child = Node(float(p[a]))
                    child.depth = leaf.depth + 1
                    leaf.children[a] = child
                self._backpropagate(paths[i], float(values[i]), min_max[i])

        visit_counts = np.zeros((batch, self.action_space_size), dtype=np.int64)
        q_values = np.zeros((batch, self.action_space_size), dtype=np.float32)
        improved_values = np.zeros(batch, dtype=np.float32)
        for i, root in enumerate(roots):
            for a, child in root.children.items():
                visit_counts[i, a] = child.visit_count
                q_values[i, a] = child.reward + self.discount * child.value() if child.visit_count > 0 else 0.0
            improved_values[i] = root.value()
        policies = visit_counts / np.maximum(1, visit_counts.sum(axis=1, keepdims=True))

        if was_training:
            self.model.train()
        return SearchResult(visit_counts, policies.astype(np.float32), improved_values, priors, q_values, roots)

    # ----------------------------------------------------------------------------------
    def _select_child(self, node: Node, min_max: MinMaxStats):
        best_score, best_action, best_child = -float("inf"), None, None
        sqrt_total = math.sqrt(max(1, node.visit_count))
        pb_c_base_term = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        for action, child in node.children.items():
            prior_score = pb_c_base_term * child.prior * sqrt_total / (1 + child.visit_count)
            if child.visit_count > 0:
                value_score = min_max.normalize(child.reward + self.discount * child.value())
            else:
                value_score = 0.0
            score = prior_score + value_score
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _backpropagate(self, path, value: float, min_max: MinMaxStats):
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            min_max.update(node.reward + self.discount * node.value())
            value = node.reward + self.discount * value


def select_action(visit_counts: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Sample an action from the visit counts (temperature 0 = greedy, used at demo time)."""
    if temperature <= 1e-3:
        max_v = visit_counts.max()
        best = np.flatnonzero(visit_counts == max_v)
        return int(rng.choice(best))
    counts = np.asarray(visit_counts, dtype=np.float64) ** (1.0 / temperature)
    probs = counts / counts.sum()
    return int(rng.choice(len(probs), p=probs))
