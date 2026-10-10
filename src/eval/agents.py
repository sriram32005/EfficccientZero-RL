"""Load a saved checkpoint of either algorithm behind one interface.

`agent.act(obs)` returns `(action, info)` where `info` carries everything the demo
overlay draws: MCTS visit counts, the network's prior policy, per-action Q values and
the root value for EfficientZero; policy distribution for REINFORCE; Q values for DQN.
"""
from __future__ import annotations

import numpy as np
import torch

from ..common.config import _to_dot
from ..common.utils import get_device
from ..dqn.model import DQNNet
from ..efficientzero.mcts import MCTS, select_action
from ..efficientzero.model import EfficientZeroNet
from ..reinforce.model import ReinforceNet


def load_checkpoint(path, device=None):
    device = device or get_device("auto")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt["cfg"] = _to_dot(ckpt["cfg"])
    return ckpt


class EZAgent:
    name = "EfficientZero"

    def __init__(self, ckpt_path, device=None, num_simulations=None, temperature=0.0, seed=0):
        self.device = device or get_device("auto")
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = ckpt["cfg"]
        self.env_steps = int(ckpt["env_steps"])
        self.training_steps = int(ckpt.get("training_steps", 0))
        self.wallclock_s = float(ckpt.get("wallclock_s", 0.0))
        self.obs_shape = tuple(ckpt["obs_shape"])
        self.action_space_size = int(ckpt["action_space_size"])
        self.model = EfficientZeroNet(self.obs_shape, self.action_space_size, self.cfg.model).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        mcts_cfg = dict(self.cfg.mcts)
        if num_simulations:
            mcts_cfg["num_simulations"] = int(num_simulations)
        self.mcts = MCTS(self.model, _to_dot(mcts_cfg), self.action_space_size, self.device)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.num_simulations = int(mcts_cfg["num_simulations"])

    @torch.no_grad()
    def act(self, obs):
        batch = torch.as_tensor(obs[None], dtype=torch.uint8, device=self.device)
        res = self.mcts.run(batch, add_noise=False, rng=self.rng)
        action = select_action(res.visit_counts[0], self.temperature, self.rng)
        info = {
            "visit_counts": res.visit_counts[0],
            "policy": res.policies[0],
            "prior": res.prior_policies[0],
            "q_values": res.q_values[0],
            "root_value": float(res.root_values[0]),
            "num_simulations": self.num_simulations,
            "action": int(action),
        }
        return int(action), info


class DQNAgent:
    name = "DQN"

    def __init__(self, ckpt_path, device=None, epsilon=0.01, seed=0):
        self.device = device or get_device("auto")
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = ckpt["cfg"]
        self.env_steps = int(ckpt["env_steps"])
        self.training_steps = int(ckpt.get("training_steps", 0))
        self.wallclock_s = float(ckpt.get("wallclock_s", 0.0))
        self.obs_shape = tuple(ckpt["obs_shape"])
        self.action_space_size = int(ckpt["action_space_size"])
        self.model = DQNNet(self.obs_shape, self.action_space_size, dueling=bool(self.cfg.dqn.dueling)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs):
        q = self.model(torch.as_tensor(obs[None], dtype=torch.uint8, device=self.device))[0].cpu().numpy()
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.action_space_size))
        else:
            action = int(np.argmax(q))
        info = {"q_values": q, "root_value": float(q.max()), "action": action, "epsilon": self.epsilon}
        return action, info


class REINFORCEAgent:
    name = "REINFORCE"

    def __init__(self, ckpt_path, device=None, seed=0, greedy=False):
        self.device = device or get_device("auto")
        self.greedy = bool(greedy)
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = ckpt["cfg"]
        self.env_steps = int(ckpt["env_steps"])
        self.training_steps = int(ckpt.get("training_steps", 0))
        self.wallclock_s = float(ckpt.get("wallclock_s", 0.0))
        self.obs_shape = tuple(ckpt["obs_shape"])
        self.action_space_size = int(ckpt["action_space_size"])
        self.model = ReinforceNet(self.obs_shape, self.action_space_size).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs):
        logits, value = self.model(torch.as_tensor(obs[None], dtype=torch.uint8, device=self.device))
        probs = torch.softmax(logits, dim=-1)[0].double().cpu().numpy()
        probs = probs / probs.sum()
        # Sample from pi (the policy that was trained). Pure argmax of a policy-gradient net
        # collapses to a single fixed action whenever the policy is still near-uniform.
        action = int(np.argmax(probs)) if self.greedy else int(self.rng.choice(len(probs), p=probs))
        info = {
            "policy": probs,
            "root_value": float(value[0, 0].item()),
            "action": action,
        }
        return action, info


def load_agent(ckpt_path, device=None, **kwargs):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent_type = ckpt.get("agent", "ez")
    del ckpt
    if agent_type == "ez":
        return EZAgent(ckpt_path, device=device, **{k: v for k, v in kwargs.items() if k in ("num_simulations", "temperature", "seed")})
    if agent_type == "reinforce":
        return REINFORCEAgent(ckpt_path, device=device, **{k: v for k, v in kwargs.items() if k in ("seed", "greedy")})
    return DQNAgent(ckpt_path, device=device, **{k: v for k, v in kwargs.items() if k in ("epsilon", "seed")})


def make_eval_env(cfg, agent_type: str, seed: int = 0, render: bool = True):
    from ..common.atari_wrappers import make_atari

    if agent_type == "ez":
        size = cfg.env.size
        stack = cfg.env.stack
    elif agent_type == "reinforce":
        size = cfg.reinforce.size
        stack = cfg.reinforce.stack
    else:
        size = cfg.dqn.size
        stack = cfg.dqn.stack
    return make_atari(
        game=cfg.env.game,
        size=size,
        grayscale=cfg.env.grayscale,
        stack=stack,
        skip=cfg.env.skip,
        episodic_life=False,
        clip_rewards=False,
        sticky_actions=cfg.env.sticky_actions,
        max_episode_steps=cfg.env.eval_max_episode_steps,
        render=render,
        seed=seed,
    )
