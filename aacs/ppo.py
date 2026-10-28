"""Proximal Policy Optimisation (PPO) baseline with the same observation history and evaluation protocol as AA-DQN."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .agents import History, evaluate_agent
from .env import OBS_DIM, CognitiveSchedulingEnv, EnvConfig


@dataclass(frozen=True)
class PPOConfig:
    name: str = "PPO"
    history: int = 4
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    rollout_steps: int = 2048
    epochs: int = 10
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    n_episodes: int = 5_000


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: PPOConfig, seed: int, n_actions: int):
        torch.manual_seed(seed)
        self.cfg, self.n_actions = cfg, n_actions
        self.q = ActorCritic(OBS_DIM * cfg.history, cfg.hidden, n_actions)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def greedy(self, s: np.ndarray) -> np.ndarray:
        logits, _ = self.q(torch.from_numpy(np.atleast_2d(s)))
        return logits.argmax(1).numpy()

    @torch.no_grad()
    def sample(self, s: np.ndarray):
        logits, v = self.q(torch.from_numpy(s[None]))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(v)

    def update(self, buf):
        c = self.cfg
        S = torch.from_numpy(np.stack(buf["s"]))
        A = torch.tensor(buf["a"])
        LP = torch.tensor(buf["lp"], dtype=torch.float32)
        R = np.array(buf["r"], np.float32)
        V = np.array(buf["v"], np.float32)
        D = np.array(buf["d"], np.float32)
        adv = np.zeros_like(R)
        last, next_v = 0.0, 0.0
        for t in reversed(range(len(R))):
            nonterm = 1.0 - D[t]
            delta = R[t] + c.gamma * next_v * nonterm - V[t]
            last = delta + c.gamma * c.gae_lambda * nonterm * last
            adv[t] = last
            next_v = V[t]
        ret = torch.from_numpy(adv + V)
        adv = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        n = len(R)
        losses = []
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, c.minibatch):
                idx = perm[i:i + c.minibatch]
                logits, v = self.q(S[idx])
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv[idx]).mean()
                vl = 0.5 * (ret[idx] - v).pow(2).mean()
                loss = pg + c.vf_coef * vl - c.ent_coef * dist.entropy().mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
                self.opt.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    def state_dict(self):
        return {"cfg": asdict(self.cfg), "q": self.q.state_dict(), "n_actions": self.n_actions, "algo": "ppo"}


def train_ppo(cfg: PPOConfig, env_cfg: EnvConfig, seed: int, eval_seeds=None, eval_every: int = 100, **_):
    agent = PPOAgent(cfg, seed, env_cfg.n_actions)
    env = CognitiveSchedulingEnv(env_cfg)
    hist = History(cfg.history)
    buf = {k: [] for k in ("s", "a", "lp", "r", "v", "d")}
    train_log, curve = [], []
    best_val, best_state, best_ep = -np.inf, None, 0
    loss = np.nan
    for ep in range(cfg.n_episodes):
        s = hist.reset(env.reset(seed * 1_000_003 + ep))
        done = False
        while not done:
            a, lp, v = agent.sample(s)
            o, r, done, info = env.step(a)
            for k, x in zip(("s", "a", "lp", "r", "v", "d"), (s, a, lp, r, v, float(done))):
                buf[k].append(x)
            s = hist.push(o)
            if len(buf["r"]) >= cfg.rollout_steps and done:
                loss = agent.update(buf)
                buf = {k: [] for k in buf}
        train_log.append({"episode": ep, "return": info["return"], "on_time_rate": info["on_time_rate"],
                          "epsilon": np.nan, "td_loss": loss})
        if eval_seeds is not None and ((ep + 1) % eval_every == 0 or ep == 0):
            rows = evaluate_agent(agent, env_cfg, eval_seeds)
            point = {"episode": ep + 1, **{k: float(np.mean([r[k] for r in rows]))
                                           for k in ("return", "on_time_rate", "mean_fatigue", "mean_attention")}}
            curve.append(point)
            if point["return"] > best_val:
                best_val, best_ep = point["return"], ep + 1
                best_state = {k: v.detach().clone() for k, v in agent.q.state_dict().items()}
    agent.best_state, agent.best_episode = best_state, best_ep
    return agent, train_log, curve
