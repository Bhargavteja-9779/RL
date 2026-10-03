"""DQN-family agents: vanilla DQN and the attention-aware variant (history + Double + Dueling)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .env import N_ACTIONS, OBS_DIM, CognitiveSchedulingEnv, EnvConfig


@dataclass(frozen=True)
class AgentConfig:
    name: str = "AA-DQN"
    history: int = 4
    double: bool = True
    dueling: bool = True
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    target_update: int = 1_000
    eps_max: float = 1.0
    eps_min: float = 0.01
    eps_reach_frac: float = 0.5
    grad_clip: float = 10.0
    n_episodes: int = 3_000
    encoder: str = "mlp"


DQN_CFG = AgentConfig(name="DQN", history=1, double=False, dueling=False)
AADQN_CFG = AgentConfig()


class GRUBody(nn.Module):
    """Recurrent encoder over the observation window (k x OBS_DIM), followed by a ReLU layer."""

    def __init__(self, k: int, hidden: int):
        super().__init__()
        self.k = k
        self.gru = nn.GRU(OBS_DIM, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x):
        _, h = self.gru(x.view(x.shape[0], self.k, OBS_DIM))
        return self.out(h[-1])


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dueling: bool, encoder: str = "mlp"):
        super().__init__()
        if encoder == "gru":
            self.body = GRUBody(in_dim // OBS_DIM, hidden)
        else:
            self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.dueling = dueling
        if dueling:
            self.v = nn.Linear(hidden, 1)
            self.a = nn.Linear(hidden, N_ACTIONS)
        else:
            self.q = nn.Linear(hidden, N_ACTIONS)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            a = self.a(h)
            return self.v(h) + a - a.mean(dim=1, keepdim=True)
        return self.q(h)


class History:
    """Fixed-length observation history; the window is zero-padded at episode start."""

    def __init__(self, k: int):
        self.k = k
        self.buf: deque = deque(maxlen=k)

    def reset(self, obs):
        self.buf.clear()
        for _ in range(self.k - 1):
            self.buf.append(np.zeros_like(obs))
        self.buf.append(obs)
        return self.get()

    def push(self, obs):
        self.buf.append(obs)
        return self.get()

    def get(self):
        return np.concatenate(self.buf).astype(np.float32)


class Replay:
    def __init__(self, cap: int, dim: int, rng: np.random.Generator):
        self.s = np.zeros((cap, dim), np.float32)
        self.s2 = np.zeros((cap, dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.n, self.i, self.rng = cap, 0, 0, rng

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = self.rng.integers(0, self.n, size=b)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.d[idx]))


class DQNAgent:
    def __init__(self, cfg: AgentConfig, seed: int):
        self.cfg = cfg
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        dim = OBS_DIM * cfg.history
        self.q = QNet(dim, cfg.hidden, cfg.dueling, cfg.encoder)
        self.qt = QNet(dim, cfg.hidden, cfg.dueling, cfg.encoder)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = Replay(cfg.buffer_size, dim, self.rng)
        self.steps = 0
        total = cfg.n_episodes * 48
        self.k_eps = math.log((cfg.eps_max - cfg.eps_min) / 0.04) / (cfg.eps_reach_frac * total)

    def epsilon(self):
        c = self.cfg
        return c.eps_min + (c.eps_max - c.eps_min) * math.exp(-self.k_eps * self.steps)

    @torch.no_grad()
    def greedy(self, s: np.ndarray) -> np.ndarray:
        return self.q(torch.from_numpy(np.atleast_2d(s))).argmax(1).numpy()

    def act(self, s):
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(N_ACTIONS))
        return int(self.greedy(s)[0])

    def update(self):
        c = self.cfg
        s, a, r, s2, d = self.replay.sample(c.batch_size)
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if c.double:
                a2 = self.q(s2).argmax(1, keepdim=True)
                q2 = self.qt(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.qt(s2).max(1).values
            y = r + c.gamma * (1.0 - d) * q2
        loss = Fnn.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        return loss.item()

    def observe(self, s, a, r, s2, d):
        self.replay.add(s, a, r, s2, d)
        self.steps += 1
        loss = None
        if self.replay.n >= self.cfg.learning_starts:
            loss = self.update()
        if self.steps % self.cfg.target_update == 0:
            self.qt.load_state_dict(self.q.state_dict())
        return loss

    def state_dict(self):
        return {"cfg": asdict(self.cfg), "q": self.q.state_dict()}


def evaluate_agent(agent: DQNAgent, env_cfg: EnvConfig, seeds, keep_traces: bool = False):
    """Greedy, batched evaluation over a fixed set of episode seeds (common random numbers)."""
    k = agent.cfg.history
    envs = [CognitiveSchedulingEnv(env_cfg) for _ in seeds]
    hists = [History(k) for _ in seeds]
    states = np.stack([h.reset(e.reset(int(s))) for e, h, s in zip(envs, hists, seeds)])
    for _ in range(env_cfg.horizon):
        acts = agent.greedy(states)
        nxt = []
        for e, h, a in zip(envs, hists, acts):
            o, _, _, _ = e.step(int(a))
            nxt.append(h.push(o))
        states = np.stack(nxt)
    rows = []
    for e, s in zip(envs, seeds):
        m = e.metrics()
        m["seed"] = int(s)
        rows.append(m)
    if keep_traces:
        return rows, [e.traces() for e in envs]
    return rows


def randomized_cfg(base: EnvConfig, rng: np.random.Generator) -> EnvConfig:
    """Domain randomisation over workload, worker profile and sensor noise."""
    n = int(rng.integers(8, 17))
    f = rng.uniform(0.7, 1.3)
    return base.with_(n_tasks=n, n_initial=n // 2, alpha=base.alpha * f, delta=base.delta * f,
                      beta=base.beta * (2 - f), eta=base.eta * (2 - f), obs_noise=rng.uniform(0.05, 0.2))


def train_agent(agent_cfg: AgentConfig, env_cfg: EnvConfig, seed: int, eval_seeds=None, eval_every: int = 100,
                log_every_episode: bool = True, randomize: bool = False):
    agent = DQNAgent(agent_cfg, seed)
    env = CognitiveSchedulingEnv(env_cfg)
    dr_rng = np.random.default_rng(seed + 777)
    hist = History(agent_cfg.history)
    train_log, curve = [], []
    best_val, best_state, best_ep = -np.inf, None, 0
    for ep in range(agent_cfg.n_episodes):
        if randomize:
            env = CognitiveSchedulingEnv(randomized_cfg(env_cfg, dr_rng))
        s = hist.reset(env.reset(seed * 1_000_003 + ep))
        done, losses = False, []
        while not done:
            a = agent.act(s)
            o, r, done, info = env.step(a)
            s2 = hist.push(o)
            loss = agent.observe(s, a, r, s2, float(done))
            if loss is not None:
                losses.append(loss)
            s = s2
        if log_every_episode:
            train_log.append({"episode": ep, "return": info["return"], "on_time_rate": info["on_time_rate"],
                              "epsilon": agent.epsilon(), "td_loss": float(np.mean(losses)) if losses else np.nan})
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
