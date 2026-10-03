"""Cognitively grounded single-worker task-scheduling simulator.

One episode is one simulated workday discretised into `horizon` decision slots.
The worker's latent cognitive state (homeostatic attention resource H, fatigue F)
evolves with workload, breaks and a circadian process; the scheduler only sees
noisy estimates of attention and fatigue (partial observability).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EASY, HARD, BREAK = 0, 1, 2
N_ACTIONS = 3
OBS_DIM = 11


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 48
    step_minutes: int = 10
    start_hour: float = 9.0
    # workload
    n_tasks: int = 12
    n_initial: int = 6
    arrival_window: float = 0.6
    diff_low: float = 0.2
    diff_high: float = 1.0
    hard_threshold: float = 0.6
    work_base: float = 1.0
    work_per_diff: float = 2.5
    slack_low: float = 0.3
    slack_high: float = 1.0
    # cognitive dynamics
    cognition: bool = True
    static_attention: float = 0.60
    static_fatigue: float = 0.30
    alpha: float = 0.02
    fatigue_coupling: float = 0.5
    beta: float = 0.30
    circadian: bool = True
    gamma: float = 0.15
    delta: float = 0.05
    eta: float = 0.20
    kappa: float = 0.5
    process_noise: float = 0.02
    obs_noise: float = 0.10
    rho0: float = 1.0
    # reward
    r_ontime: float = 1.0
    r_late_frac: float = 0.25
    r_miss: float = 0.5
    lam_fatigue: float = 0.10
    mu_attention: float = 0.05

    def with_(self, **kw) -> "EnvConfig":
        return replace(self, **kw)


def circadian(hour: np.ndarray | float) -> np.ndarray | float:
    """Normalised circadian modulation: late-morning peak, post-lunch dip at ~14:00."""
    return 0.2 * np.exp(-((hour - 10.5) ** 2) / (2 * 1.5**2)) - np.exp(-((hour - 14.0) ** 2) / (2 * 1.2**2))


class CognitiveSchedulingEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        T = cfg.horizon
        hours = cfg.start_hour + np.arange(T + 1) * cfg.step_minutes / 60.0
        self._circ = circadian(hours) if (cfg.circadian and cfg.cognition) else np.zeros(T + 1)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int) -> np.ndarray:
        c = self.cfg
        rng = np.random.default_rng(seed)
        N, T = c.n_tasks, c.horizon
        n0 = min(c.n_initial, N)
        arr = np.zeros(N, dtype=np.int64)
        arr[n0:] = np.sort(rng.integers(1, max(2, int(c.arrival_window * T)) + 1, size=N - n0))
        self.diff = rng.uniform(c.diff_low, c.diff_high, size=N)
        self.work = c.work_base + c.work_per_diff * self.diff
        slack = np.ceil(rng.uniform(c.slack_low, c.slack_high, size=N) * T).astype(np.int64)
        self.deadline = np.minimum(T, arr + slack)
        self.arrival = arr
        self.is_hard = self.diff >= c.hard_threshold
        self.remaining = self.work.copy()
        self.done_at = np.full(N, -1, dtype=np.int64)
        self.missed = np.zeros(N, dtype=bool)
        self.H = rng.uniform(0.8, 1.0)
        self.F = rng.uniform(0.0, 0.15)
        self._xi = rng.standard_normal(T)
        self._obs_eps = rng.standard_normal((T + 1, 2))
        self.t = 0
        self._log_A = np.zeros(T)
        self._log_F = np.zeros(T)
        self._log_act = np.full(T, -1, dtype=np.int64)
        self._effort = 0.0
        self._ret = 0.0
        return self._obs()

    # ------------------------------------------------------------ cognition
    def attention(self) -> float:
        if not self.cfg.cognition:
            return self.cfg.static_attention
        return float(np.clip(self.H + self.cfg.gamma * self._circ[self.t], 0.0, 1.0))

    def fatigue(self) -> float:
        return self.F if self.cfg.cognition else self.cfg.static_fatigue

    def observed_cognition(self) -> tuple[float, float]:
        s = self.cfg.obs_noise
        e = self._obs_eps[self.t]
        a = float(np.clip(self.attention() + s * e[0], 0.0, 1.0))
        f = float(np.clip(self.fatigue() + s * e[1], 0.0, 1.0))
        return a, f

    def efficiency(self, d: float) -> float:
        c = self.cfg
        return c.rho0 * self.attention() ** (0.5 + d) * (1.0 - c.kappa * self.fatigue())

    # ------------------------------------------------------------ task views
    def pending_mask(self) -> np.ndarray:
        return (self.arrival <= self.t) & (self.done_at < 0)

    def _edf_in(self, mask: np.ndarray) -> int:
        if not mask.any():
            return -1
        idx = np.flatnonzero(mask)
        return int(idx[np.lexsort((idx, self.deadline[idx]))[0]])

    def macro_to_task(self, action: int) -> int:
        pend = self.pending_mask()
        if action == BREAK or not pend.any():
            return -1
        want = self.is_hard if action == HARD else ~self.is_hard
        j = self._edf_in(pend & want)
        return j if j >= 0 else self._edf_in(pend)

    def _obs(self) -> np.ndarray:
        c = self.cfg
        t = min(self.t, c.horizon)
        a, f = self.observed_cognition()
        o = np.empty(OBS_DIM, dtype=np.float32)
        o[0] = t / c.horizon
        o[1] = a
        o[2] = f
        pend = (self.arrival <= t) & (self.done_at < 0)
        for k, cls in enumerate((~self.is_hard, self.is_hard)):
            m = pend & cls
            base = 3 + 4 * k
            o[base] = m.sum() / c.n_tasks
            o[base + 1] = self.remaining[m].sum() / 20.0
            o[base + 2] = np.clip((self.deadline[m].min() - t) / c.horizon, -1, 1) if m.any() else 1.0
            o[base + 3] = (m & (self.deadline <= t)).sum() / c.n_tasks
        return o

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        return self.step_task(self.macro_to_task(int(action)), macro=int(action))

    def step_task(self, j: int, macro: int | None = None):
        c = self.cfg
        t = self.t
        r = 0.0
        if j >= 0:
            assert self.arrival[j] <= t and self.done_at[j] < 0, "invalid task"
            d = self.diff[j]
            prog = self.efficiency(d)
            self._effort += min(prog, self.remaining[j])
            self.remaining[j] -= prog
            if self.remaining[j] <= 1e-9:
                self.remaining[j] = 0.0
                self.done_at[j] = t + 1
                val = c.r_ontime * (1.0 + d)
                r += val if t + 1 <= self.deadline[j] else c.r_late_frac * val
            if c.cognition:
                self.H = self.H - c.alpha * d * (1.0 + c.fatigue_coupling * self.F)
                self.F = self.F + c.delta * d * (1.0 - self.F)
            act = HARD if self.is_hard[j] else EASY
        else:
            if c.cognition:
                self.H = self.H + c.beta * (1.0 - self.H)
                self.F = self.F * (1.0 - c.eta)
            act = BREAK
        if c.cognition:
            self.H = float(np.clip(self.H + c.process_noise * self._xi[t], 0.0, 1.0))
            self.F = float(np.clip(self.F, 0.0, 1.0))
        self._log_act[t] = act if macro is None else macro
        self.t = t + 1
        A, F = self.attention(), self.fatigue()
        self._log_A[t] = A
        self._log_F[t] = F
        r += c.mu_attention * A - c.lam_fatigue * F
        newly_missed = (self.deadline == self.t) & (self.done_at < 0) & ~self.missed
        if newly_missed.any():
            self.missed |= newly_missed
            r -= c.r_miss * newly_missed.sum()
        self._ret += r
        done = self.t >= c.horizon
        obs = self._obs()
        info = self.metrics() if done else {}
        return obs, r, done, info

    # --------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        c = self.cfg
        on_time = (self.done_at > 0) & (self.done_at <= self.deadline)
        completed = self.done_at > 0
        n = c.n_tasks
        return {
            "return": self._ret,
            "on_time": int(on_time.sum()),
            "on_time_rate": on_time.sum() / n,
            "weighted_on_time": float((self.diff * on_time).sum() / self.diff.sum()),
            "completion_rate": completed.sum() / n,
            "miss_rate": 1.0 - on_time.sum() / n,
            "mean_attention": float(self._log_A.mean()),
            "mean_fatigue": float(self._log_F.mean()),
            "high_fatigue_frac": float((self._log_F > 0.7).mean()),
            "break_frac": float((self._log_act == BREAK).mean()),
            "effort": self._effort,
        }

    def traces(self) -> dict:
        return {"A": self._log_A.copy(), "F": self._log_F.copy(), "act": self._log_act.copy()}


# --------------------------------------------------------------- variants
NOMINAL = EnvConfig()

PROFILES = {
    "resilient": dict(alpha=0.02 * 0.7, delta=0.05 * 0.7, beta=0.30 * 1.3, eta=0.20 * 1.3),
    "nominal": {},
    "fatigue_prone": dict(alpha=0.02 * 1.3, delta=0.05 * 1.3, beta=0.30 * 0.75, eta=0.20 * 0.75),
}

ABLATIONS = {
    "full": {},
    "no_obs_noise": dict(obs_noise=0.0),
    "no_circadian": dict(circadian=False),
    "no_cognition": dict(cognition=False),
    "task_reward_only": dict(lam_fatigue=0.0, mu_attention=0.0),
}


def make_cfg(ablation: str = "full", **overrides) -> EnvConfig:
    return NOMINAL.with_(**ABLATIONS[ablation]).with_(**overrides)
