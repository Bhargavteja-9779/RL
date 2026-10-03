"""Rule-based scheduling baselines. Each policy maps the environment to a task index (-1 = break)."""
from __future__ import annotations

import itertools

import numpy as np

from .env import ACTION_SETS, EASY, HARD, CognitiveSchedulingEnv


def _pick(env: CognitiveSchedulingEnv, key) -> int:
    pend = env.pending_mask()
    if not pend.any():
        return -1
    idx = np.flatnonzero(pend)
    return int(idx[np.lexsort((idx, key[idx]))[0]])


class Policy:
    name = "policy"

    def reset(self):
        pass

    def act(self, env: CognitiveSchedulingEnv) -> int:
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"

    def act(self, env):
        return _pick(env, env.arrival)


class EDF(Policy):
    name = "EDF"

    def act(self, env):
        return _pick(env, env.deadline)


class SPT(Policy):
    name = "SPT"

    def act(self, env):
        return _pick(env, env.remaining)


class RandomMacro(Policy):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        return env.macro_to_task(ACTION_SETS["intensity"][int(self.rng.integers(3))])


class EDFPeriodic(Policy):
    """EDF with a fixed work/rest cycle: one break slot after every `period` work slots."""

    name = "EDF-Periodic"

    def __init__(self, period: int = 5):
        self.period = period

    def reset(self):
        self.since = 0

    def act(self, env):
        if self.since >= self.period:
            self.since = 0
            return -1
        j = _pick(env, env.deadline)
        self.since = self.since + 1 if j >= 0 else 0
        return j


class EDFThreshold(Policy):
    """Cognition-aware EDF: rest whenever observed fatigue/attention crosses a threshold."""

    name = "EDF-Threshold"

    def __init__(self, tau_f: float = 0.6, tau_a: float = 0.4):
        self.tau_f, self.tau_a = tau_f, tau_a

    def act(self, env):
        a, f = env.observed_cognition()
        if f > self.tau_f or a < self.tau_a:
            return -1
        return _pick(env, env.deadline)


class SPTPeriodic(EDFPeriodic):
    """SPT with a fixed work/rest cycle."""

    name = "SPT-Periodic"

    def act(self, env):
        if self.since >= self.period:
            self.since = 0
            return -1
        j = _pick(env, env.remaining)
        self.since = self.since + 1 if j >= 0 else 0
        return j


class SPTThreshold(EDFThreshold):
    """Cognition-aware SPT: rest whenever observed fatigue/attention crosses a threshold."""

    name = "SPT-Threshold"

    def act(self, env):
        a, f = env.observed_cognition()
        if f > self.tau_f or a < self.tau_a:
            return -1
        return _pick(env, env.remaining)


class CogHeuristic(Policy):
    """Hand-crafted attention-matching rule: rest on threshold, hard tasks only when attention is high."""

    name = "Cog-Heuristic"

    def __init__(self, tau_f: float = 0.6, tau_a: float = 0.4, tau_h: float = 0.7):
        self.tau_f, self.tau_a, self.tau_h = tau_f, tau_a, tau_h

    def act(self, env):
        a, f = env.observed_cognition()
        if f > self.tau_f or a < self.tau_a:
            return -1
        return env.macro_to_task(HARD if a >= self.tau_h else EASY)


def run_policy(env: CognitiveSchedulingEnv, policy: Policy, seeds, keep_traces: bool = False):
    rows, traces = [], []
    for s in seeds:
        env.reset(int(s))
        policy.reset()
        done = False
        while not done:
            _, _, done, info = env.step_task(policy.act(env))
        info["seed"] = int(s)
        rows.append(info)
        if keep_traces:
            traces.append(env.traces())
    return (rows, traces) if keep_traces else rows


GRIDS = {
    "EDF-Periodic": dict(period=[2, 3, 4, 5, 6, 8, 10]),
    "EDF-Threshold": dict(tau_f=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], tau_a=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    "Cog-Heuristic": dict(
        tau_f=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01], tau_a=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6], tau_h=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
}
GRIDS["SPT-Periodic"] = GRIDS["EDF-Periodic"]
GRIDS["SPT-Threshold"] = GRIDS["EDF-Threshold"]
CLASSES = {"EDF-Periodic": EDFPeriodic, "EDF-Threshold": EDFThreshold, "SPT-Periodic": SPTPeriodic,
           "SPT-Threshold": SPTThreshold, "Cog-Heuristic": CogHeuristic}


def tune(env: CognitiveSchedulingEnv, name: str, seeds) -> tuple[dict, float]:
    """Grid-search a parametric heuristic on tuning episodes (disjoint from test episodes)."""
    grid = GRIDS[name]
    keys = list(grid)
    best, best_val = None, -np.inf
    for vals in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, vals))
        rows = run_policy(env, CLASSES[name](**params), seeds)
        val = float(np.mean([r["return"] for r in rows]))
        if val > best_val:
            best, best_val = params, val
    return best, best_val


def fixed_baselines():
    return [FIFO(), EDF(), SPT(), RandomMacro(seed=12345)]
