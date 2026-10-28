"""Heuristic tuning/evaluation, zero-shot robustness and policy traces.

python -m experiments.run_eval heuristics   # tune + test all rule-based baselines per environment variant
python -m experiments.run_eval robustness   # nominal-trained RL + nominal-tuned heuristics under shifted conditions
python -m experiments.run_eval traces       # per-step traces on test days for behavioural analysis
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

VARIANTS = {
    "full": ("full", {}),
    "no_obs_noise": ("no_obs_noise", {}),
    "no_circadian": ("no_circadian", {}),
    "no_cognition": ("no_cognition", {}),
    "task_reward_only": ("task_reward_only", {}),
    "lam0.05": ("full", dict(lam_fatigue=0.05, mu_attention=0.025)),
    "lam0.2": ("full", dict(lam_fatigue=0.2, mu_attention=0.1)),
    "lam0.4": ("full", dict(lam_fatigue=0.4, mu_attention=0.2)),
}


def load_agent(name, s):
    """Load a trained agent (DQN family or PPO) from results/runs; returns (agent, n_actions)."""
    import torch

    from aacs.agents import AgentConfig, DQNAgent

    ck = torch.load(RES / "runs" / f"{name}_s{s}.pt", weights_only=False)
    if ck.get("algo") == "ppo":
        from aacs.ppo import PPOAgent, PPOConfig

        ag = PPOAgent(PPOConfig(**ck["cfg"]), seed=s, n_actions=ck["n_actions"])
    else:
        ag = DQNAgent(AgentConfig(**ck["cfg"]), seed=s, n_actions=ck["n_actions"])
    ag.q.load_state_dict(ck["q"])
    return ag, ck["n_actions"]


def _heur_variant(name):
    from aacs.baselines import CLASSES, fixed_baselines, run_policy, tune
    from aacs.env import CognitiveSchedulingEnv, make_cfg
    from experiments.seeds import TEST, TUNING

    abl, over = VARIANTS[name]
    env = CognitiveSchedulingEnv(make_cfg(abl, **over))
    rows, params = [], {}
    for p in fixed_baselines():
        for r in run_policy(env, p, TEST):
            rows.append({"method": p.name, **r})
    for hname, cls in CLASSES.items():
        best, val = tune(env, hname, TUNING)
        params[hname] = {"params": best, "tuning_return": val}
        for r in run_policy(env, cls(**best), TEST):
            rows.append({"method": hname, **r})
    out = RES / "heuristics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}_params.json").write_text(json.dumps(params, indent=1))
    return name, params


def heuristics(workers=2):
    with Pool(workers) as pool:
        for name, params in pool.imap_unordered(_heur_variant, list(VARIANTS)):
            print(name, params, flush=True)


ROBUST_AGENTS = [("DQN", 10), ("AA-DQN", 10), ("AA-DQN-DR", 5), ("AA-DQN-3act", 10)]
ROBUST = {
    **{f"noise{s}": dict(obs_noise=s) for s in (0.0, 0.05, 0.1, 0.2, 0.3)},
    "profile_resilient": "resilient",
    "profile_fatigue_prone": "fatigue_prone",
    "tasks8": dict(n_tasks=8, n_initial=4),
    "tasks16": dict(n_tasks=16, n_initial=8),
}


def _robust_one(cond):
    import torch

    torch.set_num_threads(1)
    from aacs.agents import AgentConfig, DQNAgent, evaluate_agent
    from aacs.baselines import CLASSES, fixed_baselines, run_policy
    from aacs.env import PROFILES, CognitiveSchedulingEnv, make_cfg
    from experiments.seeds import TEST

    spec = ROBUST[cond]
    cfg = make_cfg("full", **(PROFILES[spec] if isinstance(spec, str) else spec))
    env = CognitiveSchedulingEnv(cfg)
    params = json.loads((RES / "heuristics" / "full_params.json").read_text())
    rows = []
    pols = fixed_baselines()[:3] + [CLASSES[h](**params[h]["params"]) for h in CLASSES]  # FIFO, EDF, SPT + tuned
    for p in pols:
        for r in run_policy(env, p, TEST):
            rows.append({"condition": cond, "method": p.name, "train_seed": -1, **r})
    for name, nseeds in ROBUST_AGENTS:
        for s in range(nseeds):
            ag, nact = load_agent(name, s)
            acfg = cfg.with_(action_set="intensity") if nact == 3 else cfg
            for r in evaluate_agent(ag, acfg, TEST):
                rows.append({"condition": cond, "method": name, "train_seed": s, **r})
    return cond, rows


def robustness(workers=4):
    allrows = []
    with Pool(workers) as pool:
        for cond, rows in pool.imap_unordered(_robust_one, list(ROBUST)):
            print("robustness", cond, flush=True)
            allrows += rows
    pd.DataFrame(allrows).to_csv(RES / "robustness.csv.gz", index=False)


def traces():
    import torch

    from aacs.agents import AgentConfig, DQNAgent, evaluate_agent
    from aacs.baselines import CLASSES, run_policy
    from aacs.env import CognitiveSchedulingEnv, make_cfg
    from experiments.seeds import TEST

    cfg = make_cfg("full")
    env = CognitiveSchedulingEnv(cfg)
    params = json.loads((RES / "heuristics" / "full_params.json").read_text())
    out = {}
    from aacs.baselines import EDF
    for p in [EDF()] + [CLASSES[h](**params[h]["params"]) for h in CLASSES]:
        _, tr = run_policy(env, p, TEST, keep_traces=True)
        out[p.name] = tr
    for name in ("DQN", "AA-DQN", "AA-DQN-3act", "PPO", "PPO-3act"):
        trs = []
        for s in range(10):
            ag, nact = load_agent(name, s)
            acfg = cfg.with_(action_set="intensity") if nact == 3 else cfg
            _, tr = evaluate_agent(ag, acfg, TEST, keep_traces=True)
            trs += tr
        out[name] = trs
    arrays = {}
    for m, trs in out.items():
        key = m.replace("-", "_")
        for f in ("A", "F", "act"):
            arrays[f"{key}__{f}"] = np.stack([t[f] for t in trs])
    np.savez_compressed(RES / "traces.npz", **arrays)
    print("traces saved", list(out))


SIMSENS = {f"{p}_{tag}": {p: getattr(__import__("aacs.env", fromlist=["NOMINAL"]).NOMINAL, p) * f}
           for p in ("alpha", "beta", "delta", "eta", "kappa", "gamma", "fatigue_coupling")
           for tag, f in (("m30", 0.7), ("p30", 1.3))}


def _simsens_one(cond):
    import torch

    torch.set_num_threads(1)
    from aacs.agents import AgentConfig, DQNAgent, evaluate_agent
    from aacs.baselines import CLASSES, EDF, run_policy
    from aacs.env import CognitiveSchedulingEnv, make_cfg
    from experiments.seeds import TEST

    cfg = make_cfg("full", **SIMSENS[cond])
    env = CognitiveSchedulingEnv(cfg)
    params = json.loads((RES / "heuristics" / "full_params.json").read_text())
    rows = []
    for p in [EDF()] + [CLASSES[h](**params[h]["params"]) for h in ("EDF-Periodic", "SPT-Periodic", "SPT-Threshold")]:
        for r in run_policy(env, p, TEST):
            rows.append({"condition": cond, "method": p.name, "train_seed": -1, **r})
    for name in ("AA-DQN", "AA-DQN-3act", "PPO", "PPO-3act"):
        for s in range(10):
            ag, nact = load_agent(name, s)
            acfg = cfg.with_(action_set="intensity") if nact == 3 else cfg
            for r in evaluate_agent(ag, acfg, TEST):
                rows.append({"condition": cond, "method": name, "train_seed": s, **r})
    return cond, rows


def simsens(workers=4):
    """One-at-a-time +/-30% perturbation of each cognitive parameter, zero-shot."""
    allrows = []
    with Pool(workers) as pool:
        for cond, rows in pool.imap_unordered(_simsens_one, list(SIMSENS)):
            print("simsens", cond, flush=True)
            allrows += rows
    pd.DataFrame(allrows).to_csv(RES / "simsens.csv.gz", index=False)


def _robust_agents_only(args):
    cond, names = args
    import torch

    torch.set_num_threads(1)
    from aacs.agents import evaluate_agent
    from aacs.env import PROFILES, make_cfg
    from experiments.seeds import TEST

    spec = ROBUST[cond]
    cfg = make_cfg("full", **(PROFILES[spec] if isinstance(spec, str) else spec))
    rows = []
    for name in names:
        for s in range(10):
            ag, nact = load_agent(name, s)
            acfg = cfg.with_(action_set="intensity") if nact == 3 else cfg
            for r in evaluate_agent(ag, acfg, TEST):
                rows.append({"condition": cond, "method": name, "train_seed": s, **r})
    return cond, rows


def robustness_add(names=("PPO", "PPO-3act"), workers=4):
    """Evaluate additional agents under the robustness conditions and merge into robustness.csv.gz."""
    f = RES / "robustness.csv.gz"
    df = pd.read_csv(f)
    df = df[~df.method.isin(names)]
    new = []
    with Pool(workers) as pool:
        for cond, rows in pool.imap_unordered(_robust_agents_only, [(c, list(names)) for c in ROBUST]):
            print("robustness_add", cond, flush=True)
            new += rows
    pd.concat([df, pd.DataFrame(new)]).to_csv(f, index=False)


if __name__ == "__main__":
    {"heuristics": heuristics, "robustness": robustness, "traces": traces, "simsens": simsens, "robustness_add": robustness_add}[sys.argv[1]]()
