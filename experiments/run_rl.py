"""Train all RL configurations (resumable, parallel). Usage: python -m experiments.run_rl [--workers 4] [--only GROUP]"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def jobs():
    from aacs.agents import AADQN_CFG, DQN_CFG

    J = []
    for s in range(10):
        J.append(("main", "DQN", DQN_CFG, "full", {}, s))
        J.append(("main", "AA-DQN", AADQN_CFG, "full", {}, s))
    for abl in ("no_obs_noise", "no_circadian", "no_cognition", "task_reward_only"):
        for s in range(10):
            J.append(("ablation", f"AA-DQN@{abl}", AADQN_CFG, abl, {}, s))
    for lam in (0.05, 0.2, 0.4):
        for s in range(5):
            J.append(("pareto", f"AA-DQN@lam{lam}", AADQN_CFG, "full", dict(lam_fatigue=lam, mu_attention=lam / 2), s))
    comp = {
        "AA-DQN-noHist": replace(AADQN_CFG, name="AA-DQN-noHist", history=1),
        "AA-DQN-noDouble": replace(AADQN_CFG, name="AA-DQN-noDouble", double=False),
        "AA-DQN-noDueling": replace(AADQN_CFG, name="AA-DQN-noDueling", dueling=False),
    }
    for name, cfg in comp.items():
        for s in range(5):
            J.append(("component", name, cfg, "full", {}, s))
    sens = {
        "AA-DQN-k2": replace(AADQN_CFG, name="AA-DQN-k2", history=2),
        "AA-DQN-k8": replace(AADQN_CFG, name="AA-DQN-k8", history=8),
        "AA-DQN-lr1e-4": replace(AADQN_CFG, name="AA-DQN-lr1e-4", lr=1e-4),
        "AA-DQN-lr1e-3": replace(AADQN_CFG, name="AA-DQN-lr1e-3", lr=1e-3),
    }
    for name, cfg in sens.items():
        for s in range(3):
            J.append(("sensitivity", name, cfg, "full", {}, s))
    gru = replace(AADQN_CFG, name="AA-DQN-GRU", history=8, encoder="gru")
    for s in range(5):
        J.append(("recurrent", "AA-DQN-GRU", gru, "full", {}, s))
    return J


def tag_of(name, seed):
    return f"{name.replace('@', '__')}_s{seed}"


def run_job(job):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from aacs.agents import evaluate_agent, train_agent
    from aacs.env import make_cfg
    from experiments.seeds import TEST, VALIDATION

    group, name, acfg, abl, over, seed = job
    tag = tag_of(name, seed)
    out = RUNS / f"{tag}.json"
    if out.exists():
        return tag, "cached"
    ecfg = make_cfg(abl, **over)
    t0 = time.time()
    agent, log, curve = train_agent(acfg, ecfg, seed, eval_seeds=VALIDATION)
    train_time = time.time() - t0
    test_final = evaluate_agent(agent, ecfg, TEST)
    torch.save(agent.state_dict(), RUNS / f"{tag}_final.pt")
    agent.q.load_state_dict(agent.best_state)
    t1 = time.time()
    test = evaluate_agent(agent, ecfg, TEST)
    infer_us = (time.time() - t1) / (len(TEST) * ecfg.horizon) * 1e6
    torch.save(agent.state_dict(), RUNS / f"{tag}.pt")
    rec = dict(group=group, name=name, seed=seed, ablation=abl, env_overrides=over, agent_cfg=asdict(acfg),
               env_cfg=asdict(ecfg), train_time_s=train_time, eval_us_per_decision=infer_us,
               train_log=log, curve=curve, best_episode=agent.best_episode, test=test, test_final=test_final,
               test_mean={k: float(np.mean([r[k] for r in test])) for k in test[0] if k != "seed"},
               test_final_mean={k: float(np.mean([r[k] for r in test_final])) for k in test_final[0] if k != "seed"})
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)
    return tag, (f"{train_time:.0f}s best@{agent.best_episode} return={rec['test_mean']['return']:.3f} "
                 f"on_time={rec['test_mean']['on_time_rate']:.3f} | final return={rec['test_final_mean']['return']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    J = [j for j in jobs() if a.only is None or j[0] in a.only.split(",")]
    print(f"{len(J)} jobs", flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for tag, msg in pool.imap_unordered(run_job, J):
            print(time.strftime("%H:%M:%S"), tag, msg, flush=True)


if __name__ == "__main__":
    main()
