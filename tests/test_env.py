import numpy as np
import pytest

from aacs.baselines import EDF, FIFO, EDFThreshold, run_policy
from aacs.env import BREAK, HARD, OBS_DIM, CognitiveSchedulingEnv, make_cfg


def rollout(env, policy, seed):
    env.reset(seed)
    policy.reset()
    done, total = False, 0.0
    while not done:
        _, r, done, info = env.step_task(policy.act(env))
        total += r
    return total, info, env.traces()


def test_common_random_numbers_are_deterministic():
    env = CognitiveSchedulingEnv(make_cfg())
    a = rollout(env, EDF(), 7)
    b = rollout(env, EDF(), 7)
    assert a[0] == b[0] and np.array_equal(a[2]["A"], b[2]["A"])


def test_states_bounded_and_observation_shape():
    env = CognitiveSchedulingEnv(make_cfg())
    obs = env.reset(3)
    assert obs.shape == (OBS_DIM,)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, _, done, _ = env.step(int(rng.integers(3)))
        assert 0.0 <= env.attention() <= 1.0 and 0.0 <= env.fatigue() <= 1.0
        assert np.all(np.isfinite(obs))


def test_return_matches_sum_of_rewards():
    env = CognitiveSchedulingEnv(make_cfg())
    total, info, _ = rollout(env, FIFO(), 11)
    assert info["return"] == pytest.approx(total)


def test_breaks_recover_and_work_fatigues():
    cfg = make_cfg(process_noise=0.0)
    env = CognitiveSchedulingEnv(cfg)
    env.reset(0)
    f0 = env.F
    env.step(HARD)
    assert env.F > f0
    f1, h1 = env.F, env.H
    env.step(BREAK)
    assert env.F < f1 and env.H > h1


def test_static_capacity_variant_has_constant_cognition():
    env = CognitiveSchedulingEnv(make_cfg("no_cognition"))
    _, info, tr = rollout(env, EDF(), 5)
    assert np.allclose(tr["A"], 0.60) and np.allclose(tr["F"], 0.30)


def test_threshold_policy_never_rests_with_inactive_thresholds():
    env = CognitiveSchedulingEnv(make_cfg())
    rows = run_policy(env, EDFThreshold(tau_f=1.01, tau_a=0.0), range(5))
    assert all(r["break_frac"] == 0.0 or r["completion_rate"] == 1.0 for r in rows)


def test_metric_ranges():
    env = CognitiveSchedulingEnv(make_cfg())
    rows = run_policy(env, EDF(), range(20))
    for r in rows:
        assert 0 <= r["on_time_rate"] <= r["completion_rate"] <= 1
        assert 0 <= r["mean_fatigue"] <= 1
