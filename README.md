# Attention-aware task scheduling under partial observability

Code, data and manuscript for *"Attention-aware task scheduling under partial observability of human cognitive state: a memory-augmented deep reinforcement learning approach"* (P. N. Bhargav Teja, Lanka Sree Chathurya, Rajay Vedaraj I.S., Vellore Institute of Technology).

A single worker processes a stream of tasks during a simulated workday. Attention depletes with effort, fatigue accumulates and recovers, alertness follows a circadian rhythm, and the scheduler only sees noisy estimates of these states. Rest is an explicit action. A memory-augmented deep Q-network (AA-DQN) is benchmarked against nine rule-based schedulers, including grid-tuned EDF/SPT rules with periodic or state-triggered rest.

## Layout

| Path | Content |
|---|---|
| `aacs/env.py` | Cognitive scheduling simulator (POMDP) |
| `aacs/baselines.py` | FIFO, EDF, SPT, random, EDF/SPT-Periodic, EDF/SPT-Threshold, Cog-Heuristic, grid tuning |
| `aacs/agents.py` | DQN / AA-DQN (history, Double, Dueling, optional GRU), training, evaluation |
| `experiments/` | Seed ranges, training runner (resumable, parallel), heuristic/robustness/trace evaluation |
| `analysis/analyze.py` | All statistics, LaTeX tables, figures and `paper/numbers.tex` |
| `results/` | Compressed per-run records (`runs/*.json.gz`), model weights, heuristic results, summaries |
| `paper/` | Manuscript, supplementary material, cover letter, highlights, title page, internal review; `build.sh` builds PDFs and Word files |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest                                   # simulator tests
python -m experiments.run_eval heuristics          # tune + test rule-based baselines
python -m experiments.run_rl --workers 4           # train all agents (resumable)
python -m experiments.run_eval robustness          # zero-shot robustness
python -m experiments.run_eval traces              # behavioural traces
python -m analysis.analyze                         # tables, figures, statistics
bash paper/build.sh                                # PDFs + Word versions
```

To regenerate the analysis from the committed results without retraining, decompress nothing: `analysis/analyze.py` reads `results/runs/*.json.gz` directly (robustness and trace evaluations need the `.pt` weights, which are included).

Training one agent (5000 workdays) takes about 12 minutes on one CPU thread.
