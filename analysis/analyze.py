"""Aggregate experiment outputs into statistics, LaTeX tables and publication figures.

python -m analysis.analyze
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
SUM = RES / "summary"
for d in (FIG, TAB, SUM):
    d.mkdir(parents=True, exist_ok=True)

C = {"AA-DQN": "#2a78d6", "AA-DQN-DR": "#4a3aa7", "DQN": "#eb6834", "Cog-Heuristic": "#1baf7a", "EDF-Threshold": "#eda100",
     "EDF-Periodic": "#e87ba4", "SPT-Threshold": "#008300", "SPT-Periodic": "#e34948", "EDF": "#6b6a66", "FIFO": "#9a9993", "SPT": "#52514e", "Random": "#c3c2b7"}
MK = {"AA-DQN": "o", "AA-DQN-DR": "P", "DQN": "s", "Cog-Heuristic": "^", "EDF-Threshold": "D", "EDF-Periodic": "v",
      "SPT-Threshold": "h", "SPT-Periodic": "<", "EDF": "x",
      "FIFO": "+", "SPT": "*", "Random": "."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["Random", "FIFO", "EDF", "SPT", "EDF-Periodic", "EDF-Threshold", "SPT-Periodic", "SPT-Threshold",
         "Cog-Heuristic", "DQN", "AA-DQN"]
HEUR = ["Random", "FIFO", "EDF", "SPT", "EDF-Periodic", "EDF-Threshold", "SPT-Periodic", "SPT-Threshold", "Cog-Heuristic"]
METRICS = ["return", "on_time_rate", "weighted_on_time", "completion_rate", "mean_attention", "mean_fatigue",
           "high_fatigue_frac", "break_frac", "effort"]
RNG = np.random.default_rng(2024)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "lines.linewidth": 1.6,
})
W1, W2 = 3.5, 7.2


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def boot_ci(x, n=5000, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    b = stat(x[idx], axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


# ------------------------------------------------------------------ loading
def load_runs():
    import gzip

    recs = {}
    for f in sorted((RES / "runs").glob("*.json.gz")):
        recs[f.name[:-8]] = json.loads(gzip.decompress(f.read_bytes()))
    for f in sorted((RES / "runs").glob("*.json")):
        recs[f.stem] = json.loads(f.read_text())
    return list(recs.values())


def test_frame_final(recs, name):
    rows = []
    for r in recs:
        if r["name"] == name and "test_final" in r:
            for t in r["test_final"]:
                rows.append({"method": name, "train_seed": r["seed"], **t})
    return pd.DataFrame(rows)


def final_vs_best(recs):
    rows = []
    for n in ("DQN", "AA-DQN"):
        b, f = test_frame(recs, n), test_frame_final(recs, n)
        if b.empty or f.empty:
            continue
        be = [r["best_episode"] for r in recs if r["name"] == n]
        for lab, df in (("validation-selected", b), ("final iterate", f)):
            rr, oo = per_seed(df, "return"), per_seed(df, "on_time_rate")
            rows.append(dict(method=n, checkpoint=lab, ret=rr.mean(), ret_sd=rr.std(ddof=1), on=oo.mean(),
                             on_sd=oo.std(ddof=1), best_ep=np.mean(be)))
    df = pd.DataFrame(rows)
    df.to_csv(SUM / "final_vs_best.csv", index=False)
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Method & Checkpoint & Return & On-time rate & Mean selected episode \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.method} & {r.checkpoint} & {r.ret:.2f} $\\pm$ {r.ret_sd:.2f} & {r.on:.3f} $\\pm$ {r.on_sd:.3f} & "
                     + (f"{r.best_ep:.0f}" if r.checkpoint.startswith("val") else "3000") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "final_vs_best.tex").write_text("\n".join(lines))
    return df


def test_frame(recs, name):
    rows = []
    for r in recs:
        if r["name"] == name:
            for t in r["test"]:
                rows.append({"method": name, "train_seed": r["seed"], **t})
    return pd.DataFrame(rows)


def heur_frame(variant):
    return pd.read_csv(RES / "heuristics" / f"{variant}.csv")


def per_seed(df, metric):
    return df.groupby("train_seed")[metric].mean().values


def summarize(rl: dict, heur: pd.DataFrame):
    """rl: name -> frame with train_seed; heur: frame with method column (deterministic)."""
    rows = []
    for m in ORDER:
        if m in rl:
            df = rl[m]
            row = {"method": m, "n_seeds": df.train_seed.nunique()}
            for k in METRICS:
                v = per_seed(df, k)
                lo, hi = boot_ci(v)
                row.update({k: v.mean(), f"{k}_sd": v.std(ddof=1), f"{k}_lo": lo, f"{k}_hi": hi})
        elif m in set(heur.method):
            df = heur[heur.method == m]
            row = {"method": m, "n_seeds": 0}
            for k in METRICS:
                v = df[k].values
                lo, hi = boot_ci(v)
                row.update({k: v.mean(), f"{k}_sd": v.std(ddof=1), f"{k}_lo": lo, f"{k}_hi": hi})
        else:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_tests(rl_df, heur, metric="return", against=None):
    """Episode-level paired comparison (common random numbers): RL averaged over training seeds vs each heuristic."""
    rl_ep = rl_df.groupby("seed")[metric].mean()
    out = []
    for h in against or HEUR:
        hv = heur[heur.method == h].set_index("seed")[metric].reindex(rl_ep.index)
        d = rl_ep.values - hv.values
        w = stats.wilcoxon(rl_ep.values, hv.values, zero_method="wilcox")
        nz = d[d != 0]
        ranks = stats.rankdata(np.abs(nz))
        rbc = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
        seed_means = per_seed(rl_df, metric)
        t = stats.ttest_1samp(seed_means, hv.values.mean())
        lo, hi = boot_ci(d)
        out.append({"baseline": h, "mean_diff": d.mean(), "diff_lo": lo, "diff_hi": hi,
                    "p_wilcoxon": w.pvalue, "rank_biserial": rbc, "win_rate": float((d > 0).mean()),
                    "p_seed_t": t.pvalue, "seeds_better": int((seed_means > hv.values.mean()).sum())})
    df = pd.DataFrame(out)
    df["p_holm"] = holm(df.p_wilcoxon.values)
    return df


def fmt_p(p):
    return r"$<10^{-4}$" if p < 1e-4 else f"{p:.4f}"


# ------------------------------------------------------------------ analyses
def main_results(recs):
    rl = {m: test_frame(recs, m) for m in ("DQN", "AA-DQN")}
    heur = heur_frame("full")
    s = summarize(rl, heur)
    s.to_csv(SUM / "main_summary.csv", index=False)
    tests = {k: paired_tests(rl["AA-DQN"], heur, k) for k in ("return", "on_time_rate", "mean_fatigue", "weighted_on_time", "completion_rate")}
    tests["return_vs_DQN"] = paired_tests(rl["AA-DQN"], pd.concat([rl["DQN"].groupby("seed").mean(numeric_only=True)
                                                                   .reset_index().assign(method="DQN")]),
                                          "return", against=["DQN"])
    for k, v in tests.items():
        v.to_csv(SUM / f"tests_{k}.csv", index=False)

    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & Return & On-time rate & Weighted on-time & Mean attention & Mean fatigue & Break share \\",
             r"\midrule"]
    best = {k: (s[k].max() if k not in ("mean_fatigue",) else s[k].min()) for k in METRICS}
    for _, r in s.iterrows():
        cells = []
        for k, d in (("return", 2), ("on_time_rate", 3), ("weighted_on_time", 3), ("mean_attention", 3),
                     ("mean_fatigue", 3), ("break_frac", 3)):
            v = f"{r[k]:.{d}f} $\\pm$ {r[k + '_sd']:.{d}f}"
            if k in ("return", "on_time_rate", "weighted_on_time") and np.isclose(r[k], best[k]):
                v = r"\textbf{" + v + "}"
            cells.append(v)
        name = r["method"] + (r" (ours)" if r["method"] == "AA-DQN" else "")
        if r["method"] == "DQN":
            lines.append(r"\midrule")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "main_results.tex").write_text("\n".join(lines))

    t = tests["return"].merge(tests["on_time_rate"], on="baseline", suffixes=("_r", "_o"))
    t = t.merge(tests["weighted_on_time"].add_suffix("_w").rename(columns={"baseline_w": "baseline"}), on="baseline")
    lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Episode return} & \multicolumn{3}{c}{On-time rate (pp)} & \multicolumn{3}{c}{Weighted on-time rate (pp)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"AA-DQN vs. & $\Delta$ [95\% CI] & $r_{rb}$ & $p_{\mathrm{Holm}}$ & $\Delta$ [95\% CI] & $r_{rb}$ & $p_{\mathrm{Holm}}$ & $\Delta$ [95\% CI] & $r_{rb}$ & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    for _, r in t.iterrows():
        cells = [r.baseline]
        for suf, sc, dg in (("_r", 1, 2), ("_o", 100, 1), ("_w", 100, 1)):
            cells += [f"{sc * r['mean_diff' + suf]:+.{dg}f} [{sc * r['diff_lo' + suf]:+.{dg}f}, {sc * r['diff_hi' + suf]:+.{dg}f}]",
                      f"{r['rank_biserial' + suf]:.2f}", fmt_p(r["p_holm" + suf])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "stat_tests.tex").write_text("\n".join(lines))
    return rl, heur, s, tests


def fig_main(s):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
    ylab = list(s.method)
    y = np.arange(len(ylab))
    for ax, (k, lab) in zip(axes, (("return", "Episode return"), ("on_time_rate", "On-time completion rate"),
                                   ("mean_fatigue", "Mean fatigue (lower is better)"))):
        for i, r in s.iterrows():
            m = r.method
            ax.errorbar(r[k], i, xerr=[[r[k] - r[k + "_lo"]], [r[k + "_hi"] - r[k]]], fmt=MK[m], color=C[m],
                        ms=5 if m != "AA-DQN" else 6.5, capsize=2, lw=1.2, mec=C[m])
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [m + (" (ours)" if m == "AA-DQN" else "") for m in ylab])
    best_h = s[s.method.isin(HEUR)]["return"].max()
    axes[0].axvline(best_h, color=INK2, lw=0.6, ls=":")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_main_comparison")


def fig_curves(recs, heur):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    for name in ("DQN", "AA-DQN"):
        cur = [pd.DataFrame(r["curve"]) for r in recs if r["name"] == name]
        if not cur:
            continue
        ep = cur[0].episode.values
        for ax, k in zip(axes, ("return", "on_time_rate")):
            M = np.stack([c[k].values for c in cur])
            mu = M.mean(0)
            se = M.std(0, ddof=1) / np.sqrt(len(M))
            ax.plot(ep, mu, color=C[name], label=name + (" (ours)" if name == "AA-DQN" else ""))
            ax.fill_between(ep, mu - 1.96 * se, mu + 1.96 * se, color=C[name], alpha=0.18, lw=0)
    from aacs.baselines import CLASSES, EDF, run_policy
    from aacs.env import CognitiveSchedulingEnv, make_cfg
    from experiments.seeds import VALIDATION

    params = json.loads((RES / "heuristics" / "full_params.json").read_text())
    env = CognitiveSchedulingEnv(make_cfg("full"))
    ref = {}
    for h, pol in (("SPT-Threshold", CLASSES["SPT-Threshold"](**params["SPT-Threshold"]["params"])), ("EDF", EDF())):
        rows = run_policy(env, pol, VALIDATION)
        ref[h] = {k: np.mean([r[k] for r in rows]) for k in ("return", "on_time_rate")}
    for ax, k, lab in zip(axes, ("return", "on_time_rate"), ("Validation return", "Validation on-time rate")):
        for h, ls, txt in (("SPT-Threshold", "--", "SPT-Threshold"), ("EDF", "-.", "EDF")):
            v = ref[h][k]
            ax.axhline(v, color=C[h], lw=1.0, ls=ls)
            ax.text(5000, v, " " + txt, color=INK2, fontsize=6.5, va="center", ha="left")
        ax.set_xlabel("Training episode (simulated workdays)")
        ax.set_ylabel(lab)
        ax.set_xlim(0, 5000)
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=4)
    save(fig, "fig_learning_curves")


def fig_td(recs):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2))
    for name in ("DQN", "AA-DQN"):
        logs = [pd.DataFrame(r["train_log"]) for r in recs if r["name"] == name]
        if not logs:
            continue
        L = np.stack([l.td_loss.rolling(50, min_periods=1).mean().values for l in logs])
        R = np.stack([l["return"].rolling(50, min_periods=1).mean().values for l in logs])
        x = np.arange(L.shape[1]) + 1
        for ax, M in zip(axes, (R, L)):
            mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
            ax.plot(x, mu, color=C[name], label=name)
            ax.fill_between(x, mu - sd, mu + sd, color=C[name], alpha=0.15, lw=0)
    axes[0].set_ylabel("Training return (50-ep. mean)")
    axes[1].set_ylabel("TD loss (Huber, 50-ep. mean)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Training episode")
    axes[0].legend()
    fig.tight_layout(w_pad=3)
    save(fig, "fig_training_dynamics")


ABL = [("full", "Full model"), ("no_obs_noise", "No observation noise"), ("no_circadian", "No circadian process"),
       ("task_reward_only", "Task-only reward"), ("no_cognition", "Static capacity (no cognition)")]


def ablations(recs):
    rows, tex = [], []
    for v, lab in ABL:
        name = "AA-DQN" if v == "full" else f"AA-DQN@{v}"
        rl = test_frame(recs, name)
        if rl.empty or not (RES / "heuristics" / f"{v}.csv").exists():
            continue
        heur = heur_frame(v)
        means = heur.groupby("method")[["return", "on_time_rate", "mean_fatigue"]].mean()
        best = means["return"].idxmax()
        best_o = means["on_time_rate"].idxmax()
        t = paired_tests(rl, heur, "return", against=[best]).iloc[0]
        to = paired_tests(rl, heur, "on_time_rate", against=[best_o]).iloc[0]
        rs = per_seed(rl, "return")
        os_ = per_seed(rl, "on_time_rate")
        fs = per_seed(rl, "mean_fatigue")
        rows.append(dict(variant=v, label=lab, rl_return=rs.mean(), rl_return_sd=rs.std(ddof=1), rl_on_time=os_.mean(),
                         rl_fatigue=fs.mean(), best_heur=best, best_heur_return=means.loc[best, "return"],
                         best_heur_on=best_o, best_heur_on_time=means.loc[best_o, "on_time_rate"],
                         edf_return=means.loc["EDF", "return"], edf_on_time=means.loc["EDF", "on_time_rate"],
                         diff=t.mean_diff, diff_lo=t.diff_lo, diff_hi=t.diff_hi, p=t.p_wilcoxon,
                         diff_o=to.mean_diff, diff_o_lo=to.diff_lo, diff_o_hi=to.diff_hi, p_o=to.p_wilcoxon))
    df = pd.DataFrame(rows)
    df.to_csv(SUM / "ablations.csv", index=False)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Environment variant & AA-DQN return & Best heuristic (return) & $\Delta$ return [95\% CI] & AA-DQN on-time & Best heuristic (on-time) & $\Delta$ on-time [95\% CI] \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.label} & {r.rl_return:.2f} $\\pm$ {r.rl_return_sd:.2f} & {r.best_heur} ({r.best_heur_return:.2f}) & "
                     f"{r['diff']:+.2f} [{r.diff_lo:+.2f}, {r.diff_hi:+.2f}] & {r.rl_on_time:.3f} & "
                     f"{r.best_heur_on} ({r.best_heur_on_time:.3f}) & {r.diff_o:+.3f} [{r.diff_o_lo:+.3f}, {r.diff_o_hi:+.3f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "ablations.tex").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    y = np.arange(len(df))[::-1]
    for ax, (k, lo, hi, lab) in zip(axes, (("diff", "diff_lo", "diff_hi", r"$\Delta$ return vs. best heuristic"),
                                           ("diff_o", "diff_o_lo", "diff_o_hi", r"$\Delta$ on-time rate vs. best heuristic"))):
        ax.axvline(0, color=INK2, lw=0.8)
        ax.errorbar(df[k], y, xerr=[df[k] - df[lo], df[hi] - df[k]], fmt="o", color=C["AA-DQN"], capsize=2.5, ms=5)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, df.label)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_ablations")
    return df


def robustness():
    f = RES / "robustness.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    agg = df.groupby(["condition", "method", "train_seed"])[["return", "on_time_rate", "mean_fatigue"]].mean().reset_index()
    out = agg.groupby(["condition", "method"]).agg(ret=("return", "mean"), ret_sd=("return", "std"),
                                                   on=("on_time_rate", "mean"), fat=("mean_fatigue", "mean")).reset_index()
    out.to_csv(SUM / "robustness.csv", index=False)
    meths = [m for m in ["AA-DQN", "AA-DQN-DR", "DQN", "EDF-Periodic", "EDF-Threshold", "Cog-Heuristic", "EDF"]
             if m in set(df.method)]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    sig = [0.0, 0.05, 0.1, 0.2, 0.3]
    ax = axes[0]
    for m in meths:
        vals, lo, hi = [], [], []
        for s_ in sig:
            sub = agg[(agg.condition == f"noise{s_}") & (agg.method == m)]
            if sub.empty:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            if m in ("AA-DQN", "AA-DQN-DR", "DQN"):
                v = sub["return"].values
                a, b = boot_ci(v)
            else:
                v = df[(df.condition == f"noise{s_}") & (df.method == m)]["return"].values
                a, b = boot_ci(v)
            vals.append(v.mean()); lo.append(a); hi.append(b)
        ax.plot(sig, vals, marker=MK[m], color=C[m], ms=4, label=m)
        ax.fill_between(sig, lo, hi, color=C[m], alpha=0.12, lw=0)
    ax.axvline(0.1, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel(r"Observation-noise s.d. $\sigma_o$ at test time")
    ax.set_ylabel("Episode return")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, [l + (" (ours)" if l == "AA-DQN" else "") for l in labels], loc="upper center",
               ncol=len(labels), fontsize=6.3, bbox_to_anchor=(0.5, 1.07), handlelength=1.5, columnspacing=1.0)
    for ax, conds, labs, title in ((axes[1], ["profile_resilient", "noise0.1", "profile_fatigue_prone"],
                                    ["Resilient", "Nominal", "Fatigue-prone"], "Worker profile"),
                                   (axes[2], ["tasks8", "noise0.1", "tasks16"], ["8 tasks", "12 tasks", "16 tasks"],
                                    "Daily workload")):
        x = np.arange(len(conds))
        width = 0.8 / len(meths)
        for j, m in enumerate(meths):
            vals = [out[(out.condition == c) & (out.method == m)].ret.values for c in conds]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.plot(x + (j - (len(meths) - 1) / 2) * width, vals, MK[m], color=C[m], ms=4.5, ls="none")
        ax.set_xticks(x, labs)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylabel("Episode return")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_robustness")

    conds = [("noise0.0", r"$\sigma_o=0$"), ("noise0.1", r"Nominal ($\sigma_o=0.1$)"), ("noise0.2", r"$\sigma_o=0.2$"),
             ("noise0.3", r"$\sigma_o=0.3$"), ("profile_resilient", "Resilient worker"),
             ("profile_fatigue_prone", "Fatigue-prone worker"), ("tasks8", "8 tasks/day"), ("tasks16", "16 tasks/day")]
    lines = [r"\begin{tabular}{l" + "c" * len(meths) + "}", r"\toprule",
             "Test condition & " + " & ".join(meths) + r" \\", r"\midrule"]
    for c, lab in conds:
        vals = [out[(out.condition == c) & (out.method == m)] for m in meths]
        nums = [v.ret.values[0] if len(v) else np.nan for v in vals]
        bi = int(np.nanargmax(nums))
        cells = []
        for i, (v, n) in enumerate(zip(vals, nums)):
            cell = f"{n:.2f} / {v.on.values[0]:.3f}"
            cells.append(r"\textbf{" + cell + "}" if i == bi else cell)
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "robustness.tex").write_text("\n".join(lines))
    return out


def pareto(recs):
    pts = []
    for lam, name, var in ((0.0, "AA-DQN@task_reward_only", "task_reward_only"), (0.05, "AA-DQN@lam0.05", "lam0.05"),
                           (0.1, "AA-DQN", "full"), (0.2, "AA-DQN@lam0.2", "lam0.2"), (0.4, "AA-DQN@lam0.4", "lam0.4")):
        rl = test_frame(recs, name)
        if rl.empty:
            continue
        rl = rl[rl.train_seed < 3]
        o, f = per_seed(rl, "on_time_rate"), per_seed(rl, "mean_fatigue")
        pts.append(dict(lam=lam, on=o.mean(), on_sd=o.std(ddof=1), fat=f.mean(), fat_sd=f.std(ddof=1),
                        hf=per_seed(rl, "high_fatigue_frac").mean(), brk=per_seed(rl, "break_frac").mean()))
    p = pd.DataFrame(pts)
    p.to_csv(SUM / "pareto.csv", index=False)
    if p.empty:
        return p
    heur = heur_frame("full").groupby("method")[["on_time_rate", "mean_fatigue"]].mean()
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.errorbar(p.fat, p.on, xerr=p.fat_sd, yerr=p.on_sd, color=C["AA-DQN"], marker="o", ms=4.5, capsize=2, lw=1.4,
                label=r"AA-DQN, $\lambda\in\{0,0.05,0.1,0.2,0.4\}$")
    for _, r in p.iterrows():
        ax.annotate(rf"$\lambda$={r.lam:g}", (r.fat, r.on), textcoords="offset points", xytext=(4, 4), fontsize=6.3,
                    color=INK2)
    for h in ["EDF", "SPT", "EDF-Periodic", "EDF-Threshold", "Cog-Heuristic"]:
        ax.plot(heur.loc[h, "mean_fatigue"], heur.loc[h, "on_time_rate"], MK[h], color=C[h], ms=5.5, label=h)
    ax.set_xlabel("Mean fatigue over the workday")
    ax.set_ylabel("On-time completion rate")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_pareto")
    lines = [r"\begin{tabular}{cccccc}", r"\toprule",
             r"$\lambda$ ($\mu=\lambda/2$) & On-time rate & Mean fatigue & High-fatigue exposure & Break share \\",
             r"\midrule"]
    for _, r in p.iterrows():
        lines.append(f"{r.lam:g} & {r.on:.3f} $\\pm$ {r.on_sd:.3f} & {r.fat:.3f} $\\pm$ {r.fat_sd:.3f} & {r.hf:.3f} & {r.brk:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "pareto.tex").write_text("\n".join(lines))
    return p


def component(recs):
    names = [("AA-DQN", "AA-DQN (hybrid actions, $k{=}4$, Double, Dueling)"),
             ("AA-DQN-noHist", "-- without history ($k{=}1$)"),
             ("AA-DQN-noDouble", "-- without Double Q-learning"), ("AA-DQN-noDueling", "-- without dueling head"),
             ("AA-DQN-3act", "-- intensity-only actions \\{easy, hard, break\\}"),
             ("AA-DQN-k2", "history $k{=}2$"), ("AA-DQN-k8", "history $k{=}8$"),
             ("AA-DQN-GRU", "GRU encoder over $k{=}8$ history"),
             ("DQN", "Vanilla DQN ($k{=}1$, no Double/Dueling)")]
    rows = []
    for n, lab in names:
        rl = test_frame(recs, n)
        if rl.empty:
            continue
        r_, o_ = per_seed(rl, "return"), per_seed(rl, "on_time_rate")
        tt = [r["train_time_s"] for r in recs if r["name"] == n]
        rows.append(dict(name=n, label=lab, seeds=len(r_), ret=r_.mean(), ret_sd=r_.std(ddof=1), on=o_.mean(),
                         on_sd=o_.std(ddof=1), time=np.mean(tt)))
    df = pd.DataFrame(rows)
    df.to_csv(SUM / "component.csv", index=False)
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Configuration & Seeds & Return & On-time rate & Train time (s) \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.label} & {r.seeds} & {r.ret:.2f} $\\pm$ {r.ret_sd:.2f} & {r.on:.3f} $\\pm$ {r.on_sd:.3f} & {r.time:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "component.tex").write_text("\n".join(lines))
    return df


def behaviour():
    f = RES / "traces.npz"
    if not f.exists():
        return None
    z = np.load(f)
    hours = 9 + np.arange(48) / 6
    meths = [("AA-DQN", "AA_DQN"), ("Cog-Heuristic", "Cog_Heuristic"), ("EDF-Threshold", "EDF_Threshold"), ("EDF", "EDF")]
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.6), sharex=True)
    for j, (m, key) in enumerate(meths):
        act = z[f"{key}__act"]
        ax = axes[0, j]
        p = np.stack([(act == a).mean(0) for a in (1, 0, 4)])
        ax.stackplot(hours, p, colors=["#184f95", "#86b6ef", "#e4e3df"], labels=["Hard task", "Easy task", "Break"],
                     edgecolor="white", linewidth=0.3)
        ax.set_title(m + (" (ours)" if m == "AA-DQN" else ""))
        ax.set_ylim(0, 1)
        ax.grid(False)
        if j == 0:
            ax.set_ylabel("Action share")
        ax2 = axes[1, j]
        A, Fm = z[f"{key}__A"], z[f"{key}__F"]
        for M, col, lab in ((A, C["AA-DQN"], "Attention $A_t$"), (Fm, C["DQN"], "Fatigue $F_t$")):
            mu = M.mean(0)
            q1, q3 = np.percentile(M, [25, 75], axis=0)
            ax2.plot(hours, mu, color=col, label=lab)
            ax2.fill_between(hours, q1, q3, color=col, alpha=0.18, lw=0)
        ax2.set_ylim(0, 1)
        ax2.set_xticks([9, 11, 13, 15, 17])
        ax.set_xlim(9, 17)
        ax2.set_xlabel("Clock time (h)")
        if j == 0:
            ax2.set_ylabel("Latent state")
    axes[0, 0].legend(loc="lower left", fontsize=6, framealpha=0.85, frameon=True)
    axes[1, 0].legend(loc="lower right", fontsize=6)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "fig_behaviour")

    # timing statistics
    rows = []
    for m, key in meths + [("DQN", "DQN")]:
        if f"{key}__act" not in z:
            continue
        act, A, Fm = z[f"{key}__act"], z[f"{key}__A"], z[f"{key}__F"]
        hard = act == 1
        morning = hours < 12
        prev_F = np.concatenate([np.full((len(Fm), 1), np.nan), Fm[:, :-1]], axis=1)
        prev_A = np.concatenate([np.full((len(A), 1), np.nan), A[:, :-1]], axis=1)
        brk = act == 4
        rows.append(dict(method=m, hard_share_morning=hard[:, morning].mean(), hard_share_afternoon=hard[:, ~morning].mean(),
                         break_share_dip=brk[:, (hours >= 13) & (hours < 15)].mean(), break_share_other=brk[:, (hours < 13) | (hours >= 15)].mean(),
                         fatigue_before_break=np.nanmean(prev_F[brk]), attention_before_break=np.nanmean(prev_A[brk]),
                         attention_at_hard=np.nanmean(prev_A[hard]), end_fatigue=Fm[:, -1].mean()))
    pd.DataFrame(rows).to_csv(SUM / "behaviour.csv", index=False)
    return pd.DataFrame(rows)


def fig_dynamics():
    """Illustrate the simulator: circadian term and single-day trajectories under two policies."""
    from aacs.env import CognitiveSchedulingEnv, circadian, make_cfg
    from aacs.baselines import EDF, EDFPeriodic, run_policy

    hours = np.linspace(9, 17, 200)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2))
    axes[0].plot(hours, circadian(hours), color=INK)
    axes[0].axhline(0, color=INK2, lw=0.5)
    axes[0].set_xlabel("Clock time (h)")
    axes[0].set_ylabel("Circadian modulation $c(h)$")
    axes[0].set_title("(a) Circadian process")
    env = CognitiveSchedulingEnv(make_cfg())
    hh = 9 + np.arange(48) / 6
    for ax, pol, title in ((axes[1], EDF(), "(b) EDF, no breaks"), (axes[2], EDFPeriodic(4), "(c) EDF with periodic breaks")):
        _, tr = run_policy(env, pol, [20_000_000], keep_traces=True)
        tr = tr[0]
        ax.plot(hh, tr["A"], color=C["AA-DQN"], label="Attention $A_t$")
        ax.plot(hh, tr["F"], color=C["DQN"], label="Fatigue $F_t$")
        for t in np.flatnonzero(tr["act"] == 2):
            ax.axvspan(hh[t] - 1 / 6, hh[t], color="#e4e3df", lw=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Clock time (h)")
        ax.set_title(title)
    axes[1].legend(loc="center left", fontsize=6.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_dynamics")


def efficiency_surface():
    from aacs.env import NOMINAL as c
    A = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for d, ls in ((0.2, "-"), (0.6, "--"), (1.0, ":")):
        for F, col in ((0.0, C["AA-DQN"]), (0.8, C["DQN"])):
            ax.plot(A, c.rho0 * A ** (0.5 + d) * (1 - c.kappa * F), color=col, ls=ls, lw=1.2)
    ax.set_xlabel("Attention $A_t$")
    ax.set_ylabel(r"Work rate $\rho_t$ (units/slot)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["AA-DQN"], label="$F_t=0$"), Line2D([], [], color=C["DQN"], label="$F_t=0.8$"),
         Line2D([], [], color=INK2, ls="-", label="$d=0.2$"), Line2D([], [], color=INK2, ls="--", label="$d=0.6$"),
         Line2D([], [], color=INK2, ls=":", label="$d=1.0$")]
    ax.legend(handles=h, fontsize=6.3, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_efficiency")


def compute_cost(recs):
    rows = []
    for n in ("DQN", "AA-DQN"):
        rr = [r for r in recs if r["name"] == n]
        if rr:
            rows.append(dict(method=n, train_time_s=np.mean([r["train_time_s"] for r in rr]),
                             train_time_sd=np.std([r["train_time_s"] for r in rr]),
                             us_per_decision=np.mean([r["eval_us_per_decision"] for r in rr])))
    pd.DataFrame(rows).to_csv(SUM / "compute.csv", index=False)


def main():
    recs = load_runs()
    print(len(recs), "runs")
    fig_dynamics()
    efficiency_surface()
    rl, heur, s, tests = main_results(recs)
    print(s[["method", "return", "return_sd", "on_time_rate", "mean_fatigue", "break_frac"]].to_string())
    for k, v in tests.items():
        print(k)
        print(v.to_string())
    fig_main(s)
    fig_tradeoff(s)
    fig_curves(recs, heur)
    fig_td(recs)
    print(ablations(recs).to_string())
    r = robustness()
    if r is not None:
        print(r.to_string())
    print(pareto(recs).to_string())
    print(component(recs).to_string())
    b = behaviour()
    if b is not None:
        print(b.to_string())
    compute_cost(recs)
    print(final_vs_best(recs).to_string())
    heuristic_params_table()
    write_numbers()


if __name__ == "__main__":
    main()


def _cam(s):
    import re
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out = "".join(p[:1].upper() + p[1:] for p in parts if p)
    for d, w in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]):
        out = out.replace(d, w)
    return out


def write_numbers():
    """Emit LaTeX macros for every number quoted in the text, so prose and tables cannot drift apart."""
    N = {}
    s = pd.read_csv(SUM / "main_summary.csv").set_index("method")
    for m in s.index:
        k = _cam(m)
        r = s.loc[m]
        N[f"{k}Return"] = f"{r['return']:.2f}"
        N[f"{k}ReturnSD"] = f"{r['return_sd']:.2f}"
        N[f"{k}OnTimePct"] = f"{100 * r.on_time_rate:.1f}"
        N[f"{k}WOnTimePct"] = f"{100 * r.weighted_on_time:.1f}"
        N[f"{k}ComplPct"] = f"{100 * r.completion_rate:.1f}"
        N[f"{k}Fatigue"] = f"{r.mean_fatigue:.3f}"
        N[f"{k}Attention"] = f"{r.mean_attention:.3f}"
        N[f"{k}BreakPct"] = f"{100 * r.break_frac:.1f}"
        N[f"{k}HighFatPct"] = f"{100 * r.high_fatigue_frac:.1f}"
    heur = s.loc[[m for m in HEUR if m in s.index]]
    best = heur["return"].idxmax()
    N["bestHeur"] = best
    for suf in ("Return", "OnTimePct", "Fatigue", "BreakPct", "Attention"):
        N[f"bestHeur{suf}"] = N[f"{_cam(best)}{suf}"]
    N["aaReturn"], N["aaReturnSD"] = N["AADQNReturn"], N["AADQNReturnSD"]
    N["aaOnTimePct"], N["aaFatigue"] = N["AADQNOnTimePct"], N["AADQNFatigue"]
    N["edfOnTimePct"], N["edfFatigue"] = N["EDFOnTimePct"], N["EDFFatigue"]
    N["fatigueRedVsEdfPct"] = f"{100 * (1 - s.loc['AA-DQN', 'mean_fatigue'] / s.loc['EDF', 'mean_fatigue']):.0f}"
    N["onTimeGainVsEdfPP"] = f"{100 * (s.loc['AA-DQN', 'on_time_rate'] - s.loc['EDF', 'on_time_rate']):.1f}"
    for metric, tag in (("return", "Ret"), ("on_time_rate", "On"), ("mean_fatigue", "Fat"), ("weighted_on_time", "WOn")):
        t = pd.read_csv(SUM / f"tests_{metric}.csv").set_index("baseline")
        for b in t.index:
            r = t.loc[b]
            sc = 100 if metric in ("on_time_rate", "weighted_on_time") else 1
            dg = 1 if metric in ("on_time_rate", "weighted_on_time") else (2 if metric == "return" else 3)
            N[f"p{tag}{_cam(b)}"] = r"p_{\mathrm{Holm}}<10^{-4}" if r.p_holm < 1e-4 else rf"p_{{\mathrm{{Holm}}}}={r.p_holm:.2g}"
            k = _cam(b)
            N[f"d{tag}{k}"] = f"{sc * r.mean_diff:.{dg}f}"
            N[f"d{tag}{k}Lo"] = f"{sc * r.diff_lo:.{dg}f}"
            N[f"d{tag}{k}Hi"] = f"{sc * r.diff_hi:.{dg}f}"
            N[f"rb{tag}{k}"] = f"{r.rank_biserial:.2f}"
            N[f"win{tag}{k}Pct"] = f"{100 * r.win_rate:.1f}"
            N[f"seeds{tag}{k}"] = f"{int(r.seeds_better)}"
    t = pd.read_csv(SUM / "tests_return_vs_DQN.csv").iloc[0]
    N["dRetVsDqn"], N["dRetVsDqnLo"], N["dRetVsDqnHi"] = f"{t.mean_diff:.2f}", f"{t.diff_lo:.2f}", f"{t.diff_hi:.2f}"
    N["winRetVsDqnPct"] = f"{100 * t.win_rate:.1f}"
    if (SUM / "ablations.csv").exists():
        a = pd.read_csv(SUM / "ablations.csv")
        for _, r in a.iterrows():
            k = _cam(r.variant)
            N[f"abl{k}Diff"], N[f"abl{k}Lo"], N[f"abl{k}Hi"] = f"{r['diff']:.2f}", f"{r.diff_lo:.2f}", f"{r.diff_hi:.2f}"
            N[f"abl{k}DiffOnPP"] = f"{100 * r.diff_o:.1f}"
            N[f"abl{k}DiffOnLo"], N[f"abl{k}DiffOnHi"] = f"{100 * r.diff_o_lo:.1f}", f"{100 * r.diff_o_hi:.1f}"
            N[f"abl{k}Ret"], N[f"abl{k}OnPct"] = f"{r.rl_return:.2f}", f"{100 * r.rl_on_time:.1f}"
            N[f"abl{k}Best"], N[f"abl{k}BestRet"] = r.best_heur, f"{r.best_heur_return:.2f}"
            N[f"abl{k}BestOnPct"] = f"{100 * r.best_heur_on_time:.1f}"
            N[f"abl{k}P"] = "<10^{-4}" if r.p < 1e-4 else f"={r.p:.3f}"
            N[f"abl{k}POn"] = "<10^{-4}" if r.p_o < 1e-4 else f"={r.p_o:.3f}"
            N[f"abl{k}Fatigue"] = f"{r.rl_fatigue:.3f}"
    if (SUM / "compute.csv").exists():
        c = pd.read_csv(SUM / "compute.csv").set_index("method")
        N["trainTimeAA"] = f"{c.loc['AA-DQN', 'train_time_s']:.0f}"
        N["trainTimeDQN"] = f"{c.loc['DQN', 'train_time_s']:.0f}"
        N["inferUs"] = f"{c.loc['AA-DQN', 'us_per_decision']:.0f}"
    fb = pd.read_csv(SUM / "final_vs_best.csv")
    for _, r in fb.iterrows():
        k = _cam(r.method) + ("Sel" if r.checkpoint.startswith("val") else "Final")
        N[f"{k}Return"], N[f"{k}OnTimePct"] = f"{r.ret:.2f}", f"{100 * r.on:.1f}"
        if r.checkpoint.startswith("val"):
            N[f"{_cam(r.method)}BestEp"] = f"{r.best_ep:.0f}"
    lines = ["% auto-generated by analysis/analyze.py -- do not edit"]
    for k, v in N.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    return N


def heuristic_params_table():
    labels = {"full": "Full model", "no_obs_noise": "No observation noise", "no_circadian": "No circadian process",
              "no_cognition": "Static capacity", "task_reward_only": "Task-only reward", "lam0.05": r"$\lambda=0.05$",
              "lam0.2": r"$\lambda=0.2$", "lam0.4": r"$\lambda=0.4$"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Variant & EDF-Periodic ($p$) & EDF-Threshold ($\tau_F,\tau_A$) & Cog-Heuristic ($\tau_F,\tau_A,\tau_H$) \\",
             r"\midrule"]
    for v, lab in labels.items():
        f = RES / "heuristics" / f"{v}_params.json"
        if not f.exists():
            continue
        P = json.loads(f.read_text())
        pp = P["EDF-Periodic"]["params"]
        pt = P["EDF-Threshold"]["params"]
        pc = P["Cog-Heuristic"]["params"]
        g = lambda x: "off" if x > 1 else f"{x:g}"
        lines.append(f"{lab} & {pp['period']} & ({g(pt['tau_f'])}, {pt['tau_a']:g}) & "
                     f"({g(pc['tau_f'])}, {pc['tau_a']:g}, {pc['tau_h']:g})" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "heur_params.tex").write_text("\n".join(lines))


def fig_tradeoff(s):
    """Count-based vs difficulty-weighted on-time completion: no single rule is good at both."""
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for _, r in s.iterrows():
        m = r.method
        if m in ("Random", "FIFO"):
            continue
        ax.errorbar(r.on_time_rate, r.weighted_on_time,
                    xerr=[[r.on_time_rate - r.on_time_rate_lo], [r.on_time_rate_hi - r.on_time_rate]],
                    yerr=[[r.weighted_on_time - r.weighted_on_time_lo], [r.weighted_on_time_hi - r.weighted_on_time]],
                    fmt=MK[m], color=C[m], ms=6 if m == "AA-DQN" else 5, capsize=1.5, lw=0.8, mec=C[m])
        off = {"AA-DQN": (0, 9), "EDF-Periodic": (-6, 7), "EDF-Threshold": (-9, -6), "SPT-Threshold": (6, 4),
               "SPT-Periodic": (6, -10), "Cog-Heuristic": (6, -3), "DQN": (-7, -3), "SPT": (7, -3), "EDF": (8, -3)}
        ha = {"AA-DQN": "center", "EDF-Periodic": "right", "EDF-Threshold": "right", "DQN": "right"}.get(m, "left")
        ax.annotate(m + (" (ours)" if m == "AA-DQN" else ""), (r.on_time_rate, r.weighted_on_time),
                    textcoords="offset points", xytext=off.get(m, (4, 3)), fontsize=6.3, color=INK, ha=ha)
    ax.set_xlabel("On-time rate (share of tasks)")
    ax.set_ylabel("Difficulty-weighted on-time rate")
    fig.tight_layout()
    save(fig, "fig_tradeoff")
