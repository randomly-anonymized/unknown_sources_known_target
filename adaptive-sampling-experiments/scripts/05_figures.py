"""All figures. Reads results/*.csv only; never re-simulates."""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, SC = f"{ROOT}/results", f"{ROOT}/figures", f"{ROOT}/data/scenarios"
# ---- all styling lives in configs/style.json; edit that file, not this one --------------------
STYLE_FILE = os.environ.get("STYLE_FILE", f"{ROOT}/configs/style.json")
CFG = json.load(open(STYLE_FILE))
plt.rcParams.update(CFG["rc"])
LEGEND_FS = CFG.get("legend_fontsize", 6.5)
ANNOT_FS = CFG.get("annotation_fontsize", 5.5)
REF = CFG.get("reference_lines", {}).get("color", "k")
STYLE = CFG["policies"]


def sty(p): return dict(STYLE.get(p, CFG["default_policy"]))


def fig_tier1():
    for tag, title in [("T1-AGE", "online vs landline, share aged 65+"),
                       ("T1-BLACK", "online vs southern clinic sites, share non-Hispanic Black")]:
        d = pd.read_csv(f"{RES}/E1_rates_{tag}.csv")
        s = pd.read_csv(f"{RES}/E1_summary_{tag}.csv").set_index("quantity")["value"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for p, g in d.groupby("policy"):
            ax[0].loglog(g["T"], g.err_scalar_mean, label=p, **sty(p))
        T = np.array(sorted(d["T"].unique()))
        ax[0].loglog(T, s["kappa_lb"] / T, color=REF, ls=":", lw=1, label="lower bound $\\kappa/T$")
        ax[0].loglog(T, s["thm1_ES_bound"] / T, color=REF, ls="--", lw=1, label="Thm 1 bound")
        ax[0].loglog(T, s["open_loop_const"] / np.sqrt(T), color="#28b463", ls=":", lw=1,
                     label="open-loop $\\sqrt{2\\nu^*/\\pi T}$")
        ax[0].set_xlabel("recruits $T$"); ax[0].set_ylabel("$|\\hat p_T - p_G|$")
        ax[0].set_title(f"{tag}: {title}", fontsize=9)
        ax[0].legend(fontsize=6.5, ncol=2)
        for p, g in d.groupby("policy"):
            ax[1].semilogx(g["T"], g.S_scalar_mean, label=p, **sty(p))
        ax[1].axhline(s["kappa_lb"], color=REF, ls=":", lw=1)
        ax[1].axhline(s["thm1_ES_bound"], color=REF, ls="--", lw=1)
        ax[1].set_yscale("log"); ax[1].set_xlabel("recruits $T$"); ax[1].set_ylabel("$E|S_T|$")
        ax[1].set_title("imbalance $|S_T| = T\\,|\\hat p_T-p_G|$", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{FIG}/F1_{tag}_rates.png"); plt.close(fig)


def fig_tails():
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, tag in zip(axs, ["T1-AGE", "T1-BLACK"]):
        d = pd.read_csv(f"{RES}/E1_tails_{tag}.csv")
        for T, g in d.groupby("T"):
            ax.semilogy(g.k, np.maximum(g.empirical, 1e-4), marker="o", ms=2.5, lw=1, label=f"T={T:,}")
        g = d[d["T"] == d["T"].max()]
        ax.semilogy(g.k, g.bound, ls="--", color=REF, lw=1.2, label="Thm 1 bound")
        ax.set_xlabel("$k$"); ax.set_ylabel("$P(|S_T| \\geq k)$"); ax.set_title(tag, fontsize=9)
        ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout(); fig.savefig(f"{FIG}/F2_tails.png"); plt.close(fig)


def fig_tier2():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/E2_rates_eth_*.csv")])
    ml = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/E2_rates_ml_*.csv")])
    diag = pd.read_csv(f"{RES}/E2_diagnostics.csv").set_index(["scenario", "quantity"])["value"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, dd, name, sid in [(axs[0], d, "categorical: Hispanic / Black / other", "S-DIAB__L2_K3_eth"),
                              (axs[1], ml, "marginals: Black, Hispanic, 65+, female", "S-DIAB__ML_K4_marginals")]:
        for p, g in dd.groupby("policy"):
            g = g.sort_values("T")
            ax.loglog(g["T"], g.err_l2_mean, label=p, **sty(p))
        T = np.array(sorted(dd["T"].unique()))
        ax.loglog(T, diag[(sid, "kappa_lb")] / T, ls=":", color=REF, lw=1, label="lower bound $\\kappa/T$")
        ax.set_xlabel("recruits $T$"); ax.set_ylabel("$\\|\\hat p_T - p_G\\|_2$")
        ax.set_title(name, fontsize=9); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/F3_tier2_rates.png"); plt.close(fig)


def fig_ladder():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/E2_ladder_*.csv")])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    lad = d.sort_values("margin_c", ascending=False)
    pal = CFG.get("ladder_palette", [None] * 9)
    for i, (lid, g) in enumerate(lad.groupby("ladder", sort=False)):
        gg = g[g.policy == "ALG2(beta=0.3)"].sort_values("T")
        axs[0].loglog(gg["T"], gg.err_l2_mean, lw=1.5, color=pal[i % len(pal)],
                      label=f"{lid} (K={int(gg.K.iloc[0])}, c={gg.margin_c.iloc[0]:.3f})")
    axs[0].set_xlabel("recruits $T$"); axs[0].set_ylabel("$\\|\\hat p_T-p_G\\|_2$")
    axs[0].set_title("Algorithm 2 as the stratification is refined", fontsize=9)
    axs[0].legend(fontsize=6)
    piv = d[d.policy.isin(["ALG2(beta=0.3)", "ORACLE-CL", "N2-UNIFORM"])]
    for p, g in piv.groupby("policy"):
        gg = g.groupby("margin_c").apply(lambda x: x[x["T"] == x["T"].max()].err_l2_mean.iloc[0] *
                                         x["T"].max(), include_groups=False)
        axs[1].loglog(gg.index, gg.values, "o-", ms=3, label=p, **{k: v for k, v in sty(p).items() if k != "lw"})
    c = np.array(sorted(d.margin_c.unique()))
    axs[1].loglog(c, 0.09 / c, ls=":", color=REF, lw=1, label="$0.09/c$")
    axs[1].set_xlabel("hull margin $c$"); axs[1].set_ylabel("$\\|S_T\\|$ at $T=10^5$")
    axs[1].set_title("imbalance vs margin", fontsize=9); axs[1].legend(fontsize=LEGEND_FS)
    fig.tight_layout(); fig.savefig(f"{FIG}/F4_ladder.png"); plt.close(fig)


def fig_cost():
    e2e = pd.read_csv(f"{RES}/E3_endtoend.csv")
    ref = pd.read_csv(f"{RES}/E3_cost_reference.csv")
    last = e2e[e2e["T"] == e2e["T"].max()]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for _, r in last.iterrows():
        axs[0].scatter(r.cost_per_sample, r.err_l2_mean, s=26,
                       color=sty(r.policy).get("color", "#444"))
        axs[0].annotate(r.policy, (r.cost_per_sample, r.err_l2_mean), fontsize=ANNOT_FS,
                        xytext=(3, 3), textcoords="offset points")
    axs[0].set_yscale("log"); axs[0].set_xlabel("cost per recruit (\\$)")
    axs[0].set_ylabel("$\\|\\hat p_T-p_G\\|_2$ at $T=2\\times10^4$")
    axs[0].set_title("cost / representation frontier", fontsize=9)
    axs[1].plot(ref.r, ref.cost_adaptive, "o-", color="#1a1a1a", label="adaptive (no discard)")
    axs[1].plot(ref.r, ref.cost_quota_cheapest, "s--", color="#e67e22", label="quota on cheapest channel")
    axs[1].plot(ref.r, ref.cost_lp_discard, "^:", color="#2e86c1", label="cost-optimal mix + discard")
    axs[1].set_xlabel("offline / online cost ratio $r$"); axs[1].set_ylabel("\\$ per retained recruit")
    axs[1].set_title("when is adaptive cheaper?", fontsize=9); axs[1].legend(fontsize=LEGEND_FS)
    fig.tight_layout(); fig.savefig(f"{FIG}/F5_cost.png"); plt.close(fig)


def fig_support():
    d = pd.read_csv(f"{RES}/E3_support_selection.csv")
    base = d[d.cost_draw == 0]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for c0, g in base.groupby("c0"):
        axs[0].semilogx(g.n, g.recovery, "o-", ms=3, label=f"$c_0$={c0}")
        axs[1].semilogx(g.n, g.truly_infeasible, "o-", ms=3, label=f"$c_0$={c0}")
        axs[2].semilogx(g.n, g.margin_ok, "o-", ms=3, label=f"$c_0$={c0}")
    for ax, t in zip(axs, ["P(exact cheapest admissible support)",
                           "P(selected support infeasible under truth)",
                           "P($c(\\hat I) \\geq c_0/2$)  [Thm 3]"]):
        ax.set_xlabel("exploration samples per channel $n$"); ax.set_title(t, fontsize=8)
        ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout(); fig.savefig(f"{FIG}/F6_support_selection.png"); plt.close(fig)
    t = pd.read_csv(f"{RES}/E3_c0_tradeoff.csv")
    t = t[t["T"] == t["T"].max()]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for n, g in t.groupby("n"):
        ax.plot(g.cost_per_sample, g.err_l2, "o-", ms=4, label=f"n={n}")
        for _, r in g.iterrows():
            ax.annotate(f"$c_0$={r.c0}", (r.cost_per_sample, r.err_l2), fontsize=ANNOT_FS,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_yscale("log"); ax.set_xlabel("cost per recruit (\\$)"); ax.set_ylabel("$\\|\\hat p_T-p_G\\|_2$")
    ax.set_title("the $c_0$ trade-off ($T=2\\times10^4$)", fontsize=9); ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout(); fig.savefig(f"{FIG}/F7_c0_tradeoff.png"); plt.close(fig)


def fig_robust():
    d = pd.read_csv(f"{RES}/E1_robustness.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for v, g in d[d.variant.str.startswith(("a_", "reference"))].groupby("variant"):
        axs[0].loglog(g["T"], g.err_linf_mean, lw=1.5, label=v)
    axs[0].set_title("(a) ordering assumption violated", fontsize=8); axs[0].legend(fontsize=LEGEND_FS)
    for v, g in d[d.variant.str.startswith("b_")].groupby("variant"):
        axs[1].loglog(g["T"], g.err_linf_mean, lw=1.5, label=v.replace("b_gap_scale_", "gap x"))
    axs[1].set_title("(b) shrinking channel separation", fontsize=8); axs[1].legend(fontsize=LEGEND_FS)
    sub = d[d.variant.isin(["reference", "d_batch_10", "d_batch_50", "e_delay_10", "e_delay_50"])]
    for v, g in sub.groupby("variant"):
        axs[2].loglog(g["T"], g.err_linf_mean, lw=1.4, label=v)
    axs[2].set_title("(d,e) batched decisions / delayed labels", fontsize=8); axs[2].legend(fontsize=LEGEND_FS)
    for ax in axs:
        ax.set_xlabel("recruits $T$"); ax.set_ylabel("$|\\hat p_T - p_G|$")
    fig.tight_layout(); fig.savefig(f"{FIG}/F8_robustness.png"); plt.close(fig)


def fig_compositions():
    sc = json.load(open(f"{SC}/S-DIAB__L2_K3_eth.json"))
    mu, pG, names = np.array(sc["mu"]), np.array(sc["p_G"]), sc["arms"]
    order = np.argsort(-(mu[:, 0] + mu[:, 1]))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for k, lab in enumerate(sc["labels"]):
        ax.bar(x, mu[order, k], bottom=bottom, label=lab, width=.75,
               color=CFG["strata_palette"][k % len(CFG["strata_palette"])])
        bottom += mu[order, k]
    for k, cum in enumerate(np.cumsum(pG)[:-1]):
        ax.axhline(cum, color=REF, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=LEGEND_FS + 0.5)
    ax.set_ylabel("share of the diabetes population reached")
    ax.set_title("NHIS 2023: ethnic composition of each recruitment channel (dashed = target)", fontsize=9)
    ax.legend(fontsize=LEGEND_FS + 1, ncol=3, loc="upper right"); ax.set_ylim(0, 1.22)
    fig.tight_layout(); fig.savefig(f"{FIG}/F0_channel_compositions.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for f in [fig_compositions, fig_tier1, fig_tails, fig_tier2, fig_ladder, fig_cost, fig_support, fig_robust]:
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, type(e).__name__, e)
