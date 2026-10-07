"""Manuscript figures (vector PDF) for the Experiments section and its appendix.
Reads results/*.csv only; never re-simulates. Styling lives in configs/style-paper.json.

  figures/paper/fig_main.pdf         two sources | many sources | error at a fixed budget   (main text)
  figures/paper/figA_twosource.pdf   both two-source instances, error and E|S_T|
  figures/paper/figA_tails.pdf       tail of |S_T| vs the Theorem-1 bound
  figures/paper/figA_robust.pdf      Algorithm 1 under violated assumptions and operational constraints
  figures/paper/figA_multidim.pdf    multi-label target, Black x age target, margin ladder over T
  figures/paper/figA_cost.pdf        support selection and c0 trade-off
  figures/paper/figA_ladder.pdf      imbalance against the hull margin (six targets)
  figures/paper/figA_budget.pdf      error at a fixed budget, all variants, two cost ratios
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results"
OUT = f"{ROOT}/figures/paper"
CFG = json.load(open(os.environ.get("STYLE_FILE", f"{ROOT}/configs/style-paper.json")))
plt.rcParams.update(CFG["rc"])
REF = CFG["reference_lines"]
SEQ = CFG["sequence_palette"]
FULL_W = 6.75                                  # AISTATS text width (in)

LADDER_LABEL = {"L1_K2_black": "Black / other", "L2_K3_eth": "diagnosed diabetes",
                "L3_K3_age": "age", "L5_K4_agesex": "age $\\times$ sex",
                "L4_K4_blackage": "Black $\\times$ age", "L6_K6_ethage": "ethnicity $\\times$ age"}


def sty(p, **over):
    s = dict(CFG["policies"].get(p, CFG["default_policy"]))
    for k in ("marker", "ms", "label"):
        s.pop(k, None)
    s.update(over)
    return s


def label(p):
    return CFG["policies"].get(p, {}).get("label", p)


def plot_rates(ax, d, policies, ycol, etc_at=None):
    """Error curves over T. ETC is tuned to one horizon, so it is drawn only at that horizon."""
    for p in policies:
        g = d[d.policy == p].sort_values("T")
        if g.empty:
            continue
        if p.startswith("ETC"):
            r = g.iloc[(g["T"] - etc_at).abs().argmin()]
            s = CFG["policies"][p]
            ax.plot([r["T"]], [r[ycol]], ls="none", marker=s["marker"], ms=s["ms"], color=s["color"],
                    label=label(p), zorder=5)
        else:
            ax.loglog(g["T"], g[ycol], label=label(p), **sty(p))


def load_rates(files):
    return pd.concat([pd.read_csv(f"{RES}/{f}") for f in files], ignore_index=True)


def summary(tag):
    return pd.read_csv(f"{RES}/E1_summary_{tag}.csv").set_index("quantity")["value"]


def two_source_bounds(ax, T, s):
    q = s["kappa_lb"]                          # q = min_a min(p_a, 1 - p_a)  (Proposition 1)
    ax.loglog(T, q / (2 * T), color=REF["color"], lw=REF["lw"], ls=":", label="Prop. 1 lower bound")
    ax.loglog(T, s["thm1_ES_bound"] / T, color=REF["color"], lw=REF["lw"], ls="--",
              label="Thm. 1 (proof constants)")


# ------------------------------------------------------------------ main-text figure
def fig_main():
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 1.72))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.21, top=0.70, wspace=0.34)
    # (a) two sources
    d = pd.read_csv(f"{RES}/E1_rates_T1-AGE.csv"); s = summary("T1-AGE")
    plot_rates(ax[0], d, ["ALG1", "ORACLE-OL", "ETC(eps=0.1)", "N1-CHEAPEST", "N2-UNIFORM"],
               "err_scalar_mean", etc_at=1e5)
    T = np.array(sorted(d["T"].unique()), float)
    two_source_bounds(ax[0], T, s)
    ax[0].set_ylim(5e-7, 3)
    ax[0].set_ylabel("$|\\widehat p_T-p_G|$", labelpad=1)
    ax[0].set_title("(a) 2 channels (online, landline); share 65+", pad=3)
    # (b) sixteen channels, diagnosed diabetes target (ethnic composition)
    d = load_rates(["E2_rates_eth_ucb.csv", "E2_rates_eth_ref.csv"])
    plot_rates(ax[1], d, ["ALG2(beta=1)", "ALG2(beta=0.3)", "ALG2(greedy)", "ORACLE-CL", "ORACLE-OL",
                          "ETC(eps=0.1)", "N1-CHEAPEST", "N2-UNIFORM"], "err_l2_mean", etc_at=1e5)
    ax[1].set_ylim(5e-6, 0.6)
    ax[1].set_ylabel("$\\|\\widehat p_T-p_G\\|_2$", labelpad=1)
    ax[1].set_title("(b) 16 channels; diagnosed diabetes ($K=3$)", pad=3)
    for a in ax[:2]:
        a.set_xlabel("recruits $T$", labelpad=1)
        a.set_xlim(8, 1.3e5)
    # one legend for (a) and (b): the thick black line is Algorithm 1 in (a) and Algorithm 2 in (b)
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    hl = dict(zip(l1, h1)); hl.update({k: v for k, v in zip(l0, h0) if k not in hl})
    ours = "Alg. 1 in (a), Alg. 2 in (b)"
    hl[ours] = hl.pop(label("ALG1")); hl.pop(label("ALG2(beta=1)"))
    keys = [ours, label("ALG2(beta=0.3)"), label("ALG2(greedy)"), label("ORACLE-CL"), label("ORACLE-OL"),
            label("ETC(eps=0.1)"), label("N1-CHEAPEST"), label("N2-UNIFORM"), "Prop. 1 lower bound",
            "Thm. 1 (proof constants)"]
    fig.legend([hl[k] for k in keys], keys, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4, handlelength=2.4, fontsize=5.9)
    # (c) cost-aware sampling: cost per recruit against the sample size (Black/other target, median costs)
    plot_cost_per_recruit(ax[2], ["ALG3+2(n=1600,c0=0.02)", "MYOPIC-COST", "ALG2(beta=0.3)", "ORACLE-CL(I*)"],
                          labels={"ALG3+2(n=1600,c0=0.02)": "Alg. 3 $\\to$ 2", "MYOPIC-COST": "greedy rule",
                                  "ALG2(beta=0.3)": "Alg. 2, all", "ORACLE-CL(I*)": "oracle on $I^\\star$"})
    ax[2].set_title("(c) 16 channels; Black / other, cost", pad=3)
    ax[2].legend(loc="lower left", fontsize=5.4, bbox_to_anchor=(0.0, 0.11), ncol=2, columnspacing=0.8,
                 handlelength=1.8)
    fig.savefig(f"{OUT}/fig_main.pdf", bbox_inches=None); plt.close(fig)


BUDGET_STYLE = {
    "ALG3+2(n=1600,c0=0.02)": dict(label="Alg. 3 $\\to$ Alg. 2 ($n=1600$)", color="#000000", lw=1.7),
    "ALG3+2(n=800,c0=0.02)": dict(label="Alg. 3 $\\to$ Alg. 2 ($n=800$)", color="#555555", lw=1.0, ls="--"),
    "ALG3+2(n=400,c0=0.02)": dict(label="Alg. 3 $\\to$ Alg. 2 ($n=400$)", color="#999999", lw=1.0, ls=":"),
    "MYOPIC-COST": dict(label="greedy rule", color="#D55E00", lw=1.2, marker="s", ms=2.0),
    "ALG2(beta=0.3)": dict(label="Alg. 2, all channels", color="#555555", lw=1.0, ls="-."),
    "ORACLE-CL(I*)": dict(label="oracle closed loop on $I^\\star$", color="#0072B2", lw=1.0),
}
B_ISTAR = 91.27                                 # cost per recruit of I* (Black/other, c0 = 0.02)


def _style(p, labels):
    st = dict(BUDGET_STYLE[p]); lab = st.pop("label")
    if "marker" in st:
        st["markevery"] = 3
    return st, (labels or {}).get(p, lab)


def plot_cost_per_recruit(ax, policies, labels=None):
    d = pd.read_csv(f"{RES}/E3_endtoend.csv")
    for p in policies:
        g = d[(d.policy == p) & (d["T"] >= 1000)].sort_values("T")
        st, lab = _style(p, labels)
        ax.semilogx(g["T"], g.cost_per_sample, label=lab, **st)
    ax.axhline(B_ISTAR, color=REF["color"], lw=REF["lw"], ls=":")
    ax.annotate("$B(I^\\star)$", (3.2e5, B_ISTAR), textcoords="offset points", xytext=(-2, -7), ha="right",
                fontsize=5.4)
    ax.set_xlabel("recruits $T$", labelpad=1)
    ax.set_ylabel("\\$ per recruit", labelpad=1)
    ax.set_xlim(1e3, 3.2e5); ax.set_ylim(80, 210)


def plot_error_T(ax, policies, labels=None):
    d = pd.read_csv(f"{RES}/E3_endtoend.csv")
    for p in policies:
        g = d[(d.policy == p) & (d["T"] >= 1000)].sort_values("T")
        st, lab = _style(p, labels)
        ax.loglog(g["T"], g.err_l2_mean, label=lab, **st)
    ax.set_xlabel("recruits $T$", labelpad=1)
    ax.set_ylabel("$\\|\\widehat p_T-p_G\\|_2$", labelpad=1)
    ax.set_xlim(1e3, 3.2e5)


def plot_budget(ax, policies, labels=None):
    d = pd.read_csv(f"{RES}/E3_budget.csv")
    for p in policies:
        g = d[d.policy == p].sort_values("budget")
        st, lab = _style(p, labels)
        st.pop("markevery", None)
        ax.loglog(g.budget, g.err_mean, label=lab, **st)
    ax.set_xlabel("total budget (\\$)", labelpad=1)
    ax.set_ylabel("$\\|\\widehat p_T-p_G\\|_2$", labelpad=1)
    ax.set_xlim(0.95e6, 1.7e7)
    ticks = [1e6, 2e6, 4e6, 8e6, 16e6]
    ax.set_xticks(ticks); ax.set_xticklabels(["1M", "2M", "4M", "8M", "16M"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_ladder():
    """Imbalance at T = 1e5 against the hull margin (formerly Figure 1(c))."""
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.1))
    lad = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/E2_ladder_*.csv")])
    last = lad[lad["T"] == lad["T"].max()]
    for p in ["ALG2(beta=0.3)", "ORACLE-CL"]:
        g = last[last.policy == p].sort_values("margin_c")
        ax.loglog(g.margin_c, g.S_norm_mean, marker="o", ms=2.8, label=label(p), **sty(p))
    g = last[last.policy == "ORACLE-CL"].sort_values("margin_c")
    offset = {"L5_K4_agesex": (-8, 3)}
    for _, r in g.iterrows():
        ax.annotate(f"$K$={int(r.K)}", (r.margin_c, r.S_norm_mean), textcoords="offset points",
                    xytext=offset.get(r.ladder, (0, -8)), ha="center", fontsize=5.2, color="#0072B2")
    c = np.logspace(np.log10(1.0e-3), np.log10(0.16), 50)
    ax.loglog(c, 0.15 / c, color=REF["color"], lw=REF["lw"], ls=":", label="$0.15/c$")
    ax.set_xlabel("hull margin $c$ of the target"); ax.set_ylabel("$\\|S_T\\|_2$ at $T=10^5$")
    ax.set_xlim(9e-4, 0.25); ax.set_ylim(0.25, 800)
    ax.legend(loc="upper right", fontsize=5.6)
    fig.tight_layout(); fig.savefig(f"{OUT}/figA_ladder.pdf"); plt.close(fig)


def fig_budget():
    """Black/other target: cost per recruit and error against T, and error at a fixed budget."""
    pols = ["ALG3+2(n=1600,c0=0.02)", "ALG3+2(n=800,c0=0.02)", "ALG3+2(n=400,c0=0.02)",
            "MYOPIC-COST", "ALG2(beta=0.3)", "ORACLE-CL(I*)"]
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.2))
    plot_cost_per_recruit(ax[0], pols); ax[0].set_title("(a) cost per recruit")
    plot_error_T(ax[1], pols); ax[1].set_title("(b) error against the sample size")
    plot_budget(ax[2], pols); ax[2].set_title("(c) error at a fixed budget")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=5.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6); fig.savefig(f"{OUT}/figA_budget.pdf"); plt.close(fig)


# ------------------------------------------------------------------ appendix figures
def fig_twosource():
    fig, ax = plt.subplots(2, 2, figsize=(FULL_W, 4.4))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.86, hspace=0.42, wspace=0.18)
    titles = {"T1-AGE": "online vs landline, share aged 65+",
              "T1-BLACK": "online vs southern clinics, share non-Hispanic Black"}
    pols = ["ALG1", "ORACLE-OL", "ORACLE-OL-iid", "ETC(eps=0.1)", "N1-CHEAPEST", "N1-EXPENSIVE", "N2-UNIFORM"]
    for i, tag in enumerate(["T1-AGE", "T1-BLACK"]):
        d = pd.read_csv(f"{RES}/E1_rates_{tag}.csv"); s = summary(tag)
        T = np.array(sorted(d["T"].unique()), float)
        plot_rates(ax[i, 0], d, pols, "err_scalar_mean", etc_at=1e5)
        two_source_bounds(ax[i, 0], T, s)
        ax[i, 0].loglog(T, s["open_loop_const"] / np.sqrt(T), color="#009E73", lw=0.6, ls="-", alpha=0.5,
                        label="$\\sqrt{2\\nu^\\star/(\\pi T)}$")
        ax[i, 0].set_ylabel("$|\\widehat p_T-p_G|$"); ax[i, 0].set_title(titles[tag])
        plot_rates(ax[i, 1], d, pols, "S_scalar_mean", etc_at=1e5)
        ax[i, 1].axhline(s["kappa_lb"] / 2, color=REF["color"], lw=REF["lw"], ls=":")
        ax[i, 1].axhline(s["thm1_ES_bound"], color=REF["color"], lw=REF["lw"], ls="--")
        ax[i, 1].set_ylabel("$E|S_T|$"); ax[i, 1].set_title(titles[tag])
        for a in ax[i]:
            a.set_xlabel("recruits $T$"); a.set_xlim(8, 1.3e5)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    fig.savefig(f"{OUT}/figA_twosource.pdf", bbox_inches=None); plt.close(fig)


def fig_tails():
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.0))
    for a, tag, title in zip(ax, ["T1-AGE", "T1-BLACK"], ["online vs landline (65+)", "online vs southern clinics (Black)"]):
        d = pd.read_csv(f"{RES}/E1_tails_{tag}.csv")
        for j, (T, g) in enumerate(d.groupby("T")):
            g = g[g.empirical > 0]
            a.semilogy(g.k, g.empirical, marker="o", ms=2, lw=0.9, color=SEQ[j + 1], label=f"$T={T:,}$")
        g = d[d["T"] == d["T"].max()]
        a.semilogy(g.k, g.bound, color=REF["color"], lw=1.0, ls="--", label="Theorem 1 bound")
        a.set_xlabel("$k$"); a.set_ylabel("$P(|S_T|\\geq k)$"); a.set_title(title)
        a.set_ylim(2e-4, 1.5)
    ax[0].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/figA_tails.pdf"); plt.close(fig)


def fig_robust():
    d = pd.read_csv(f"{RES}/E1_robustness.csv")
    ref = d[d.variant == "reference"].sort_values("T")
    fig, ax = plt.subplots(1, 4, figsize=(FULL_W, 1.8))
    ax[0].loglog(ref["T"], ref.err_linf_mean, color="#000000", lw=1.3, label="reference")
    g = d[d.variant == "a_ordering_violated"].sort_values("T")
    ax[0].loglog(g["T"], g.err_linf_mean, color="#D55E00", label="both channels below $p_G$")
    ax[0].set_title("(a) ordering violated")
    for j, sc in enumerate(["1.0", "0.5", "0.25", "0.125"]):
        g = d[d.variant == f"b_gap_scale_{sc}"].sort_values("T")
        ax[1].loglog(g["T"], g.err_linf_mean, color=SEQ[j], label=f"separation $\\times{sc}$")
    ax[1].set_title("(b) shrinking channel separation")
    ax[2].loglog(ref["T"], ref.err_linf_mean, color="#000000", lw=1.3, label="reference")
    for j, (v, lab) in enumerate([("d_batch_10", "batches of 10"), ("d_batch_50", "batches of 50"),
                                  ("e_delay_10", "labels delayed 10"), ("e_delay_50", "labels delayed 50")]):
        g = d[d.variant == v].sort_values("T")
        ax[2].loglog(g["T"], g.err_linf_mean, color=SEQ[j + 1], ls="-" if "batch" in v else "--", label=lab)
    ax[2].set_title("(c) batched decisions, delayed labels")
    g = d[d.variant == "c_drift"]
    for p in ["ALG1", "ORACLE-OL", "ETC(eps=0.1)"]:
        h = g[g.policy == p].sort_values("T")
        if p.startswith("ETC"):
            r = h.iloc[-1]; s = CFG["policies"][p]
            ax[3].plot([r["T"]], [r.err_linf_mean], ls="none", marker=s["marker"], ms=s["ms"], color=s["color"],
                       label="explore-then-commit")
        else:
            ax[3].loglog(h["T"], h.err_linf_mean, label=label(p), **sty(p))
    ax[3].set_title("(d) $\\pm 5$ pp drift in both channels")
    for a in ax:
        a.set_xlabel("recruits $T$"); a.legend(loc="lower left", fontsize=5.2)
        a.set_ylim(2e-5, 0.3)
    ax[0].set_ylabel("$|\\widehat p_T-p_G|$")
    fig.tight_layout(w_pad=0.5); fig.savefig(f"{OUT}/figA_robust.pdf"); plt.close(fig)


def fig_multidim():
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.15, top=0.74, wspace=0.22)
    d = load_rates(["E2_rates_ml_ucb.csv", "E2_rates_ml_ref.csv"])
    plot_rates(ax[0], d, ["ALG2(beta=1)", "ALG2(beta=0.3)", "ALG2(beta=0.1)", "ALG2(greedy)", "ORACLE-CL",
                          "ORACLE-OL", "ETC(eps=0.1)", "N1-CHEAPEST", "N2-UNIFORM", "N3-POOLPROP"],
               "err_l2_mean", etc_at=1e5)
    ax[0].set_title("(a) marginal goals: Black, Hispanic, 65+, female")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    d = pd.read_csv(f"{RES}/E2_rates_blackage.csv")
    plot_rates(ax[1], d, ["ALG2(beta=0.3)", "ORACLE-CL", "N1-CHEAPEST", "N2-UNIFORM"], "err_l2_mean")
    ax[1].set_title("(b) Black $\\times$ age ($K=4$, $c=0.0086$)")
    ax[1].legend(loc="lower left")
    lad = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/E2_ladder_*.csv")])
    order = lad.drop_duplicates("ladder").sort_values("margin_c", ascending=False)
    for j, (_, r) in enumerate(order.iterrows()):
        for p, ls, lw in [("ALG2(beta=0.3)", "-", 1.0), ("ORACLE-CL", ":", 0.9)]:
            g = lad[(lad.ladder == r.ladder) & (lad.policy == p)].sort_values("T")
            ax[2].loglog(g["T"], g.err_l2_mean, color=SEQ[j], ls=ls, lw=lw,
                         label=f"{LADDER_LABEL[r.ladder]}, $c$={r.margin_c:.4f}" if p.startswith("ALG2") else None)
    ax[2].set_title("(c) Alg. 2 ($0.3\\,r_a$, solid) and oracle (dotted)")
    ax[2].legend(loc="lower left", fontsize=5.0)
    for a in ax:
        a.set_xlabel("recruits $T$"); a.set_xlim(8, 1.3e5)
    ax[0].set_ylabel("$\\|\\widehat p_T-p_G\\|_2$")
    fig.savefig(f"{OUT}/figA_multidim.pdf", bbox_inches=None); plt.close(fig)


def fig_cost():
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 1.95))
    sel = pd.read_csv(f"{RES}/E3_support_selection.csv")
    base = sel[sel.cost_draw == 0]
    for j, (c0, g) in enumerate(base.groupby("c0")):
        g = g.sort_values("n")
        ax[0].semilogx(g.n, g.truly_infeasible, marker="o", ms=2.2, color=SEQ[j + 1], label=f"$c_0={c0:g}$")
        ax[1].semilogx(g.n, g.margin_ok, marker="o", ms=2.2, color=SEQ[j + 1], label=f"$c_0={c0:g}$")
    ax[0].set_title("(a) $P(p_G\\notin$ true hull of $\\widehat I)$")
    ax[1].set_title("(b) $P(c(\\widehat I)\\geq c_0/2)$")
    for a in ax[:2]:
        a.set_xlabel("exploration samples per channel $n$"); a.set_ylim(-0.03, 1.03); a.legend(loc="center right")
    tr = pd.read_csv(f"{RES}/E3_c0_tradeoff.csv")
    tr = tr[tr["T"] == tr["T"].max()]
    for j, (n, g) in enumerate(tr.groupby("n")):
        g = g.sort_values("c0")
        ax[2].semilogy(g.cost_per_sample, g.err_l2, marker="o", ms=2.2, color=SEQ[j + 1], label=f"$n={n}$")
        if n == 1600:                             # c0 increases left to right on every curve
            for _, r in g.iterrows():
                ax[2].annotate(f"$c_0$={r.c0:g}", (r.cost_per_sample, r.err_l2), textcoords="offset points",
                               xytext={0.01: (-2, 6), 0.02: (3, -7), 0.05: (0, -7)}.get(r.c0, (-2, -7)),
                               ha={0.01: "right", 0.02: "left", 0.05: "center"}.get(r.c0, "right"), va="center",
                               fontsize=4.8, color=SEQ[j + 1])
    ax[2].set_xlabel("\\$ per recruit"); ax[2].set_ylabel("$\\|\\widehat p_T-p_G\\|_2$ at $T=10^5$")
    ax[2].set_title("(c) margin threshold $c_0$")
    ax[2].set_xlim(92, 205)
    ax[2].legend(loc="upper right")
    fig.tight_layout(w_pad=0.5); fig.savefig(f"{OUT}/figA_cost.pdf"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in [fig_main, fig_twosource, fig_tails, fig_robust, fig_multidim, fig_cost, fig_ladder, fig_budget]:
        f(); print("wrote", f.__name__)
