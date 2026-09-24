"""Tier 3 (sec. 6): cost-aware sampling. Stages: e31 (support selection), e32 (end-to-end),
e33 (c0 trade-off), e34 (myopic-rule counterexample, Example 2), refs (cost references / break-even table)."""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.env import categorical_env
from src.geometry import enumerate_supports, support_table, feasible_allocation
from src.policies import (Alg2UCB, OracleCL, Alg3ThenAlg2, MyopicCost, FixedArm, RoundRobin,
                          OpenLoopHerding, Alg1Sign, simplex_heights_batch)
from src.runner import run, log_checkpoints, lp_discard_cost
from src.theory import thm3_cost_bound, thm3_n

RES, SC = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "scenarios")
SID = "S-DIAB__L2_K3_eth"
N_GRID = [25, 50, 100, 200, 400, 800, 1600]
C0_GRID = [0.01, 0.02, 0.05, 0.10]
R_SEL, T_E2E, R_E2E = 250, 20_000, 200
COST_RANGE = {72.0: (3.9, 251.2), 199.0: (19.1, 839.0)}     # JMIR 2020 meta-analysis ranges


def scenario():
    sc = json.load(open(os.path.join(SC, SID + ".json")))
    env = categorical_env(np.array(sc['mu']), np.array(sc['p_G']), np.array(sc['cost']),
                          names=sc['arms'], label_names=sc['labels'])
    env.c_margin = sc['margin_c']
    return sc, env


def geometry_batch(muhat, p_G, supports):
    """Per support: barycentric weights and margin for every replication. Cost-independent."""
    R, m, K = muhat.shape
    Q = np.zeros((R, len(supports), K)); M = np.full((R, len(supports)), -np.inf)
    for s, I in enumerate(supports):
        V = muhat[:, list(I), :]
        A = np.swapaxes(V, 1, 2)
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-10
        q = np.zeros((R, K))
        if ok.any():
            q[ok] = np.linalg.solve(A[ok], np.broadcast_to(p_G, (int(ok.sum()), K))[..., None])[..., 0]
        res = np.linalg.norm(np.einsum('rkj,rj->rk', A, q) - p_G, axis=1)
        h = simplex_heights_batch(V)
        marg = np.min(q * h, axis=1)
        good = ok & (res < 1e-8) & (q.min(axis=1) >= 0)
        Q[:, s, :] = q
        M[:, s] = np.where(good, marg, -np.inf)
    return Q, M


def stage_e31():
    sc, env = scenario()
    mu, p_G, cost, names = env.mu, env.p_G, env.cost, env.names
    m, K = mu.shape
    supports = enumerate_supports(m, K)
    truth = support_table(mu, p_G, cost, supports)
    true_margin = np.array([c for _, c, _, _ in truth])
    true_q = [q for _, _, q, _ in truth]
    rng = np.random.default_rng(5)
    cost_draws = [cost] + [np.exp([rng.uniform(*np.log(COST_RANGE[c])) for c in cost]) for _ in range(9)]
    rows = []
    for n in N_GRID:
        muhat = np.stack([rng.multinomial(n, mu[a], size=R_SEL) / n for a in range(m)], axis=1)
        Q, M = geometry_batch(muhat, p_G, supports)
        for ci, cvec in enumerate(cost_draws):
            true_cost = np.array([np.inf if q is None or q.min() < 0 else cvec[list(I)] @ q
                                  for (I, _, q, _), q in zip(truth, true_q)])
            for c0 in C0_GRID:
                adm = np.where((true_margin >= c0) & np.isfinite(true_cost))[0]
                if adm.size == 0:
                    continue
                order = adm[np.argsort(true_cost[adm])]
                s_star, C_star = order[0], true_cost[order[0]]
                gap2 = true_cost[order[1]] - C_star if order.size > 1 else np.inf
                C_any = np.nanmin(true_cost[np.isfinite(true_cost)])
                est_cost = np.einsum('rsk,sk->rs', Q, np.array([cvec[list(I)] for I in supports]))
                admissible = M >= 0.75 * c0
                est_cost = np.where(admissible, est_cost, np.inf)
                pick = np.argmin(est_cost, axis=1)
                fb = ~np.isfinite(est_cost[np.arange(R_SEL), pick])
                sel_true_cost = np.where(fb, np.nan, true_cost[pick])
                sel_margin = np.where(fb, np.nan, true_margin[pick])
                truly_feasible = np.isfinite(sel_true_cost) & (sel_margin > 0)
                good = sel_true_cost[truly_feasible]
                rows.append(dict(cost_draw=ci, c0=c0, n=n,
                                 recovery=float(np.mean(pick[~fb] == s_star)) if (~fb).any() else np.nan,
                                 near_opt_1usd=float(np.mean(good <= C_star + 1.0)) if good.size else np.nan,
                                 fallback=float(fb.mean()),
                                 truly_infeasible=float(np.mean(~truly_feasible)),
                                 realised_cost=float(np.mean(good)) if good.size else np.nan,
                                 gap_vs_admissible=float(np.mean(good) - C_star) if good.size else np.nan,
                                 gap_vs_any=float(np.mean(good) - C_any) if good.size else np.nan,
                                 margin_ok=float(np.nanmean(sel_margin >= c0 / 2)),
                                 mean_sel_margin=float(np.nanmean(sel_margin)),
                                 n_admissible=int(adm.size),
                                 C_star=float(C_star), C_any=float(C_any), gap2=float(gap2),
                                 thm3_bound=float(thm3_cost_bound(n, K, m, c0, cvec.max())),
                                 thm3_n=int(thm3_n(c0, K, m))))
        print(f"  n={n} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E3_support_selection.csv", index=False)
    base = df[df.cost_draw == 0]
    for v in ['recovery', 'near_opt_1usd', 'gap_vs_admissible', 'truly_infeasible', 'margin_ok', 'fallback']:
        print(f"-- {v}")
        print(base.pivot_table(index='n', columns='c0', values=v).to_string(float_format=lambda x: f"{x:7.3f}"))
    print("admissible supports / C_star / gap-to-2nd:")
    print(base.groupby('c0')[['n_admissible', 'C_star', 'C_any', 'gap2']].first().to_string(float_format=lambda x: f"{x:,.3f}"))


def stage_e32():
    sc, env = scenario()
    mu, p_G, cost = env.mu, env.p_G, env.cost
    supports = enumerate_supports(env.m, env.K)
    truth = support_table(mu, p_G, cost, supports)
    adm = sorted([t for t in truth if np.isfinite(t[3]) and t[1] >= 0.05], key=lambda z: z[3])
    I_star = list(adm[0][0])
    print(f"  cheapest admissible support (c0=0.05): {[env.names[i] for i in I_star]} ${adm[0][3]:.2f}")
    x_cost = np.array(sc['x_star_mincost'])
    pols = [Alg3ThenAlg2(n=100, c0=0.05, beta=0.3), Alg3ThenAlg2(n=25, c0=0.05, beta=0.3),
            Alg2UCB(0.3), MyopicCost(oracle=True), OracleCL(arms=I_star), OracleCL(),
            OpenLoopHerding(x_cost, "ORACLE-OL-mincost"),
            FixedArm(int(np.argmin(cost)), "N1-CHEAPEST"), RoundRobin()]
    frames, cps = [], log_checkpoints(T_E2E, 20)
    for pol in pols:
        df, _ = run(env, pol, T_E2E, R_E2E, seed=6001, checkpoints=cps)
        df["scenario"], df["experiment"] = SID, "E3.2"
        df["fallback_rate"] = (float(np.mean(pol.fallback)) if getattr(pol, "fallback", None) is not None else np.nan)
        frames.append(df)
        print(f"  {pol.name:<24} err={df.err_l2_mean.iloc[-1]:.5f} cost/n=${df.cost_per_sample.iloc[-1]:6.2f}", flush=True)
    pd.concat(frames).to_csv(f"{RES}/E3_endtoend.csv", index=False)


def stage_e33():
    sc, env = scenario()
    rows = []
    for c0 in C0_GRID:
        for n in [25, 100, 400]:
            pol = Alg3ThenAlg2(n=n, c0=c0, beta=0.3, tag=f"ALG3+2(n={n},c0={c0})")
            df, _ = run(env, pol, T_E2E, 100, seed=7001, checkpoints=[2000, 5000, T_E2E])
            for _, r in df.iterrows():
                rows.append(dict(c0=c0, n=n, T=int(r["T"]), err_l2=r["err_l2_mean"],
                                 cost_per_sample=r["cost_per_sample"], cost_total=r["cost_mean"],
                                 fallback=float(np.mean(pol.fallback)) if pol.fallback is not None else np.nan,
                                 explore_frac=env.m * n / r["T"]))
            print(f"  c0={c0} n={n}: err={rows[-1]['err_l2']:.5f} cost/n=${rows[-1]['cost_per_sample']:.1f} "
                  f"fallback={rows[-1]['fallback']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/E3_c0_tradeoff.csv", index=False)


def stage_e34():
    """Example 2 of the manuscript: the myopic cost-ratio rule pays 21/9 per recruit, support selection ~1.
    Arms A (p=0.4, $1), B (p=0.9, $1), C (p=0, $4); target share p_G = 0.5 of the designated group."""
    mu = np.array([[0.4, 0.6], [0.9, 0.1], [0.0, 1.0]])
    env = categorical_env(mu, np.array([0.5, 0.5]), np.array([1.0, 1.0, 4.0]), names=["A", "B", "C"])
    T, R = 100_000, 100
    pols = [MyopicCost(oracle=True, tie_arm=1, tag="MYOPIC-COST(oracle,tie=B)"),
            MyopicCost(oracle=True, tag="MYOPIC-COST(oracle,tie=least-sampled)"),
            Alg1Sign(arm_high=1, arm_low=0, coord=0), Alg3ThenAlg2(n=1000, c0=0.1, beta=1.0)]
    rows = []
    for pol in pols:
        df, _ = run(env, pol, T, R, seed=8001, checkpoints=[T])
        r = df.iloc[-1]
        rows.append(dict(policy=pol.name if pol.name != "ALG1" else "ALG1 on {A,B}", T=T, R=R,
                         cost_per_sample=r.cost_per_sample, err_share=r.err_linf_mean,
                         pull_A=r.pull_A, pull_B=r.pull_B, pull_C=r.pull_C,
                         fallback=float(np.mean(pol.fallback)) if getattr(pol, "fallback", None) is not None else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E3_myopic_counterexample.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"analytic: myopic 21/9 = {21/9:.4f}, support {{A,B}} = 1")


def stage_refs():
    sc, env = scenario()
    mu, p_G, cost = env.mu, env.p_G, env.cost
    rows = []
    for r in [1.0, 1.5, 2.0, 2.76, 4.0, 8.0]:
        cv = np.where(cost == 72.0, 72.0, 72.0 * r)
        x, C_ad = feasible_allocation(mu, p_G, cv)
        C_disc, _ = lp_discard_cost(mu, p_G, cv)
        cheap = int(np.argmin(cv))
        C_q = float(np.max(p_G / np.maximum(mu[cheap], 1e-12)) * cv[cheap])
        rows.append(dict(r=r, cost_adaptive=C_ad, cost_lp_discard=C_disc, cost_quota_cheapest=C_q,
                         adaptive_beats_quota=bool(C_ad <= C_q),
                         adaptive_beats_lp_discard=bool(C_ad <= C_disc),
                         arms_used=", ".join(f"{env.names[i]}:{x[i]:.2f}" for i in np.argsort(-x)[:4] if x[i] > 1e-3)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/E3_cost_reference.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    dict(e31=stage_e31, e32=stage_e32, e33=stage_e33, e34=stage_e34, refs=stage_refs)[sys.argv[1]]()
