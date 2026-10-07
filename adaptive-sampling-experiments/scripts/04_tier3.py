"""Tier 3 (sec. 6): cost-aware sampling. Stages: e31 (support selection), e32 <key> / e32merge (cost per
recruit and error against the sample size T; Figure 1(c)), e33 (c0 trade-off), e35 <key> / e35merge
(error at a fixed total budget), refs (cost references /
break-even table; not used in the manuscript).

The cost experiments use the Black/other target of the diabetes population and the meta-analysis medians:
online recruits at $72 and offline recruits at OFFLINE_COST (default $199; can be overridden through the
environment variable of the same name). No policy discards recruits; the exploration samples of
Algorithm 3 are kept in the sample and also used by Algorithm 2 to estimate the source means."""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.env import categorical_env
from src.geometry import enumerate_supports, support_table, feasible_allocation
from src.policies import (Alg2UCB, OracleCL, Alg3ThenAlg2, GreedyCost, FixedArm, RoundRobin,
                          OpenLoopHerding, Alg1Sign, simplex_heights_batch)
from src.runner import run, log_checkpoints, lp_discard_cost, run_budget
from src.theory import thm3_cost_bound, thm3_n

RES, SC = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "scenarios")
SID = "S-DIAB__L1_K2_black"
N_GRID = [25, 50, 100, 200, 400, 800, 1600]
C0_GRID = [0.01, 0.02, 0.05, 0.10]
C0_MAIN, N_MAIN = 0.02, 1600                                    # Algorithm 3 in Figure 1(c)
R_SEL, T_E2E, R_E2E = 250, 300_000, 200
T_C0 = 100_000                                                  # e33 horizon
COST_RANGE = {72.0: (3.9, 251.2), 199.0: (19.1, 839.0)}     # JMIR 2020 meta-analysis ranges
OFFLINE_COST = float(os.environ.get("OFFLINE_COST", 199.0))    # offline cost per recruit in the cost experiments
BUDGETS = np.geomspace(1e6, 1.6e7, 13)                         # e35: total budgets ($)


def scenario():
    sc = json.load(open(os.path.join(SC, SID + ".json")))
    cost = np.where(np.array(sc['cost']) == 72.0, 72.0, OFFLINE_COST)
    env = categorical_env(np.array(sc['mu']), np.array(sc['p_G']), cost,
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
    cost_draws = [cost] + [np.exp([rng.uniform(*np.log(COST_RANGE[72.0 if c == 72.0 else 199.0])) for c in cost])
                           for _ in range(9)]
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


def cheapest_admissible(env, c0):
    truth = support_table(env.mu, env.p_G, env.cost, enumerate_supports(env.m, env.K))
    adm = sorted([t for t in truth if np.isfinite(t[3]) and t[1] >= c0], key=lambda z: z[3])
    return list(adm[0][0]), adm[0][3]


E32_POLICIES = {
    "alg3": lambda env: Alg3ThenAlg2(n=N_MAIN, c0=C0_MAIN, beta=0.3),
    "alg3n800": lambda env: Alg3ThenAlg2(n=800, c0=C0_MAIN, beta=0.3),
    "alg3n400": lambda env: Alg3ThenAlg2(n=400, c0=C0_MAIN, beta=0.3),
    "alg2": lambda env: Alg2UCB(0.3),
    "myopic": lambda env: GreedyCost(beta=0.3),               # greedy rule (Algorithm 4), optimistic estimated means
    "oracle": lambda env: OracleCL(arms=cheapest_admissible(env, C0_MAIN)[0]),
    "oracleall": lambda env: OracleCL(),
    "olmincost": lambda env: OpenLoopHerding(feasible_allocation(env.mu, env.p_G, env.cost)[0], "ORACLE-OL-mincost"),
    "online": lambda env: FixedArm(int(np.argmin(env.cost)), "N1-CHEAPEST"),
    "uniform": lambda env: RoundRobin(),
}


def stage_e32(key):
    """Cost per recruit and error against T. Writes results/E3_endtoend_parts/<key>.csv."""
    sc, env = scenario()
    pol = E32_POLICIES[key](env)
    if key == "oracle":
        pol.name = "ORACLE-CL(I*)"
    cps = np.union1d(log_checkpoints(T_E2E, 40), [20_000, 100_000])
    df, _ = run(env, pol, T_E2E, R_E2E, seed=6001, checkpoints=cps)
    df["scenario"], df["experiment"] = SID, "E3.2"
    df["fallback_rate"] = (float(np.mean(pol.fallback)) if getattr(pol, "fallback", None) is not None else np.nan)
    out = os.path.join(RES, "E3_endtoend_parts"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{key}.csv"), index=False)
    for T in [20_000, 100_000, T_E2E]:
        r = df[df["T"] == T].iloc[0]
        print(f"  {pol.name:<24} T={T:>7} err={r.err_l2_mean:.2e} (med {r.err_l2_med:.2e}) cost/n=${r.cost_per_sample:6.2f}",
              flush=True)


def stage_e32_merge():
    import glob
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "E3_endtoend_parts", "*.csv")))])
    df.to_csv(f"{RES}/E3_endtoend.csv", index=False)
    print(df[df["T"] == 100_000][["policy", "err_l2_mean", "err_l2_med", "cost_per_sample"]].to_string(index=False))


def stage_e33():
    sc, env = scenario()
    rows = []
    for c0 in C0_GRID:
        for n in [400, 800, 1600]:
            pol = Alg3ThenAlg2(n=n, c0=c0, beta=0.3, tag=f"ALG3+2(n={n},c0={c0})")
            df, _ = run(env, pol, T_C0, 100, seed=7001, checkpoints=[30_000, T_C0])
            for _, r in df.iterrows():
                rows.append(dict(c0=c0, n=n, T=int(r["T"]), err_l2=r["err_l2_mean"],
                                 cost_per_sample=r["cost_per_sample"], cost_total=r["cost_mean"],
                                 fallback=float(np.mean(pol.fallback)) if pol.fallback is not None else np.nan,
                                 explore_frac=env.m * n / r["T"]))
            print(f"  c0={c0} n={n}: err={rows[-1]['err_l2']:.5f} cost/n=${rows[-1]['cost_per_sample']:.1f} "
                  f"fallback={rows[-1]['fallback']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/E3_c0_tradeoff.csv", index=False)


E35_POLICIES = {k: E32_POLICIES[k] for k in ["alg3", "alg3n800", "alg3n400", "alg2", "myopic", "oracle"]}


def stage_e35(key):
    """Error ||p_hat - p_G||_2 at fixed total budgets. Writes results/E3_budget_parts/<offline cost>_<key>.csv."""
    sc, env = scenario()
    mu, p_G, cost, names = env.mu, env.p_G, env.cost, env.names
    out = os.path.join(RES, "E3_budget_parts"); os.makedirs(out, exist_ok=True)
    R, rows = 200, []
    pol = E35_POLICIES[key](env)
    err, Tb = run_budget(env, pol, BUDGETS, R, seed=9002, T_cap=int(BUDGETS[-1] / cost.min()) + 10)
    fb = float(np.mean(pol.fallback)) if getattr(pol, "fallback", None) is not None else np.nan
    for b_i, b in enumerate(BUDGETS):
        rows.append(dict(policy=pol.name if key != "oracle" else "ORACLE-CL(I*)", budget=b,
                         err_mean=np.nanmean(err[:, b_i]), err_median=np.nanmedian(err[:, b_i]),
                         err_q90=np.nanquantile(err[:, b_i], .9), n_mean=np.nanmean(Tb[:, b_i]),
                         fallback=fb))
    df = pd.DataFrame(rows); df.insert(0, "offline_cost", OFFLINE_COST)
    df.to_csv(os.path.join(out, f"{OFFLINE_COST:g}_{key}.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def stage_e35_merge():
    import glob
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "E3_budget_parts", "*.csv")))])
    df.to_csv(f"{RES}/E3_budget.csv", index=False)
    print(df.pivot_table(index=["offline_cost", "policy"], columns="budget", values="err_mean").to_string(
        float_format=lambda v: f"{v:.1e}"))


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
    if sys.argv[1] in ("e32", "e35"):
        dict(e32=stage_e32, e35=stage_e35)[sys.argv[1]](sys.argv[2])
    else:
        dict(e31=stage_e31, e32merge=stage_e32_merge, e33=stage_e33, e35merge=stage_e35_merge,
             refs=stage_refs)[sys.argv[1]]()
