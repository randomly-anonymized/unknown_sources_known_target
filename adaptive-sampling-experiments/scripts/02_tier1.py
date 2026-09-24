"""Tier 1 (sec. 2-3): two recruitment channels, one binary attribute.
E1.1 rates | E1.2 tail vs Theorem 1 | E1.3 Lemma-1 decomposition | E1.4 lower/upper bounds
E1.5 cost vs discard and reweighting | E1.6 robustness."""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.env import categorical_env, DriftingEnvironment
from src.policies import (Alg1Sign, OracleCL, OpenLoopHerding, OpenLoopIID, ETC, FixedArm, RoundRobin,
                          BatchWrapper, DelayWrapper)
from src.runner import run, log_checkpoints, quota_discard, largest_remainder, poststratification_stats, lp_discard_cost
from src.theory import thm1_best, thm1_tail_bound, lower_bound_kappa_binary, open_loop_constant

T_MAX, R_REPS, SEED = 100_000, 2000, 20260923
RES = os.path.join(ROOT, "results")


def load(sid):
    return json.load(open(os.path.join(ROOT, "data", "scenarios", sid + ".json")))


def tier1_env(sc):
    """Order the two arms so that arm 0 is BELOW and arm 1 ABOVE the target on coordinate 0."""
    mu = np.array(sc['mu']); p_G = np.array(sc['p_G']); cost = np.array(sc['cost'])
    K = mu.shape[1]
    if K > 2:                                   # collapse to the designated group vs rest
        raise ValueError("tier-1 scenarios must be binary")
    order = np.argsort(mu[:, 0])
    mu, cost = mu[order], cost[order]
    names = [sc['arms'][i] for i in order]
    env = categorical_env(mu, p_G, cost, names=names, label_names=sc['labels'])
    return env, names


def policies_for(env, x_star):
    return [Alg1Sign(arm_high=1, arm_low=0, coord=0),
            OpenLoopHerding(x_star, "ORACLE-OL"), OpenLoopIID(x_star, "ORACLE-OL-iid"),
            ETC(eps=0.1, cost_aware=True), FixedArm(int(np.argmin(env.cost)), "N1-CHEAPEST"),
            RoundRobin(), FixedArm(int(np.argmax(env.cost)), "N1-EXPENSIVE")]


def stage_rates(which):
    all_rates, all_tails, all_summary, all_cost = [], [], [], []
    for sid in [which]:
        sc = load(sid)
        env, names = tier1_env(sc)
        p_low, p_high = env.mu[0, 0], env.mu[1, 0]
        p_G = env.p_G[0]
        x_star = np.array([(p_high - p_G) / (p_high - p_low), (p_G - p_low) / (p_high - p_low)])
        kappa = lower_bound_kappa_binary(env.mu[:, 0])
        nu, ol_const = open_loop_constant(x_star, env.mu[:, 0])
        bestES, lam, rho, C = thm1_best(p_low, p_G, p_high)
        print(f"\n=== {sid}: arms {names} p_low={p_low:.4f} p_G={p_G:.4f} p_high={p_high:.4f}")
        print(f"    x*={x_star.round(4)}  gap_min={min(p_G-p_low, p_high-p_G):.4f}  kappa={kappa:.4f} "
              f"Thm1 E|S|<= {bestES:.2f} (lam={lam:.3f}, C={C:.1f}) open-loop const={ol_const:.4f}")
        cps = log_checkpoints(T_MAX, 30)
        keep = (100, 1000, 10_000, 100_000)
        for pol in policies_for(env, x_star):
            df, extra = run(env, pol, T_MAX, R_REPS, seed=SEED + hash(sid) % 1000,
                            checkpoints=cps, keep_paths=keep if pol.name == "ALG1" else ())
            df["scenario"] = sid
            df["err_scalar_mean"] = df.err_linf_mean
            df["S_scalar_mean"] = df["T"] * df["err_linf_mean"]
            all_rates.append(df)
            if pol.name == "ALG1":
                for t, pth in extra["paths"].items():
                    if not isinstance(pth, dict):
                        continue
                    s_scalar = np.abs(pth["S"][:, 0])
                    ks = np.arange(0, 25)
                    emp = [(s_scalar >= k).mean() for k in ks]
                    bnd = thm1_tail_bound(ks, p_low, p_G, p_high)
                    all_tails.append(pd.DataFrame(dict(scenario=sid, T=t, k=ks, empirical=emp, bound=bnd)))
                st = extra["state"]
                alloc = st.alloc[:, 0]; samp = st.S[:, 0] - st.alloc[:, 0]
                all_summary.append(dict(scenario=sid, quantity="ALG1_decomposition_corr",
                                        value=float(np.corrcoef(alloc, samp)[0, 1])))
                all_summary.append(dict(scenario=sid, quantity="ALG1_E|S_T|", value=float(np.abs(st.S[:, 0]).mean())))
            # decomposition for ETC as contrast
            if pol.name.startswith("ETC"):
                st = extra["state"]
                all_summary.append(dict(scenario=sid, quantity="ETC_decomposition_corr",
                                        value=float(np.corrcoef(st.alloc[:, 0], st.S[:, 0] - st.alloc[:, 0])[0, 1])))
        all_summary += [dict(scenario=sid, quantity=k, value=v) for k, v in
                        dict(p_low=p_low, p_G=p_G, p_high=p_high, x_star_high=x_star[1], kappa_lb=kappa,
                             thm1_ES_bound=bestES, thm1_lambda=lam, thm1_C=C, open_loop_const=ol_const,
                             cost_low=env.cost[0], cost_high=env.cost[1]).items()]

        # ---- E1.5 cost: adaptive vs quota-discard vs reweighting, over the cost ratio r
        cheap = int(np.argmin(env.cost)); exp_ = 1 - cheap
        f_discard = float(max(p_G / env.mu[cheap, 0], (1 - p_G) / (1 - env.mu[cheap, 0])))
        x_cheap = x_star[cheap]
        r_star = (f_discard - x_cheap) / max(1 - x_cheap, 1e-12)
        all_summary += [dict(scenario=sid, quantity="quota_factor", value=f_discard),
                        dict(scenario=sid, quantity="r_star_breakeven", value=float(r_star)),
                        dict(scenario=sid, quantity="r_base", value=float(env.cost[exp_] / env.cost[cheap]))]
        for N in [200, 500, 1000, 2000, 5000]:
            contacts, quota = quota_discard(env, cheap, N, 400, seed=SEED + N)
            for r in [1.0, 1.5, 2.0, 2.76, 4.0, 8.0]:
                lc, hc = env.cost[cheap], env.cost[cheap] * r
                adaptive = N * (x_cheap * lc + (1 - x_cheap) * hc)
                quota_cost = float(contacts.mean()) * lc
                naive_cost = N * lc
                all_cost.append(dict(scenario=sid, N=N, r=r, cost_adaptive=adaptive, cost_quota=quota_cost,
                                     cost_naive_cheapest=naive_cost,
                                     contacts_quota=float(contacts.mean()),
                                     cost_lp_discard=N * lp_discard_cost(env.mu, env.p_G, [lc, hc])[0]))
    tag = which.split('__')[-1]
    pd.concat(all_rates).to_csv(f"{RES}/E1_rates_{tag}.csv", index=False)
    pd.concat(all_tails).to_csv(f"{RES}/E1_tails_{tag}.csv", index=False)
    pd.DataFrame(all_summary).to_csv(f"{RES}/E1_summary_{tag}.csv", index=False)
    pd.DataFrame(all_cost).to_csv(f"{RES}/E1_cost_{tag}.csv", index=False)
    print(f"wrote E1_*_{tag}.csv")


def stage_example1():
    all_summary, all_rates = [], []
    # ---- Example 1 of the manuscript (sanity reproduction)
    env_ex = categorical_env(np.array([[0.2, 0.8], [0.8, 0.2]]), np.array([0.4, 0.6]), np.array([3.0, 1.0]),
                             names=["PHONE", "ONLINE"])
    contacts, _ = quota_discard(env_ex, 1, 1000, 500, seed=7)
    x_star_ex = np.array([(0.8 - 0.4) / 0.6, (0.4 - 0.2) / 0.6])
    all_summary += [dict(scenario="EXAMPLE-1", quantity="cost_adaptive_N1000",
                         value=float(1000 * (x_star_ex[0] * 3 + x_star_ex[1] * 1))),
                    dict(scenario="EXAMPLE-1", quantity="cost_discard_N1000_sim", value=float(contacts.mean() * 1.0)),
                    dict(scenario="EXAMPLE-1", quantity="online_share_x*", value=float(x_star_ex[1]))]
    df_ex, _ = run(env_ex, Alg1Sign(1, 0, 0), 20000, 800, seed=11, checkpoints=log_checkpoints(20000, 20))
    df_ex["scenario"] = "EXAMPLE-1"; df_ex["err_scalar_mean"] = df_ex.err_linf_mean
    df_ex["S_scalar_mean"] = df_ex["T"] * df_ex["err_linf_mean"]
    all_rates.append(df_ex)
    pd.concat(all_rates).to_csv(f"{RES}/E1_rates_example1.csv", index=False)
    pd.DataFrame(all_summary).to_csv(f"{RES}/E1_summary_example1.csv", index=False)
    print("wrote example-1 files")


def stage_robust():
    all_robust = []
    # ---- E1.6 robustness (on the T1-AGE instance)
    sc = load("S-DIAB__T1-AGE"); env, names = tier1_env(sc)
    p_low, p_high, p_G = env.mu[0, 0], env.mu[1, 0], env.p_G[0]
    x_star = np.array([(p_high - p_G) / (p_high - p_low), (p_G - p_low) / (p_high - p_low)])
    cps = log_checkpoints(20000, 18)
    # (a) ordering violated: both channels below the target (ONLINE vs PORTAL on the Black stratum)
    scb = load("S-DIAB__L1_K2_black")
    i_on, i_po = scb['arms'].index('ONLINE'), scb['arms'].index('PORTAL')
    mu_bad = np.array(scb['mu'])[[i_po, i_on]]
    env_bad = categorical_env(mu_bad, np.array(scb['p_G']), np.array(scb['cost'])[[i_po, i_on]],
                              names=['PORTAL', 'ONLINE'])
    d, _ = run(env_bad, Alg1Sign(1, 0, 0), 20000, 500, seed=31, checkpoints=cps)
    d["variant"] = "a_ordering_violated"; all_robust.append(d)
    # (b) shrinking gap: mu(s) = p_G + s (mu - p_G)
    for s in [1.0, 0.5, 0.25, 0.125]:
        mus = env.p_G + s * (env.mu - env.p_G)
        e2 = categorical_env(mus, env.p_G, env.cost, names=names)
        d, _ = run(e2, Alg1Sign(1, 0, 0), 20000, 500, seed=41, checkpoints=cps)
        d["variant"] = f"b_gap_scale_{s}"; all_robust.append(d)
    # (c) non-stationary channels: +-5 points of linear drift, ordering preserved
    mu0 = env.mu.copy(); mu1 = env.mu.copy()
    mu1[0, 0] += 0.05; mu1[0, 1] -= 0.05; mu1[1, 0] -= 0.05; mu1[1, 1] += 0.05
    env_dr = DriftingEnvironment(mu0, mu1, env.p_G, env.cost, 20000, names=names)
    for pol in [Alg1Sign(1, 0, 0), ETC(eps=0.1), OpenLoopHerding(x_star, "ORACLE-OL")]:
        d, _ = run(env_dr, pol, 20000, 500, seed=51, checkpoints=cps)
        d["variant"] = "c_drift"; all_robust.append(d)
    # (d) batched decisions and (e) delayed labels
    for B in [10, 50]:
        d, _ = run(env, BatchWrapper(Alg1Sign(1, 0, 0), B), 20000, 500, seed=61, checkpoints=cps)
        d["variant"] = f"d_batch_{B}"; all_robust.append(d)
    for dl in [10, 50]:
        d, _ = run(env, DelayWrapper(Alg1Sign(1, 0, 0), dl), 20000, 500, seed=71, checkpoints=cps)
        d["variant"] = f"e_delay_{dl}"; all_robust.append(d)
    d, _ = run(env, Alg1Sign(1, 0, 0), 20000, 500, seed=81, checkpoints=cps)
    d["variant"] = "reference"; all_robust.append(d)

    pd.concat(all_robust).to_csv(f"{RES}/E1_robustness.csv", index=False)
    print("wrote E1_robustness.csv")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "age":
        stage_rates("S-DIAB__T1-AGE")
    elif stage == "black":
        stage_rates("S-DIAB__T1-BLACK")
    elif stage == "example1":
        stage_example1()
    elif stage == "robust":
        stage_robust()
    else:
        raise SystemExit("stage must be one of: age black example1 robust")
