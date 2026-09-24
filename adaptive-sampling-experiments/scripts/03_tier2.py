"""Tier 2 (sec. 5): multi-dimensional targets. Staged so each stage runs in a few minutes.
stages: e21a1 e21a2 (categorical headline) | e21b (Black x age) | e22a e22b (multi-label)
        e24a e24b (granularity ladder) | e25 (downstream estimation)"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.env import Environment, categorical_env
from src.policies import Alg2UCB, OracleCL, OpenLoopHerding, ETC, FixedArm, RoundRobin, HerdingAllocation
from src.runner import run, log_checkpoints, poststratification_stats
from src.theory import thm2_constants, lower_bound_kappa, ucb_radius

RES, SC = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "scenarios")
T_MAIN, R_MAIN = 100_000, 250
T_LADDER, R_LADDER = 100_000, 120
T_RESP, R_RESP = 20_000, 200
POOLS = os.path.join(SC, "pools_S-DIAB.npz")


def load(sid):
    return json.load(open(os.path.join(SC, sid + ".json")))


def env_from_scenario(sc):
    e = categorical_env(np.array(sc['mu']), np.array(sc['p_G']), np.array(sc['cost']),
                        names=sc['arms'], label_names=sc['labels'])
    e.c_margin = sc['margin_c']
    return e


def multilabel_env(sc):
    z = np.load(POOLS, allow_pickle=True)
    ml, w, mask, arms = z['ml'], z['w'], z['arm_mask'], [str(a) for a in z['arms']]
    keep = [arms.index(a) for a in sc['arms']]
    pats, inv = np.unique(ml, axis=0, return_inverse=True)
    inv = inv.ravel()
    probs = np.zeros((len(keep), len(pats)))
    for i, a in enumerate(keep):
        sel = mask[a]
        np.add.at(probs[i], inv[sel], w[sel])
        probs[i] /= probs[i].sum()
    e = Environment(probs, pats, np.array(sc['p_G']), np.array(sc['cost']), names=sc['arms'],
                    categorical=False, label_names=sc['labels'])
    e.c_margin = sc['margin_c']
    return e


def respondent_env(sc, ladder_key):
    z = np.load(POOLS, allow_pickle=True)
    w, mask, arms = z['w'], z['arm_mask'], [str(a) for a in z['arms']]
    lab = z[f'lab_{ladder_key}'].astype(float)
    out, onames = z['outcomes'], [str(x) for x in z['outcome_names']]
    keep = [arms.index(a) for a in sc['arms']]
    probs = np.zeros((len(keep), len(w)))
    for i, a in enumerate(keep):
        probs[i, mask[a]] = w[mask[a]]
        probs[i] /= probs[i].sum()
    e = Environment(probs, lab, np.array(sc['p_G']), np.array(sc['cost']), names=sc['arms'],
                    outcomes=out, outcome_names=onames, label_names=sc['labels'])
    e.c_margin = sc['margin_c']
    theta = (out * w[:, None]).sum(0) / w.sum()
    return e, theta, onames


UCB_SET = lambda: [Alg2UCB(1.0), Alg2UCB(0.3), Alg2UCB(0.1), Alg2UCB(0.0, tag="ALG2(greedy)")]
REF_SET = lambda env, sc: [OracleCL(), OpenLoopHerding(np.array(sc['x_star_mincost']), "ORACLE-OL"),
                           ETC(eps=0.1), FixedArm(int(np.argmin(env.cost)), "N1-CHEAPEST"),
                           RoundRobin(), HerdingAllocation(np.array(sc['pool_share']), "N3-POOLPROP")]


def run_set(env, sc, sid, pols, tag, T, R, exp, seed=1000, extra_cols=None):
    cps, frames = log_checkpoints(T, 26), []
    for pol in pols:
        df, _ = run(env, pol, T, R, seed=seed, checkpoints=cps)
        df["scenario"], df["experiment"] = sid, exp
        for k, v in (extra_cols or {}).items():
            df[k] = v
        frames.append(df)
        print(f"  {sid} {pol.name:<18} err(T)={df.err_l2_mean.iloc[-1]:.5f} "
              f"T*err={T*df.err_l2_mean.iloc[-1]:7.1f} cost/n=${df.cost_per_sample.iloc[-1]:6.1f} "
              f"unc={df.uncertain_mean.iloc[-1]:.0f}", flush=True)
    pd.concat(frames).to_csv(f"{RES}/E2_rates_{tag}.csv", index=False)


def diagnostics():
    rows = []
    for sid in ["S-DIAB__L2_K3_eth", "S-DIAB__L4_K4_blackage", "S-DIAB__ML_K4_marginals"]:
        sc = load(sid)
        env = multilabel_env(sc) if "ML_" in sid else env_from_scenario(sc)
        cst = thm2_constants(sc['margin_c'], env.R2, env.K, env.m)
        kap = (lower_bound_kappa(env.arm_probs, patterns=env.patterns) if "ML_" in sid
               else lower_bound_kappa(env.mu))
        rows += [dict(scenario=sid, quantity=k, value=v) for k, v in
                 dict(margin_c=sc['margin_c'], R2=env.R2, kappa_lb=kap,
                      N_c=cst.get('N_c', np.nan), M_c=cst.get('M_c', np.nan),
                      thm2_bound_S=cst.get('bound_S', np.nan),
                      r_at_1e3=float(ucb_radius(1000, env.K, env.m)),
                      r_at_1e5=float(ucb_radius(100000, env.K, env.m)),
                      ell0_max=float(env.patterns.sum(1).max())).items()]
    pd.DataFrame(rows).to_csv(f"{RES}/E2_diagnostics.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index='quantity', columns='scenario', values='value').to_string())


def downstream():
    sid = "S-DIAB__L2_K3_eth"; sc = load(sid)
    envR, theta, onames = respondent_env(sc, "L2_K3_eth")
    envS = env_from_scenario(sc)
    rows = []
    for mode, env in [("respondent", envR), ("stratum", envS)]:
        for pol in [Alg2UCB(0.3), OracleCL(), FixedArm(int(np.argmin(env.cost)), "N1-CHEAPEST"), RoundRobin()]:
            df, extra = run(env, pol, T_RESP, R_RESP, seed=4000, checkpoints=[1000, T_RESP],
                            track_outcomes=(mode == "respondent"), keep_paths=(1000, T_RESP))
            for t, pth in extra["paths"].items():
                counts = np.round(pth["counts"]).astype(int)
                deff, ess, fail = poststratification_stats(counts, env.p_G)
                row = dict(scenario=sid, mode=mode, policy=pol.name, T=t,
                           err_l2=float(np.linalg.norm(pth["S"] / t, axis=1).mean()),
                           deff=float(np.nanmean(deff)), ess_frac=float(np.nanmean(ess)) / t,
                           ps_fail=float(fail.mean()),
                           cost_per_sample=float(df[df["T"] == t].cost_per_sample.iloc[0]))
                if mode == "respondent" and pth["out"] is not None:
                    est = pth["out"]
                    for j, nm in enumerate(onames):
                        row[f"bias_{nm}"] = float(est[:, j].mean() - theta[j])
                        row[f"sd_{nm}"] = float(est[:, j].std())
                        row[f"rmse_{nm}"] = float(np.sqrt(((est[:, j] - theta[j]) ** 2).mean()))
                        row[f"truth_{nm}"] = float(theta[j])
                rows.append(row)
            print(f"  {mode} {pol.name} done", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/E2_downstream.csv", index=False)


LADDERS = ["L1_K2_black", "L2_K3_eth", "L3_K3_age", "L5_K4_agesex", "L4_K4_blackage", "L6_K6_ethage"]


def ladder(which):
    frames = []
    for lid in which:
        sid = f"S-DIAB__{lid}"; sc = load(sid); env = env_from_scenario(sc)
        cps = log_checkpoints(T_LADDER, 20)
        for pol in [Alg2UCB(0.3), OracleCL(), RoundRobin()]:
            df, _ = run(env, pol, T_LADDER, R_LADDER, seed=3000, checkpoints=cps)
            df["scenario"], df["experiment"], df["ladder"] = sid, "E2.4", lid
            df["margin_c"], df["K"] = sc['margin_c'], sc['K']
            frames.append(df)
        print(f"  ladder {lid:<16} c={sc['margin_c']:.4f} "
              f"errT={frames[-3].err_l2_mean.iloc[-1]:.5f}(ucb) {frames[-2].err_l2_mean.iloc[-1]:.5f}(oracle)", flush=True)
    pd.concat(frames).to_csv(f"{RES}/E2_ladder_{which[0]}.csv", index=False)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "diag":
        diagnostics()
    elif st == "e21a1":
        sc = load("S-DIAB__L2_K3_eth"); run_set(env_from_scenario(sc), sc, sc['id'], UCB_SET(), "eth_ucb", T_MAIN, R_MAIN, "E2.1")
    elif st == "e21a2":
        sc = load("S-DIAB__L2_K3_eth"); e = env_from_scenario(sc); run_set(e, sc, sc['id'], REF_SET(e, sc), "eth_ref", T_MAIN, R_MAIN, "E2.1")
    elif st == "e21b":
        sc = load("S-DIAB__L4_K4_blackage"); e = env_from_scenario(sc)
        run_set(e, sc, sc['id'], [Alg2UCB(0.3), OracleCL(), RoundRobin(), FixedArm(int(np.argmin(e.cost)), "N1-CHEAPEST")],
                "blackage", T_MAIN, R_MAIN, "E2.1")
    elif st == "e22a":
        sc = load("S-DIAB__ML_K4_marginals"); run_set(multilabel_env(sc), sc, sc['id'], UCB_SET(), "ml_ucb", T_MAIN, R_MAIN, "E2.2")
    elif st == "e22b":
        sc = load("S-DIAB__ML_K4_marginals"); e = multilabel_env(sc); run_set(e, sc, sc['id'], REF_SET(e, sc), "ml_ref", T_MAIN, R_MAIN, "E2.2")
    elif st == "e24a":
        ladder(LADDERS[:3])
    elif st == "e24b":
        ladder(LADDERS[3:])
    elif st == "e25":
        downstream()
    else:
        raise SystemExit("unknown stage")
