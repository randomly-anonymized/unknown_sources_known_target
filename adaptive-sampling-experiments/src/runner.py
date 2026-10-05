"""Simulation driver: runs a policy on an environment for T rounds x R replications and records metrics."""
import numpy as np
import pandas as pd
from .policies import State


def log_checkpoints(T, n=30, start=10):
    cps = np.unique(np.round(np.logspace(np.log10(start), np.log10(T), n)).astype(int))
    return cps[(cps >= 1) & (cps <= T)]


def run(env, policy, T, R, seed, checkpoints=None, track_outcomes=False, common_uniforms=True,
        keep_paths=(), batch_uniform=4096):
    """Returns (df_metrics, extras). One row per (checkpoint, replication-summary)."""
    checkpoints = log_checkpoints(T) if checkpoints is None else np.asarray(checkpoints)
    rng = np.random.default_rng(seed)
    st = State(R, env.m, env.K)
    policy.reset(env, T, R)
    p_G = env.p_G
    J = 0 if (env.outcomes is None or not track_outcomes) else env.outcomes.shape[1]
    osum = np.zeros((R, J)) if J else None
    rows, paths = [], {c: {} for c in keep_paths}
    cps = set(int(c) for c in checkpoints)
    u_block = None
    for t in range(T):
        if common_uniforms:
            if t % batch_uniform == 0:
                u_block = rng.random((batch_uniform, R))
            u = u_block[t % batch_uniform]
        else:
            u = rng.random(R)
        if hasattr(env, 'set_time'):
            env.set_time(t)
        a = policy.choose(t, st, rng)
        pat = env.draw_patterns(a, u)
        Y = env.labels(pat)
        st.S += Y - p_G
        st.alloc += env.mu[a] - p_G
        st.N[np.arange(R), a] += 1
        st.Ssum[np.arange(R), a] += Y
        st.cost += env.cost[a]
        if J:
            osum += env.outcome(pat)
        tt = t + 1
        if tt in cps:
            err = st.S / tt
            l2 = np.linalg.norm(err, axis=1)
            linf = np.abs(err).max(axis=1)
            row = dict(T=tt,
                       err_l2_mean=l2.mean(), err_l2_med=np.median(l2), err_l2_q90=np.quantile(l2, .9),
                       err_linf_mean=linf.mean(), err_linf_med=np.median(linf),
                       err_tv_mean=0.5 * np.abs(err).sum(axis=1).mean(),
                       S_norm_mean=(tt * l2).mean(), S_norm_q90=np.quantile(tt * l2, .9),
                       S_norm_max=(tt * l2).max(),
                       cost_mean=st.cost.mean(), cost_per_sample=st.cost.mean() / tt,
                       uncertain_mean=st.uncertain.mean(),
                       alloc_err=np.linalg.norm(st.alloc / tt, axis=1).mean(),
                       samp_err=np.linalg.norm((st.S - st.alloc) / tt, axis=1).mean())
            for j in range(env.m):
                row[f"pull_{env.names[j]}"] = st.N[:, j].mean() / tt
            if J:
                for j, nm in enumerate(env.outcome_names):
                    row[f"out_{nm}_mean"] = (osum[:, j] / tt).mean()
                    row[f"out_{nm}_sd"] = (osum[:, j] / tt).std()
            rows.append(row)
        if tt in paths:
            paths[tt] = dict(S=st.S.copy(), N=st.N.copy(), counts=(st.S + tt * p_G).copy(),
                             out=None if not J else osum / tt)
    df = pd.DataFrame(rows)
    df.insert(0, "policy", policy.name)
    extras = dict(state=st, paths=paths, fallback=getattr(policy, "fallback", None),
                  selected=getattr(policy, "selected", None))
    return df, extras


def quota_discard(env, arm, N_target, R, seed, max_factor=40, chunk=4000):
    """Cheap-arm + discard (quota screening): contacts needed until every stratum quota is filled.
    Exact, vectorised over replications. Returns (contacts (R,), quota (K,))."""
    rng = np.random.default_rng(seed)
    K = env.K
    quota = largest_remainder(env.p_G, N_target)
    cum = np.cumsum(env.mu[arm]); cum[-1] = 1.0
    counts = np.zeros((R, K), dtype=np.int64)
    contacts = np.zeros(R, dtype=np.int64)
    done = np.zeros(R, bool)
    processed = 0
    while (not done.all()) and processed < max_factor * max(N_target, 1):
        B = int(chunk)
        act = np.where(~done)[0]
        u = rng.random((act.size, B))
        strat = np.searchsorted(cum, u, side="right").clip(0, K - 1)
        onehot = np.zeros((act.size, B, K), dtype=np.int32)
        np.put_along_axis(onehot, strat[:, :, None], 1, axis=2)
        run = counts[act][:, None, :] + np.cumsum(onehot, axis=1)
        ok = (run >= quota[None, None, :]).all(axis=2)          # (len(act), B)
        hit = ok.any(axis=1)
        first = np.argmax(ok, axis=1)
        contacts[act[hit]] = processed + first[hit] + 1
        done[act[hit]] = True
        counts[act] = run[:, -1, :]
        processed += B
    contacts[~done] = max_factor * max(N_target, 1)
    return contacts, quota


def lp_discard_cost(mu, p_G, cost):
    """Cheapest contact mix that yields one representative retained respondent when discarding is allowed:
    min sum_a l_a n_a  s.t.  sum_a n_a mu_a >= p_G, n >= 0.   (Relaxation of the no-discard LP.)"""
    from scipy.optimize import linprog
    mu = np.asarray(mu, float)
    r = linprog(np.asarray(cost, float), A_ub=-mu.T, b_ub=-np.asarray(p_G, float),
                bounds=[(0, None)] * mu.shape[0])
    return (float(r.fun), r.x) if r.status == 0 else (np.inf, None)


def largest_remainder(p, N):
    raw = np.asarray(p, float) * N
    base = np.floor(raw).astype(int)
    rem = N - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return base


def poststratification_stats(counts, p_G):
    """counts: (R,K) realised stratum counts. Returns (design effect, ESS, failure mask)."""
    T = counts.sum(axis=1, keepdims=True)
    phat = counts / T
    fail = (counts == 0).any(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        deff = np.where(fail, np.nan, np.sum(p_G[None, :] ** 2 / np.where(phat > 0, phat, np.nan), axis=1))
    return deff, T[:, 0] / deff, fail


def run_budget(env, policy, budgets, R, seed, T_cap):
    """Run a policy until each replication has spent the largest budget. For every budget b, record
    ||p_hat - p_G||_2 of the sample bought with at most b (the recruit that would exceed b is excluded).
    Returns (err (R, len(budgets)), T (R, len(budgets)))."""
    budgets = np.asarray(budgets, float)
    rng = np.random.default_rng(seed)
    st = State(R, env.m, env.K)
    policy.reset(env, T_cap, R)
    nb = len(budgets)
    err = np.full((R, nb), np.nan); Tb = np.full((R, nb), np.nan)
    j = np.zeros(R, dtype=int)
    for t in range(T_cap):
        a = policy.choose(t, st, rng)
        Y = env.labels(env.draw_patterns(a, rng.random(R)))
        new_cost = st.cost + env.cost[a]
        while True:
            m = (j < nb) & (new_cost > budgets[np.minimum(j, nb - 1)])
            if not m.any():
                break
            err[m, j[m]] = np.linalg.norm(st.S[m], axis=1) / max(t, 1)
            Tb[m, j[m]] = t
            j[m] += 1
        st.S += Y - env.p_G
        st.alloc += env.mu[a] - env.p_G
        st.N[np.arange(R), a] += 1
        st.Ssum[np.arange(R), a] += Y
        st.cost = new_cost
        if (j >= nb).all():
            break
    return err, Tb


def cheap_and_discard_budget(mu_cheap, p_G, ell, budgets, R, seed):
    """Cheap & discard as in Example 1 of the manuscript, under a total budget b. The planned sample
    size is the one the budget buys on average, N = floor((b / ell) / f) with f = max_k p_G,k / mu_k
    (Example 1's arithmetic, using the composition of the cheap channel). Recruits are drawn from the
    cheap channel one at a time and kept while their group's quota round(N p_G) is open; the procedure
    stops when all quotas are full or the budget is spent. Every contact is paid.
    Returns (err (R, nb), retained (R, nb), spent (R, nb), completed (R, nb))."""
    rng = np.random.default_rng(seed)
    mu_cheap = np.asarray(mu_cheap, float); p_G = np.asarray(p_G, float)
    K = len(p_G); f = float(np.max(p_G / mu_cheap))
    cum = np.cumsum(mu_cheap); cum[-1] = 1.0
    nb = len(budgets)
    err = np.zeros((R, nb)); kept_n = np.zeros((R, nb)); spent = np.zeros((R, nb)); completed = np.zeros((R, nb), bool)
    for b_i, b in enumerate(budgets):
        M = int(b // ell)
        q = largest_remainder(p_G, int(M / f))
        strata = np.searchsorted(cum, rng.random((R, M)), side="right").clip(0, K - 1)
        onehot = np.zeros((R, M, K), dtype=np.int32)
        np.put_along_axis(onehot, strata[:, :, None], 1, axis=2)
        run_counts = np.cumsum(onehot, axis=1)                      # (R, M, K) counts after each contact
        full = (run_counts >= q[None, None, :]).all(axis=2)
        done = full.any(axis=1)
        stop = np.where(done, np.argmax(full, axis=1) + 1, M)        # contacts actually paid
        kept = np.minimum(run_counts[np.arange(R), stop - 1], q[None, :])
        n = kept.sum(axis=1)
        err[:, b_i] = np.linalg.norm(kept / np.maximum(n, 1)[:, None] - p_G, axis=1)
        kept_n[:, b_i] = n; spent[:, b_i] = stop * ell; completed[:, b_i] = done
    return err, kept_n, spent, completed
