"""Fast correctness tests for the simulation stack (run: python3 tests/test_all.py)."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.geometry import margin, margin_bruteforce, simplex_margin, feasible_allocation
from src.env import categorical_env
from src.policies import Alg1Sign, OracleCL, OpenLoopHerding, RoundRobin, select_supports_batch, simplex_heights_batch
from src.runner import run, quota_discard, largest_remainder, poststratification_stats, lp_discard_cost
from src.theory import thm1_best, lower_bound_kappa_binary
from src.geometry import enumerate_supports, support_table

ok = lambda name: print(f"  PASS  {name}")


def t1_margin():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu = rng.dirichlet(np.ones(4), size=8); p = mu.mean(0)
        a, b = margin(mu, p), margin_bruteforce(mu, p, n_dir=60000)
        assert a <= b + 1e-6 and (b - a) < 0.10 * max(b, 1e-3) + 1e-3, (a, b)  # brute force is an upper bound, converges slowly
    V = rng.dirichlet(np.ones(4), size=4)
    assert abs(simplex_margin(V, V.mean(0))[0] - margin(V, V.mean(0))) < 1e-9
    boundary = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    assert abs(margin(boundary, np.array([0.5, 0.5, 0.0]))) < 1e-9
    ok("margin() matches brute force; simplex margin agrees; boundary margin = 0")


def t2_lemma1():
    """p_hat - p_G = (y_T - x*)(p_O - p_P) + (1/T) sum (Y_t - p_{A_t}) to machine precision."""
    mu = np.array([[0.30, 0.70], [0.78, 0.22]]); p_G = np.array([0.47, 0.53])
    env = categorical_env(mu, p_G, np.array([72., 199.]))
    df, ex = run(env, Alg1Sign(1, 0, 0), 2000, 64, seed=3, checkpoints=[2000], keep_paths=(2000,))
    st = ex["state"]; T = 2000
    x_star = (p_G[0] - mu[0, 0]) / (mu[1, 0] - mu[0, 0])
    y_T = st.N[:, 1] / T
    lhs = st.S[:, 0] / T
    rhs = (y_T - x_star) * (mu[1, 0] - mu[0, 0]) + (st.S[:, 0] - st.alloc[:, 0]) / T
    assert np.abs(lhs - rhs).max() < 1e-12, np.abs(lhs - rhs).max()
    ok("Lemma 1 decomposition exact")


def t3_herding():
    mu = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]]); p_G = np.array([0.5, 0.5])
    env = categorical_env(mu, p_G, np.ones(3))
    x = np.array([0.3, 0.45, 0.25])
    df, ex = run(env, OpenLoopHerding(x), 5000, 8, seed=4, checkpoints=[5000], keep_paths=(5000,))
    N = ex["state"].N / 5000
    assert np.abs(N - x).max() <= 1.0 / 5000 + 1e-9
    ok("open-loop herding allocation error <= 1/T")


def t4_thm1():
    mu = np.array([[0.2, 0.8], [0.8, 0.2]]); p_G = np.array([0.4, 0.6])
    env = categorical_env(mu, p_G, np.array([1., 3.]))
    bound = thm1_best(0.2, 0.4, 0.8)[0]
    for T in [500, 5000]:
        df, ex = run(env, Alg1Sign(1, 0, 0), T, 400, seed=5, checkpoints=[T], keep_paths=(T,))
        ES = np.abs(ex["paths"][T]["S"][:, 0]).mean()
        assert ES <= bound, (T, ES, bound)
        assert ES >= 0.5 * lower_bound_kappa_binary(mu[:, 0])
    ok("Algorithm 1: empirical E|S_T| inside [kappa, Thm-1 bound]")


def t5_alg3_limit():
    sc = json.load(open(f"{ROOT}/data/scenarios/S-DIAB__L2_K3_eth.json"))
    mu, p_G, cost = np.array(sc['mu']), np.array(sc['p_G']), np.array(sc['cost'])
    sup = enumerate_supports(mu.shape[0], mu.shape[1]); c0 = 0.05
    truth = support_table(mu, p_G, cost, sup)
    best = min([t for t in truth if np.isfinite(t[3]) and t[1] >= 0.75 * c0], key=lambda z: z[3])[0]
    sel, fb = select_supports_batch(np.broadcast_to(mu, (3,) + mu.shape).copy(), p_G, cost, sup, c0)
    assert not fb.any() and tuple(sorted(sel[0])) == tuple(sorted(best)), (sel[0], best)
    ok("Algorithm 3 with exact means returns argmin{C(I) : c(I) >= 3c0/4}")


def t6_weights():
    p_G = np.array([0.2, 0.5, 0.3])
    counts = np.array([[100, 300, 600], [200, 500, 300]])
    deff, ess, fail = poststratification_stats(counts, p_G)
    assert not fail.any() and abs(deff[1] - 1.0) < 1e-12 and deff[0] > 1
    w = p_G / (counts[0] / counts[0].sum())
    assert abs((counts[0] * w).sum() / counts[0].sum() - 1) < 1e-12
    ok("post-stratification: weights reproduce p_G, deff = 1 on a perfect sample")


def t7_quota():
    mu = np.array([[0.4, 0.6], [0.8, 0.2]]); p_G = np.array([0.5, 0.5])
    env = categorical_env(mu, p_G, np.array([72., 199.]))
    contacts, quota = quota_discard(env, 0, 1000, 200, seed=6)
    assert quota.sum() == 1000 and abs(quota / 1000 - p_G).max() <= 1e-3
    expected = 1000 * max(p_G / mu[0])
    assert abs(contacts.mean() - expected) / expected < 0.05
    assert lp_discard_cost(mu, p_G, [72., 199.])[0] <= feasible_allocation(mu, p_G, [72., 199.])[1] + 1e-9
    ok("quota baseline: exact quotas, contacts match theory, LP-discard <= LP-no-discard")


def t8_sampler():
    rng = np.random.default_rng(7)
    mu = rng.dirichlet(np.ones(5), size=3)
    env = categorical_env(mu, mu.mean(0), np.ones(3))
    n = 200_000
    for a in range(3):
        pat = env.draw_patterns(np.full(n, a), rng.random(n))
        emp = np.bincount(pat, minlength=5) / n
        se = np.sqrt(mu[a] * (1 - mu[a]) / n)
        assert np.all(np.abs(emp - mu[a]) < 4 * se + 1e-12)
    ok("sampler reproduces arm distributions within 4 s.e.")


def t9_bookkeeping():
    sc = json.load(open(f"{ROOT}/data/scenarios/S-DIAB__L2_K3_eth.json"))
    env = categorical_env(np.array(sc['mu']), np.array(sc['p_G']), np.array(sc['cost']), names=sc['arms'])
    env.c_margin = sc['margin_c']
    from src.policies import Alg2UCB
    for pol in [Alg2UCB(0.3), OracleCL(), RoundRobin()]:
        df, ex = run(env, pol, 500, 16, seed=8, checkpoints=[500], keep_paths=(500,))
        st = ex["state"]
        assert np.allclose(st.N.sum(1), 500)
        assert np.allclose(st.Ssum.sum(axis=(1, 2)), 500)
        assert np.allclose(st.S, st.Ssum.sum(1) - 500 * env.p_G)
        assert np.allclose(st.cost, (st.N * env.cost).sum(1))
    ok("state bookkeeping consistent for every policy")


if __name__ == "__main__":
    print("running tests")
    for f in [t1_margin, t2_lemma1, t3_herding, t4_thm1, t5_alg3_limit, t6_weights, t7_quota, t8_sampler, t9_bookkeeping]:
        f()
    print("all tests passed")
