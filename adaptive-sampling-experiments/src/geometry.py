"""Convex-geometry utilities: affine-hull margins, feasible/min-cost allocations, support enumeration."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog, nnls


def hyperplane_basis(K):
    """Orthonormal basis (K x (K-1)) of {x in R^K : sum(x) = 0}."""
    A = np.concatenate([np.ones((K, 1)), np.eye(K)[:, :K - 1]], axis=1)
    Q, _ = np.linalg.qr(A)
    return Q[:, 1:K]


def margin(mu, p_G, categorical=True):
    """Signed radius of the largest ball centred at p_G inside conv(mu), in the ambient l2 metric
    but restricted to the affine hull (relative interior when Y is categorical).
    > 0: interior with that margin.  < 0: p_G outside (magnitude ~ distance to the hull)."""
    mu = np.atleast_2d(np.asarray(mu, float))
    p_G = np.asarray(p_G, float)
    K = mu.shape[1]
    P = (mu - p_G) @ hyperplane_basis(K) if categorical else (mu - p_G)
    d = P.shape[1]
    if d == 1:
        lo, hi = P[:, 0].min(), P[:, 0].max()
        return float(min(-lo, hi)) if (lo < 0 < hi) else -float(min(abs(lo), abs(hi)))
    if P.shape[0] <= d:
        return -float(np.linalg.norm(P, axis=1).min())
    try:
        h = ConvexHull(P)
    except Exception:
        return -float(np.linalg.norm(P, axis=1).min())
    A, b = h.equations[:, :-1], h.equations[:, -1]
    dist = -b / np.linalg.norm(A, axis=1)
    return float(dist.min())


def margin_bruteforce(mu, p_G, n_dir=20000, seed=0, categorical=True):
    """c = min_{||u||=1} max_a <u, mu_a - p_G> over affine-hull directions. Test oracle for margin()."""
    mu = np.asarray(mu, float); p_G = np.asarray(p_G, float); K = mu.shape[1]
    B = hyperplane_basis(K) if categorical else np.eye(K)
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_dir, B.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return float(np.min(np.max((U @ B.T) @ (mu - p_G).T, axis=1)))


def simplex_margin(V, p_G):
    """(margin, barycentric weights) of p_G in the simplex with vertices V (K x K)."""
    V = np.asarray(V, float); p_G = np.asarray(p_G, float); K = V.shape[1]
    A = np.vstack([V.T, np.ones(K)])
    b = np.append(p_G, 1.0)
    q, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(A @ q - b) > 1e-9:
        return -np.inf, None
    if q.min() < 0:
        return -abs(q.min()), q
    marg = []
    for i in range(K):
        others = np.delete(V, i, axis=0)
        base = others[0]
        D = (others[1:] - base).T
        if D.size:
            Q, _ = np.linalg.qr(D)
            d = V[i] - base
            h = np.linalg.norm(d - Q @ (Q.T @ d))
        else:
            h = np.linalg.norm(V[i] - base)
        marg.append(q[i] * h)
    return float(min(marg)), q


def feasible_allocation(mu, p_G, cost=None):
    """Min-cost (or any) feasible allocation x in the simplex with sum_a x_a mu_a = p_G."""
    mu = np.asarray(mu, float); m = mu.shape[0]
    c = np.zeros(m) if cost is None else np.asarray(cost, float)
    r = linprog(c, A_eq=np.vstack([mu.T, np.ones(m)]), b_eq=np.append(p_G, 1.0), bounds=[(0, 1)] * m)
    if r.status != 0:
        return None, np.inf
    return r.x, float(np.asarray(c, float) @ r.x)


def linf_projection_allocation(mu, target):
    """argmin_x ||sum_a x_a mu_a - target||_inf over the simplex (LP). Always returns a point."""
    mu = np.asarray(mu, float); m, K = mu.shape
    # variables [x (m), t]
    c = np.zeros(m + 1); c[-1] = 1.0
    A_ub = np.vstack([np.hstack([mu.T, -np.ones((K, 1))]),
                      np.hstack([-mu.T, -np.ones((K, 1))])])
    b_ub = np.concatenate([target, -target])
    A_eq = np.zeros((1, m + 1)); A_eq[0, :m] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                bounds=[(0, 1)] * m + [(0, None)])
    return r.x[:m] if r.status == 0 else np.full(m, 1.0 / m)


def closest_point_in_hull(mu, p_G):
    """(distance from p_G to conv(mu), minimising mixture)."""
    mu = np.asarray(mu, float); m = mu.shape[0]
    big = 1e3
    A = np.vstack([mu.T, big * np.ones((1, m))])
    b = np.append(p_G, big)
    x, _ = nnls(A, b)
    s = x.sum()
    x = x / s if s > 0 else np.full(m, 1.0 / m)
    return float(np.linalg.norm(mu.T @ x - p_G)), x


def enumerate_supports(m, K):
    return list(itertools.combinations(range(m), K))


def support_table(mu, p_G, cost, supports):
    """[(I, margin, q, cost)] for every K-subset, using the given (true or estimated) means."""
    out = []
    for I in supports:
        c_I, q = simplex_margin(mu[list(I)], p_G)
        if q is None or q.min() < 0:
            out.append((I, c_I if q is not None else -np.inf, None, np.inf))
        else:
            out.append((I, c_I, q, float(np.asarray(cost, float)[list(I)] @ q)))
    return out
