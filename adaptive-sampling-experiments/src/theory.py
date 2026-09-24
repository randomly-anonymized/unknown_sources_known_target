"""Theoretical constants and reference curves from the manuscript, plus the two lower bounds."""
import numpy as np


# ---------- Theorem 1 (two sources, binary label) ----------
def thm1_rho_C(lam, p_low, p_G, p_high):
    """rho(lam), C(lam) from the potential argument (Appendix A). p_low < p_G < p_high."""
    g_P = p_low * np.exp(lam * (1 - p_G)) + (1 - p_low) * np.exp(-lam * p_G)
    g_O = p_high * np.exp(-lam * (1 - p_G)) + (1 - p_high) * np.exp(lam * p_G)
    rho = max(g_P, g_O)
    if rho >= 1:
        return None
    return rho, 1.0 + np.exp(2 * lam) / (1 - rho)


def thm1_best(p_low, p_G, p_high, lam_grid=None):
    """Best (over lam) bound on E|S_T| and the (lam, rho, C) achieving it."""
    lam_grid = np.linspace(1e-4, 5.0, 50000) if lam_grid is None else lam_grid
    best = None
    for lam in lam_grid:
        r = thm1_rho_C(lam, p_low, p_G, p_high)
        if r is None:
            continue
        rho, C = r
        val = (np.log(C) + 1.0) / lam                      # E|S| <= (log C + 1)/lam by Jensen
        if best is None or val < best[0]:
            best = (val, lam, rho, C)
    return best


def thm1_tail_bound(k, p_low, p_G, p_high, lam_grid=None):
    """inf_lam C(lam) exp(-lam k): the Theorem-1 tail bound, optimised per k."""
    lam_grid = np.linspace(1e-4, 5.0, 4000) if lam_grid is None else lam_grid
    k = np.atleast_1d(np.asarray(k, float))
    best = np.full(k.shape, np.inf)
    for lam in lam_grid:
        r = thm1_rho_C(lam, p_low, p_G, p_high)
        if r is None:
            continue
        _, C = r
        best = np.minimum(best, C * np.exp(-lam * k))
    return np.minimum(best, 1.0)


# ---------- Theorem 2 (multi-dimensional, UCB) ----------
def ucb_radius(n, K, m, delta=0.05, beta=1.0):
    n = np.maximum(np.asarray(n, float), 1.0)
    return beta * np.sqrt((K / (2 * n)) * np.log(np.pi ** 2 * m * K * n ** 2 / (3 * delta)))


def N_of_c(c, K, m, delta=0.05):
    """Smallest n with r(n) <= c/4 (forced-exploration budget per arm in Theorem 2)."""
    if c <= 0:
        return np.inf
    n = 1
    while ucb_radius(n, K, m, delta) > c / 4:
        n = int(n * 1.05) + 1
        if n > 1e12:
            return np.inf
    return n


def thm2_constants(c, R2, K, m, delta=0.05, eta=0.05):
    """Remark-1 constants with R^2 computed from the actual support (see notes: NOT ell_0+1 in general)."""
    if c <= 0:
        return dict(valid=False)
    R = np.sqrt(R2)
    gamma = c / 2
    lam = c / (4 * R2)
    kappa = c ** 2 / (32 * R2)
    rho = np.exp(-kappa)
    L = max(R2 / gamma, gamma)
    B = np.exp(lam * (L + R))
    C = 1 + B / (1 - rho)
    Nc = N_of_c(c, K, m, delta)
    Mc = m * Nc
    C1 = 4 * R2 / c
    C0 = (R + c / 8) * Mc + C1 * np.log(C)
    return dict(valid=True, R2=R2, lam=lam, kappa=kappa, rho=rho, B=B, C=C, N_c=Nc, M_c=Mc,
                C0=C0, C1=C1, bound_S=C0 + C1 * np.log(1 / eta))


def thm3_cost_bound(n, K, m, c0, ell_max, delta=0.05):
    return (3 * ell_max / c0) * np.sqrt((K / (2 * n)) * np.log(2 * m * K / delta))


def thm3_n(c0, K, m, delta=0.05):
    return int(np.ceil((8 * K / c0 ** 2) * np.log(2 * m * K / delta)))


# ---------- lower bounds (see design note 10.3) ----------
def lower_bound_kappa(mu, patterns=None):
    """kappa = (1/2) min_a E||Y - Y'||_2 for i.i.d. Y, Y' ~ arm a.  E||S_T|| >= kappa for every policy.
    For categorical Y this equals (1/sqrt(2)) min_a (1 - ||mu_a||^2)."""
    mu = np.asarray(mu, float)
    if patterns is None:
        return float((1 / np.sqrt(2)) * np.min(1 - (mu ** 2).sum(axis=1)))
    P = np.asarray(patterns, float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)          # pairwise ||y - y'||
    vals = [0.5 * float(p @ D @ p) for p in mu]
    return float(min(vals))


def lower_bound_kappa_binary(p_arms):
    """Sharper 1-D bound: E|S_T| >= min_a min(p_a, 1-p_a)."""
    p = np.asarray(p_arms, float)
    return float(np.min(np.minimum(p, 1 - p)))


def open_loop_constant(x_star, p_arms):
    """nu* from Lemma 8 (binary case); E|p_hat - p_G| ~ sqrt(2 nu*/pi)/sqrt(T) for open-loop policies."""
    x = np.asarray(x_star, float); p = np.asarray(p_arms, float)
    nu = float(np.sum(x * p * (1 - p)))
    return nu, np.sqrt(2 * nu / np.pi)
