"""Sampling policies, batched over R replications. Every policy exposes choose(t, st, rng) -> (R,) arms."""
import numpy as np
from .geometry import feasible_allocation, linf_projection_allocation, simplex_margin, enumerate_supports
from .theory import ucb_radius


class State:
    def __init__(self, R, m, K):
        self.R, self.m, self.K = R, m, K
        self.S = np.zeros((R, K))            # sum_t (Y_t - p_G)
        self.N = np.zeros((R, m))            # pulls per arm
        self.Ssum = np.zeros((R, m, K))      # sum of observed labels per arm
        self.alloc = np.zeros((R, K))        # sum_t (mu_{A_t} - p_G)   (for the Lemma-1 decomposition)
        self.cost = np.zeros(R)
        self.uncertain = np.zeros(R)         # rounds with r_{A}(N_A) > c/4

    def muhat(self):
        return np.where(self.N[:, :, None] > 0, self.Ssum / np.maximum(self.N[:, :, None], 1), 0.0)


class Policy:
    name = "policy"
    uses_truth = False
    def reset(self, env, T, R): pass
    def choose(self, t, st, rng): raise NotImplementedError


# ---------------- proposed ----------------
class Alg1Sign(Policy):
    """Algorithm 1 (sec. 2). Two arms, binary label; uses only the ordering of the two arms."""
    name = "ALG1"
    def __init__(self, arm_high, arm_low, coord):
        self.hi, self.lo, self.k = arm_high, arm_low, coord
    def choose(self, t, st, rng):
        return np.where(st.S[:, self.k] > 0, self.lo, self.hi)


class Alg2UCB(Policy):
    """Algorithm 2 (sec. 5): index <u_t, muhat_a - p_G> - beta * r_a(N_a); -inf for unsampled arms."""
    def __init__(self, beta=1.0, delta=0.05, arms=None, tag=None):
        self.beta, self.delta, self.arms = beta, delta, arms
        self.name = tag or f"ALG2(beta={beta:g})"
    def reset(self, env, T, R):
        self.env = env
        self.idx = np.arange(env.m) if self.arms is None else np.asarray(self.arms)
        self.c_ref = getattr(env, "c_margin", np.nan)
    def choose(self, t, st, rng):
        e = self.env
        S = st.S[:, :]
        nrm = np.linalg.norm(S, axis=1, keepdims=True)
        u = np.where(nrm > 0, S / np.maximum(nrm, 1e-300), 0.0)
        mh = st.muhat()[:, self.idx, :]
        N = st.N[:, self.idx]
        idxv = np.einsum('rk,rmk->rm', u, mh - e.p_G)
        if self.beta > 0:
            idxv = idxv - ucb_radius(N, e.K, len(self.idx), self.delta, self.beta)
        idxv = np.where(N == 0, -np.inf, idxv)
        a_loc = np.argmin(idxv, axis=1)
        zero = (nrm[:, 0] == 0)
        if zero.any():
            a_loc = np.where(zero, np.argmin(N, axis=1), a_loc)
        a = self.idx[a_loc]
        if np.isfinite(self.c_ref) and self.c_ref > 0:
            st.uncertain += ucb_radius(st.N[np.arange(st.R), a], e.K, len(self.idx), self.delta, 1.0) > self.c_ref / 4
        return a


class MyopicCost(Policy):
    """Cost-ratio heuristic rejected in sec. 6: argmin_a <S_t, muhat_a - p_G> / ell_a."""
    def __init__(self, beta=0.3, delta=0.05, oracle=False, tie_arm=None, tag=None):
        self.beta, self.delta, self.oracle = beta, delta, oracle
        self.tie_arm = tie_arm                  # arm used when S_t = 0 (default: least sampled)
        self.name = tag or "MYOPIC-COST" + ("(oracle)" if oracle else "")
        self.uses_truth = oracle
    def reset(self, env, T, R): self.env = env
    def choose(self, t, st, rng):
        e = self.env
        mh = np.broadcast_to(e.mu, (st.R, e.m, e.K)) if self.oracle else st.muhat()
        drift = np.einsum('rk,rmk->rm', st.S, mh - e.p_G) / e.cost[None, :]
        if not self.oracle and self.beta > 0:
            drift = drift - ucb_radius(st.N, e.K, e.m, self.delta, self.beta) * np.linalg.norm(st.S, axis=1, keepdims=True) / e.cost[None, :]
        drift = np.where(st.N == 0, -np.inf, drift)
        a = np.argmin(drift, axis=1)
        zero = (np.linalg.norm(st.S, axis=1) == 0)
        if zero.any():
            a = np.where(zero, np.argmin(st.N, axis=1) if self.tie_arm is None else self.tie_arm, a)
        return a


class Alg3ThenAlg2(Policy):
    """Algorithm 3 (min-cost support from m*n exploration samples, retained) then Algorithm 2 on it."""
    def __init__(self, n, c0, beta=0.3, delta=0.05, tag=None):
        self.n, self.c0, self.beta, self.delta = n, c0, beta, delta
        self.name = tag or f"ALG3+2(n={n},c0={c0:g})"
    def reset(self, env, T, R):
        self.env = env
        self.supports = enumerate_supports(env.m, env.K)
        self.selected = None
        self.fallback = np.zeros(R, bool)
        self.explore_len = env.m * self.n
    def choose(self, t, st, rng):
        e = self.env
        if t < self.explore_len:
            return np.full(st.R, t % e.m)
        if self.selected is None:
            self.selected, self.fallback = select_supports_batch(
                st.muhat(), e.p_G, e.cost, self.supports, self.c0)
        u = np.where(np.linalg.norm(st.S, axis=1, keepdims=True) > 0,
                     st.S / np.maximum(np.linalg.norm(st.S, axis=1, keepdims=True), 1e-300), 0.0)
        mh = st.muhat()
        idxv = np.einsum('rk,rmk->rm', u, mh - e.p_G) - ucb_radius(st.N, e.K, e.K, self.delta, self.beta)
        mask = np.zeros((st.R, e.m), bool)
        mask[np.arange(st.R)[:, None], self.selected] = True
        mask[self.fallback] = True
        idxv = np.where(mask, idxv, np.inf)
        idxv = np.where((st.N == 0) & mask, -np.inf, idxv)
        return np.argmin(idxv, axis=1)


def select_supports_batch(muhat, p_G, cost, supports, c0):
    """Vectorised Algorithm 3. Returns (selected (R,K) arm indices, fallback mask (R,))."""
    R, m, K = muhat.shape
    best_cost = np.full(R, np.inf)
    best = np.zeros((R, K), dtype=int)
    for I in supports:
        V = muhat[:, list(I), :]                       # (R,K,K) rows = vertices
        A = np.swapaxes(V, 1, 2)                       # (R,K,K) columns = vertices
        rhs = np.broadcast_to(p_G, (R, K))[..., None]
        ok = np.abs(np.linalg.det(A)) > 1e-12
        q = np.full((R, K), -1.0)
        if ok.any():
            q[ok] = np.linalg.solve(A[ok], rhs[ok])[..., 0]
        res = np.linalg.norm(np.einsum('rkj,rj->rk', A, q) - p_G, axis=1)
        h = simplex_heights_batch(V)                   # (R,K)
        marg = np.min(q * h, axis=1)
        feas = (q.min(axis=1) >= 0) & (res < 1e-9) & (marg >= 0.75 * c0)
        c_I = np.where(feas, q @ np.asarray(cost, float)[list(I)], np.inf)
        upd = c_I < best_cost
        best_cost[upd] = c_I[upd]
        best[upd] = np.array(I)
    fallback = ~np.isfinite(best_cost)
    return best, fallback


def simplex_heights_batch(V):
    """Heights h_i = dist(v_i, aff(v_j, j != i)) for batched simplices V (R,K,K) (rows = vertices)."""
    R, K, _ = V.shape
    if K == 1:
        return np.ones((R, 1))
    from math import factorial
    def vol2(Vs):
        k = Vs.shape[1]
        if k == 1:
            return np.ones(Vs.shape[0])
        E = Vs[:, 1:, :] - Vs[:, :1, :]
        G = np.einsum('rik,rjk->rij', E, E)
        return np.sqrt(np.maximum(np.linalg.det(G), 0.0)) / factorial(k - 1)
    Vfull = vol2(V)
    H = np.zeros((R, K))
    for i in range(K):
        Vi = np.delete(V, i, axis=1)
        vf = vol2(Vi)
        H[:, i] = np.where(vf > 0, (K - 1) * Vfull / np.maximum(vf, 1e-300), 0.0)
    return H


# ---------------- oracles ----------------
class OracleCL(Policy):
    """argmin_a <S_t, mu_a - p_G> with the true means (best achievable closed loop)."""
    name = "ORACLE-CL"; uses_truth = True
    def __init__(self, arms=None): self.arms = arms
    def reset(self, env, T, R):
        self.env = env
        self.idx = np.arange(env.m) if self.arms is None else np.asarray(self.arms)
        self.D = (env.mu[self.idx] - env.p_G).T         # (K, m')
    def choose(self, t, st, rng):
        v = st.S @ self.D
        a = self.idx[np.argmin(v, axis=1)]
        zero = (np.linalg.norm(st.S, axis=1) == 0)
        if zero.any():
            a = np.where(zero, self.idx[np.argmin(st.N[:, self.idx], axis=1)], a)
        return a


class OpenLoopHerding(Policy):
    """Deterministic herding on a fixed allocation x: argmax_a (t+1) x_a - N_a. Allocation error O(1/T)."""
    uses_truth = True
    def __init__(self, x, tag="ORACLE-OL"):
        self.x = np.asarray(x, float); self.name = tag
    def choose(self, t, st, rng):
        return np.argmax((t + 1) * self.x[None, :] - st.N, axis=1)


class OpenLoopIID(Policy):
    uses_truth = True
    def __init__(self, x, tag="ORACLE-OL-iid"):
        self.x = np.asarray(x, float); self.name = tag
    def choose(self, t, st, rng):
        return np.searchsorted(np.cumsum(self.x), rng.random(st.R), side="right").clip(0, len(self.x) - 1)


# ---------------- naive / corrections ----------------
class FixedArm(Policy):
    def __init__(self, a, tag): self.a, self.name = a, tag
    def choose(self, t, st, rng): return np.full(st.R, self.a)


class RoundRobin(Policy):
    name = "N2-UNIFORM"
    def reset(self, env, T, R): self.m = env.m
    def choose(self, t, st, rng): return np.full(st.R, t % self.m)


class HerdingAllocation(Policy):
    """Naive allocation proportional to a fixed weight vector (e.g. channel pool sizes)."""
    def __init__(self, w, tag="N3-POOLPROP"):
        self.x = np.asarray(w, float) / np.sum(w); self.name = tag
    def choose(self, t, st, rng):
        return np.argmax((t + 1) * self.x[None, :] - st.N, axis=1)


class ETC(Policy):
    """Explore-then-commit: round-robin n0 per arm (retained), then open-loop herding on the
    allocation that makes the *final* composition equal p_G under the estimated means."""
    def __init__(self, eps=0.1, cost_aware=True, tag=None):
        self.eps, self.cost_aware = eps, cost_aware
        self.name = tag or f"ETC(eps={eps:g})"
    def reset(self, env, T, R):
        self.env, self.T, self.R = env, T, R
        self.n0 = max(1, int(self.eps * T / env.m))
        self.explore_len = self.n0 * env.m
        self.x = None; self.N0 = None
    def choose(self, t, st, rng):
        e = self.env
        if t < self.explore_len:
            return np.full(st.R, t % e.m)
        if self.x is None:
            mh = st.muhat()
            self.N0 = st.N.copy()
            rem = self.T - self.explore_len
            self.x = np.zeros((st.R, e.m))
            explored_mix = np.einsum('rm,rmk->rk', st.N, mh)
            tgt = (self.T * e.p_G[None, :] - explored_mix) / max(rem, 1)
            for r in range(st.R):
                x, _ = feasible_allocation(mh[r], tgt[r], e.cost if self.cost_aware else None)
                if x is None or np.any(np.asarray(x) < -1e-9):
                    x = linf_projection_allocation(mh[r], tgt[r])
                self.x[r] = np.clip(x, 0, None) / max(np.sum(np.clip(x, 0, None)), 1e-12)
        return np.argmax((t + 1 - self.explore_len) * self.x - (st.N - self.N0), axis=1)


class BatchWrapper(Policy):
    """Operational constraint: the allocation decision is revised only every B recruits."""
    def __init__(self, base, B):
        self.base, self.B = base, B
        self.name = f"{base.name}+batch{B}"
        self.uses_truth = base.uses_truth
    def reset(self, env, T, R):
        self.base.reset(env, T, R); self.last = None
    def choose(self, t, st, rng):
        if t % self.B == 0 or self.last is None:
            self.last = self.base.choose(t, st, rng)
        return self.last


class DelayWrapper(Policy):
    """Operational constraint: the stratum of a recruit is revealed only d rounds later."""
    def __init__(self, base, d):
        self.base, self.d = base, d
        self.name = f"{base.name}+delay{d}"
        self.uses_truth = base.uses_truth
    def reset(self, env, T, R):
        self.base.reset(env, T, R); self.buf = []
    def choose(self, t, st, rng):
        import copy
        snap = (st.S.copy(), st.N.copy(), st.Ssum.copy())
        self.buf.append(snap)
        if len(self.buf) > self.d + 1:
            self.buf.pop(0)
        S0, N0, Ss0 = self.buf[0]
        shadow = State(st.R, st.m, st.K)
        shadow.S, shadow.N, shadow.Ssum = S0, N0, Ss0
        a = self.base.choose(t, shadow, rng)
        st.uncertain = shadow.uncertain
        return a
