"""Sampling environment: m arms, each a distribution over label patterns (+ optional outcomes)."""
import numpy as np


class Environment:
    """arm_probs: (m, P) distribution of each arm over P label patterns.
       patterns:   (P, K) label vectors (one-hot for categorical Y, multi-hot otherwise).
       outcomes:   (P, J) optional real covariates attached to each pattern (respondent mode)."""

    def __init__(self, arm_probs, patterns, p_G, cost, names=None, outcomes=None,
                 outcome_names=None, categorical=True, label_names=None):
        self.arm_probs = np.asarray(arm_probs, float)
        self.patterns = np.asarray(patterns, float)
        self.p_G = np.asarray(p_G, float)
        self.cost = np.asarray(cost, float)
        self.m, self.P = self.arm_probs.shape
        self.K = self.patterns.shape[1]
        self.names = list(names) if names is not None else [f"arm{i}" for i in range(self.m)]
        self.label_names = list(label_names) if label_names is not None else [f"y{k}" for k in range(self.K)]
        self.outcomes = None if outcomes is None else np.asarray(outcomes, float)
        self.outcome_names = list(outcome_names) if outcome_names is not None else []
        self.categorical = categorical
        self.mu = self.arm_probs @ self.patterns                      # (m, K) true arm means
        self._cum = np.cumsum(self.arm_probs, axis=1)
        self._cum[:, -1] = 1.0
        self.R2 = float(np.max(np.sum((self.patterns - self.p_G) ** 2, axis=1)))

    def draw_patterns(self, arms, u):
        """arms: (R,) arm indices, u: (R,) uniforms -> pattern indices (R,)."""
        arms = np.asarray(arms); out = np.empty(arms.shape[0], dtype=np.int64)
        for a in np.unique(arms):
            sel = arms == a
            out[sel] = np.searchsorted(self._cum[a], u[sel], side="right")
        return np.minimum(out, self.P - 1)

    def labels(self, pattern_idx):
        return self.patterns[pattern_idx]

    def outcome(self, pattern_idx):
        return None if self.outcomes is None else self.outcomes[pattern_idx]


def categorical_env(mu, p_G, cost, names=None, label_names=None):
    """Stratum-level environment: patterns are the K unit vectors, arm_probs = mu."""
    mu = np.asarray(mu, float)
    K = mu.shape[1]
    return Environment(mu, np.eye(K), p_G, cost, names=names, categorical=True, label_names=label_names)


class DriftingEnvironment(Environment):
    """Non-stationary channels: arm means move linearly from mu0 to mu1 over the horizon."""
    def __init__(self, mu0, mu1, p_G, cost, T, names=None, label_names=None):
        super().__init__(np.asarray(mu0, float), np.eye(np.asarray(mu0).shape[1]), p_G, cost,
                         names=names, label_names=label_names)
        self.mu0, self.mu1, self.T = np.asarray(mu0, float), np.asarray(mu1, float), T
        self.mu = self.mu0.copy()

    def set_time(self, t):
        w = min(max(t / max(self.T - 1, 1), 0.0), 1.0)
        self.mu = (1 - w) * self.mu0 + w * self.mu1
        self._cum = np.cumsum(self.mu, axis=1)
        self._cum[:, -1] = 1.0
