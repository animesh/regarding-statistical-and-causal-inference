"""
Unit tests for the SCM module (Pearl 2009 simulation).

Tests cover:
* DiscreteDistribution — marginalisation, conditioning, probability queries
* d-separation — fork, chain, collider structures
* Backdoor criterion and adjustment
* Front-door criterion and adjustment
* Verify that adjustments recover ground-truth causal effects
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from scm import (
    SCM,
    DiscreteDistribution,
    d_separated,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def make_uniform_dist(variables):
    """Return a uniform joint distribution over binary variables."""
    n = len(variables)
    cards = {v: 2 for v in variables}
    probs = np.ones(tuple(2 for _ in variables)) / (2 ** n)
    return DiscreteDistribution(variables, cards, probs)


def make_uniform_scm(edges, variables):
    dag = nx.DiGraph(edges)
    dist = make_uniform_dist(variables)
    return SCM("test", dag, dist)


# ────────────────────────────────────────────────────────────────────────────
# DiscreteDistribution tests
# ────────────────────────────────────────────────────────────────────────────

class TestDiscreteDistribution:

    def test_sum_to_one(self):
        dist = make_uniform_dist(["X", "Y"])
        assert np.isclose(dist._table.sum(), 1.0)

    def test_marginal(self):
        # Uniform over X, Y → marginals are also uniform
        dist = make_uniform_dist(["X", "Y"])
        m = dist.marginal(["X"])
        assert m.variables == ["X"]
        assert np.isclose(m.p({"X": 0}), 0.5)
        assert np.isclose(m.p({"X": 1}), 0.5)

    def test_condition(self):
        # Chain: P(Z=1)=0.5, P(X=1|Z)=0.7 if Z=1 else 0.3
        p_Z = np.array([0.5, 0.5])
        p_X_Z = np.array([[0.7, 0.3], [0.3, 0.7]])
        joint = np.zeros((2, 2))
        for z in range(2):
            for x in range(2):
                joint[z, x] = p_Z[z] * p_X_Z[z, x]
        dist = DiscreteDistribution(["Z", "X"], {"Z": 2, "X": 2}, joint)
        c = dist.condition({"Z": 1})
        assert c.variables == ["X"]
        assert np.isclose(c.p({"X": 1}), 0.7)
        assert np.isclose(c.p({"X": 0}), 0.3)

    def test_conditional_p(self):
        dist = make_uniform_dist(["X", "Y"])
        # Uniform: P(Y=1|X=0) = P(X=0,Y=1) / P(X=0) = 0.25 / 0.5 = 0.5
        assert np.isclose(dist.conditional_p({"Y": 1}, {"X": 0}), 0.5)

    def test_joint_p(self):
        dist = make_uniform_dist(["X", "Y"])
        assert np.isclose(dist.p({"X": 0, "Y": 1}), 0.25)

    def test_to_dataframe(self):
        dist = make_uniform_dist(["X"])
        df = dist.to_dataframe()
        assert list(df.columns) == ["X", "P"]
        assert len(df) == 2
        assert np.isclose(df["P"].sum(), 1.0)


# ────────────────────────────────────────────────────────────────────────────
# d-separation tests
# ────────────────────────────────────────────────────────────────────────────

class TestDSeparation:

    def test_fork_unconditional(self):
        # Z -> X, Z -> Y: X and Y are NOT d-separated unconditionally
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
        assert d_separated(dag, ["X"], ["Y"]) is False

    def test_fork_conditional(self):
        # Z -> X, Z -> Y: X and Y ARE d-separated given Z
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
        assert d_separated(dag, ["X"], ["Y"], ["Z"]) is True

    def test_chain_unconditional(self):
        # X -> Z -> Y: X and Y are NOT d-separated unconditionally
        dag = nx.DiGraph([("X", "Z"), ("Z", "Y")])
        assert d_separated(dag, ["X"], ["Y"]) is False

    def test_chain_conditional(self):
        # X -> Z -> Y: X and Y ARE d-separated given Z
        dag = nx.DiGraph([("X", "Z"), ("Z", "Y")])
        assert d_separated(dag, ["X"], ["Y"], ["Z"]) is True

    def test_collider_unconditional(self):
        # X -> Z <- Y: X and Y ARE d-separated unconditionally (collider blocks)
        dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
        assert d_separated(dag, ["X"], ["Y"]) is True

    def test_collider_conditional(self):
        # X -> Z <- Y: conditioning on Z OPENS the path → NOT d-separated
        dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
        assert d_separated(dag, ["X"], ["Y"], ["Z"]) is False

    def test_scm_wrapper(self):
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y")])
        model = make_uniform_scm([("Z", "X"), ("Z", "Y")], ["Z", "X", "Y"])
        assert model.is_d_separated("X", "Y") is False
        assert model.is_d_separated("X", "Y", "Z") is True


# ────────────────────────────────────────────────────────────────────────────
# Backdoor criterion tests
# ────────────────────────────────────────────────────────────────────────────

class TestBackdoorCriterion:

    def _make_model(self):
        """Z -> X, Z -> Y, X -> Y"""
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dist = make_uniform_dist(["Z", "X", "Y"])
        return SCM("bd_test", dag, dist)

    def test_z_satisfies_backdoor(self):
        model = self._make_model()
        assert model.satisfies_backdoor("X", "Y", ["Z"]) is True

    def test_empty_set_fails_backdoor(self):
        model = self._make_model()
        assert model.satisfies_backdoor("X", "Y", []) is False

    def test_descendant_fails_backdoor(self):
        # Y is a descendant of X; using Y as adjustment violates condition (i)
        model = self._make_model()
        assert model.satisfies_backdoor("X", "Y", ["Y"]) is False

    def test_backdoor_adjustment_equals_ground_truth(self):
        """Backdoor adjustment should recover P(Y|do(X)) exactly."""
        # Ground-truth parameters
        p_Z1 = 0.40
        p_X_Z = {0: 0.25, 1: 0.75}
        p_Y_XZ = {(0, 0): 0.40, (1, 0): 0.60,
                  (0, 1): 0.20, (1, 1): 0.50}

        joint = np.zeros((2, 2, 2))
        for z in range(2):
            pz = p_Z1 if z == 1 else 1 - p_Z1
            for x in range(2):
                px_z = p_X_Z[z] if x == 1 else 1 - p_X_Z[z]
                for y in range(2):
                    py_xz = p_Y_XZ[(x, z)] if y == 1 else 1 - p_Y_XZ[(x, z)]
                    joint[z, x, y] = pz * px_z * py_xz

        dist = DiscreteDistribution(["Z", "X", "Y"], {"Z": 2, "X": 2, "Y": 2}, joint)
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        model = SCM("bd_test", dag, dist)

        # Ground truth
        true_do1 = p_Y_XZ[(1, 0)] * (1 - p_Z1) + p_Y_XZ[(1, 1)] * p_Z1
        true_do0 = p_Y_XZ[(0, 0)] * (1 - p_Z1) + p_Y_XZ[(0, 1)] * p_Z1

        adj_1 = model.backdoor_adjustment("X", "Y", ["Z"], 1, 1)
        adj_0 = model.backdoor_adjustment("X", "Y", ["Z"], 0, 1)

        assert np.isclose(adj_1, true_do1, atol=1e-9)
        assert np.isclose(adj_0, true_do0, atol=1e-9)

    def test_simpsons_paradox(self):
        """
        Classic Simpson's Paradox: crude effect is negative (drug appears
        harmful), but backdoor-adjusted effect is positive (drug is beneficial).
        """
        # Healthy (G=0) rarely take drug; Sick (G=1) often take drug
        data = {
            (0, 0): [90,  360],   # healthy, no drug
            (0, 1): [5,   45],    # healthy, drug
            (1, 0): [40,  10],    # sick, no drug
            (1, 1): [225, 225],   # sick, drug
        }
        total = sum(v[0] + v[1] for v in data.values())
        joint = np.zeros((2, 2, 2))
        for (g, x), counts in data.items():
            for y, cnt in enumerate(counts):
                joint[g, x, y] = cnt / total

        dist = DiscreteDistribution(["G", "X", "Y"], {"G": 2, "X": 2, "Y": 2}, joint)
        dag = nx.DiGraph([("G", "X"), ("G", "Y"), ("X", "Y")])
        model = SCM("Simpson", dag, dist)

        crude_drug   = dist.conditional_p({"Y": 1}, {"X": 1})
        crude_nodrug = dist.conditional_p({"Y": 1}, {"X": 0})
        crude_effect = crude_drug - crude_nodrug

        causal_drug   = model.backdoor_adjustment("X", "Y", ["G"], 1, 1)
        causal_nodrug = model.backdoor_adjustment("X", "Y", ["G"], 0, 1)
        causal_effect = causal_drug - causal_nodrug

        # The paradox: crude shows harm, causal shows benefit
        assert crude_effect < 0, f"Expected negative crude effect, got {crude_effect}"
        assert causal_effect > 0, f"Expected positive causal effect, got {causal_effect}"


# ────────────────────────────────────────────────────────────────────────────
# Front-door criterion tests
# ────────────────────────────────────────────────────────────────────────────

class TestFrontdoorCriterion:

    def _make_fd_model(self):
        """X -> M -> Y (no hidden confounder in observable skeleton)."""
        dag = nx.DiGraph([("X", "M"), ("M", "Y")])
        dist = make_uniform_dist(["X", "M", "Y"])
        return SCM("fd_test", dag, dist)

    def test_m_satisfies_frontdoor(self):
        model = self._make_fd_model()
        assert model.satisfies_frontdoor("X", "Y", ["M"]) is True

    def test_direct_path_fails_frontdoor(self):
        # If X -> Y is present, M does NOT intercept all causal paths
        dag = nx.DiGraph([("X", "M"), ("M", "Y"), ("X", "Y")])
        dist = make_uniform_dist(["X", "M", "Y"])
        model = SCM("fd_fail", dag, dist)
        assert model.satisfies_frontdoor("X", "Y", ["M"]) is False

    def test_frontdoor_adjustment_equals_ground_truth(self):
        """Front-door adjustment recovers P(Y|do(X)) despite hidden confounder."""
        p_U_fd = np.array([0.5, 0.5])
        p_X_U_fd = {0: 0.15, 1: 0.70}
        p_M_X_fd = {0: 0.05, 1: 0.90}
        p_Y_MU_fd = {(0, 0): 0.02, (1, 0): 0.20,
                     (0, 1): 0.20, (1, 1): 0.65}

        joint_full = np.zeros((2, 2, 2, 2))
        for u in range(2):
            for x in range(2):
                for m in range(2):
                    for y in range(2):
                        px_u  = p_X_U_fd[u] if x == 1 else 1 - p_X_U_fd[u]
                        pm_x  = p_M_X_fd[x] if m == 1 else 1 - p_M_X_fd[x]
                        py_mu = p_Y_MU_fd[(m, u)] if y == 1 else 1 - p_Y_MU_fd[(m, u)]
                        joint_full[u, x, m, y] = p_U_fd[u] * px_u * pm_x * py_mu

        dist_full = DiscreteDistribution(
            ["U", "X", "M", "Y"], {"U": 2, "X": 2, "M": 2, "Y": 2}, joint_full
        )
        dist_obs = dist_full.marginal(["X", "M", "Y"])

        dag_obs = nx.DiGraph([("X", "M"), ("M", "Y")])
        model = SCM("fd_test", dag_obs, dist_obs)

        def ground_truth(x_val):
            result = 0.0
            for u in range(2):
                for m in range(2):
                    pm_x  = p_M_X_fd[x_val] if m == 1 else 1 - p_M_X_fd[x_val]
                    py_mu = p_Y_MU_fd[(m, u)]
                    result += p_U_fd[u] * pm_x * py_mu
            return result

        true_1 = ground_truth(1)
        true_0 = ground_truth(0)
        fd_1 = model.frontdoor_adjustment("X", "Y", ["M"], 1, 1)
        fd_0 = model.frontdoor_adjustment("X", "Y", ["M"], 0, 1)

        assert np.isclose(fd_1, true_1, atol=1e-9), f"{fd_1} != {true_1}"
        assert np.isclose(fd_0, true_0, atol=1e-9), f"{fd_0} != {true_0}"

    def test_invalid_z_raises_for_backdoor(self):
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        dist = make_uniform_dist(["Z", "X", "Y"])
        model = SCM("test", dag, dist)
        with pytest.raises(ValueError, match="backdoor criterion"):
            model.backdoor_adjustment("X", "Y", [], 1, 1)


# ────────────────────────────────────────────────────────────────────────────
# Integration: verify consistency of probability tables
# ────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_chain_marginals_consistent(self):
        """Marginalising a chain SCM gives consistent marginals."""
        p_Z = np.array([0.5, 0.5])
        p_X_Z = np.array([[0.7, 0.3], [0.3, 0.7]])
        p_Y_X = np.array([[0.8, 0.2], [0.2, 0.8]])
        joint = np.zeros((2, 2, 2))
        for z in range(2):
            for x in range(2):
                for y in range(2):
                    joint[z, x, y] = p_Z[z] * p_X_Z[z, x] * p_Y_X[x, y]
        dist = DiscreteDistribution(["Z", "X", "Y"], {"Z": 2, "X": 2, "Y": 2}, joint)
        # P(X) = Σ_z P(Z=z) * P(X|Z=z)
        p_X1_expected = 0.5 * 0.3 + 0.5 * 0.7  # = 0.5
        p_X1_actual = dist.marginal(["X"]).p({"X": 1})
        assert np.isclose(p_X1_actual, p_X1_expected)

    def test_backdoor_distribution_sums_to_one(self):
        """Backdoor-adjusted P(Y|do(X)) should sum to 1 over Y."""
        p_Z1 = 0.4
        p_X_Z = {0: 0.25, 1: 0.75}
        p_Y_XZ = {(0, 0): 0.40, (1, 0): 0.60,
                  (0, 1): 0.20, (1, 1): 0.50}
        joint = np.zeros((2, 2, 2))
        for z in range(2):
            pz = p_Z1 if z == 1 else 1 - p_Z1
            for x in range(2):
                px_z = p_X_Z[z] if x == 1 else 1 - p_X_Z[z]
                for y in range(2):
                    py_xz = p_Y_XZ[(x, z)] if y == 1 else 1 - p_Y_XZ[(x, z)]
                    joint[z, x, y] = pz * px_z * py_xz
        dist = DiscreteDistribution(["Z", "X", "Y"], {"Z": 2, "X": 2, "Y": 2}, joint)
        dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        model = SCM("bd", dag, dist)
        bd_dist = model.backdoor_distribution("X", "Y", ["Z"], 1)
        assert np.isclose(bd_dist.sum(), 1.0)
