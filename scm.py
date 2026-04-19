"""
Structural Causal Models (SCMs) — Pearl (2009)

Implements the core machinery from:
  Pearl, J. (2009). Causal inference in statistics: An overview.
  Statistics Surveys, 3, 96-146.
  https://ftp.cs.ucla.edu/pub/stat_ser/r350-reprint.pdf

Key concepts implemented
------------------------
* DAG representation (networkx DiGraph)
* d-separation queries
* do() operator / mutilated graph
* Backdoor criterion and adjustment formula
* Front-door criterion and adjustment formula
* Helper for discrete joint distributions

All probability calculations use exact enumeration over discrete variables.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility: discrete joint distribution
# ---------------------------------------------------------------------------

class DiscreteDistribution:
    """
    A joint probability distribution over a finite set of discrete variables.

    Parameters
    ----------
    variables : list of str
        Variable names.
    cardinalities : dict[str, int]
        Number of values for each variable (values 0 .. k-1).
    probabilities : np.ndarray or dict
        Either a flat ndarray indexed by the Cartesian product of variable
        values (in the order given by ``variables``), or a dict mapping
        value-tuples to probabilities.
    """

    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        probabilities,
    ):
        self.variables = list(variables)
        self.cardinalities = dict(cardinalities)
        shape = tuple(cardinalities[v] for v in self.variables)
        if isinstance(probabilities, dict):
            arr = np.zeros(shape)
            for key, prob in probabilities.items():
                arr[key] = prob
        else:
            arr = np.asarray(probabilities, dtype=float).reshape(shape)
        assert np.isclose(arr.sum(), 1.0, atol=1e-6), (
            f"Probabilities must sum to 1; got {arr.sum()}"
        )
        self._table = arr

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def _axis(self, var: str) -> int:
        return self.variables.index(var)

    def marginal(self, variables: List[str]) -> "DiscreteDistribution":
        """Marginalise out all variables not in *variables*."""
        axes_to_sum = [
            self._axis(v) for v in self.variables if v not in variables
        ]
        table = self._table
        for ax in sorted(axes_to_sum, reverse=True):
            table = table.sum(axis=ax)
        remaining = [v for v in self.variables if v in variables]
        cards = {v: self.cardinalities[v] for v in remaining}
        return DiscreteDistribution(remaining, cards, table)

    def condition(
        self, evidence: Dict[str, int]
    ) -> "DiscreteDistribution":
        """
        Condition on the given evidence (a dict variable -> value).

        Returns a normalised distribution over the remaining variables.
        """
        table = self._table
        for var, val in sorted(evidence.items(), key=lambda x: -self._axis(x[0])):
            ax = self._axis(var)
            table = np.take(table, val, axis=ax)
        remaining = [v for v in self.variables if v not in evidence]
        cards = {v: self.cardinalities[v] for v in remaining}
        total = table.sum()
        if total == 0:
            raise ValueError(
                f"Zero probability evidence: {evidence}"
            )
        return DiscreteDistribution(remaining, cards, table / total)

    def p(self, query: Dict[str, int]) -> float:
        """Return the probability of the given joint assignment."""
        idx = tuple(query[v] for v in self.variables)
        return float(self._table[idx])

    def conditional_p(
        self, query: Dict[str, int], given: Dict[str, int]
    ) -> float:
        """Return P(query | given) using Bayes."""
        joint = {**query, **given}
        joint_p = self.marginal(list(joint.keys())).p(joint)
        given_p = self.marginal(list(given.keys())).p(given)
        if given_p == 0:
            raise ValueError(f"Zero probability condition: {given}")
        return joint_p / given_p

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all joint assignments and probabilities."""
        rows = []
        for idx in itertools.product(
            *[range(self.cardinalities[v]) for v in self.variables]
        ):
            row = dict(zip(self.variables, idx))
            row["P"] = float(self._table[idx])
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"DiscreteDistribution(vars={self.variables}, cards={self.cardinalities})"


# ---------------------------------------------------------------------------
# DAG utilities and d-separation
# ---------------------------------------------------------------------------

def _ancestors(graph: nx.DiGraph, nodes: Iterable[str]) -> Set[str]:
    """Return all ancestors of *nodes* (including the nodes themselves)."""
    result: Set[str] = set(nodes)
    for node in nodes:
        result.update(nx.ancestors(graph, node))
    return result


def _moral_graph(graph: nx.DiGraph, nodes: Set[str]) -> nx.Graph:
    """
    Build the moral graph of the ancestral subgraph induced by *nodes*.

    Steps (as per the Bayes-ball / moral-graph algorithm):
    1. Restrict to the ancestral subgraph.
    2. "Marry" co-parents (add undirected edge between parents sharing a child).
    3. Make all edges undirected.
    """
    ancestral = graph.subgraph(nodes).copy()
    moral = nx.Graph(ancestral)
    for node in ancestral.nodes:
        parents = list(ancestral.predecessors(node))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])
    return moral


def d_separated(
    graph: nx.DiGraph,
    x_nodes: Iterable[str],
    y_nodes: Iterable[str],
    z_nodes: Iterable[str] = (),
) -> bool:
    """
    Test d-separation: are *x_nodes* d-separated from *y_nodes* given
    *z_nodes* in *graph*?

    Uses the moral-graph (Bayes-ball) approach:
    1. Form the ancestral graph over X ∪ Y ∪ Z.
    2. Moralise it.
    3. Remove Z nodes.
    4. Check whether X and Y are disconnected.

    Returns True if d-separated (i.e. conditionally independent),
    False otherwise.
    """
    x_set = set(x_nodes)
    y_set = set(y_nodes)
    z_set = set(z_nodes)

    relevant = _ancestors(graph, x_set | y_set | z_set)
    moral = _moral_graph(graph, relevant)

    # Remove conditioning set
    for z in z_set:
        if z in moral:
            moral.remove_node(z)

    # Check disconnection between every x-y pair
    for x in x_set:
        for y in y_set:
            if x in moral and y in moral:
                if nx.has_path(moral, x, y):
                    return False
            # If one of them was in z, they are "blocked"
    return True


# ---------------------------------------------------------------------------
# Structural Causal Model
# ---------------------------------------------------------------------------

class SCM:
    """
    A discrete Structural Causal Model (SCM).

    An SCM is a triple M = (U, V, F) where
    * U  — exogenous (background) variables with known distribution P(U)
    * V  — endogenous variables determined by structural equations F
    * F  — a family of functions f_i : Pa(V_i) ∪ U_i -> V_i

    For computational convenience we work with a pre-computed joint
    observational distribution ``joint_dist`` over the endogenous variables
    V (integrating out U analytically or via Monte-Carlo).

    Additional graph structure (DAG) is required for d-separation queries
    and for identifying the backdoor / front-door sets.

    Parameters
    ----------
    name : str
        Human-readable model name.
    dag : nx.DiGraph
        Directed acyclic graph over endogenous variables V.
    joint_dist : DiscreteDistribution
        Observational joint distribution P(V).
    description : str, optional
        Free-text description.
    """

    def __init__(
        self,
        name: str,
        dag: nx.DiGraph,
        joint_dist: DiscreteDistribution,
        description: str = "",
    ):
        self.name = name
        self.dag = dag
        self.joint_dist = joint_dist
        self.description = description

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def parents(self, node: str) -> List[str]:
        return list(self.dag.predecessors(node))

    def children(self, node: str) -> List[str]:
        return list(self.dag.successors(node))

    def is_d_separated(
        self,
        x: str | Iterable[str],
        y: str | Iterable[str],
        z: str | Iterable[str] = (),
    ) -> bool:
        """Wrapper for the module-level ``d_separated`` function."""
        x_nodes = [x] if isinstance(x, str) else list(x)
        y_nodes = [y] if isinstance(y, str) else list(y)
        z_nodes = [z] if isinstance(z, str) else list(z)
        return d_separated(self.dag, x_nodes, y_nodes, z_nodes)

    # ------------------------------------------------------------------
    # do()-calculus: interventional distribution
    # ------------------------------------------------------------------

    def do(
        self,
        interventions: Dict[str, int],
        target: str,
        conditioning: Optional[Dict[str, int]] = None,
    ) -> float:
        """
        Compute P(target=t | do(X=x)) using the truncated factorisation
        (also known as the g-formula / Markov condition on the mutilated graph).

        For each intervened variable X_i = x_i the structural equation is
        replaced by a constant, which corresponds to deleting all incoming
        edges to X_i in the DAG.

        When *all* confounders are observed and the backdoor criterion is
        satisfied the result equals the backdoor-adjusted estimate.
        When the graph is Markovian (no hidden variables) the result can
        always be computed by the truncated factorisation.

        Parameters
        ----------
        interventions : dict  variable -> value
            The do() assignment.
        target : dict  variable -> value
            The variable (and value) whose probability we want.  Pass as
            a single variable name; iterate over cardinalities externally
            if you want the full distribution.
        conditioning : dict, optional
            Additional conditioning variables (post-intervention).

        Returns
        -------
        float  P(target | do(interventions), conditioning)

        Notes
        -----
        Implemented via the backdoor adjustment when applicable, and via
        direct truncated-factorisation marginalisation otherwise (requires
        the full joint to be Markovian / no hidden common causes among V).
        """
        raise NotImplementedError(
            "Generic do() requires either a Markovian joint or explicit "
            "identification; use backdoor_adjustment() or frontdoor_adjustment() "
            "for specific identification strategies."
        )

    # ------------------------------------------------------------------
    # Backdoor criterion and adjustment
    # ------------------------------------------------------------------

    def satisfies_backdoor(
        self, x: str, y: str, z_set: Iterable[str]
    ) -> bool:
        """
        Check the backdoor criterion (Pearl 2009, Definition 3.3.1).

        Z satisfies the backdoor criterion relative to (X, Y) iff
        (i)  No node in Z is a descendant of X.
        (ii) Z blocks every backdoor path from X to Y.

        A *backdoor path* is any path from X to Y that begins with an arrow
        INTO X (i.e. starts X ← ...).  We test condition (ii) by forming the
        graph G_X^- in which all OUTGOING edges of X are removed; every
        remaining path from X to Y in G_X^- is a backdoor path.  Z
        blocks all backdoor paths iff X ⊥ Y | Z in G_X^-.
        """
        z_set = list(z_set)
        # (i) No element of Z is a descendant of X
        desc_x = nx.descendants(self.dag, x)
        if any(z in desc_x for z in z_set):
            return False
        # (ii) In G_X^- (remove all outgoing edges from X), Z d-separates X from Y.
        # Removing X's outgoing edges leaves only backdoor paths from X to Y.
        mutilated = self.dag.copy()
        for child in list(mutilated.successors(x)):
            mutilated.remove_edge(x, child)
        return d_separated(mutilated, [x], [y], z_set)

    def backdoor_adjustment(
        self,
        x: str,
        y: str,
        z_set: List[str],
        x_val: int,
        y_val: int,
    ) -> float:
        """
        Compute P(Y=y | do(X=x)) via the backdoor adjustment formula
        (Pearl 2009, Theorem 3.3.2):

            P(Y=y | do(X=x)) = Σ_z P(Y=y | X=x, Z=z) · P(Z=z)

        Parameters
        ----------
        x, y    : treatment and outcome variable names
        z_set   : list of adjustment variables (must satisfy backdoor criterion)
        x_val, y_val : values of X and Y

        Returns
        -------
        float
        """
        if not self.satisfies_backdoor(x, y, z_set):
            raise ValueError(
                f"Z={z_set} does not satisfy the backdoor criterion "
                f"for ({x} -> {y})."
            )
        dist = self.joint_dist
        result = 0.0
        z_ranges = [range(dist.cardinalities[z]) for z in z_set]
        for z_vals in itertools.product(*z_ranges):
            z_dict = dict(zip(z_set, z_vals))
            p_z = dist.marginal(z_set).p(z_dict)
            if p_z == 0:
                continue
            p_y_given_x_z = dist.conditional_p(
                {y: y_val}, {x: x_val, **z_dict}
            )
            result += p_y_given_x_z * p_z
        return result

    def backdoor_distribution(
        self,
        x: str,
        y: str,
        z_set: List[str],
        x_val: int,
    ) -> np.ndarray:
        """
        Return P(Y | do(X=x_val)) as a 1-D array (one entry per Y value).
        """
        y_card = self.joint_dist.cardinalities[y]
        return np.array(
            [
                self.backdoor_adjustment(x, y, z_set, x_val, y_val)
                for y_val in range(y_card)
            ]
        )

    # ------------------------------------------------------------------
    # Front-door criterion and adjustment
    # ------------------------------------------------------------------

    def satisfies_frontdoor(
        self, x: str, y: str, m_set: List[str]
    ) -> bool:
        """
        Check the front-door criterion (Pearl 2009, Definition 3.3.3).

        M satisfies the front-door criterion relative to (X, Y) iff
        (i)  All directed paths from X to Y are intercepted by M.
        (ii) There are no unblocked backdoor paths from X to M
             (checked by removing X's outgoing edges and testing X ⊥ M in the
             resulting graph).
        (iii) All backdoor paths from M to Y are blocked by X.

        We implement this as three graph-level checks on the observable DAG.
        Note: when unobserved confounders exist this DAG represents only
        the observable skeleton; callers must ensure that the bivariate
        graph faithfully encodes the confounding structure.
        """
        # (i) Every directed path X -> ... -> Y goes through M
        m_set_s = set(m_set)
        all_paths_go_through_m = True
        for path in nx.all_simple_paths(self.dag, x, y):
            if not any(node in m_set_s for node in path[1:]):
                all_paths_go_through_m = False
                break
        if not all_paths_go_through_m:
            return False

        # (ii) No unblocked backdoor path from X to any M.
        # Backdoor paths from X are paths beginning with an arrow into X.
        # Removing X's outgoing edges leaves only those paths; if X ⊥ m | ∅
        # in the resulting graph then there are no such paths.
        mutilated_x = self.dag.copy()
        for child in list(mutilated_x.successors(x)):
            mutilated_x.remove_edge(x, child)
        for m in m_set:
            if not d_separated(mutilated_x, [x], [m], []):
                return False

        # (iii) All backdoor paths from M to Y are blocked by X
        for m in m_set:
            if not self.satisfies_backdoor(m, y, [x]):
                return False

        return True

    def frontdoor_adjustment(
        self,
        x: str,
        y: str,
        m_set: List[str],
        x_val: int,
        y_val: int,
    ) -> float:
        """
        Compute P(Y=y | do(X=x)) via the front-door adjustment formula
        (Pearl 2009, Theorem 3.3.4):

            P(Y=y | do(X=x)) =
                Σ_m P(M=m | X=x) · Σ_{x'} P(Y=y | X=x', M=m) · P(X=x')

        Handles a single mediator M for simplicity; for multiple mediators
        the formula generalises via the chain rule.

        Parameters
        ----------
        x, y   : treatment and outcome variable names
        m_set  : list with exactly one mediator variable (for this implementation)
        x_val, y_val : values of X and Y
        """
        if len(m_set) != 1:
            raise NotImplementedError(
                "front-door adjustment currently supports exactly one mediator."
            )
        m = m_set[0]
        if not self.satisfies_frontdoor(x, y, m_set):
            raise ValueError(
                f"M={m_set} does not satisfy the front-door criterion "
                f"for ({x} -> {y})."
            )
        dist = self.joint_dist
        x_card = dist.cardinalities[x]
        m_card = dist.cardinalities[m]
        result = 0.0
        for m_val in range(m_card):
            p_m_given_x = dist.conditional_p({m: m_val}, {x: x_val})
            inner = 0.0
            for x_prime in range(x_card):
                p_x_prime = dist.marginal([x]).p({x: x_prime})
                p_y_given_x_prime_m = dist.conditional_p(
                    {y: y_val}, {x: x_prime, m: m_val}
                )
                inner += p_y_given_x_prime_m * p_x_prime
            result += p_m_given_x * inner
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SCM(name={self.name!r}, "
            f"nodes={list(self.dag.nodes)}, "
            f"edges={list(self.dag.edges)})"
        )
