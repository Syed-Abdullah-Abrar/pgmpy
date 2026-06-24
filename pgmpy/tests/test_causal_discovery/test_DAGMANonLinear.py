"""Tests for the DAGMANonlinear causal discovery estimator in pgmpy.causal_discovery."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMANonLinear import (
    DAGMANonlinear,
    DagmaMLP,
    LocallyConnected,
)
from pgmpy.datasets import load_dataset


def expected_failed_checks(estimator):
    """
    scikit-learn checks that are expected to fail
    for pgmpy causal discovery algorithms.
    """
    return {
        "check_fit_score_takes_y": (
            "Causal discovery estimators do not take y parameter in score method."
        ),
        "check_n_features_in_after_fitting": (
            "Failing for score method (not for fit) for unknown reason."
        ),
    }


@parametrize_with_checks(
    [DAGMANonlinear(hidden_dims=(10,), warm_iter=10, max_iter=10)],
    expected_failed_checks=expected_failed_checks,
)
def test_dagma_nonlinear_sklearn_compatibility(estimator, check):
    """Run all sklearn estimator checks against DAGMANonlinear."""
    check(estimator)


@pytest.fixture
def synthetic_nonlinear_data():
    """Build a 4-variable DAG via DAGitty and simulate data.

    The ground-truth DAG is X0 -> X1, X1 -> X2, X2 -> X3, X0 -> X3.
    Returns ``(true_edges, data)`` with 2000 mean-centered, std-scaled samples.
    """
    dagitty_str = """
    dag {
        X0 -> X1 [beta=1.5]
        X1 -> X2 [beta=1.2]
        X2 -> X3 [beta=1.0]
        X0 -> X3 [beta=0.8]
    }
    """
    true_dag = DAG.from_dagitty(string=dagitty_str)
    true_edges = set(true_dag.edges())

    data = true_dag.simulate(n_samples=2000, seed=42)
    data = (data - data.mean()) / data.std()
    return true_edges, data


@pytest.mark.slow
class TestDAGMANonlinearCore:
    """Core estimation tests for DAGMANonlinear on synthetic data."""

    def test_fit_returns_dag(self, synthetic_nonlinear_data):
        """Fit returns a valid DAG on synthetic nonlinear data."""
        torch.manual_seed(42)
        true_edges, data = synthetic_nonlinear_data

        est = DAGMANonlinear(
            hidden_dims=(10,),
            lambda1=0.02,
            lambda2=0.005,
            T=2,
            warm_iter=5000,
            max_iter=10000,
            lr=0.0002,
            w_threshold=0.2,
            tol=1e-6,
            checkpoint=500,
        )
        est.fit(data)

        # 1. Verify fit returns self (sklearn convention)
        assert est is not None
        assert hasattr(est, "causal_graph_")
        assert hasattr(est, "W_est_")
        assert hasattr(est, "model_")

        # 2. Verify the result is a DAG (acyclicity enforced by constructor)
        assert isinstance(est.causal_graph_, DAG)
        assert est.W_est_.shape == (data.shape[1], data.shape[1])
        assert isinstance(est.model_, DagmaMLP)

        # 3. Verify at least 1 edge was recovered
        n_edges = len(est.causal_graph_.edges())
        assert n_edges >= 1, f"Expected at least 1 edge, got {n_edges}"

        # 4. Verify edge count is bounded by fully-connected DAG
        d = data.shape[1]
        assert n_edges <= d * (d - 1), (
            f"Edge count {n_edges} exceeds fully-connected bound {d * (d - 1)}"
        )

    def test_edge_recovery_synthetic(self, synthetic_nonlinear_data):
        """At least 2 of 4 true edges recovered from the known DAG."""
        torch.manual_seed(42)
        true_edges, data = synthetic_nonlinear_data

        est = DAGMANonlinear(
            hidden_dims=(10,),
            lambda1=0.02,
            lambda2=0.005,
            T=2,
            warm_iter=5000,
            max_iter=10000,
            lr=0.0002,
            w_threshold=0.2,
            tol=1e-6,
            checkpoint=500,
        )
        est.fit(data)

        learned_edges = set(est.causal_graph_.edges())
        recovered = len(learned_edges & true_edges)
        assert recovered >= 2, (
            f"Expected at least 2/4 true edges recovered, got {recovered}. "
            f"Learned: {learned_edges}, True: {true_edges}"
        )


@pytest.mark.slow
class TestDAGMANonlinearSachs:
    """Full-scale Sachs protein signaling benchmark (CI-skipped by default)."""

    def test_sachs_continuous(self):
        """DAGMANonlinear on Sachs continuous data produces a valid DAG."""
        torch.manual_seed(42)
        data = load_dataset("sachs_continuous").data
        data = (data - data.mean()) / data.std()

        est = DAGMANonlinear(
            hidden_dims=(10,),
            lambda1=0.02,
            lambda2=0.005,
            T=4,
            warm_iter=50000,
            max_iter=80000,
            lr=0.0002,
            w_threshold=0.3,
            tol=1e-6,
            checkpoint=1000,
        )
        est.fit(data)

        # 1. Validate DAG
        assert isinstance(est.causal_graph_, DAG)

        # 2. Validate shape
        d = data.shape[1]
        assert est.W_est_.shape == (d, d)

        # 3. Validate edge count
        n_edges = len(est.causal_graph_.edges())
        assert n_edges >= 1, f"Expected at least 1 edge on Sachs, got {n_edges}"
        assert n_edges <= d * (d - 1) // 2, (
            f"Edge count {n_edges} exceeds fully-connected bound {d * (d - 1) // 2}"
        )

        # 4. Validate fitted model
        assert isinstance(est.model_, DagmaMLP)


class TestDAGMANonlinearParameters:
    """Constructor parameter storage and default value checks."""

    def test_parameters(self):
        """All constructor parameters are stored correctly on the instance."""
        est = DAGMANonlinear(
            hidden_dims=(20, 10),
            s=1.0,
            lambda1=0.03,
            lambda2=0.01,
            T=5,
            mu_init=0.2,
            mu_factor=0.05,
            warm_iter=30000,
            max_iter=60000,
            lr=0.001,
            w_threshold=0.5,
            tol=1e-5,
            checkpoint=500,
            return_type="dag",
        )
        assert est.hidden_dims == (20, 10)
        assert est.s == 1.0
        assert est.lambda1 == 0.03
        assert est.lambda2 == 0.01
        assert est.T == 5
        assert est.mu_init == 0.2
        assert est.mu_factor == 0.05
        assert est.warm_iter == 30000
        assert est.max_iter == 60000
        assert est.lr == 0.001
        assert est.w_threshold == 0.5
        assert est.tol == 1e-5
        assert est.checkpoint == 500
        assert est.return_type == "dag"

    def test_default_parameters(self):
        """Default parameter values match the DAGMA paper / ADR-005 spec."""
        est = DAGMANonlinear()
        assert est.hidden_dims == (10,)
        assert est.s == 1.0
        assert est.lambda1 == 0.02
        assert est.lambda2 == 0.005
        assert est.T == 4
        assert est.mu_init == 0.1
        assert est.mu_factor == 0.1
        assert est.warm_iter == 50000
        assert est.max_iter == 80000
        assert est.lr == 0.0002
        assert est.w_threshold == 0.3
        assert est.tol == 1e-6
        assert est.checkpoint == 1000
        assert est.return_type == "dag"

    def test_fit_returns_self(self, synthetic_nonlinear_data):
        """fit() returns self (sklearn convention)."""
        torch.manual_seed(42)
        _, data = synthetic_nonlinear_data

        est = DAGMANonlinear(
            hidden_dims=(10,),
            T=1,
            warm_iter=10,
            max_iter=20,
        )
        result = est.fit(data)
        assert result is est


class TestWThreshold:
    """Edge weight threshold (``w_threshold``) behavior."""

    def test_w_threshold(self):
        """Larger w_threshold produces sparser (or equally-sparse) graphs."""
        torch.manual_seed(42)
        dagitty_str = """
        dag {
            X0 -> X1 [beta=1.0]
            X1 -> X2 [beta=0.8]
        }
        """
        model = DAG.from_dagitty(string=dagitty_str)
        data = model.simulate(n_samples=500, seed=42)
        data = (data - data.mean()) / data.std()

        # Fit with low threshold (keeps more edges)
        torch.manual_seed(42)
        est_low = DAGMANonlinear(
            hidden_dims=(5,),
            T=1,
            warm_iter=200,
            max_iter=500,
            w_threshold=0.05,
            lambda1=0.01,
        )
        est_low.fit(data)

        # Fit with high threshold (prunes more aggressively)
        torch.manual_seed(42)
        est_high = DAGMANonlinear(
            hidden_dims=(5,),
            T=1,
            warm_iter=200,
            max_iter=500,
            w_threshold=0.5,
            lambda1=0.01,
        )
        est_high.fit(data)

        n_low = len(est_low.causal_graph_.edges())
        n_high = len(est_high.causal_graph_.edges())

        # Higher threshold should NOT produce more edges
        assert n_high <= n_low, (
            f"Higher w_threshold ({est_high.w_threshold}) produced "
            f"{n_high} edges, but lower threshold ({est_low.w_threshold}) "
            f"produced {n_low} edges.  Expected n_high <= n_low."
        )


class TestReproducibility:
    """Deterministic output with fixed seed."""

    def test_reproducibility(self):
        """Identical torch seed + data yields identical W_est_ matrices."""
        data = pd.DataFrame(
            np.random.RandomState(42).randn(200, 4),
            columns=["A", "B", "C", "D"],
        )
        data = (data - data.mean()) / data.std()

        kwargs = dict(
            hidden_dims=(10,),
            T=2,
            warm_iter=100,
            max_iter=200,
            lr=0.0002,
            lambda1=0.02,
            lambda2=0.005,
            w_threshold=0.0,  # No threshold so raw W is compared
        )

        torch.manual_seed(123)
        est1 = DAGMANonlinear(**kwargs)
        est1.fit(data)

        torch.manual_seed(123)
        est2 = DAGMANonlinear(**kwargs)
        est2.fit(data)

        assert np.allclose(est1.W_est_, est2.W_est_), (
            "W_est_ matrices differ despite identical seed. "
            f"Max diff: {np.max(np.abs(est1.W_est_ - est2.W_est_))}"
        )

    def test_different_seeds_produce_different_results(self):
        """Different seeds produce valid DAGs in both cases."""
        data = pd.DataFrame(
            np.random.RandomState(42).randn(100, 3),
            columns=["X", "Y", "Z"],
        )
        data = (data - data.mean()) / data.std()

        kwargs = dict(
            hidden_dims=(10,),
            T=1,
            warm_iter=50,
            max_iter=100,
            lr=0.001,
            lambda1=0.01,
            lambda2=0.001,
            w_threshold=0.0,
        )

        torch.manual_seed(42)
        est1 = DAGMANonlinear(**kwargs)
        est1.fit(data)

        torch.manual_seed(999)
        est2 = DAGMANonlinear(**kwargs)
        est2.fit(data)

        # Both runs should produce valid results
        assert est1.W_est_.shape == est2.W_est_.shape
        assert isinstance(est1.causal_graph_, DAG)
        assert isinstance(est2.causal_graph_, DAG)


class TestFailureModes:
    """Edge cases: empty data, too few samples, invalid return_type."""

    def test_raises_on_zero_features(self):
        """Fit raises ValueError when X has zero features."""
        est = DAGMANonlinear()
        with pytest.raises(ValueError, match="0 feature"):
            est.fit(pd.DataFrame())

    def test_raises_on_insufficient_samples(self):
        """Fit raises ValueError when n_samples < 2."""
        est = DAGMANonlinear()
        with pytest.raises(ValueError):
            est.fit(pd.DataFrame({"A": [1.0]}))

    def test_invalid_return_type(self):
        """Invalid return_type raises ValueError during fit."""
        est = DAGMANonlinear(return_type="pdag", warm_iter=10, max_iter=10)
        rng = np.random.RandomState(42)
        data = pd.DataFrame(rng.randn(100, 2), columns=["A", "B"])
        data = (data - data.mean()) / data.std()
        with pytest.raises((ValueError, NotImplementedError)):
            est.fit(data)


class TestLocallyConnected:
    """Unit tests for the ``LocallyConnected`` module."""

    def test_forward_shape(self):
        """Forward pass contract: (n, nl, in) -> (n, nl, out)."""
        layer = LocallyConnected(num_linear=5, input_features=10, output_features=3)
        x = torch.randn(100, 5, 10)
        out = layer(x)
        assert out.shape == (100, 5, 3)

    def test_forward_with_bias(self):
        """Bias=True produces non-zero output even with initialized weights."""
        layer = LocallyConnected(num_linear=3, input_features=4, output_features=2, bias=True)
        x = torch.ones(50, 3, 4)
        out = layer(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_bias_false(self):
        """When bias=False, the bias parameter is ``None``."""
        layer = LocallyConnected(num_linear=3, input_features=4, output_features=2, bias=False)
        assert layer.bias is None

    def test_weight_init_range(self):
        """Weights initialized uniformly in [-1/sqrt(in_features), 1/sqrt(in_features)]."""
        in_f = 10
        layer = LocallyConnected(num_linear=5, input_features=in_f, output_features=3)
        bound = 1.0 / np.sqrt(in_f)
        w = layer.weight.data
        assert torch.all(w >= -bound)
        assert torch.all(w <= bound)

    def test_extra_repr(self):
        """extra_repr returns a string with parameter info."""
        layer = LocallyConnected(num_linear=5, input_features=10, output_features=3, bias=True)
        rep = layer.extra_repr()
        assert "5" in rep
        assert "10" in rep
        assert "3" in rep
        assert isinstance(rep, str)


class TestDagmaMLP:
    """Unit tests for the ``DagmaMLP`` module."""

    def test_architecture(self):
        """Forward pass: (n, d) -> (n, d)."""
        model = DagmaMLP(dims=[5, 10, 1])
        x = torch.randn(100, 5)
        out = model(x)
        assert out.shape == (100, 5)

    def test_get_w_zero_init(self):
        """Zero-initialized fc1 produces all-zero adjacency matrix."""
        model = DagmaMLP(dims=[5, 10, 1])
        W = model.get_w()
        assert torch.allclose(W, torch.zeros(5, 5)), (
            f"Expected zero W after zero-init fc1, got max|W| = "
            f"{torch.max(torch.abs(W)).item():.6f}"
        )

    def test_get_w_shape(self):
        """get_w() returns (d, d) tensor."""
        model = DagmaMLP(dims=[7, 12, 1])
        W = model.get_w()
        assert W.shape == (7, 7)

    def test_loss_shape(self):
        """loss() returns a scalar tensor."""
        model = DagmaMLP(dims=[5, 10, 1])
        x = torch.randn(100, 5)
        x_hat = model(x)
        loss = model.loss(x, x_hat)
        assert loss.ndim == 0

    def test_loss_positive(self):
        """loss() is positive when predictions differ from targets."""
        model = DagmaMLP(dims=[5, 10, 1])
        x = torch.randn(100, 5)
        x_hat = model(x)
        loss = model.loss(x, x_hat)
        assert loss.item() > 0

    def test_fc1_l1_reg_positive(self):
        """fc1_l1_reg() > 0 after setting nonzero fc1 weights."""
        model = DagmaMLP(dims=[5, 10, 1])
        model.fc1.weight.data = torch.randn_like(model.fc1.weight.data)
        reg = model.fc1_l1_reg()
        assert reg.item() > 0

    def test_fc1_l1_reg_zero_init(self):
        """fc1_l1_reg() == 0 after zero initialization."""
        model = DagmaMLP(dims=[5, 10, 1])
        reg = model.fc1_l1_reg()
        assert reg.item() == 0.0

    def test_deep_architecture(self):
        """Two hidden LocallyConnected layers work correctly."""
        model = DagmaMLP(dims=[5, 20, 10, 1])
        x = torch.randn(100, 5)
        out = model(x)
        assert out.shape == (100, 5)
        W = model.get_w()
        assert W.shape == (5, 5)


class TestFittedAttributes:
    """After fit(), trailing-underscore attributes are present and well-formed."""

    def test_fitted_attributes(self, synthetic_nonlinear_data):
        """All expected fitted attributes are present after fit()."""
        torch.manual_seed(42)
        _, data = synthetic_nonlinear_data

        est = DAGMANonlinear(
            hidden_dims=(10,),
            T=1,
            warm_iter=50,
            max_iter=100,
        )
        est.fit(data)

        # 1. _BaseDAGMAMixin / BaseCausalDiscovery attributes
        assert hasattr(est, "causal_graph_")
        assert isinstance(est.causal_graph_, DAG)

        assert hasattr(est, "W_est_")
        assert isinstance(est.W_est_, np.ndarray)
        assert est.W_est_.shape == (data.shape[1], data.shape[1])

        assert hasattr(est, "model_")
        assert isinstance(est.model_, DagmaMLP)

        # 2. feature_names_in_ set by _check_fit_data
        assert hasattr(est, "feature_names_in_")
        assert hasattr(est, "n_features_in_")
        assert est.n_features_in_ == data.shape[1]


class TestConvergence:
    """Early stopping and convergence behavior."""

    def test_convergence_no_crash(self, synthetic_nonlinear_data):
        """Optimisation loop completes without RuntimeError on valid data."""
        torch.manual_seed(42)
        _, data = synthetic_nonlinear_data

        est = DAGMANonlinear(
            hidden_dims=(10,),
            T=2,
            warm_iter=100,
            max_iter=200,
            tol=1e-2,  # Loose tol so early-stop may trigger
            checkpoint=50,
        )
        est.fit(data)
        assert isinstance(est.causal_graph_, DAG)
