r"""DAGMA Non-linear causal discovery via MLP-based structural equations.

Implements the DAGMA non-linear algorithm that parameterises each structural
equation with a multi-layer perceptron (MLP).  The weighted adjacency matrix
is extracted from the first-layer weights via the L2 norm across hidden units,
and the log-determinant acyclicity barrier is applied to that induced matrix.
"""
from __future__ import annotations

import copy
import logging
import math

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.causal_discovery._base import BaseCausalDiscovery, _BaseDAGMAMixin

torch = _safe_import("torch")
tqdm = _safe_import("tqdm.auto", "tqdm")

logger = logging.getLogger(__name__)


class LocallyConnected(torch.nn.Module):
    r"""Batched independent linear transformations.

    Equivalent to a 1-D local convolution with filter size 1.  Each of
    ``num_linear`` independent linear maps transforms from
    ``input_features`` to ``output_features``.

    Parameters
    ----------
    num_linear : int
        Number of independent linear maps (typically the data dimension
        :math:`d`).
    input_features : int
        Dimensionality of each input vector.
    output_features : int
        Dimensionality of each output vector.
    bias : bool, optional
        Whether to include an additive bias.  Default ``True``.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Shape ``(num_linear, input_features, output_features)``.
    bias : torch.nn.Parameter or None
        Shape ``(num_linear, output_features)`` if *bias* is ``True``.
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.empty(num_linear, input_features, output_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(num_linear, output_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        r"""Initialise weights and bias uniformly.

        Weights are drawn from
        :math:`\mathcal{U}\bigl(-1 / \sqrt{m_1},\; 1 / \sqrt{m_1}\bigr)`
        where :math:`m_1` = ``input_features``.
        """
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        r"""Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Shape :math:`(n, d, m_1)`.

        Returns
        -------
        torch.Tensor
            Shape :math:`(n, d, m_2)`.
        """
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        r"""Human-readable parameter summary."""
        return (
            f"num_linear={self.num_linear}, "
            f"in_features={self.input_features}, "
            f"out_features={self.output_features}, "
            f"bias={self.bias is not None}"
        )


class DagmaMLP(torch.nn.Module):
    r"""MLP that models the structural equations for non-linear DAGMA.

    The first layer ``fc1`` maps from :math:`\mathbb{R}^d` to
    :math:`\mathbb{R}^{d \times m_1}`.  Its weights are reshaped and used
    to induce the weighted adjacency matrix :math:`W` via the L2 norm across
    the hidden-unit dimension.  Remaining layers are ``LocallyConnected``
    modules applied per-variable.

    Parameters
    ----------
    dims : list of int
        Layer dimensions ``[d, m_1, m_2, ..., 1]``.  The first entry is
        the number of variables :math:`d`; the last entry must be ``1``.
    bias : bool, optional
        Whether fully-connected and locally-connected layers include a bias
        term.  Hard-coded to ``True`` following ADR-004.  Default ``True``.

    Attributes
    ----------
    dims : list of int
        Layer dimensions passed at construction.
    d : int
        Number of variables (first entry of *dims*).
    fc1 : torch.nn.Linear
        First fully-connected layer, shape ``(d*m_1, d)``, zero-initialised.
    fc2 : torch.nn.ModuleList
        List of ``LocallyConnected`` layers for hidden-to-output mapping.
    """

    def __init__(self, dims, bias=True):
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"dims must have at least 2 entries, got {len(dims)}")
        if dims[-1] != 1:
            raise ValueError(f"Last entry of dims must be 1, got {dims[-1]}")

        self.dims = dims
        self.d = dims[0]

        # fc1: d → d×m₁  (zero-initialised so W starts at 0)
        self.fc1 = torch.nn.Linear(self.d, self.d * dims[1], bias=bias)
        torch.nn.init.zeros_(self.fc1.weight)
        if self.fc1.bias is not None:
            torch.nn.init.zeros_(self.fc1.bias)

        # fc2: chain of LocallyConnected layers
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[idx + 1], dims[idx + 2], bias=bias))
        self.fc2 = torch.nn.ModuleList(layers)

    def forward(self, x):
        r"""Apply the structural equations to data.

        Parameters
        ----------
        x : torch.Tensor
            Shape :math:`(n, d)`.

        Returns
        -------
        torch.Tensor
            Shape :math:`(n, d)` — the predicted values :math:`\hat{X}`.
        """
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(dim=2)

    def h_func(self, s=1.0):
        r"""Log-determinant acyclicity barrier.

        Computes :math:`h(W) = -\log\det(s I - A) + d \log s` where
        :math:`A_{ij} = \| \mathrm{fc1.weight}_{j,\cdot,i} \|_2^2`.

        Parameters
        ----------
        s : float, optional
            M-matrix domain parameter.  Default ``1.0``.

        Returns
        -------
        torch.Tensor
            Scalar barrier value.  The current parameters encode a DAG iff
            the value is finite (i.e., the ``slogdet`` sign is positive).
        """
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()  # [i, j]
        I = torch.eye(self.d, device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)
        sign, logdet = torch.slogdet(s * I - A)
        if sign <= 0:
            # M-matrix domain violated — caller should handle rollback.
            return -torch.tensor(float("inf"), device=self.fc1.weight.device)
        h = -logdet + self.d * math.log(s)
        return h

    @torch.no_grad()
    def get_w(self):
        r"""Extract the induced weighted adjacency matrix.

        .. math::
            W_{ij} = \bigl\|
                \mathrm{fc1.weight}_{j,\cdot,i}
            \bigr\|_2

        Returns
        -------
        torch.Tensor
            Shape :math:`(d, d)` weighted adjacency matrix on the same
            device as ``fc1.weight``.
        """
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()  # [i, j]
        W = torch.sqrt(A)
        return W

    def loss(self, X, X_hat):
        r"""Log-MSE loss (negative log-likelihood proxy for continuous data).

        .. math::
            \mathcal{L} = \frac{d}{2}
            \log\!\Bigl(\frac{1}{n}\sum (\hat{X} - X)^2\Bigr)

        Parameters
        ----------
        X : torch.Tensor
            Ground-truth data, shape :math:`(n, d)`.
        X_hat : torch.Tensor
            Predicted values, shape :math:`(n, d)`.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        n, d = X.shape
        return 0.5 * d * torch.log((1.0 / n) * torch.sum((X_hat - X) ** 2))

    def fc1_l1_reg(self):
        r"""L1 norm of the first fully-connected layer weights.

        Returns
        -------
        torch.Tensor
            Scalar :math:`\|\mathrm{fc1.weight}\|_1`.
        """
        return torch.sum(torch.abs(self.fc1.weight))


class DAGMANonlinear(_BaseDAGMAMixin, BaseCausalDiscovery):
    r"""DAGMA Non-linear causal discovery via MLP structural equations.

    Learns a Directed Acyclic Graph (DAG) from observational data by jointly
    optimising a log-MSE score and a log-determinant acyclicity barrier over
    the parameters of a multi-layer perceptron (``DagmaMLP``).  The weighted
    adjacency matrix is extracted from the first-layer weights via the L2
    norm across hidden units.

    Unlike :class:`DAGMALinear`, this estimator implements its own dual-loop
    optimisation (it does **not** use ``_BaseDAGMAMixin._optimize``, which
    targets the linear case where a single ``(d, d)`` matrix is the
    optimisation variable).  It reuses ``_log_det_barrier``,
    ``_convert_to_dag``, and ``_resolve_device_and_dtype`` from the mixin.

    The algorithm uses a central path method with model rollback: when the
    barrier function detects a domain violation (the induced adjacency matrix
    leaves the M-matrix domain), the model is restored from a checkpoint, the
    learning rate is halved, and the domain parameter *s* is reset to 1.0.
    This safeguard is critical for non-convex MLP optimisation, where the
    invexity guarantees that protect the linear case no longer hold.

    Parameters
    ----------
    hidden_dims : tuple of int, optional (default=(10,))
        Sizes of hidden layers in each MLP.  The full ``dims`` list passed
        to ``DagmaMLP`` is ``[d, *hidden_dims, 1]``, constructed at fit time.

    s : float or sequence of float, optional (default=1.0)
        M-matrix domain parameter.  Must satisfy :math:`s \ge 1`.  When a
        scalar, the same value is used for all *T* outer iterations.  When a
        sequence (list or tuple), one value is consumed per outer iteration;
        if the sequence is shorter than *T*, its last value is repeated.

    lambda1 : float, optional (default=0.02)
        L1 penalty coefficient on the first-layer weights to enforce
        sparsity in the estimated graph.  Higher values promote sparsity by
        shrinking edge weights to zero.

    lambda2 : float, optional (default=0.005)
        L2 penalty coefficient (weight decay) applied to all model
        parameters via the Adam optimiser.  Helps prevent overfitting and
        stabilises training.

    T : int, optional (default=4)
        Number of DAGMA outer iterations (mu-decay steps).  Each outer
        iteration runs an inner Adam optimisation at the current barrier
        weight :math:`\mu`.

    mu_init : float, optional (default=0.1)
        Initial barrier penalty weight :math:`\mu`.  Controls how strongly
        the acyclicity constraint is enforced at the start of optimisation.

    mu_factor : float, optional (default=0.1)
        Decay factor for :math:`\mu` after each outer iteration
        (:math:`\mu \leftarrow \mu \times \text{mu_factor}`).  Smaller
        values decay faster, reaching the DAG constraint sooner.

    warm_iter : int, optional (default=50000)
        Number of inner Adam steps for the first :math:`T-1` outer
        iterations.  These warm-up iterations optimise the objective at
        larger mu values before the final fine-tuning.

    max_iter : int, optional (default=80000)
        Number of inner Adam steps for the final outer iteration.  This is
        typically larger than *warm_iter* to allow fine convergence when
        :math:`\mu` is small.

    lr : float, optional (default=0.0002)
        Learning rate for the Adam optimiser.

    w_threshold : float, optional (default=0.3)
        Edges with absolute weight below this value are pruned from the
        final graph.  Higher values produce sparser graphs.

    tol : float, optional (default=1e-6)
        Relative tolerance for early stopping inside the inner optimisation
        loop.  Smaller values require tighter convergence.

    checkpoint : int, optional (default=1000)
        Frequency (in inner iterations) at which convergence is checked and
        progress is logged (when ``verbose`` is ``True``).

    return_type : str, optional (default="dag")
        The type of graph to return.  Must be either ``"dag"`` or
        ``"cpdag"``.

    verbose : bool, optional (default=False)
        If ``True``, prints loss and barrier values every ``checkpoint``
        iterations.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The estimated Directed Acyclic Graph.

    W_est_ : np.ndarray
        The estimated weighted adjacency matrix, shape ``(d, d)``.

    model_ : DagmaMLP
        The fitted MLP model.

    n_features_in_ : int
        The number of features in the dataset used to learn the causal graph.

    feature_names_in_ : pd.Index
        The feature names in the dataset used to learn the causal graph.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.causal_discovery import DAGMANonlinear
    >>> data = load_dataset("sachs_continuous").data
    >>> data = (data - data.mean()) / data.std()
    >>> est = DAGMANonlinear(hidden_dims=(10,), T=2, warm_iter=5000,
    ...                      max_iter=10000)
    >>> est.fit(data)  # doctest: +SKIP
    >>> print(list(est.causal_graph_.edges()))  # doctest: +SKIP

    References
    ----------
    .. [1] DAGMA: Learning DAGs via M-matrices and a Log-Determinant
           Acyclicity Characterization.  Kevin Bello, Bryon Aragam, Pradeep
           Ravikumar.  NeurIPS 2022.
    """

    def __init__(
        self,
        hidden_dims=(10,),
        s=1.0,
        lambda1=0.02,
        lambda2=0.005,
        T=4,
        mu_init=0.1,
        mu_factor=0.1,
        warm_iter=50000,
        max_iter=80000,
        lr=0.0002,
        w_threshold=0.3,
        tol=1e-6,
        checkpoint=1000,
        return_type: str = "dag",
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.s = s
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.T = T
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.warm_iter = warm_iter
        self.max_iter = max_iter
        self.lr = lr
        self.w_threshold = w_threshold
        self.tol = tol
        self.checkpoint = checkpoint
        self.return_type = return_type
        self.verbose = verbose

    def _minimize(self, model, X_tensor, mu, s, lambda1, lambda2, lr, max_iter, pbar, lr_decay=False):
        r"""Run Adam inner optimisation for a fixed barrier weight :math:`\mu`.

        Parameters
        ----------
        model : DagmaMLP
            The MLP model whose parameters are optimised.
        X_tensor : torch.Tensor
            Input data, shape ``(n, d)``.
        mu : float
            Current barrier penalty weight.
        s : float
            Current M-matrix domain parameter.
        lambda1 : float
            L1 penalty coefficient on ``fc1`` weights.
        lambda2 : float
            L2 weight-decay coefficient (applied via Adam).
        lr : float
            Learning rate for Adam.
        max_iter : int
            Maximum number of Adam steps.
        pbar : tqdm or None
            Progress bar to update.
        lr_decay : bool, optional
            If ``True``, apply an ``ExponentialLR`` scheduler with
            ``gamma=0.8`` every 1000 steps.  This is enabled after a
            rollback to stabilise recovery.

        Returns
        -------
        success : bool
            ``True`` if optimisation finished without leaving the M-matrix
            domain.  ``False`` if the barrier went negative, signalling that
            the caller should roll back and retry.
        """
        vprint = print if self.verbose else lambda *a, **k: None

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.99, 0.999),
            weight_decay=mu * lambda2,
        )

        scheduler = None
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()

            h_val = model.h_func(s)
            if h_val.item() < 0:
                vprint(f"  h negative ({h_val.item():.4e}) at inner iter {i}")
                return False

            X_hat = model(X_tensor)
            score = model.loss(X_tensor, X_hat)
            l1_reg = lambda1 * model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()

            # Exponential LR decay: step every 1000 iters (official behaviour).
            if scheduler is not None and (i + 1) % 1000 == 0:
                scheduler.step()

            if i % self.checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()
                vprint(f"  Inner iter {i}: h={h_val.item():.4e}  score={score.item():.4e}  obj={obj_new:.4e}")
                if abs((obj_prev - obj_new) / max(abs(obj_prev), 1e-16)) <= self.tol:
                    if pbar is not None:
                        pbar.update(max_iter - i)
                    break
                obj_prev = obj_new

            if pbar is not None:
                pbar.update(1)

        return True

    def _fit(self, X: pd.DataFrame):
        r"""Fit the DAGMANonlinear model to *X*.

        The algorithm uses a central path method that optimises a sequence of
        unconstrained problems.  As :math:`\mu` decays to zero, the solution
        converges to a DAG.  A model rollback mechanism handles M-matrix
        domain violations during non-convex MLP optimisation.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.
        """
        device, dtype = self._resolve_device_and_dtype()

        # Step 1: Convert data to torch tensor.
        # _check_fit_data guarantees that X is a pd.DataFrame by this point.
        X_np = X.values.astype(np.float64)

        X_tensor = torch.from_numpy(X_np).to(device=device, dtype=dtype)
        n, d = X_tensor.shape

        # Step 2: Build the MLP model.
        dims = [d] + list(self.hidden_dims) + [1]
        model = DagmaMLP(dims=dims, bias=True)

        # Ensure model parameters are on the correct device and dtype.
        # PyTorch may default to float32; cast explicitly when needed.
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)

        # Step 3: Dual-loop optimisation with model rollback.
        mu = self.mu_init

        # s scheduling: scalar → repeat T times; sequence → use as-is.
        if isinstance(self.s, (int, float)):
            s_vals = [self.s] * self.T
        else:
            s_vals = list(self.s)
            if len(s_vals) < self.T:
                s_vals = s_vals + [s_vals[-1]] * (self.T - len(s_vals))

        total_steps = (self.T - 1) * self.warm_iter + self.max_iter
        vprint = print if self.verbose else lambda *a, **k: None

        with tqdm.tqdm(total=total_steps, disable=not self.verbose) as pbar:
            for t in range(self.T):
                vprint(f"\nDAGMA iter t={t + 1}/{self.T}  mu={mu:.4e}  s={s_vals[t]}")
                vprint("-" * 40)

                inner_iter = self.max_iter if t == self.T - 1 else self.warm_iter
                s_cur = s_vals[t]
                lr_cur = self.lr
                lr_decay = False
                success = False

                # Checkpoint model state before inner optimisation.
                state_dict_copy = copy.deepcopy(model.state_dict())

                while not success:
                    success = self._minimize(
                        model,
                        X_tensor,
                        mu=mu,
                        s=s_cur,
                        lambda1=self.lambda1,
                        lambda2=self.lambda2,
                        lr=lr_cur,
                        max_iter=int(inner_iter),
                        pbar=pbar,
                        lr_decay=lr_decay,
                    )

                    if not success:
                        # Rollback: restore model, halve lr, widen domain,
                        # and enable exponential LR decay for stabilisation.
                        model.load_state_dict(state_dict_copy)
                        lr_cur *= 0.5
                        lr_decay = True
                        if lr_cur < 1e-10:
                            logger.warning(
                                "Learning rate too small (%.2e); stopping retries at outer iter %d.", lr_cur, t
                            )
                            break
                        s_cur = 1.0  # Reset to widest M-matrix domain.
                        vprint(f"  Rollback: lr -> {lr_cur:.2e}, s -> {s_cur}")

                mu *= self.mu_factor

        # Step 4: Extract adjacency and convert to pgmpy DAG.
        W_tensor = model.get_w()
        W_est = W_tensor.detach().cpu().numpy()

        self.model_ = model
        self.W_est_ = W_est

        feature_names = list(X.columns)
        self.causal_graph_ = self._convert_to_dag(W_est, feature_names, self.w_threshold, self.return_type)

        return self
