from typing import Union
from types import SimpleNamespace
from io import BytesIO
import os
import sys
from urllib.request import urlopen
from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import warnings

from .utils import (
    L2star, L2ctr, L2ext, L2per, L2sym, L2mix,
    L2star_weighted, L2ctr_weighted, L2ext_weighted, L2per_weighted,
    L2sym_weighted, L2mix_weighted,
)
from .models import *


_DISCREPANCY = {
    'L2star': L2star, 'L2ctr': L2ctr, 'L2ext': L2ext, 'L2per': L2per, 'L2sym': L2sym, 'L2mix': L2mix,
    'L2star_weighted': L2star_weighted, 'L2ctr_weighted': L2ctr_weighted, 'L2ext_weighted': L2ext_weighted,
    'L2per_weighted': L2per_weighted, 'L2sym_weighted': L2sym_weighted, 'L2mix_weighted': L2mix_weighted,
}

class MPMC(AbstractLDDiscreteDistribution):
    """Low-discrepancy generator trained by MPMC. Produces nbatch independent
    pointsets of size n in [0,1]^d.

    Requires PyTorch and PyTorch Geometric. Install with:

    python -m pip install "qmcpy[mpmc]" qmcpy-install-mpmc

    For GPU support or platform-specific details, see
    https://pytorch.org/get-started/locally/

    Examples:
        >>> mpmc = MPMC(
        ...     dimension=2,
        ...     randomize='false',
        ...     seed=7,
        ...     epochs=100,
        ...     use_pretrained=False,
        ...     prompt_on_missing=False,
        ... )
        >>> points = mpmc.gen_samples(n=50)
        >>> points.shape
        (1, 50, 2)
        >>> print(mpmc)
        MPMC Generator Object
            dim             2
            randomize       FALSE
            loss_fn         L2star
            epochs          100
            lr              0.001
            nlayers         3
            nhid            32
            weight_decay    1e-06
            radius          0.35
            nbatch          1
            use_pretrained  False
        <BLANKLINE>
    """

    def __init__(
        self,
        randomize: str = 'shift',
        seed: Union[None, int, np.random.SeedSequence] = None,
        dimension: int = 2,
        replications: int = 1,
        d_max: Union[None, int] = None,
        lr: float = 1e-3,
        nlayers: int = 3,
        weight_decay: float = 1e-6,
        nhid: int = 32,
        epochs: int = 50_000,
        start_reduce: int = 40_000,
        radius: float = 0.35,
        nbatch: int = 1,
        loss_fn: str = 'L2star',
        weights: Union[None, list, np.ndarray, torch.Tensor] = None,
        use_pretrained: bool = True,
        pretrained_local_dir: Union[None, str] = None,
        pretrained_base_url: str = 'https://github.com/QMCSoftware/LDData/tree/main/pregenerated_pointsets/mpmc',
        prompt_on_missing: bool = True,
    ) -> None:
        """Initialize an MPMC discrete distribution.

        Args:
            randomize (str): `'shift'`/`'true'` for a random shift, or
                `'false'`/`'none'`/`'no'` for no randomization.
            seed (Union[None, int, np.random.SeedSequence]): Seed the random
                number generator for reproducibility.
            dimension (int): Dimension of the generated pointsets.
            replications (int): Number of independent pointsets to
                generate. Ignored if `nbatch` is set.
            d_max (Union[None, int]): Unused; kept for backward compatibility.
                `self.d_max` always mirrors `dimension`.
            lr (float): Learning rate for the MPMC network optimizer.
            nlayers (int): Number of message-passing layers in the MPMC
                network.
            weight_decay (float): Weight decay (L2 regularization) for the
                optimizer.
            nhid (int): Hidden dimension of the MPMC network layers.
            epochs (int): Number of training epochs.
            start_reduce (int): Epoch at which learning-rate reduction
                begins.
            radius (float): Radius parameter for the discrepancy loss.
            nbatch (int): Number of independent pointsets to train and
                generate (overrides `replications` when not `None`).
            loss_fn (str): Name of the discrepancy loss to train against; one
                of the keys in `qmcpy.discrete_distribution.mpmc.utils`'s
                discrepancy registry (e.g. `'L2star'`), optionally suffixed
                `'_weighted'`.
            weights (Union[None, list, np.ndarray, torch.Tensor]): Per-
                coordinate weights, required when `loss_fn` names a weighted
                discrepancy (or supplying them switches `loss_fn` to its
                weighted variant automatically).
            use_pretrained (bool): If `True`, load a pretrained pointset
                generator instead of training a new one, when one is
                available for the requested `dimension`/`nbatch`.
            pretrained_local_dir (Union[None, str]): Local directory to search for (and
                cache) pretrained generators. Defaults to a package cache
                directory when `None`.
            pretrained_base_url (str): Base URL to download pretrained
                generators from when not already cached locally.
            prompt_on_missing (bool): If `True`, prompt interactively before
                training a new generator when no pretrained one is found.
        """
        self.mimics = 'StdUniform'
        self.low_discrepancy = True

        self.parameters = [
            'dim', 'randomize', 'loss_fn', 'epochs', 'lr', 'nlayers', 'nhid',
            'weight_decay', 'radius', 'nbatch', 'use_pretrained'
        ]

        # core config
        self.dim = int(dimension)
        self.lr = float(lr)
        self.nlayers = int(nlayers)
        self.weight_decay = float(weight_decay)
        self.nhid = int(nhid)
        self.epochs = int(epochs)
        self.start_reduce = int(start_reduce)
        self.radius = float(radius)
        self.loss_fn = str(loss_fn)
        self.nbatch = int(nbatch) if nbatch is not None else int(replications)
        self.d_max = self.dim  # kept for compat
        self.use_pretrained = bool(use_pretrained)
        self.pretrained_local_dir = pretrained_local_dir
        self.pretrained_base_url = str(pretrained_base_url).rstrip('/')
        self.prompt_on_missing = bool(prompt_on_missing)
        self._pretrained_n_values = {16, 32, 64, 128, 256, 512, 1024}
        self._pretrained_d_values = {2, 3, 5, 8, 10}

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weights = None
        if weights is not None:
            # accept list/np/torch → torch.float32 on device
            self.weights = torch.as_tensor(weights, dtype=torch.float32, device=self.torch_device)
            if self.weights.dim() != 1 or self.weights.numel() != self.dim:
                raise ValueError(f"weights must be 1-D length d={self.dim}; got {tuple(self.weights.shape)}")

        # ensure weighted function name & presence of weights are consistent
        is_weighted_name = self.loss_fn.endswith('weighted')
        if is_weighted_name and self.weights is None:
            raise ValueError("Must specify `weights` for weighted loss function.")
        if (self.weights is not None) and (not is_weighted_name):
            warnings.warn("`weights` provided; switching to weighted discrepancy.", stacklevel=1)
            self.loss_fn = self.loss_fn + '_weighted'

        if (self.weights is None) and (self.dim > 5):
            warnings.warn("Product coordinate weights are recommended for dimension > 5.", stacklevel=1)

        # init AbstractLDDiscreteDistribution base class (sets self.rng, self.d, etc.)
        # AbstractDiscreteDistribution.__init__ signature is
        #   __init__(self, dimension, replications, seed, d_limit, n_limit)
        # so pass the number of replications (nbatch) and reasonable limits.
        super(MPMC, self).__init__(int(self.dim), self.nbatch, seed, d_limit=np.inf, n_limit=np.inf)

        # randomization mode
        rnd = str(randomize).strip().upper()
        if rnd in ('TRUE', 'SHIFT'):
            self.randomize = 'SHIFT'
        elif rnd in ('FALSE', 'NONE', 'NO'):
            self.randomize = 'FALSE'
        else:
            raise ParameterError(f"randomize must be in {{'shift','false'}}; got '{randomize}'")

        # pre-draw shifts if needed: shape (nbatch, d)
        if self.randomize == 'SHIFT':
            self.shift = self.rng.uniform(size=(self.nbatch, self.d))

        # backward-compat mirror
        self.replications = self.nbatch

    # --------------------------
    # Core API
    # --------------------------
    def _gen_samples(self, n_min, n_max, return_binary, warn, return_unrandomized=False):
        if n_min != 0:
            raise ParameterError("MPMC requires n_min=0 as it does not support indexing subsequencing")
        if return_binary is not False:
            raise ParameterError("MPMC requires return_binary=False")
        n = int(n_max-n_min)
        x = self._try_load_pretrained(n)
        if x is None:
            # training config
            args = SimpleNamespace(
                lr=self.lr,
                nlayers=self.nlayers,
                weight_decay=self.weight_decay,
                nhid=self.nhid,
                nbatch=self.nbatch,
                epochs=self.epochs,
                start_reduce=self.start_reduce,
                radius=self.radius,
                nsamples=n,
                dim=self.dim,
                loss_fn=self.loss_fn,
                weights=self.weights,
            )
            x = self._train(args)  # (nbatch, n, d)

        if self.randomize == 'FALSE':
            return x

        xr = (x + self.shift[:, None, :]) % 1.0

        return (xr, x) if return_unrandomized else xr

    def _pretrained_filename(self, n):
        return f"mpmc_d{self.dim}_n{n}_{self.loss_fn}.npy"

    def _ask_train_from_scratch(self):
        base_msg = "Pre-trained configuration not found; please train from scratch"
        prompt = base_msg + ". Continue training? [Y/N]: "
        try:
            if not self.prompt_on_missing:
                print(base_msg)
                return True
            if not hasattr(sys, 'stdin') or sys.stdin is None or not sys.stdin.isatty():
                print(base_msg)
                return True
            answer = input(prompt).strip().lower()
            if answer in ('y', 'yes', ''):
                return True
            if answer in ('n', 'no'):
                return False
            print("Unrecognized response; defaulting to training from scratch.")
            return True
        except Exception:
            print(base_msg)
            return True

    def _load_pretrained_array(self, n):
        fname = self._pretrained_filename(n)
        if self.pretrained_local_dir:
            local_path = os.path.join(self.pretrained_local_dir, fname)
            if os.path.isfile(local_path):
                return np.load(local_path)

        url = f"{self.pretrained_base_url}/{fname}"
        with urlopen(url, timeout=10) as resp:
            return np.load(BytesIO(resp.read()))

    def _try_load_pretrained(self, n):
        if not self.use_pretrained:
            return None
        if self.dim not in self._pretrained_d_values or n not in self._pretrained_n_values:
            return None

        try:
            pts = self._load_pretrained_array(n)
        except Exception:
            should_train = self._ask_train_from_scratch()
            if not should_train:
                raise RuntimeError("Pre-trained configuration not found and training declined by user.")
            return None

        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, self.dim):
            warnings.warn(
                f"Pre-trained file has shape {pts.shape}, expected {(n, self.dim)}; training from scratch.",
                stacklevel=1,
            )
            return None

        print("Pre-trained configuration already available; loading points.")
        if self.nbatch == 1:
            return pts[None, :, :]

        warnings.warn(
            "Using the same pre-trained point set for each batch replication.",
            stacklevel=1,
        )
        return np.repeat(pts[None, :, :], self.nbatch, axis=0)

    def __repr__(self):
        out = f"{self.__class__.__name__} Generator Object\n"
        for p in self.parameters:
            p_val = getattr(self, p)
            out += f"    {p:<15} {str(p_val)}\n"
        return out

    def _spawn(self, child_seed, dimension):
        """Spawn a child generator with same config (QMCPy hook)."""
        child_weights = None
        if self.weights is not None:
            child_weights = self.weights.detach().cpu().tolist()
            if dimension < len(child_weights):
                child_weights = child_weights[:dimension]
            elif dimension > len(child_weights):
                child_weights = child_weights + [child_weights[-1]] * (dimension - len(child_weights))

        return MPMC(
            randomize=self.randomize,
            seed=child_seed,
            dimension=dimension,
            nbatch=self.nbatch,
            lr=self.lr,
            nlayers=self.nlayers,
            weight_decay=self.weight_decay,
            nhid=self.nhid,
            epochs=self.epochs,
            start_reduce=self.start_reduce,
            radius=self.radius,
            loss_fn=self.loss_fn,
            weights=child_weights,
            use_pretrained=self.use_pretrained,
            pretrained_local_dir=self.pretrained_local_dir,
            pretrained_base_url=self.pretrained_base_url,
            prompt_on_missing=self.prompt_on_missing,
        )

    # --------------------------
    # Training
    # --------------------------
    def _train(self, args: SimpleNamespace):
        """Train an MPMC network and return its generated pointsets.

        Args:
            args (SimpleNamespace): Training configuration, carrying `dim`,
                `nhid`, `nlayers`, `nsamples`, `nbatch`, `radius`, `loss_fn`,
                `weights`, `lr`, `weight_decay`, `epochs`, and `start_reduce`.

        Returns:
            np.ndarray: shape `(nbatch, nsamples, dim)`
        """
        model = MPMC_net(
            dim=args.dim, nhid=args.nhid, nlayers=args.nlayers,
            nsamples=args.nsamples, nbatch=args.nbatch,
            radius=args.radius, loss_fn=args.loss_fn, weights=args.weights
        ).to(self.torch_device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float('inf')
        patience = 0
        end_result = None

        # adaptive schedule
        reduce_point = 10

        for epoch in tqdm(range(args.epochs),
                          desc=f"Training: N={args.nsamples}, d={args.dim}, loss={args.loss_fn}"):

            model.train()
            optimizer.zero_grad()
            loss, X = model()          # X: (nbatch, n, d)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    # compute batch discrepancies using the configured loss
                    fn = _DISCREPANCY[args.loss_fn]
                    if args.loss_fn.endswith('weighted'):
                        batched = fn(X, args.weights)
                    else:
                        batched = fn(X)
                    min_disc = batched.min().item()

                if min_disc < best_loss:
                    best_loss = min_disc
                    end_result = X.detach().cpu().numpy()

                # LR schedule after start_reduce
                if (epoch + 1) >= args.start_reduce:
                    if min_disc > best_loss:
                        patience += 1
                    if patience == reduce_point:
                        patience = 0
                        for g in optimizer.param_groups:
                            g['lr'] = max(g['lr'] / 10.0, 1e-6)
                        # stop early if lr already tiny
                        if optimizer.param_groups[0]['lr'] <= 1e-6:
                            break

        if end_result is None:
            end_result = X.detach().cpu().numpy()
        return end_result
