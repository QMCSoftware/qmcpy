from typing import Union
import numpy as np
import torch
import gpytorch


class ExactGPyTorchRegressionModel(gpytorch.models.ExactGP):
    """Exact Gaussian process regression model backed by GPyTorch.

    Wraps ``gpytorch.models.ExactGP`` with fitting, chunked prediction, and
    incremental data addition, optionally on the GPU.
    """

    allowed_likelihood_types = (
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.GaussianLikelihoodWithMissingObs,
        gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
    )

    def __init__(self, x_t, y_t, prior_mean, prior_cov, likelihood, use_gpu=False) -> None:
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        if isinstance(y_t, np.ndarray):
            y_t = torch.from_numpy(y_t)
        if not (x_t.ndim == 2 and y_t.ndim == 1 and len(x_t) == len(y_t)):
            raise AssertionError
        super(ExactGPyTorchRegressionModel, self).__init__(x_t, y_t, likelihood)
        if not (isinstance(
            self.likelihood, ExactGPyTorchRegressionModel.allowed_likelihood_types
        )):
            raise AssertionError
        self.mean_module, self.covar_module = prior_mean, prior_cov
        self.d = x_t.shape[1]
        self.use_gpu = use_gpu
        if self.use_gpu:
            if not (torch.cuda.is_available()):
                raise AssertionError
            self = self.cuda()
            self.likelihood = self.likelihood.cuda()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate the GP prior at the given inputs.

        Args:
            x (torch.Tensor): Inputs of shape ``(n, d)``.

        Returns:
            gpytorch.distributions.MultivariateNormal: Prior distribution at ``x``.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, optimizer: torch.optim.Optimizer, mll: gpytorch.mlls.MarginalLogLikelihood, training_iter: int, verbose: int = 0):
        """Fit the model hyperparameters by maximizing the marginal log likelihood.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer over the model parameters.
            mll (gpytorch.mlls.MarginalLogLikelihood): Objective to maximize.
            training_iter (int): Number of optimizer steps.
            verbose (int): Print progress every ``verbose`` iterations; ``0`` is silent.
        """
        self.train()
        self.likelihood.train()
        if verbose:
            print("\tgpytorch model fitting")
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.__call__(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            if verbose and (i + 1) % verbose == 0:
                print("\t\titer %-3d of %d" % (i + 1, training_iter))
                for name, val in self.named_parameters():
                    print("\t\t\t%s %.2e" % (name.ljust(50, "."), val))
            optimizer.step()

    def predict(self, x: Union[np.ndarray, torch.Tensor], noise_const: float = 0, chunk_size: int = 2**15) -> tuple:
        """Predict the posterior mean and standard deviation at new inputs.

        Inputs are processed in chunks so large batches do not exhaust memory.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Inputs of shape ``(n, d)``.
            noise_const (float): Observation noise assumed at each new input.
            chunk_size (int): Number of inputs evaluated per batch.

        Returns:
            tuple: Posterior mean and standard deviation, each of length ``n``.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not (x.ndim == 2 and x.shape[1] == self.d):
            raise AssertionError
        self.eval()
        self.likelihood.eval()
        n = len(x)
        mean_post, std_post = np.zeros(n, dtype=float), np.zeros(n, dtype=float)
        for i in range(0, n, chunk_size):
            lchunk, uchunk = i, min(i + chunk_size, n)
            noise_chunk = noise_const * torch.ones(uchunk - lchunk)
            mean_post[lchunk:uchunk], std_post[lchunk:uchunk] = self._predict_batch(
                x[lchunk:uchunk], noise_chunk
            )
        return mean_post, std_post

    def _predict_batch(self, x, noise):
        if self.use_gpu:
            x, noise = x.cuda(), noise.cuda()
        with torch.no_grad():
            # some issue with gpytorch == 1.15.2 making this call very slow by not obeying torch.no_grad() in some way
            # https://github.com/cornellius-gp/gpytorch/issues/2736
            observed_pred = self.likelihood(self.__call__(x), noise=noise)
        mean_post = observed_pred.mean
        std_post = observed_pred.stddev
        if self.use_gpu:
            mean_post, std_post = mean_post.cpu(), std_post.cpu()
        if self.use_gpu:
            del x, noise, observed_pred
            torch.cuda.empty_cache()
        return mean_post.numpy(), std_post.numpy()

    def add_data(self, x_t_new: Union[np.ndarray, torch.Tensor], y_t_new: Union[np.ndarray, torch.Tensor]) -> "ExactGPyTorchRegressionModel":
        """Add observations to the training set and condition the model on them.

        Args:
            x_t_new (Union[np.ndarray, torch.Tensor]): New inputs of shape ``(n, d)``.
            y_t_new (Union[np.ndarray, torch.Tensor]): New responses of length ``n``.

        Returns:
            ExactGPyTorchRegressionModel: Fantasy model conditioned on the combined
            training set. The receiver is left unchanged.
        """
        if isinstance(x_t_new, np.ndarray):
            x_t_new = torch.from_numpy(x_t_new)
        if isinstance(y_t_new, np.ndarray):
            y_t_new = torch.from_numpy(y_t_new)
        if not (
            x_t_new.ndim == 2
            and x_t_new.shape[1] == self.d
            and y_t_new.ndim == 1
            and len(x_t_new) == len(y_t_new)
        ):
            raise AssertionError
        if self.use_gpu:
            x_t_new, y_t_new = x_t_new.cuda(), y_t_new.cuda()
        fantasy_model = self.get_fantasy_model(x_t_new, y_t_new)
        if self.use_gpu:
            del x_t_new, y_t_new
            torch.cuda.empty_cache()
        return fantasy_model
