import QuantLib as ql
import numpy as np


def generate_quantlib_paths(
    initial_value: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    sampler_type: str = "IIDStdUniform",
    seed: int = 7,
) -> tuple:
    """
    Generate Geometric Brownian Motion paths using QuantLib.

    Args:
        initial_value: Initial value of the GBM process (S_0)
        mu: Drift parameter
        sigma: Volatility parameter (note: NOT diffusion coefficient)
        maturity: Final time T
        n_steps: Number of discretization time steps
        n_paths: Number of paths to generate
        sampler_type: Type of sampler ('IIDStdUniform' or 'Sobol')
        seed: Random seed for reproducibility

    Returns:
        tuple: (paths, gbm) where paths has shape (n_paths, n_steps+1)
               (includes initial value at t=0) and gbm is the
               GeometricBrownianMotionProcess object

    Raises:
        ValueError: If sampler_type is not 'IIDStdUniform' or 'Sobol'
    """
    gbm = ql.GeometricBrownianMotionProcess(initial_value, mu, sigma)
    times = ql.TimeGrid(maturity, n_steps)
    dimension = n_steps
    if sampler_type == "IIDStdUniform":
        uniform_rng = ql.UniformRandomGenerator(seed)
        sequence_gen = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(n_steps, uniform_rng)
        )
        path_gen = ql.GaussianPathGenerator(gbm, maturity, n_steps, sequence_gen, False)
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            sample_path = path_gen.next().value()
            paths[i, :] = np.array([sample_path[j] for j in range(n_steps + 1)])
        return paths, gbm
    elif sampler_type == "Sobol":
        uniform_rsg = ql.UniformLowDiscrepancySequenceGenerator(dimension, seed)
        sequence_gen = ql.GaussianLowDiscrepancySequenceGenerator(uniform_rsg)
        path_gen = ql.GaussianSobolMultiPathGenerator(
            gbm, list(times), sequence_gen, False
        )
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value  # Set initial value
        for i in range(n_paths):
            sample_path = path_gen.next().value()
            # For 1D process, get the first (and only) path
            path_values = sample_path[0]
            paths[i, :] = np.array([path_values[j] for j in range(n_steps + 1)])
        return paths, gbm
    else:
        raise ValueError(
            f"Unsupported sampler type: {sampler_type}.  Use 'IIDStdUniform' or 'Sobol'"
        )
