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
        sampler_type: Type of sampler ('IIDStdUniform' or randomized 'Sobol')
        seed: Random seed for IID sampling or the Sobol scramble

    Returns:
        tuple: (paths, gbm) where paths has shape (n_paths, n_steps+1)
               (includes initial value at t=0) and gbm is the
               GeometricBrownianMotionProcess object

    Raises:
        ValueError: If sampler_type is not 'IIDStdUniform' or 'Sobol'

    References:
        Peter Jaeckel. Monte Carlo Methods in Finance. Wiley, 2002.
            Source of the `ql.SobolRsg.Jaeckel` direction integers used by
            both the IID and Sobol branches below.

        Brent Burley. Practical Hash-based Owen Scrambling. Journal of
            Computer Graphics Techniques (JCGT), 9(4), 1-20, 2020.
            https://jcgt.org/published/0009/04/01/
            Algorithm implemented by `ql.Burley2020SobolRsg` and
            `ql.InvCumulativeBurley2020SobolGaussianRsg`; this is what makes
            `seed` actually change the Sobol scramble.

        Art B. Owen. Randomly permuted (t,m,s)-nets and (t,s)-sequences. In
            Monte Carlo and Quasi-Monte Carlo Methods in Scientific
            Computing, Springer, 1995.
            Nested/Owen scrambling that Burley's hash-based method
            approximates.

        Peter E. Kloeden, Eckhard Platen. Numerical Solution of Stochastic
            Differential Equations. Springer, 1992.
            Euler-Maruyama scheme implemented by the vectorized
            `S_{j+1} = S_j * (1 + mu*dt + sigma*sqrt(dt)*Z_j)` update in the
            Sobol branch.
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
        # UniformLowDiscrepancySequenceGenerator is deterministic for the fixed
        # Jaeckel direction integers, so changing its `seed` does not create an
        # independent replication. Burley2020SobolRsg applies a seeded Owen-style
        # scramble; keep the underlying Sobol seed fixed and vary the scramble.
        uniform_rsg = ql.Burley2020SobolRsg(dimension, 0, ql.SobolRsg.Jaeckel, seed)
        gaussian_rsg = ql.InvCumulativeBurley2020SobolGaussianRsg(uniform_rsg)
        normals = np.asarray(
            [gaussian_rsg.nextSequence().value() for _ in range(n_paths)]
        )

        # QuantLib's GeometricBrownianMotionProcess uses Euler evolution:
        # S_{j+1} = S_j * (1 + mu*dt + sigma*sqrt(dt)*Z_j). Vectorizing this
        # avoids millions of Python-to-QuantLib calls in the benchmark notebook.
        time_grid = np.asarray(list(times), dtype=float)
        dt = np.diff(time_grid)
        factors = 1 + mu * dt + sigma * np.sqrt(dt) * normals

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = initial_value
        paths[:, 1:] = initial_value * np.cumprod(factors, axis=1)
        return paths, gbm
    else:
        raise ValueError(
            f"Unsupported sampler type: {sampler_type}.  Use 'IIDStdUniform' or 'Sobol'"
        )
