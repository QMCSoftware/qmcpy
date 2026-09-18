import gc

import numpy as np
import qmcpy as qp

if __package__:  # Imported as demos.GBM.gbm_code.qmcpy_util (e.g. by pytest)
    from . import config as cf
else:  # Compatibility symlink imported with demos/GBM on sys.path.
    import config as cf


def create_qmcpy_sampler(
    sampler_type: str, dimension: int, replications: int = 1, seed: int = 42
):
    """
    Create a sampler instance based on type and dimension.

    Args:
        sampler_type: Type of sampler ('IIDStdUniform', 'Sobol', 'Lattice', 'Halton')
        dimension: Dimension of the sampler (typically n_steps)
        replications: Number of independent replications
        seed: Random seed for reproducibility

    Returns:
        QMCPy sampler instance
    """
    if sampler_type == "IIDStdUniform":
        return qp.IIDStdUniform(dimension, replications, seed=seed)
    elif sampler_type == "Sobol":
        return qp.Sobol(dimension, replications, seed=seed)
    elif sampler_type == "Lattice":
        return qp.Lattice(dimension, replications, seed=seed)
    elif sampler_type == "Halton":
        # Keep the default 'LMS DP' randomization but trim the digit array; see
        # cf.HALTON_DIGITS for why this costs no accuracy.
        return qp.Halton(dimension, replications, seed=seed, t=cf.HALTON_DIGITS)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def generate_qmcpy_paths(
    initial_value: float,
    mu: float,
    diffusion: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    sampler_type: str = "IIDStdUniform",
    replications: int = 1,
    seed: int = 42,
    decomp_type: str = "PCA",
    monitoring_times=None,
):
    """
    Generate Geometric Brownian Motion paths using QMCPy with multiple replications.

    Halton uses consecutive batches from one sampler to limit its two temporary
    digit buffers to ``cf.HALTON_MAX_DIGIT_BYTES`` (or one point if larger).
    Paths and randomizations are preserved up to floating-point rounding in the
    transform. Permutation tables and returned paths require additional memory.
    Before constructing Halton, collect unreachable measures so repeated timing
    calls do not retain their permutation tables. Timings include this cleanup.

    Args:
        initial_value: Initial value of the GBM process (S_0)
        mu: Drift parameter
        diffusion: Diffusion coefficient (sigma^2)
        maturity: Final time T
        n_steps: Number of discretization time steps
        n_paths: Number of paths to generate per replication
        sampler_type: Type of sampler ('IIDStdUniform', 'Sobol', 'Lattice', 'Halton')
        replications: Number of independent replications
        seed: Random seed for reproducibility
        decomp_type: Path construction, 'PCA' (QMCPy's default), 'Cholesky', or
            'BrownianBridge'. All three describe the same process and give the
            same distribution of S_T; they differ in which low-discrepancy
            coordinate drives which feature of the path.
        monitoring_times: Optional custom sampling times for
            decomp_type='BrownianBridge'. Passing this with PCA/Cholesky raises
            ParameterError. Pass this to make BrownianBridge share their evenly
            spaced grid instead of its own default van der Corput times.

    Returns:
        tuple: (paths, gbm) where paths has shape (n_paths, n_steps) if replications is None,
               or (replications, n_paths, n_steps) if replications>=1,
               and gbm is the GeometricBrownianMotion object
    """
    if sampler_type == "Halton":
        # GBM has a self-reference; timeit disables automatic cyclic collection.
        gc.collect()
    sampler = create_qmcpy_sampler(sampler_type, n_steps, replications, seed)
    gbm = qp.GeometricBrownianMotion(
        sampler,
        t_final=maturity,
        initial_value=initial_value,
        drift=mu,
        diffusion=diffusion,
        decomp_type=decomp_type,
        monitoring_times=monitoring_times,
    )
    n_paths = int(n_paths)
    if sampler_type == "Halton":
        bytes_per_path = 2 * sampler.replications * sampler.d * int(sampler.t) * 8
        batch_size = max(1, cf.HALTON_MAX_DIGIT_BYTES // bytes_per_path)
        if batch_size < n_paths <= sampler.n_limit:
            shape = (n_paths, sampler.d)
            if not sampler.no_replications:
                shape = (sampler.replications,) + shape
            paths = np.empty(shape, dtype=np.float64)
            for start in range(0, n_paths, batch_size):
                stop = min(start + batch_size, n_paths)
                paths[..., start:stop, :] = gbm.gen_samples(n_min=start, n_max=stop)
            return paths, gbm
    paths = gbm.gen_samples(n_paths)
    return paths, gbm
