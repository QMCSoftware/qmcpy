import QuantLib as ql
import numpy as np

if __package__:  # Imported as demos.GBM.gbm_code.quantlib_util (e.g. by pytest)
    from . import config as cf
else:  # Compatibility symlink imported with demos/GBM on sys.path.
    import config as cf

# Flat term structures need a reference date, but nothing here depends on which
# date it is: the process reads rates by year fraction, so the paths are the
# same whenever the demo runs. Using a fixed date also avoids mutating
# ql.Settings.instance().evaluationDate, which is global state.
REFERENCE_DATE = ql.Date(1, 1, 2000)


def build_quantlib_process(
    initial_value: float, mu: float, sigma: float, scheme: str = "exact"
):
    """
    Build the QuantLib stochastic process used to evolve GBM paths.

    Args:
        initial_value: Initial value of the GBM process (S_0)
        mu: Drift parameter
        sigma: Volatility parameter (note: NOT diffusion coefficient)
        scheme: 'exact' for the exact GBM solution, 'euler' for Euler-Maruyama

    Returns:
        A QuantLib one-dimensional stochastic process

    Raises:
        ValueError: If scheme is not 'exact' or 'euler'

    Notes:
        With 'exact', `BlackScholesMertonProcess.evolve()` applies
        `S_{j+1} = S_j * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z_j)`, because
        `GeneralizedBlackScholesProcess` overrides `apply(x, dx)` as
        `x*exp(dx)`. For constant drift and volatility that recursion *is* the
        exact GBM solution on the time grid, so no discretization bias is
        introduced. Setting the risk-free rate to `mu` and the dividend yield
        to zero makes the process drift equal `mu`.

        With 'euler', `GeometricBrownianMotionProcess.evolve()` applies
        `S_{j+1} = S_j * (1 + mu*dt + sigma*sqrt(dt)*Z_j)`, which carries the
        usual O(dt) discretization bias. It is kept so the demo can show what
        the choice of scheme costs.
    """
    if scheme == "exact":
        day_count = ql.Actual365Fixed()
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(initial_value)),
            ql.YieldTermStructureHandle(
                ql.FlatForward(REFERENCE_DATE, 0.0, day_count)  # dividend yield
            ),
            ql.YieldTermStructureHandle(
                ql.FlatForward(REFERENCE_DATE, mu, day_count)  # risk-free rate
            ),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(REFERENCE_DATE, ql.NullCalendar(), sigma, day_count)
            ),
        )
    if scheme == "euler":
        return ql.GeometricBrownianMotionProcess(initial_value, mu, sigma)
    raise ValueError(f"Unsupported scheme: {scheme}.  Use 'exact' or 'euler'")


def _low_discrepancy_normals(
    sampler_type: str, dimension: int, n_paths: int, seed: int
) -> np.ndarray:
    """
    Draw standard normals from a randomized QuantLib low-discrepancy sequence.

    Args:
        sampler_type: 'Sobol' or 'Halton'
        dimension: Sequence dimension (one coordinate per time step)
        n_paths: Number of points to draw
        seed: Seed for the randomization

    Returns:
        Array of shape (n_paths, dimension)

    Raises:
        ValueError: If sampler_type is not 'Sobol' or 'Halton'

    Notes:
        Both branches randomize on `seed` so that replications are independent.
        `UniformLowDiscrepancySequenceGenerator` would not: it ignores its seed
        for the fixed Jaeckel direction integers and returns the same sequence
        every time.
    """
    if sampler_type == "Sobol":
        # Burley2020SobolRsg applies a seeded Owen-style scramble; keep the
        # underlying Sobol' seed fixed at cf.SOBOL_DIRECTION_SEED (which must be
        # non-zero, see config.py) and vary only the scramble.
        uniform_rsg = ql.Burley2020SobolRsg(
            dimension, cf.SOBOL_DIRECTION_SEED, ql.SobolRsg.Jaeckel, seed
        )
        gaussian_rsg = ql.InvCumulativeBurley2020SobolGaussianRsg(uniform_rsg)
    elif sampler_type == "Halton":
        # randomStart=True draws a seeded random starting index per dimension,
        # which is what makes distinct seeds independent randomizations. Note
        # this is a random start, not a scramble, so it is a weaker
        # randomization than QMCPy's scrambled Halton.
        uniform_rsg = ql.HaltonRsg(dimension, seed, True, False)
        gaussian_rsg = ql.InvCumulativeHaltonGaussianRsg(uniform_rsg)
    else:
        raise ValueError(
            f"Unsupported low-discrepancy sampler: {sampler_type}.  "
            "Use 'Sobol' or 'Halton'"
        )
    return np.asarray([gaussian_rsg.nextSequence().value() for _ in range(n_paths)])


def _evolve_vectorized(
    normals: np.ndarray,
    initial_value: float,
    mu: float,
    sigma: float,
    dt: np.ndarray,
    scheme: str,
) -> np.ndarray:
    """
    Apply the QuantLib evolution recursion to a whole block of normals at once.

    Args:
        normals: Standard normals of shape (n_paths, n_steps)
        initial_value: Initial value of the GBM process (S_0)
        mu: Drift parameter
        sigma: Volatility parameter
        dt: Time step widths of shape (n_steps,)
        scheme: 'exact' or 'euler'

    Returns:
        Paths of shape (n_paths, n_steps + 1), including the initial value

    Notes:
        These formulas reproduce the corresponding QuantLib `evolve()` step for
        step (see build_quantlib_process); vectorizing them only avoids
        millions of Python-to-QuantLib calls in the benchmark notebook.
    """
    paths = np.empty((normals.shape[0], normals.shape[1] + 1))
    paths[:, 0] = initial_value
    if scheme == "exact":
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * normals
        paths[:, 1:] = initial_value * np.exp(np.cumsum(increments, axis=1))
    else:
        factors = 1 + mu * dt + sigma * np.sqrt(dt) * normals
        paths[:, 1:] = initial_value * np.cumprod(factors, axis=1)
    return paths


def generate_quantlib_paths(
    initial_value: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    sampler_type: str = "IIDStdUniform",
    seed: int = 7,
    scheme: str = "exact",
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
        sampler_type: Sampler to use ('IIDStdUniform', 'Sobol', or 'Halton')
        seed: Random seed for IID sampling or for the low-discrepancy
            randomization
        scheme: 'exact' for the exact GBM solution (default), 'euler' for
            Euler-Maruyama; see build_quantlib_process()

    Returns:
        tuple: (paths, process) where paths has shape (n_paths, n_steps+1)
               (includes initial value at t=0) and process is the QuantLib
               stochastic process object

    Raises:
        ValueError: If sampler_type is not 'IIDStdUniform', 'Sobol', or
            'Halton', or if scheme is not 'exact' or 'euler'

    References:
        Peter Jaeckel. Monte Carlo Methods in Finance. Wiley, 2002.
            Source of the `ql.SobolRsg.Jaeckel` direction integers used by
            the Sobol branch.

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

        John C. Hull. Options, Futures, and Other Derivatives. Pearson, 10th
            edition, 2017.
            Exact lognormal solution reproduced by the 'exact' scheme.

        Peter E. Kloeden, Eckhard Platen. Numerical Solution of Stochastic
            Differential Equations. Springer, 1992.
            Euler-Maruyama scheme used by the 'euler' scheme.
    """
    process = build_quantlib_process(initial_value, mu, sigma, scheme)
    times = ql.TimeGrid(maturity, n_steps)

    if sampler_type == "IIDStdUniform":
        # QuantLib's own path generator, one path per call.
        uniform_rng = ql.UniformRandomGenerator(seed)
        sequence_gen = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(n_steps, uniform_rng)
        )
        path_gen = ql.GaussianPathGenerator(
            process, maturity, n_steps, sequence_gen, False
        )
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            sample_path = path_gen.next().value()
            paths[i, :] = np.array([sample_path[j] for j in range(n_steps + 1)])
        return paths, process
    elif sampler_type in ("Sobol", "Halton"):
        normals = _low_discrepancy_normals(sampler_type, n_steps, n_paths, seed)
        dt = np.diff(np.asarray(list(times), dtype=float))
        paths = _evolve_vectorized(normals, initial_value, mu, sigma, dt, scheme)
        return paths, process
    else:
        raise ValueError(
            f"Unsupported sampler type: {sampler_type}.  "
            "Use 'IIDStdUniform', 'Sobol', or 'Halton'"
        )
