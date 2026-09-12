is_debug = False

# Random seeds for reproducibility
QMCPY_SEED = 42
QUANTLIB_SEED = 7

# Seed for QuantLib's Sobol' direction integers, held fixed so that only the
# Owen-style scramble varies between replications. It must be non-zero: the
# Jaeckel table only covers the first 32 dimensions, and above that a seed of 0
# makes QuantLib draw the remaining direction integers from a clock-seeded
# generator, which would make the demo's 252-dimensional paths differ on every
# run. Reported upstream as
# https://github.com/lballabio/QuantLib/issues/2732; drop this pin if it is
# resolved. See also demos/GBM/gbm_code/quantlib_util.py.
SOBOL_DIRECTION_SEED = 1

# Halton's work scales with its base-b digit count. Thirty-two digits exceed
# this demo's resolution needs while retaining the default 'LMS DP' scramble.
HALTON_DIGITS = 32


def get_experiment_configurations() -> dict:
    """
    Define experimental parameter ranges for GBM simulations.

    Returns:
        dict: Configuration dictionary with 'time_steps' and 'paths' experiments,
              each containing 'fixed_paths'/'fixed_steps', 'range', and 'series_name'
    """
    return {
        "time_steps": {
            "fixed_paths": 2**3 if is_debug else 2**12,
            "range": (
                [2**i for i in range(4, 7)]
                if is_debug
                else [2**i for i in range(4, 10)]
            ),
            "series_name": "Time Steps",
        },
        "paths": {
            "fixed_steps": 2**5 if is_debug else 252,
            "range": (
                [2**i for i in range(6, 9)]
                if is_debug
                else [2**i for i in range(9, 15)]
            ),
            "series_name": "Paths",
        },
    }


def get_sampler_configurations() -> dict:
    """
    Define sampler types available for testing.

    Returns:
        dict: Dictionary with 'all_samplers' (QMCPy samplers) and
              'quantlib_samplers' (QuantLib-supported samplers)
    """
    return {
        "all_samplers": ["IIDStdUniform", "Sobol", "Halton", "Lattice"],
        "quantlib_samplers": ["IIDStdUniform", "Sobol", "Halton"],
    }


def get_gbm_parameters() -> dict:
    """
    Define base Geometric Brownian Motion parameters.

    Returns:
        dict: Parameters including 'initial_value' (S_0), 'mu' (drift),
              'sigma' (volatility), and 'maturity' (time horizon T)
    """
    return {"initial_value": 100, "mu": 0.05, "sigma": 0.2, "maturity": 1.0}
