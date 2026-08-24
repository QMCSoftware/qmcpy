is_debug = False

# Random seeds for reproducibility
QMCPY_SEED = 42
QUANTLIB_SEED = 7


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
        "all_samplers": ["IIDStdUniform", "Sobol", "Lattice", "Halton"],
        "quantlib_samplers": ["IIDStdUniform", "Sobol"],
    }


def get_gbm_parameters() -> dict:
    """
    Define base Geometric Brownian Motion parameters.

    Returns:
        dict: Parameters including 'initial_value' (S_0), 'mu' (drift),
              'sigma' (volatility), and 'maturity' (time horizon T)
    """
    return {"initial_value": 100, "mu": 0.05, "sigma": 0.2, "maturity": 1.0}
