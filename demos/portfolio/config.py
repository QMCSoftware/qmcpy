"""Shared configuration for the portfolio demo notebooks.

data_portfolio_allocation.ipynb (which downloads and caches this range) and
portfolio_allocation_demo.ipynb (which reads those same cached files) must
agree on these values, or they silently read/write mismatched filenames.
"""

import os

data_dir = 'data' + os.sep

start_date = '2014-01-01'          # first date of price/return history to load
end_date = '2026-09-30'            # last date of price/return history to load
train_end_date = '2021-12-31'      # in-sample cutoff: fit/optimize weights on [start_date, train_end_date]
test_start_date = '2022-01-01'     # OOS start: evaluate those weights on [test_start_date, end_date]

# QMC point-set constructions compared throughout; each is also used via its
# simplex-transformed '<name>_simplex' variant (see portfolio_allocation_demo.ipynb).
base_sampler_types = ['iid', 'lattice', 'sobol', 'halton', 'faure', 'korobov', 'kronecker']

trading_days_per_year = 252  # used to annualize daily returns/volatility/risk-free rates
