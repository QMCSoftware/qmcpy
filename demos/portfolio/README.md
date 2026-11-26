# Portfolio allocation demos

| File or directory | Description |
|---|---|
| [`backtest_util.py`](backtest_util.py) | Core backtest computation: asset-metadata loading, point-in-time universe construction, periodic/event-driven rebalancing, and Sharpe-ratio computation. Unit tested directly (`test/test_tm_demo_portfolio.py`). |
| [`config.py`](config.py) | Shared configuration (price-history date range, train/test split, base sampler list) used by both notebooks below. |
| [`data/`](data/) | Committed market-price, benchmark, and risk-free-rate datasets used by the portfolio notebooks; not reproducible bit-for-bit if regenerated, since `data_portfolio_allocation.ipynb` downloads adjusted closes that Yahoo revises after the fact. The log returns derived from the prices are gitignored and rebuilt locally when missing. |
| [`data_portfolio_allocation.ipynb`](data_portfolio_allocation.ipynb) | Downloads market and Treasury-bill data and prepares the CSV datasets consumed by the main demo. |
| [`pa_util.py`](pa_util.py) | Generic, non-portfolio-specific plotting/display utilities: figures, interactive filterable tables, and syntax-highlighted source display (`show_source`). |
| [`portfolio_allocation_demo.ipynb`](portfolio_allocation_demo.ipynb) | Compares IID and low-discrepancy portfolio-weight sampling, simplex mappings, Sharpe ratios, runtime, and backtests. |
| [`portfolio_allocation_to_do.md`](portfolio_allocation_to_do.md) | Tracks completed and remaining work for the portfolio demo, documentation, tests, and poster. |
| [`sampler_util.py`](sampler_util.py) | Sampler-comparison utilities: generating simplex-transformed portfolio weights from several low-discrepancy/IID samplers, and evaluating/timing them against each other. |
