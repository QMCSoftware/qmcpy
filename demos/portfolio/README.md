# Portfolio allocation demos

| File or directory | Description |
|---|---|
| [`data/`](data/) | Regenerated (gitignored) market-price, log-return, and risk-free-rate datasets used by the portfolio notebooks; not reproducible bit-for-bit, since `data_portfolio_allocation.ipynb` downloads adjusted closes that Yahoo revises after the fact. |
| [`data_portfolio_allocation.ipynb`](data_portfolio_allocation.ipynb) | Downloads market and Treasury-bill data and prepares the CSV datasets consumed by the main demo. |
| [`portfolio_allocation_demo.ipynb`](portfolio_allocation_demo.ipynb) | Compares IID and low-discrepancy portfolio-weight sampling, simplex mappings, Sharpe ratios, runtime, and backtests. |
| [`portfolio_allocation_to_do.md`](portfolio_allocation_to_do.md) | Tracks completed and remaining work for the portfolio demo, documentation, tests, and poster. |
