# QMCPy 2.2 Portfolio allocation to-do

## Demo

1. What are the differences between the two notebooks in `demos/portfolio`? Keep `portfolio_allocation_demo` and merge content from `archive.ipynb` into it if necessary. Afterwards, delete `archive.ipynb`. --DONE
2. Add Sobol and Halton samplers for weights generation in the demo --DONE
3. Create a single function to generate weights for all sampler types --DONE. See `gen_weights*()`.
4. Update visualizations in the demo with all QMC samplers and risk categories --DONE
5. Make a function for collecting data (?) --DONE (`evaluate_sampler_sharpe`)
6. Update start_date and end_date --DONE
7. Save price datasets to `data` folder --DONE
8. Save figures as .png in `images` --DONE
9. Use same colors for same samplers across all visualizations --DONE
10. Use white background for all visualizations --DONE
11. Implement a more optimal simplex transformation for low-discrepancy sequences --DONE. See `SimplexUniform` in `simplex_uniform.py`.
12. Clean up demo --TODO
14. Improve documentation --TODO
15. Implement multiple randomized backtests --TODO
16. Look into Sortino ratio --TODO
17. Explain how we choose tickers, testing/training split: were train_end_date/test_start_date/tickers1 etc. chosen by objective criteria, or subjectively? --TODO
18. Handle tickers with a later listing date (e.g. MRNA, ABNB): their missing early price history gets silently dropped (not held as cash) when concatenated, understating portfolio value until a sudden jump at listing; fix via point-in-time universe construction. --TODO

## Documentation

1. Stop checking in .png or binary files into repository --DONE
2. Create `mkdocs.yml` documentation for the demo and make it like a blog --TODO 
3. Add references to the demo--DONE
4. Add more text in Markdown cells before code cells to the demo --TODO

## Tests

1. Utilize QMCPy's `replications` parameter for averaged results --DONE
2. Implement out-of-sample backtesting --DONE
3. Make tb_portfolio_allocation_demo.py work in CI tests on Windows --DONE
4. Implement rebalancing



## Poster

1. Update poster in Overleaf --TODO
