n = 16
sampler = qp.Lattice(2**7, seed=42)
plot_paths("BM", sampler, t_final=1, initial_value=1, drift=0, diffusion=1, n=n)
plot_paths("GBM", sampler, t_final=1, initial_value=1, drift=0, diffusion=1, n=n)
