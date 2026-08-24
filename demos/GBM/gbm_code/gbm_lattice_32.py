gbm_lattice = plot_paths(
    "GBM",
    qp.Lattice(2**8, seed=42),
    t_final=5,
    initial_value=50,
    drift=0.1,
    diffusion=0.2,
    n=32,
)
