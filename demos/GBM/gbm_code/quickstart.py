import numpy as np
import matplotlib.pyplot as plt
import qmcpy as qp

sampler = qp.Lattice(252, seed=42)  # daily steps for 1 year
gbm = qp.GeometricBrownianMotion(
    sampler, t_final=1, initial_value=1, drift=0.05, diffusion=0.2
)
paths = gbm.gen_samples(16)
t = np.linspace(0, 1.0, paths.shape[1])
plt.plot(t, paths[:5].T, alpha=0.8)
plt.xlabel("$t$")
plt.ylabel("$S_t$")
plt.title("GBM paths")
plt.show()
