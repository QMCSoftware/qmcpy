# Generate GBM samples for theoretical validation
import qmcpy as qp
import numpy as np

S0, mu, sigma, T, n_samples = 100.0, 0.05, 0.20, 1.0, 2**12
diffusion = sigma**2
sampler = qp.Lattice(5, seed=42)
qp_gbm = qp.GeometricBrownianMotion(
    sampler, t_final=T, initial_value=S0, drift=mu, diffusion=diffusion
)
paths = qp_gbm.gen_samples(n_samples)
S_T = paths[:, -1]  # Final values only

# Calculate theoretical vs empirical sample moments
theo_mean = S0 * np.exp(mu * T)
theo_var = S0**2 * np.exp(2 * mu * T) * (np.exp(diffusion * T) - 1)
qp_emp_mean = np.mean(S_T)
qp_emp_var = np.var(S_T, ddof=1)
print(f"Mean: {qp_emp_mean:.3f} (theoretical: {theo_mean:.3f})")
print(f"Variance: {qp_emp_var:.3f} (theoretical: {theo_var:.3f})")
qp_gbm
