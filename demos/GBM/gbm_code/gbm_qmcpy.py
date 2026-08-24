import qmcpy as qp

qp_gbm = qp.GeometricBrownianMotion(qp.Lattice(2, seed=42))
qp_gbm.gen_samples(n=4)
