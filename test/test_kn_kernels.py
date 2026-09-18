from qmcpy import (
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
    KernelShiftInvar,
    KernelShiftInvarCombined,
)
from qmcpy.kernel.si_dsi_kernels import AbstractSIDSIKernel
from qmcpy.util import MethodImplementationError
from qmcpy.util.transforms import tf_exp_eps, tf_exp_eps_inv
import unittest


class KernelsTest(unittest.TestCase):

    def test_get_per_dim_components_raises_on_abstract_base(self):
        kernel = KernelShiftInvar(d=2)
        with self.assertRaises(MethodImplementationError):
            AbstractSIDSIKernel.get_per_dim_components(kernel, None, None, None, None)

    def test_si_dsi_kernel_weights_alias_lengthscales(self):
        for KernelClass in [
            KernelShiftInvar,
            KernelShiftInvarCombined,
            KernelDigShiftInvar,
            KernelDigShiftInvarAdaptiveAlpha,
            KernelDigShiftInvarCombined,
            ]:
            d = 3
            kernel = KernelClass(
                d = d,
                weights = [1/j**2 for j in range(1,d+1)])
            with self.assertRaises(ValueError) as ae:
                kernel = KernelClass(
                d = d,
                lengthscales = [1/j**2 for j in range(1,d+1)],
                weights = [1/j**2 for j in range(1,d+1)],)
            kernel = KernelClass(
                d = d,
                shape_weights = [1,])
            with self.assertRaises(ValueError) as ae:
                kernel = KernelClass(
                    d = d,
                    shape_weights = [1,],
                    shape_lengthscales = [1,])
            kernel = KernelClass(
                d = d,
                tfs_weights = (tf_exp_eps_inv, tf_exp_eps),
            )
            with self.assertRaises(ValueError) as ae:
                kernel = KernelClass(
                d = d,
                tfs_weights = (tf_exp_eps_inv, tf_exp_eps),
                tfs_lengthscales = (tf_exp_eps_inv, tf_exp_eps),
                )
            kernel = KernelClass(
                d = d,
                requires_grad_weights = True,
                )
            with self.assertRaises(ValueError) as ae:
                kernel = KernelClass(
                    d = d,
                    requires_grad_weights = True,
                    requires_grad_lengthscales = True,
                )


if __name__ == "__main__":
    unittest.main()
