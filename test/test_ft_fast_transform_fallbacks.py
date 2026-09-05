import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from qmcpy import (
    fftbr,
    fftbr_torch,
    fwht,
    fwht_torch,
    ifftbr,
    ifftbr_torch,
    omega_fftbr,
    omega_fftbr_torch,
    omega_fwht,
    omega_fwht_torch,
)


class TestFastTransformFallbacks(unittest.TestCase):

    def test_non_torch_transforms_basic(self):
        rng = np.random.default_rng(11)
        x = rng.random(8) + 1j * rng.random(8)
        y = fftbr(x)
        self.assertEqual(y.shape, x.shape)
        xr = ifftbr(y)
        self.assertEqual(xr.shape, x.shape)

        a = rng.random(8)
        b = fwht(a)
        self.assertEqual(b.shape, a.shape)

        omega = omega_fftbr(3)
        self.assertEqual(omega.shape[0], 2**3)
        omega2 = omega_fwht(3)
        self.assertEqual(omega2.shape[0], 2**3)

    def test_torch_transforms_or_fallbacks(self):
        if torch is None:
            calls = (
                (fftbr_torch, np.zeros(8, dtype=complex)),
                (ifftbr_torch, np.zeros(8, dtype=complex)),
                (fwht_torch, np.zeros(8)),
                (omega_fftbr_torch, 3),
                (omega_fwht_torch, 3),
            )
            for transform, argument in calls:
                with self.subTest(transform=transform.__name__):
                    with self.assertRaisesRegex(ModuleNotFoundError, "requires torch"):
                        transform(argument)
            return

        complex_x = torch.zeros(8, dtype=torch.complex64)
        real_x = torch.zeros(8)
        self.assertEqual(fftbr_torch(complex_x).shape, complex_x.shape)
        self.assertEqual(ifftbr_torch(complex_x).shape, complex_x.shape)
        self.assertEqual(fwht_torch(real_x).shape, real_x.shape)
        self.assertEqual(omega_fftbr_torch(3).shape[0], 2**3)
        self.assertEqual(omega_fwht_torch(3).shape[0], 2**3)


if __name__ == "__main__":
    unittest.main()
