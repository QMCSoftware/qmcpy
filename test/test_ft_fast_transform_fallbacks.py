import unittest

import numpy as np

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

    def test_torch_fallbacks_raise(self):
        with self.assertRaises(Exception):
            fftbr_torch()
        with self.assertRaises(Exception):
            ifftbr_torch()
        with self.assertRaises(Exception):
            fwht_torch()
        with self.assertRaises(Exception):
            omega_fftbr_torch()
        with self.assertRaises(Exception):
            omega_fwht_torch()


if __name__ == "__main__":
    unittest.main()
