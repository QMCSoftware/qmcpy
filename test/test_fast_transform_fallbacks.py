import numpy as np
import pytest

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


def test_non_torch_transforms_basic():
    rng = np.random.default_rng(11)
    x = rng.random(8) + 1j * rng.random(8)
    y = fftbr(x)
    assert y.shape == x.shape
    xr = ifftbr(y)
    assert xr.shape == x.shape

    a = rng.random(8)
    b = fwht(a)
    assert b.shape == a.shape

    omega = omega_fftbr(3)
    assert omega.shape[0] == 2**3
    omega2 = omega_fwht(3)
    assert omega2.shape[0] == 2**3


def test_torch_fallbacks_raise():
    with pytest.raises(Exception):
        fftbr_torch()
    with pytest.raises(Exception):
        ifftbr_torch()
    with pytest.raises(Exception):
        fwht_torch()
    with pytest.raises(Exception):
        omega_fftbr_torch()
    with pytest.raises(Exception):
        omega_fwht_torch()
