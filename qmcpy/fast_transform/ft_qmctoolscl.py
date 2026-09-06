import qmctoolscl
import numpy as np


def _parse_ft_input(x):
    if x.ndim == 1:
        d, n = 1, len(x)
        x = x[None, :]
        shape = (n,)
    elif x.ndim == 2:
        d, n = x.shape
        shape = (d, n)
    else:
        shape = x.shape
        n = shape[-1]
        x = x.reshape(-1, n)
        d = x.shape[0]
    if not ((n & (n - 1)) == 0):  # require n is 0 or a power of 2
        raise AssertionError
    return x, shape, d, n, n // 2


def fftbr_qmctoolscl(x: np.ndarray):
    r"""QMCToolsCL implementation of the 1 dimensional Bit-Reversed-Order
    (BRO) Fast Fourier Transform (FFT) along the last dimension. Requires the
    last dimension of x is already in BRO, so we can skip the first step of the
    decimation-in-time FFT. Requires the size of the last dimension is a power
    of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.random(8)+1j*rng.random(8)
        >>> with np.printoptions(precision=2):
        ...     fftbr_qmctoolscl(x)
        array([ 1.07+1.14j, -0.32+0.26j, -0.27+0.22j, -0.27-0.34j,  0.1 +0.16j,
               -0.15+0.17j,  0.5 +0.24j,  0.06-0.02j])
        >>> fftbr_qmctoolscl(rng.random((2,3,4,5,8))+1j*rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run BRO-FFT.

    Returns:
        BRO-FFT values.
    """
    x, shape, d, n, n_half = _parse_ft_input(x)
    if n <= 1:
        return x.reshape(shape).copy() + 0 * 1j
    twiddler = np.empty(n, dtype=np.float64)
    twiddlei = np.empty(n, dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.fft_bro_1d_radix2(1, d, int(n_half), twiddler, twiddlei, xr, xi)
    xc = xr + 1j * xi
    return xc.reshape(shape)


def ifftbr_qmctoolscl(x: np.ndarray):
    r"""QMCToolsCL implementation of the 1 dimensional Bit-Reversed-Order
    (BRO) Inverse Fast Fourier Transform (IFFT) along the last dimension.
    Outputs an array in bit-reversed order, so we can skip the last step of the
    decimation-in-time IFFT. Requires the size of the last dimension is a power
    of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.random(8)+1j*rng.random(8)
        >>> with np.printoptions(precision=2):
        ...     ifftbr_qmctoolscl(x)
        array([ 1.07+1.14j, -0.29-0.22j,  0.3 +0.06j, -0.09+0.02j,  0.03+0.54j,
               -0.04-0.33j, -0.19+0.26j, -0.08+0.36j])
        >>> ifftbr_qmctoolscl(rng.random((2,3,4,5,8))+1j*rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run BRO-IFFT.

    Returns:
        BRO-IFFT values.
    """
    x, shape, d, n, n_half = _parse_ft_input(x)
    if n <= 1:
        return x.reshape(shape).copy() + 0 * 1j
    twiddler = np.empty(n, dtype=np.float64)
    twiddlei = np.empty(n, dtype=np.float64)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmctoolscl.ifft_bro_1d_radix2(1, d, int(n_half), twiddler, twiddlei, xr, xi)
    xc = xr + 1j * xi
    return xc.reshape(shape)


def fwht_qmctoolscl(x: np.ndarray):
    r"""QMCToolsCL implementation of the 1 dimensional Fast Walsh Hadamard
    Transform (FWHT) along the last dimension. Requires the size of the last
    dimension is a power of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> with np.printoptions(precision=2):
        ...     fwht_qmctoolscl(rng.random(8))
        array([ 1.07, -0.29,  0.12,  0.08,  0.1 , -0.45,  0.1 , -0.03])
        >>> fwht_qmctoolscl(rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run FWHT.

    Returns:
        FWHT values.
    """
    x, shape, d, n, n_half = _parse_ft_input(x)
    if n <= 1:
        return x.reshape(shape).copy()
    xcp = x.copy()
    qmctoolscl.fwht_1d_radix2(1, d, int(n_half), xcp)
    return xcp.reshape(shape)
