import numpy as np
import scipy.fft
import itertools


def fftbr(x: np.ndarray) -> np.ndarray:
    r"""1 dimensional Bit-Reversed-Order (BRO) Fast Fourier Transform (FFT)
    along the last dimension. Requires the last dimension of x is already in
    BRO, so we can skip the first step of the decimation-in-time FFT. Requires
    the size of the last dimension is a power of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.random(8)+1j*rng.random(8)
        >>> with np.printoptions(precision=2):
        ...     fftbr(x)
        array([ 1.07+1.14j, -0.32+0.26j, -0.27+0.22j, -0.27-0.34j,  0.1 +0.16j,
               -0.15+0.17j,  0.5 +0.24j,  0.06-0.02j])
        >>> fftbr(rng.random((2,3,4,5,8))+1j*rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run BRO-FFT.

    Returns:
        np.ndarray: BRO-FFT values.
    """
    n = x.shape[-1]
    if not (n & (n - 1) == 0):  # require n is a power of 2
        raise AssertionError
    m = int(np.log2(n))
    shape = list(x.shape)
    ndim = x.ndim
    twos = [2] * m
    xs = x.reshape(shape[:-1] + twos)
    pdims = tuple(
        itertools.chain(range(ndim - 1), range(m + ndim - 2, ndim - 2, -1))
    )  # [i for i in range(ndim-1)]+[i+ndim-1 for i in range(m-1,-1,-1)]
    xrf = np.moveaxis(xs, np.arange(len(pdims)), pdims)
    xr = np.ascontiguousarray(xrf).reshape(shape)
    return scipy.fft.fft(xr, norm="ortho")


def ifftbr(x: np.ndarray) -> np.ndarray:
    r"""1 dimensional Bit-Reversed-Order (BRO) Inverse Fast Fourier Transform
    (IFFT) along the last dimension. Outputs an array in bit-reversed order, so
    we can skip the last step of the decimation-in-time IFFT. Requires the size
    of the last dimension is a power of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.random(8)+1j*rng.random(8)
        >>> with np.printoptions(precision=2):
        ...     ifftbr(x)
        array([ 1.07+1.14j, -0.29-0.22j,  0.3 +0.06j, -0.09+0.02j,  0.03+0.54j,
               -0.04-0.33j, -0.19+0.26j, -0.08+0.36j])
        >>> ifftbr(rng.random((2,3,4,5,8))+1j*rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run BRO-IFFT.

    Returns:
        np.ndarray: BRO-IFFT values.
    """
    n = x.shape[-1]
    if not (n & (n - 1) == 0):  # require n is a power of 2
        raise AssertionError
    m = int(np.log2(n))
    shape = list(x.shape)
    ndim = x.ndim
    twos = [2] * m
    x = scipy.fft.ifft(x, norm="ortho")
    xs = x.reshape(shape[:-1] + twos)
    pdims = tuple(itertools.chain(range(ndim - 1), range(m + ndim - 2, ndim - 2, -1)))
    xrf = np.moveaxis(xs, np.arange(len(pdims)), pdims)
    xr = np.ascontiguousarray(xrf).reshape(shape)
    return xr


def fwht(x: np.ndarray) -> np.ndarray:
    r"""1 dimensional Fast Walsh Hadamard Transform (FWHT) along the last
    dimension. Requires the size of the last dimension is a power of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> with np.printoptions(precision=2):
        ...     fwht(rng.random(8))
        array([ 1.07, -0.29,  0.12,  0.08,  0.1 , -0.45,  0.1 , -0.03])
        >>> fwht(rng.random((2,3,4,5,8))).shape
        (2, 3, 4, 5, 8)

    Args:
        x (np.ndarray): Array of samples at which to run FWHT.

    Returns:
        np.ndarray: FWHT values.
    """
    y = x.copy() + 0.0
    n = x.shape[-1]
    if n <= 1:
        return y
    if not (n & (n - 1) == 0):  # require n is a power of 2
        raise AssertionError
    m = int(np.log2(n))
    it = np.arange(n, dtype=np.int64).reshape(
        [2] * m
    )  # 2 x 2 x ... x 2 array (size 2^m)
    idx0 = [slice(None)] * (m - 1) + [0]
    idx1 = [slice(None)] * (m - 1) + [1]
    for k in range(m):
        eps0 = it[tuple(idx0[-(k + 1) :])].flatten()
        eps1 = it[tuple(idx1[-(k + 1) :])].flatten()
        y0, y1 = y[..., eps0], y[..., eps1]
        y[..., eps0], y[..., eps1] = (y0 + y1) / np.sqrt(2), (y0 - y1) / np.sqrt(2)
    return y


def omega_fwht(m: int) -> np.ndarray:
    r"""A useful when efficiently updating FWHT values after doubling the
    sample size.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> m = 3
        >>> x1 = rng.random((3,5,7,2**m))
        >>> x2 = rng.random((3,5,7,2**m))
        >>> x = np.concatenate([x1,x2],axis=-1)
        >>> ytrue = fwht(x)
        >>> omega = omega_fwht(m)
        >>> y1 = fwht(x1)
        >>> y2 = fwht(x2)
        >>> y = np.concatenate([y1+omega*y2,y1-omega*y2],axis=-1)/np.sqrt(2)
        >>> np.allclose(y,ytrue)
        True

    Args:
        m (int): Size $2^m$ output.

    Returns:
        np.ndarray: $\left(1\right)_{k=0}^{2^m}$.
    """
    return np.ones(2**m)


def omega_fftbr(m: int) -> np.ndarray:
    r"""A useful when efficiently updating FFT values after doubling the
    sample size.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> m = 3
        >>> x1 = rng.random((3,5,7,2**m))+1j*rng.random((3,5,7,2**m))
        >>> x2 = rng.random((3,5,7,2**m))+1j*rng.random((3,5,7,2**m))
        >>> x = np.concatenate([x1,x2],axis=-1)
        >>> ytrue = fftbr(x)
        >>> omega = omega_fftbr(m)
        >>> y1 = fftbr(x1)
        >>> y2 = fftbr(x2)
        >>> y = np.concatenate([y1+omega*y2,y1-omega*y2],axis=-1)/np.sqrt(2)
        >>> np.allclose(y,ytrue)
        True

    Args:
        m (int): Size $2^m$ output.

    Returns:
        np.ndarray: $\left(e^{- \pi \mathrm{i} k / 2^m}\right)_{k=0}^{2^m}$.
    """
    return np.exp(-np.pi * 1j * np.arange(2**m) / 2**m)
