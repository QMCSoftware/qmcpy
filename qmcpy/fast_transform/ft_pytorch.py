import torch
import numpy as np
import itertools


def fftbr_torch(x: torch.Tensor):
    r"""Torch implementation of the 1 dimensional Bit-Reversed-Order (BRO)
    Fast Fourier Transform (FFT) along the last dimension. Requires the last
    dimension of x is already in BRO, so we can skip the first step of the
    decimation-in-time FFT. Requires the size of the last dimension is a power
    of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = torch.from_numpy(rng.random(8)+1j*rng.random(8)).to(torch.complex64).requires_grad_()
        >>> y = fftbr_torch(x)
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     y.detach()
        tensor([ 1.07+1.14j, -0.32+0.26j, -0.27+0.22j, -0.27-0.34j,  0.10+0.16j,
                -0.15+0.17j,  0.50+0.24j,  0.06-0.02j])
        >>> v = torch.abs(torch.sum(y**2))
        >>> dvdx = torch.autograd.grad(v,x)[0]
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     x.detach()
        tensor([0.25+0.65j, 0.73+0.60j, 0.06+0.20j, 0.61+0.39j, 0.45+0.06j, 0.26+0.09j,
                0.35+0.39j, 0.32+0.85j])
        >>> print("%.4f"%v.detach())
        2.5584
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     dvdx.detach()
        tensor([1.30+0.49j, 1.21+1.45j, 0.79+1.22j, 0.41+0.11j, 1.71+0.61j, 0.79+0.69j,
                0.19+0.51j, 0.12+0.90j])
        >>> fftbr_torch(torch.rand((2,3,4,5,8))+1j*torch.rand((2,3,4,5,8))).shape
        torch.Size([2, 3, 4, 5, 8])

    Args:
        x (torch.Tensor): Array of samples at which to run BRO-FFT.

    Returns:
        BRO-FFT values.
    """
    n = x.size(-1)
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
    xrf = torch.permute(xs, pdims)
    xr = xrf.contiguous().view(shape)
    return torch.fft.fft(xr, norm="ortho")


def ifftbr_torch(x: torch.Tensor):
    r"""Torch implementation of the 1 dimensional Bit-Reversed-Order (BRO)
    Inverse Fast Fourier Transform (IFFT) along the last dimension. Outputs an
    array in bit-reversed order, so we can skip the last step of the
    decimation-in-time IFFT. Requires the size of the last dimension is a power
    of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = torch.from_numpy(rng.random(8)+1j*rng.random(8)).to(torch.complex64).requires_grad_()
        >>> y = ifftbr_torch(x)
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     y.detach()
        tensor([ 1.07+1.14j, -0.29-0.22j,  0.30+0.06j, -0.09+0.02j,  0.03+0.54j,
                -0.04-0.33j, -0.19+0.26j, -0.08+0.36j])
        >>> v = torch.abs(torch.sum(y**2))
        >>> dvdx = torch.autograd.grad(v,x)[0]
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     x.detach()
        tensor([0.25+0.65j, 0.73+0.60j, 0.06+0.20j, 0.61+0.39j, 0.45+0.06j, 0.26+0.09j,
                0.35+0.39j, 0.32+0.85j])
        >>> print("%.4f"%v.detach())
        2.5656
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     dvdx.detach()
        tensor([ 1.15+0.79j,  1.51+1.01j,  0.60+0.86j,  0.06+0.54j, -0.10+0.90j,  0.47+1.37j,
                 0.37+0.20j,  0.83+1.70j])
        >>> ifftbr_torch(torch.rand((2,3,4,5,8))+1j*torch.rand((2,3,4,5,8))).shape
        torch.Size([2, 3, 4, 5, 8])

    Args:
        x (torch.Tensor): Array of samples at which to run BRO-IFFT.

    Returns:
        BRO-IFFT values.
    """
    n = x.size(-1)
    if not (n & (n - 1) == 0):  # require n is a power of 2
        raise AssertionError
    m = int(np.log2(n))
    shape = list(x.shape)
    ndim = x.ndim
    twos = [2] * m
    x = torch.fft.ifft(x, norm="ortho")
    xs = x.reshape(shape[:-1] + twos)
    pdims = tuple(itertools.chain(range(ndim - 1), range(m + ndim - 2, ndim - 2, -1)))
    xrf = torch.permute(xs, pdims)
    xr = xrf.contiguous().view(shape)
    return xr


def _fwht_torch(x):
    y = x.clone() + 0.0
    n = x.size(-1)
    if n <= 1:
        return y
    if not (n & (n - 1) == 0):  # require n is a power of 2
        raise AssertionError
    m = int(np.log2(n))
    it = torch.arange(n, dtype=torch.int64, device=x.device).reshape(
        [2] * m
    )  # 2 x 2 x ... x 2 array (size 2^m)
    idx0 = [slice(None)] * (m - 1) + [0]
    idx1 = [slice(None)] * (m - 1) + [1]
    for k in range(m):
        eps0 = it[tuple(idx0[-(k + 1) :])].flatten()
        eps1 = it[tuple(idx1[-(k + 1) :])].flatten()
        y0, y1 = y[(Ellipsis, eps0)], y[(Ellipsis, eps1)]
        y[(Ellipsis, eps0)], y[(Ellipsis, eps1)] = (y0 + y1) / np.sqrt(2), (
            y0 - y1
        ) / np.sqrt(2)
    return y


class _FWHTB2Ortho(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _fwht_torch(x)

    @staticmethod
    def backward(ctx, dx):
        return _fwht_torch(dx)


def fwht_torch(x: torch.Tensor):
    r"""Torch implementation of the 1 dimensional Fast Walsh Hadamard
    Transform (FWHT) along the last dimension. Requires the size of the last
    dimension is a power of 2.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = torch.from_numpy(rng.random(8)).float().requires_grad_()
        >>> y = fwht_torch(x)
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     y.detach()
        tensor([ 1.07, -0.29,  0.12,  0.08,  0.10, -0.45,  0.10, -0.03])
        >>> v = torch.sum(y**2)
        >>> dvdx = torch.autograd.grad(v,x)[0]
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     x.detach()
        tensor([0.25, 0.73, 0.06, 0.61, 0.45, 0.26, 0.35, 0.32])
        >>> print("%.4f"%v.detach())
        1.4694
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     dvdx.detach()
        tensor([0.50, 1.47, 0.11, 1.23, 0.90, 0.51, 0.70, 0.64])
        >>> fwht_torch(torch.rand((2,3,4,5,8))).shape
        torch.Size([2, 3, 4, 5, 8])

    Args:
        x (torch.Tensor): Array of samples at which to run FWHT.

    Returns:
        FWHT values.
    """
    return _FWHTB2Ortho.apply(x)


def omega_fwht_torch(m: int, device=None):
    r"""Torch implementation useful when efficiently updating FWHT values
    after doubling the sample size.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> m = 3
        >>> x1 = torch.from_numpy(rng.random((3,5,7,2**m)))
        >>> x2 = torch.from_numpy(rng.random((3,5,7,2**m)))
        >>> x = torch.cat([x1,x2],axis=-1)
        >>> ytrue = fwht_torch(x)
        >>> omega = omega_fwht_torch(m)
        >>> y1 = fwht_torch(x1)
        >>> y2 = fwht_torch(x2)
        >>> y = torch.cat([y1+omega*y2,y1-omega*y2],axis=-1)/np.sqrt(2)
        >>> np.allclose(y,ytrue)
        True

    Args:
        m (int): Size $2^m$ output.

    Returns:
        $\left(1\right)_{k=0}^{2^m}$.
    """
    if device is None:
        device = "cpu"
    return torch.ones(2**m, device=device)


def omega_fftbr_torch(m: int, device=None):
    r"""Torch implementation useful when efficiently updating FFT values after
    doubling the sample size.

    Examples:
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> m = 3
        >>> x1 = torch.from_numpy(rng.random((3,5,7,2**m))+1j*rng.random((3,5,7,2**m)))
        >>> x2 = torch.from_numpy(rng.random((3,5,7,2**m))+1j*rng.random((3,5,7,2**m)))
        >>> x = torch.cat([x1,x2],axis=-1)
        >>> ytrue = fftbr_torch(x)
        >>> omega = omega_fftbr_torch(m)
        >>> y1 = fftbr_torch(x1)
        >>> y2 = fftbr_torch(x2)
        >>> y = torch.cat([y1+omega*y2,y1-omega*y2],axis=-1)/np.sqrt(2)
        >>> np.allclose(y,ytrue)
        True

    Args:
        m (int): Size $2^m$ output.

    Returns:
        $\left(e^{- \pi \mathrm{i} k / 2^m}\right)_{k=0}^{2^m}$.
    """
    if device is None:
        device = "cpu"
    return torch.exp(-torch.pi * 1j * torch.arange(2**m, device=device) / 2**m)
