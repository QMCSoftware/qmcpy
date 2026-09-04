import numpy as np
from .exceptions_warnings import ParameterError
from .torch_numpy_ops import get_npt


def k4sumterm(x, t, cutoff=1e-8):
    r"""
    $$K_4(x) = \sum_{a=0}^{t-1} \frac{x_a}{2^{3a}}$$

    where $x_a$ is the bit at index $a$ in the binary expansion of $x$
    e.g. $x = 6$ with $t=3$ has $(x_0,x_1,x_2) = (1,1,0)$

    Examples:
        >>> t = 3
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> x = rng.integers(low=0,high=2**t,size=(5,4))
        >>> with np.printoptions(precision=2):
        ...     k4sumterm(x,t)
        array([[ 1.11,  0.89,  1.11, -0.89],
               [ 1.11,  1.14,  1.11, -0.86],
               [-0.89,  0.86, -1.11,  0.89],
               [-1.11,  0.89, -0.89,  0.89],
               [-1.14, -0.89, -0.89, -0.86]])
        >>> import torch
        >>> with torch._tensor_str.printoptions(precision=2):
        ...     k4sumterm(torch.from_numpy(x),t)
        tensor([[ 1.11,  0.89,  1.11, -0.89],
                [ 1.11,  1.14,  1.11, -0.86],
                [-0.89,  0.86, -1.11,  0.89],
                [-1.11,  0.89, -0.89,  0.89],
                [-1.14, -0.89, -0.89, -0.86]])

    Args:
        x (Union[np.ndarray torch.Tensor]): Integer arrays.
        t (int): Number of bits in each integer.

    Returns:
        y (Union[np.ndarray torch.Tensor]): The $K_4$ sum term.
    """
    total = 0.0
    for a in range(0, t):
        factor = 1 / float(2.0 ** (3 * a))
        if factor < cutoff:
            break
        total += (-1.0) ** ((x >> (t - a - 1)) & 1) * factor
    return total


WEIGHTEDWALSHFUNCSPOS = {
    2: lambda beta, xf, xb, t: -beta * xf + 5 / 2 * (1 - 2 ** (-beta)),
    3: lambda beta, xf, xb, t: beta * xf**2
    - 5 * (1 - 2 ** (-beta)) * xf
    + 43 / 18 * (1 - 2 ** (-2 * beta)),
    4: lambda beta, xf, xb, t: -2 / 3 * beta * xf**3
    + 5 * (1 - 2 ** (-beta)) * xf**2
    - 43 / 9 * (1 - 2 ** (-2 * beta)) * xf
    + 701 / 294 * (1 - 2 ** (-3 * beta))
    + beta * (1 / 48 * k4sumterm(xb, t) - 1 / 42),
}

WEIGHTEDWALSHFUNCSZEROS = {
    2: 5 / 2,
    3: 43 / 18,
    4: 701 / 294,
}


def weighted_walsh_funcs(alpha, xb, t):
    r"""
    Weighted walsh functions

    $$\sum_{k=0}^\infty \mathrm{wal}_k(x) 2^{-\mu_\alpha(k)}$$

    where $\mathrm{wal}_k$ is the $k^\text{th}$ Walsh function
    and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of $1$ bits in the binary expansion of $k$
    e.g. $k=13=1101_2$ has 1-bit indexes $(4,3,1)$ so

    $$\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots$$

    Examples:
        >>> t = 3
        >>> rng = np.random.Generator(np.random.SFC64(11))
        >>> xb = rng.integers(low=0,high=2**t,size=(2,3))
        >>> available_alpha = list(WEIGHTEDWALSHFUNCSPOS.keys())
        >>> available_alpha
        [2, 3, 4]
        >>> for alpha in available_alpha:
        ...     y = weighted_walsh_funcs(alpha,xb,t)
        ...     with np.printoptions(precision=2):
        ...         print("alpha = %d\n%s"%(alpha,y))
        alpha = 2
        [[1.81 1.38 1.81]
         [0.62 1.81 2.5 ]]
        alpha = 3
        [[1.85 1.43 1.85]
         [0.62 1.85 2.39]]
        alpha = 4
        [[1.85 1.43 1.85]
         [0.62 1.85 2.38]]
        >>> import torch
        >>> for alpha in available_alpha:
        ...     y = weighted_walsh_funcs(alpha,torch.from_numpy(xb),t)
        ...     with torch._tensor_str.printoptions(precision=2):
        ...         print("alpha = %d\n%s"%(alpha,y))
        alpha = 2
        tensor([[1.81, 1.38, 1.81],
                [0.62, 1.81, 2.50]])
        alpha = 3
        tensor([[1.85, 1.43, 1.85],
                [0.62, 1.85, 2.39]])
        alpha = 4
        tensor([[1.85, 1.43, 1.85],
                [0.62, 1.85, 2.38]])

    Args:
        alpha (int): Weighted walsh functions order.
        xb (Union[np.ndarray, torch.Tensor]): Integer points at which to evaluate the weighted Walsh function.
        t (int): Number of bits in each integer in xb.

    Returns:
        y (Union[np.ndarray, torch.Tensor]): Weighted Walsh function values.

    **References:**

    1.  Dick, Josef.
        "Walsh spaces containing smooth functions and quasi–Monte Carlo rules of arbitrary high order."
        SIAM Journal on Numerical Analysis 46.3 (2008): 1519-1553.

    2.  Dick, Josef.
        "The decay of the Walsh coefficients of smooth functions."
        Bulletin of the Australian Mathematical Society 80.3 (2009): 430-453.
    """
    assert isinstance(alpha, int)
    assert alpha in WEIGHTEDWALSHFUNCSPOS, (
        "alpha = %d not in WEIGHTEDWALSHFUNCSPOS" % alpha
    )
    assert alpha in WEIGHTEDWALSHFUNCSZEROS, (
        "alpha = %d not in WEIGHTEDWALSHFUNCSZEROS" % alpha
    )
    if isinstance(xb, np.ndarray):
        np_or_torch = np
        y = np.ones(xb.shape)
    else:
        import torch

        np_or_torch = torch
        y = torch.ones(xb.shape, device=xb.device)
    pidxs = xb > 0
    y[~pidxs] = WEIGHTEDWALSHFUNCSZEROS[alpha]
    xfpidxs = (2 ** (-t)) * xb[pidxs]
    betapidxs = -np_or_torch.floor(np_or_torch.log2(xfpidxs))
    y[pidxs] = WEIGHTEDWALSHFUNCSPOS[alpha](betapidxs, xfpidxs, xb[pidxs], t)
    return y


def to_bin(x, t):
    r"""
    Convert floating point representations of digital net samples in base $b=2$ to binary representations.

    Examples:
        >>> xf = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(5))
        >>> xf
        array([0.62509547, 0.8972138 , 0.77568569, 0.22520719, 0.30016628])
        >>> xb = to_bin(xf,2)
        >>> xb
        array([2, 3, 3, 0, 1], dtype=uint64)
        >>> to_bin(xb,2)
        array([2, 3, 3, 0, 1], dtype=uint64)
        >>> import torch
        >>> xftorch = torch.from_numpy(xf)
        >>> xftorch
        tensor([0.6251, 0.8972, 0.7757, 0.2252, 0.3002], dtype=torch.float64)
        >>> xbtorch = to_bin(xftorch,2)
        >>> xbtorch
        tensor([2, 3, 3, 0, 1])
        >>> to_bin(xbtorch,2)
        tensor([2, 3, 3, 0, 1])


    Args:
        x (Union[np.ndarray, torch.Tensor]): floating point representation of samples.
        t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.

    Returns:
        xb (Unioin[np.ndarray,torch.Tensor]): binary representation of samples with `dtype` either `np.uint64` or `torch.int64`.
    """
    npt = get_npt(x)
    if npt == np:
        if npt.issubdtype(x.dtype, npt.floating):
            return npt.floor((x % 1) * 2.0**t).astype(npt.uint64)
        elif npt.issubdtype(x.dtype, npt.integer):
            return x
        else:
            raise ParameterError("x.dtype must be float or int, got %s" % str(x.dtype))
    else:  # npt==torch
        if npt.is_floating_point(x):
            return npt.floor((x % 1) * 2.0**t).to(npt.int64)
        elif (not npt.is_floating_point(x)) and (not npt.is_complex(x)):  # int type
            return x
        else:
            raise ParameterError("x.dtype must be float or int, got %s" % str(x.dtype))


def to_float(x, t):
    r"""
    Convert binary representations of digital net samples in base $b=2$ to floating point representations.

    Examples:
        >>> xb = np.arange(8,dtype=np.uint64)
        >>> xb
        array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)
        >>> to_float(xb,3)
        array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875])
        >>> xbtorch = bin_from_numpy_to_torch(xb)
        >>> xbtorch
        tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> to_float(xbtorch,3)
        tensor([0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750])

    Args:
        x (Union[np.ndarray, torch.Tensor]): binary representation of samples with `dtype` either `np.uint64` or `torch.int64`.
        t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.

    Returns:
        xf (Unioin[np.ndarray,torch.Tensor]): floating point representation of samples.
    """
    npt = get_npt(x)
    if npt == np:  # npt==torch
        if x.dtype == np.uint64:
            return x.astype(np.float64) * 2.0 ** (-t)
        elif npt.is_floating_point(x):
            return x
        else:
            raise ParameterError("x.dtype must be np.uint64, got %s" % str(x.dtype))
    else:
        if x.dtype == npt.int64:
            return x.to(npt.get_default_dtype()) * 2.0 ** (-t)
        elif npt.is_floating_point(x):
            return x
        else:
            raise ParameterError("x.dtype must be torch.int64, got %s" % str(x.dtype))


def bin_from_numpy_to_torch(xb):
    r"""
    Convert `numpy.uint64` to `torch.int64`, useful for converting binary samples from `DigitalNetB2` to torch representations.

    Examples:
        >>> xb = np.arange(8,dtype=np.uint64)
        >>> xb
        array([0, 1, 2, 3, 4, 5, 6, 7], dtype=uint64)
        >>> bin_from_numpy_to_torch(xb)
        tensor([0, 1, 2, 3, 4, 5, 6, 7])

    Args:
        xb (Union[np.ndarray]): binary representation of samples with `dtype=np.uint64`

    Returns:
        xbtorch (Unioin[torch.Tensor]): binary representation of samples with `dtype=torch.int64`.
    """
    assert xb.dtype == np.uint64
    assert xb.max() <= (2**63 - 1), "require all xb < 2^63"
    import torch

    return torch.from_numpy(xb.astype(np.int64))
