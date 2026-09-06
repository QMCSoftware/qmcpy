import numpy as np


class Polynomial:
    """
    >>> p = Polynomial([3,5,0,1,2])
    >>> x = np.random.rand(3,4)
    >>> y = p(x)
    >>> y_true = 3*x**4 + 5*x**3 + 0*x**2 + 1*x**1 + 2*x**0
    >>> assert np.allclose(y,y_true,atol=1e-12)
    """

    def __init__(self, coeffs) -> None:
        """Polynomial evaluation with Horner's rule

        Args:
            coeffs (list or np.ndarray or torch.Tensor): vector of coefficients
            e.g. coeffs = [a, b, c] corresponds to the quadratic polynomial
            a*x**2 + b*x + c
        """
        if not (isinstance(coeffs, list)):
            raise AssertionError
        self.order = len(coeffs)
        if not (self.order >= 1):
            raise AssertionError
        self.coeffs = coeffs

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            powers = np.arange(self.order - 1, -1, -1, dtype=x.dtype)
            coeffs = np.array(self.coeffs, dtype=x.dtype)
        else:
            import torch

            powers = torch.arange(
                self.order - 1, -1, -1, dtype=x.dtype, device=x.device
            )
            coeffs = torch.tensor(self.coeffs, dtype=x.dtype, device=x.device)
        return (x[..., None] ** powers * coeffs).sum(-1)


BERNOULLIPOLYSDICT = {
    # 0:  Polynomial([1]),
    1: Polynomial([1, -1 / 2]),
    2: Polynomial([1, -1, 1 / 6]),
    3: Polynomial([1, -3 / 2, 1 / 2, 0]),
    4: Polynomial([1, -2, 1, 0, -1 / 30]),
    5: Polynomial([1, -5 / 2, 5 / 3, 0, -1 / 6, 0]),
    6: Polynomial([1, -3, 5 / 2, 0, -1 / 2, 0, 1 / 42]),
    7: Polynomial([1, -7 / 2, 7 / 2, 0, -7 / 6, 0, 1 / 6, 0]),
    8: Polynomial([1, -4, 14 / 3, 0, -7 / 3, 0, 2 / 3, 0, -1 / 30]),
    9: Polynomial([1, -9 / 2, 6, 0, -21 / 5, 0, 2, 0, -3 / 10, 0]),
    10: Polynomial([1, -5, 15 / 2, 0, -7, 0, 5, 0, -3 / 2, 0, 5 / 66]),
}


def bernoulli_poly(n: int, x):
    r"""$n^\text{th}$ Bernoulli polynomial

    Examples:
        >>> x = np.arange(6).reshape((2,3))/6
        >>> available_n = list(BERNOULLIPOLYSDICT.keys())
        >>> available_n
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for n in available_n:
        ...     y = bernoulli_poly(n,x)
        ...     with np.printoptions(precision=2):
        ...         print("n = %d\n%s"%(n,y))
        n = 1
        [[-0.5  -0.33 -0.17]
         [ 0.    0.17  0.33]]
        n = 2
        [[ 0.17  0.03 -0.06]
         [-0.08 -0.06  0.03]]
        n = 3
        [[ 0.    0.05  0.04]
         [ 0.   -0.04 -0.05]]
        n = 4
        [[-0.03 -0.01  0.02]
         [ 0.03  0.02 -0.01]]
        n = 5
        [[ 0.00e+00 -2.19e-02 -2.06e-02]
         [ 1.39e-17  2.06e-02  2.19e-02]]
        n = 6
        [[ 0.02  0.01 -0.01]
         [-0.02 -0.01  0.01]]
        n = 7
        [[ 0.00e+00  2.28e-02  2.24e-02]
         [-1.39e-17 -2.24e-02 -2.28e-02]]
        n = 8
        [[-0.03 -0.02  0.02]
         [ 0.03  0.02 -0.02]]
        n = 9
        [[ 0.   -0.04 -0.04]
         [ 0.    0.04  0.04]]
        n = 10
        [[ 0.08  0.04 -0.04]
         [-0.08 -0.04  0.04]]
        >>> import scipy.special
        >>> for n in available_n:
        ...     bpoly_coeffs = BERNOULLIPOLYSDICT[n].coeffs
        ...     bpoly_coeffs_true = scipy.special.bernoulli(n)*scipy.special.comb(n,np.arange(n,-1,-1))
        ...     assert np.allclose(bpoly_coeffs_true,bpoly_coeffs,atol=1e-12)

    Args:
        n (int): Polynomial order.
        x (Union[np.ndarray, torch.Tensor]): Points at which to evaluate the
            Bernoulli polynomial.

    Returns:
        Bernoulli polynomial values.
    """
    if not (isinstance(n, int)):
        raise AssertionError
    if not (n in BERNOULLIPOLYSDICT):
        raise AssertionError("n = %d not in BERNOULLIPOLYSDICT" % n)
    bpoly = BERNOULLIPOLYSDICT[n]
    y = bpoly(x)
    return y
