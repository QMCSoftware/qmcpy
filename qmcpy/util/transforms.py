from __future__ import annotations

from typing import TYPE_CHECKING, Union
import types
import numpy as np
import scipy.special
from .torch_numpy_ops import get_npt

if TYPE_CHECKING:
    import torch

EPS64 = float(np.finfo(np.float64).eps)


def insert_batch_dims(param: Union[np.ndarray, torch.Tensor], ndims: int, k: int) -> Union[np.ndarray, torch.Tensor]:
    """Insert singleton dimensions into a parameter so it broadcasts against batched inputs.

    Args:
        param (Union[np.ndarray, torch.Tensor]): Parameter to reshape.
        ndims (int): Number of singleton dimensions to insert.
        k (int): Position at which to insert them.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``param`` with ``ndims`` singleton axes
            inserted after its first ``k`` axes.
    """
    ones = [1] * ndims
    return param.reshape(list(param.shape[:k]) + ones + list(param.shape[k:]))


def tf_exp(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Exponential transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``exp(x)``.
    """
    npt = get_npt(x)
    return npt.exp(x)


def tf_exp_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the exponential transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``log(x)``.
    """
    npt = get_npt(x)
    return npt.log(x)


def tf_exp_eps(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Exponential transform offset by machine epsilon.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``exp(x) + eps``, kept strictly positive.
    """
    return tf_exp(x) + EPS64


def tf_exp_eps_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the epsilon-offset exponential transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``log(x - eps)``.
    """
    return tf_exp_inv(x - EPS64)


def tf_square(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Square transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``x**2``.
    """
    return x**2


def tf_square_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the square transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``sqrt(x)``.
    """
    npt = get_npt(x)
    return npt.sqrt(x)


def tf_square_eps(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Square transform offset by machine epsilon.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``x**2 + eps``, kept strictly positive.
    """
    return tf_square(x) + EPS64


def tf_square_eps_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the epsilon-offset square transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``sqrt(x - eps)``.
    """
    return tf_square_inv(x - EPS64)


def tf_explinear(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Exponential-linear (softplus) transform.

    Behaves like ``exp(x)`` for small ``x`` and like ``x`` for large ``x``, so it
    maps the real line to the positive reals without overflowing.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``log(1 + exp(x))``, computed stably.
    """
    npt = get_npt(x)
    if npt == np:
        return -scipy.special.log_expit(-x)
    else:
        return -npt.nn.functional.logsigmoid(-x)


def tf_explinear_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the exponential-linear transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``log(expm1(x))``, falling back to ``x`` once ``x >= 34``
            where the two agree to machine precision.
    """
    npt = get_npt(x)
    return npt.where(x < 34, npt.log(npt.expm1(x)), x)


def tf_explinear_eps(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Exponential-linear transform offset by machine epsilon.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``tf_explinear(x) + eps``, kept strictly positive.
    """
    return tf_explinear(x) + EPS64


def tf_explinear_eps_inv(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse of the epsilon-offset exponential-linear transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``tf_explinear_inv(x - eps)``.
    """
    return tf_explinear_inv(x - EPS64)


def tf_identity(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Identity transform.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input values.

    Returns:
        Union[np.ndarray, torch.Tensor]: ``x`` unchanged.
    """
    return x


def parse_assign_param(
    pname: str,
    param: Union[float, np.ndarray, torch.Tensor],
    shape_param: list,
    requires_grad_param: bool,
    tfs_param: tuple,
    endsize_ops: list,
    constraints: list,
    torchify: bool,
    npt: types.ModuleType,
    nptkwargs: dict,
) -> tuple:
    """Validate and normalize one kernel parameter, returning it in array form.

    A scalar is broadcast to ``shape_param``; an array-like is converted to the
    backend array type and checked against the supplied constraints.

    Args:
        pname (str): Parameter name, used in error messages.
        param (Union[float, np.ndarray, torch.Tensor]): Value to normalize.
        shape_param (list): Target shape used when ``param`` is a scalar.
        requires_grad_param (bool): Whether the torch parameter requires a gradient.
        tfs_param (tuple): Pair of forward and inverse transforms for this parameter.
        endsize_ops (list): Permitted sizes for the trailing dimension.
        constraints (list): Constraints the parameter must satisfy.
        torchify (bool): Return a ``torch.Tensor`` rather than an ``np.ndarray``.
        npt (types.ModuleType): Array backend, either ``numpy`` or ``torch``.
        nptkwargs (dict): Backend keyword arguments such as ``dtype`` and ``device``.

    Returns:
        tuple: The normalized parameter, its shape, and its transformed value.
    """
    if np.isscalar(param):
        param = param * npt.ones(shape_param, **nptkwargs)
    else:
        if torchify:
            if not isinstance(param, npt.Tensor):
                param = npt.tensor(param)
            param = npt.atleast_1d(param)
            if not (isinstance(param, npt.Tensor)):
                raise AssertionError(
                    "%s must be a scalar or torch.Tensor" % pname
                )
        else:
            if not isinstance(param, npt.ndarray):
                param = npt.array(param)
            param = npt.atleast_1d(param)
            if not (isinstance(param, npt.ndarray)):
                raise AssertionError(
                    "%s must be a scalar or np.ndarray" % pname
                )
    shape_param = list(param.shape)
    if not (len(shape_param) >= 1):
        raise AssertionError("invalid shape_%s = %s" % (pname, str(shape_param)))
    if not (len(tfs_param) == 2):
        raise AssertionError("tfs_scale should be a tuple of length 2")
    if not (callable(tfs_param[0])):
        raise AssertionError("tfs_scale[0] should be a callable e.g. torch.log")
    if not (callable(tfs_param[1])):
        raise AssertionError("tfs_scale[1] should be a callable e.g. torch.exp")
    raw_param = tfs_param[0](param)
    if torchify:
        if not (isinstance(requires_grad_param, bool)):
            raise AssertionError
        if requires_grad_param:
            raw_param = 1.0 * raw_param
        raw_param = npt.nn.Parameter(raw_param, requires_grad=requires_grad_param)
    if not (shape_param[-1] in endsize_ops):
        raise AssertionError("%s not in %s" % (
            str(shape_param[-1]),
            str(endsize_ops),
        ))
    if "POSITIVE" in constraints:
        if not ((param > 0).all()):
            raise AssertionError("%s must be positive" % pname)
    if "NON-NEGATIVE" in constraints:
        if not ((param >= 0).all()):
            raise AssertionError("%s must be non-negative" % pname)
    if "INTEGER" in constraints:
        if not ((param % 1 == 0).all()):
            raise AssertionError("%s must be integers" % pname)
    return raw_param
