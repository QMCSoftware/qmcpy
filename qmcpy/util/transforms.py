import numpy as np
import scipy.special
from .torch_numpy_ops import get_npt

EPS64 = float(np.finfo(np.float64).eps)


def insert_batch_dims(param, ndims, k):
    ones = [1] * ndims
    return param.reshape(list(param.shape[:k]) + ones + list(param.shape[k:]))


def tf_exp(x):
    npt = get_npt(x)
    return npt.exp(x)


def tf_exp_inv(x):
    npt = get_npt(x)
    return npt.log(x)


def tf_exp_eps(x):
    return tf_exp(x) + EPS64


def tf_exp_eps_inv(x):
    return tf_exp_inv(x - EPS64)


def tf_square(x):
    return x**2


def tf_square_inv(x):
    npt = get_npt(x)
    return npt.sqrt(x)


def tf_square_eps(x):
    return tf_square(x) + EPS64


def tf_square_eps_inv(x):
    return tf_square_inv(x - EPS64)


def tf_explinear(x):
    npt = get_npt(x)
    if npt == np:
        return -scipy.special.log_expit(-x)
    else:
        return -npt.nn.functional.logsigmoid(-x)


def tf_explinear_inv(x):
    npt = get_npt(x)
    return npt.where(x < 34, npt.log(npt.expm1(x)), x)


def tf_explinear_eps(x):
    return tf_explinear(x) + EPS64


def tf_explinear_eps_inv(x):
    return tf_explinear_inv(x - EPS64)


def tf_identity(x):
    return x


def parse_assign_param(
    pname,
    param,
    shape_param,
    requires_grad_param,
    tfs_param,
    endsize_ops,
    constraints,
    torchify,
    npt,
    nptkwargs,
):
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
