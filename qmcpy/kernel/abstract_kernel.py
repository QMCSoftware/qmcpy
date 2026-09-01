from ..util import MethodImplementationError
import numpy as np
from ..util.transforms import (
    tf_exp_eps,
    tf_exp_eps_inv,
    parse_assign_param,
    insert_batch_dims,
)


class AbstractKernel(object):

    def __new__(cls, *args, **kwargs):
        if (
            ("torchify" in kwargs and kwargs["torchify"])
            or ("base_kernel" in kwargs and kwargs["base_kernel"].torchify)
            or (
                len(args) > 0
                and isinstance(args[0], AbstractKernel)
                and args[0].torchify
            )
        ):
            import torch

            x = type(cls.__name__, (cls, torch.nn.Module), {})
            instance = super().__new__(x)
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, d, torchify, device, compile_call, compile_call_kwargs):
        super().__init__()
        # dimension
        assert d % 1 == 0 and d > 0, "dimension d must be a positive int"
        self.d = d
        # torchify
        self.torchify = torchify
        if self.torchify:
            import torch

            self.npt = torch
            self.nptarray = torch.tensor
            self.nptarraytype = torch.Tensor
            self.device = torch.device(device)
            self.nptkwargs = {"device": device}
        else:
            self.npt = np
            self.nptarray = np.array
            self.nptarraytype = np.ndarray
            self.device = None
            self.nptkwargs = {}
        self.batch_param_names = []
        if compile_call:
            assert self.torchify, "compile_call requires torchify is True"
            import torch

            self.compiled_parsed___call__ = torch.compile(
                self.parsed___call__, **compile_call_kwargs
            )
        else:
            self.compiled_parsed___call__ = self.parsed___call__

    @property
    def nbdim(self):
        empty = self.npt.empty((0, self.d), **self.nptkwargs)
        v = self.__call__(empty, empty)
        nbdim = v.ndim - 1
        return nbdim

    @property
    def batch_params(self):
        return {pname: getattr(self, pname) for pname in self.batch_param_names}

    def get_batch_params(self, ndim):
        return {
            pname: insert_batch_dims(batch_param, ndim, -1)
            for pname, batch_param in self.batch_params.items()
        }

    def __call__(self, x0, x1, beta0=None, beta1=None, c=None, **kwargs):
        r"""
        Evaluate the kernel with (optional) partial derivatives

        $$\sum_{\ell=1}^p c_{\ell} \partial_{\boldsymbol{x}_0}^{\boldsymbol{\beta}_{\ell 0}} \partial_{\boldsymbol{x}_1}^{\boldsymbol{\beta}_{\ell 1}} K(\boldsymbol{x}_0,\boldsymbol{x}_1).$$

        Args:
            x0 (Union[np.ndarray, torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel with
            x1 (Union[np.ndarray, torch.Tensor]): Shape `x1.shape=(...,d)` second input to kernel with
            beta0 (Union[np.ndarray, torch.Tensor]): Shape `beta0.shape=(p,d)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_0$.
            beta1 (Union[np.ndarray, torch.Tensor]): Shape `beta1.shape=(p,d)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_1$.
            c (Union[np.ndarray, torch.Tensor]): Shape `c.shape=(p,)` coefficients of derivatives.
            kwargs (dict): keyword arguments to parsed call
        Returns:
            k (Union[np.ndarray, torch.Tensor]): Shape `y.shape=(x0+x1).shape[:-1]` kernel evaluations.
        """
        assert isinstance(x0, self.nptarraytype)
        assert isinstance(x0, self.nptarraytype)
        assert (
            x0.shape[-1] == self.d
        ), "the size of the last dimension of x0 must equal d=%d, got x0.shape=%s" % (
            self.d,
            str(tuple(x0.shape)),
        )
        assert (
            x1.shape[-1] == self.d
        ), "the size of the last dimension of x1 must equal d=%d, got x1.shape=%s" % (
            self.d,
            str(tuple(x1.shape)),
        )
        if beta0 is None:
            beta0 = self.npt.zeros((1, self.d), dtype=int, **self.nptkwargs)
        if beta1 is None:
            beta1 = self.npt.zeros((1, self.d), dtype=int, **self.nptkwargs)
        if not isinstance(beta0, self.nptarraytype):
            beta0 = self.nptarray(beta0)
        if not isinstance(beta1, self.nptarraytype):
            beta1 = self.nptarray(beta1)
        beta0 = self.npt.atleast_2d(beta0)
        beta1 = self.npt.atleast_2d(beta1)
        assert (
            beta0.ndim == 2 and beta1.ndim == 2
        ), "beta0 and beta1 must both be 2 dimensional"
        p = beta0.shape[0]
        assert beta0.shape == (
            p,
            self.d,
        ), "expected beta0.shape=(%d,%d) but got beta0.shape=%s" % (
            p,
            self.d,
            str(tuple(beta0.shape)),
        )
        assert beta1.shape == (
            p,
            self.d,
        ), "expected beta1.shape=(%d,%d) but got beta1.shape=%s" % (
            p,
            self.d,
            str(tuple(beta1.shape)),
        )
        assert (beta0 % 1 == 0).all() and (beta0 >= 0).all(), "require int beta0 >= 0"
        assert (beta1 % 1 == 0).all() and (beta1 >= 0).all(), "require int beta1 >= 0"
        if c is None:
            c = self.npt.ones(p, **self.nptkwargs)
        if not isinstance(c, self.nptarraytype):
            c = self.nptarray(c)
        c = self.npt.atleast_1d(c)
        assert c.shape == (p,), "expected c.shape=(%d,) but got c.shape=%s" % (
            p,
            str(tuple(c.shape)),
        )
        if not self.AUTOGRADKERNEL:
            batch_params = self.get_batch_params(max(x0.ndim - 1, x1.ndim - 1))
            k = self.compiled_parsed___call__(
                x0, x1, beta0, beta1, c, batch_params, **kwargs
            )
        else:
            if (beta0 == 0).all() and (beta1 == 0).all():
                batch_params = self.get_batch_params(max(x0.ndim - 1, x1.ndim - 1))
                k = c.sum() * self.compiled_parsed___call__(
                    x0, x1, batch_params, **kwargs
                )
            else:  # requires autograd, so self.npt=torch
                assert self.torchify, "autograd requires torchify=True"
                import torch

                incoming_grad_enabled = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                incoming_grad_enabled_params = {
                    pname: param.requires_grad
                    for pname, param in self.named_parameters()
                }
                if not incoming_grad_enabled:
                    for pname, param in self.named_parameters():
                        param.requires_grad_(False)
                if (beta0 > 0).any():
                    tileshapex0 = tuple(
                        self.npt.ceil(
                            self.npt.tensor(x1.shape[:-1])
                            / self.npt.tensor(x0.shape[:-1])
                        ).to(int)
                    )
                    x0gs = [
                        self.npt.tile(
                            x0[..., j].clone().requires_grad_(True), tileshapex0
                        )
                        for j in range(self.d)
                    ]
                    for x0gj in x0gs:
                        x0gj.requires_grad_(True)
                    x0g = self.npt.stack(x0gs, dim=-1)
                else:
                    x0g = x0
                if (beta1 > 0).any():
                    tileshapex1 = tuple(
                        self.npt.ceil(
                            self.npt.tensor(x0.shape[:-1])
                            / self.npt.tensor(x1.shape[:-1])
                        ).to(int)
                    )
                    x1gs = [
                        self.npt.tile(
                            x1[..., j].clone().requires_grad_(True), tileshapex1
                        )
                        for j in range(self.d)
                    ]
                    for x1gj in x1gs:
                        x1gj.requires_grad_(True)
                    x1g = self.npt.stack(x1gs, dim=-1)
                else:
                    x1g = x1
                batch_params = self.get_batch_params(max(x0.ndim - 1, x1.ndim - 1))
                k = 0.0
                k_base = self.compiled_parsed___call__(x0g, x1g, batch_params, **kwargs)
                for l in range(p):
                    if (beta0[l] > 0).any() or (beta1[l] > 0).any():
                        k_part = k_base.clone()
                        for j0 in range(self.d):
                            for _ in range(beta0[l, j0]):
                                k_part = torch.autograd.grad(
                                    k_part,
                                    x0gs[j0],
                                    grad_outputs=torch.ones_like(
                                        k_part, requires_grad=True
                                    ),
                                    create_graph=True,
                                )[0]
                        for j1 in range(self.d):
                            for _ in range(beta1[l, j1]):
                                k_part = torch.autograd.grad(
                                    k_part,
                                    x1gs[j1],
                                    grad_outputs=torch.ones_like(
                                        k_part, requires_grad=True
                                    ),
                                    create_graph=True,
                                )[0]
                    else:
                        k_part = k_base
                    k += c[l] * k_part
                if not incoming_grad_enabled:
                    for pname, param in self.named_parameters():
                        param.requires_grad_(incoming_grad_enabled_params[pname])
                if (not incoming_grad_enabled) or (
                    (not any(incoming_grad_enabled_params.values()))
                    and (not x0.requires_grad)
                    and (not x1.requires_grad)
                ):
                    k = k.detach()
                torch.set_grad_enabled(incoming_grad_enabled)
        return k

    def parsed___call__(self, *args, **kwargs):
        raise MethodImplementationError(self, "parsed___call__")

    def single_integral_01d(self, x):
        r"""
        Evaluate the integral of the kernel over the unit cube

        $$\tilde{K}(\boldsymbol{x}) = \int_{[0,1]^d} K(\boldsymbol{x},\boldsymbol{z}) \; \mathrm{d} \boldsymbol{z}.$$

        Args:
            x (Union[np.ndarray, torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel with

        Returns:
            tildek (Union[np.ndarray, torch.Tensor]): Shape `y.shape=x.shape[:-1]` integral kernel evaluations.
        """
        if self.npt == np:
            assert isinstance(x, np.ndarray)
        else:  # self.npt==torch
            assert isinstance(x, self.npt.Tensor)
        assert (
            x.shape[-1] == self.d
        ), "the size of the last dimension of x must equal d=%d, got x.shape=%s" % (
            self.d,
            str(tuple(x.shape)),
        )
        batch_params = self.get_batch_params(x.ndim - 1)
        return self.parsed_single_integral_01d(x, batch_params)

    def parsed_single_integral_01d(self, x, batch_params):
        raise MethodImplementationError(self, "parsed_single_integral_01d")

    def double_integral_01d(self):
        r"""
        Evaluate the integral of the kernel over the unit cube

        $$\tilde{K} = \int_{[0,1]^d} \int_{[0,1]^d} K(\boldsymbol{x},\boldsymbol{z}) \; \mathrm{d} \boldsymbol{x} \; \mathrm{d} \boldsymbol{z}.$$

        Returns:
            tildek (Union[np.ndarray, torch.Tensor]): Double integral kernel evaluations.
        """
        raise MethodImplementationError(self, "double_integral_01d")

    def rel_pairwise_dist_func(self, x0, x1, lengthscales):
        return self.npt.linalg.norm((x0 - x1) / (np.sqrt(2) * lengthscales), 2, -1)

    def parse_assign_param(
        self,
        pname,
        param,
        shape_param,
        requires_grad_param,
        tfs_param,
        endsize_ops,
        constraints,
    ):
        return parse_assign_param(
            pname=pname,
            param=param,
            shape_param=shape_param,
            requires_grad_param=requires_grad_param,
            tfs_param=tfs_param,
            endsize_ops=endsize_ops,
            constraints=constraints,
            torchify=self.torchify,
            npt=self.npt,
            nptkwargs=self.nptkwargs,
        )


class AbstractKernelScaleLengthscales(AbstractKernel):

    def __init__(
        self,
        d,
        scale=1.0,
        lengthscales=1.0,
        shape_scale=None,
        shape_lengthscales=None,
        tfs_scale=(tf_exp_eps_inv, tf_exp_eps),
        tfs_lengthscales=(tf_exp_eps_inv, tf_exp_eps),
        torchify=False,
        requires_grad_scale=True,
        requires_grad_lengthscales=True,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Lengthscales $\boldsymbol{\gamma}$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`.
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method.
            compile_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            torchify=torchify,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
        )
        self.raw_scale = self.parse_assign_param(
            pname="scale",
            param=scale,
            shape_param=shape_scale,
            requires_grad_param=requires_grad_scale,
            tfs_param=tfs_scale,
            endsize_ops=[1],
            constraints=["POSITIVE"],
        )
        self.tfs_scale = tfs_scale
        self.batch_param_names.append("scale")
        self.raw_lengthscales = self.parse_assign_param(
            pname="lengthscales",
            param=lengthscales,
            shape_param=[self.d] if shape_lengthscales is None else shape_lengthscales,
            requires_grad_param=requires_grad_lengthscales,
            tfs_param=tfs_lengthscales,
            endsize_ops=[1, self.d],
            constraints=["POSITIVE"],
        )
        self.tfs_lengthscales = tfs_lengthscales
        self.batch_param_names.append("lengthscales")

    @property
    def scale(self):
        return self.tfs_scale[1](self.raw_scale)

    @property
    def lengthscales(self):
        return self.tfs_lengthscales[1](self.raw_lengthscales)
