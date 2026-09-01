from .abstract_kernel import AbstractKernelScaleLengthscales
from ..util.transforms import tf_exp_eps, tf_exp_eps_inv, tf_identity
from ..util.shift_invar_ops import BERNOULLIPOLYSDICT, bernoulli_poly
from ..util.dig_shift_invar_ops import (
    to_bin,
    bin_from_numpy_to_torch,
    to_float,
    weighted_walsh_funcs
)
from ..util import ParameterError, MethodImplementationError
import numpy as np
import scipy.special


class AbstractSIDSIKernel(AbstractKernelScaleLengthscales):

    AUTOGRADKERNEL = False

    def __init__(
        self,
        d,
        scale,
        lengthscales,
        alpha,
        shape_alpha,
        alpha_endsize_ops,
        shape_scale,
        shape_lengthscales,
        tfs_alpha,
        tfs_scale,
        tfs_lengthscales,
        torchify,
        requires_grad_alpha,
        requires_grad_scale,
        requires_grad_lengthscales,
        device,
        compile_call,
        compile_call_kwargs,
        weights,
        shape_weights,
        tfs_weights,
        requires_grad_weights,
    ):
        # alias lengthscales with weights
        if weights is not None:
            if lengthscales is not None:
                raise ValueError("weights is an alias for lengthscales, so leave lengthscales=None if passing in weights")
            lengthscales = weights
        if shape_weights is not None:
            if shape_lengthscales is not None:
                raise ValueError("shape_weights is an alias for shape_lengthscales, so leave shape_lengthscales=None if passing in shape_weights")
            shape_lengthscales = shape_weights
        if tfs_weights is not None:
            if tfs_lengthscales is not None:
                raise ValueError("tfs_weights is an alias for tfs_lengthscales, so leave tfs_lengthscales=None if passing in tfs_weights")
            tfs_lengthscales = tfs_weights
        if requires_grad_weights is not None:
            if requires_grad_lengthscales is not None:
                raise ValueError("requires_grad_weights is an alias for requires_grad_lengthscales, so leave requires_grad_lengthscales=None if passing in requires_grad_weights")
            requires_grad_lengthscales = requires_grad_weights
        # default requires_grad values
        if requires_grad_alpha is None:
            requires_grad_alpha = True
        if requires_grad_scale is None:
            requires_grad_scale = True
        if requires_grad_lengthscales is None:
            requires_grad_lengthscales = True
        # default lengthscales and check if None
        input_lengthscales_is_none = lengthscales is None
        # default transforms
        if input_lengthscales_is_none:
            lengthscales = 1.0
        if tfs_alpha is None:
            tfs_alpha = (tf_exp_eps_inv, tf_exp_eps)
        if tfs_scale is None:
            tfs_scale = (tf_exp_eps_inv, tf_exp_eps)
        if tfs_lengthscales is None:
            tfs_lengthscales = (tf_exp_eps_inv, tf_exp_eps)
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            tfs_scale=tfs_scale,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
        )
        self.raw_alpha = self.parse_assign_param(
            pname="alpha",
            param=alpha,
            shape_param=shape_alpha,
            requires_grad_param=requires_grad_alpha,
            tfs_param=tfs_alpha,
            endsize_ops=alpha_endsize_ops,
            constraints=["POSITIVE"],
        )
        self.tfs_alpha = tfs_alpha
        if input_lengthscales_is_none:
            lengthscales_new = 2 ** (1 / self.d) - 1.0
            raw_lengthscales_new = self.tfs_lengthscales[0](
                self.nptarray([lengthscales_new])
            )
            if self.torchify:
                self.raw_lengthscales.data[:] = raw_lengthscales_new
            else:
                self.raw_lengthscales[:] = raw_lengthscales_new

    @property
    def alpha(self):
        return self.tfs_alpha[1](self.raw_alpha)

    def parsed_single_integral_01d(self, x, batch_params):
        return batch_params["scale"][..., 0] + 0 * x[..., 0]

    def double_integral_01d(self):
        return self.scale[..., 0]

    def combine_per_dim_components_raw_m1(
        self, kparts, beta0, beta1, c, batch_params, stable
    ):
        scale = batch_params["scale"][..., 0]
        lengthscales = batch_params["lengthscales"]
        ind = 1.0 * ((beta0 + beta1) == 0)
        v = 1
        p = lengthscales[..., None, :] * kparts
        sc = scale * c.sum(-1)
        if stable:
            v = p[..., 0]
            icp = ind[..., 0]
            for j in range(1, p.shape[-1]):
                v = v * (ind[..., j] + p[..., j]) + icp * p[..., j]
                icp = icp * ind[..., j]
            v = v - 1.0 * (icp == 0)
            v = scale * (v * c).sum(-1)
        else:
            v = scale * ((ind + p).prod(-1) * c).sum(-1) - sc
        return sc, v

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        raise MethodImplementationError(self, "get_per_dim_components")

    def combine_per_dim_components(self, kparts, beta0, beta1, c, batch_params, stable):
        sc, v = self.combine_per_dim_components_raw_m1(
            kparts, beta0, beta1, c, batch_params, stable
        )
        k = sc + v
        return k

    def parsed___call__(self, x0, x1, beta0, beta1, c, batch_params, stable=False):
        kparts = self.get_per_dim_components(x0, x1, beta0, beta1)
        k = self.combine_per_dim_components(
            kparts, beta0, beta1, c, batch_params, stable
        )
        return k


class KernelShiftInvar(AbstractSIDSIKernel):
    r""" 
    Shift invariant kernel with 
    smoothness $\boldsymbol{\alpha}$, product weights (lengthscales) $\boldsymbol{\gamma}$, and scale $S$:
    
    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \tilde{K}_{\alpha_j}((x_j - z_j) \mod 1))\right), \\ 
        \tilde{K}_\alpha(x) &= (-1)^{\alpha+1}\frac{(2 \pi)^{2 \alpha}}{(2\alpha)!} B_{2\alpha}(x)
    \end{aligned}$$

    where $B_n$ is the $n^\text{th}$ Bernoulli polynomial.
    
    Examples:
        >>> from qmcpy import Lattice, fftbr, ifftbr
        >>> n = 8
        >>> d = 4
        >>> lat = Lattice(d,seed=11)
        >>> x = lat(n)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('float64')
        >>> kernel = KernelShiftInvar(
        ...     d = d, 
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        91.23444453396341
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [91.23 -2.32  5.69  5.69 12.7  -4.78 -4.78 12.7 ]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[91.23 -2.32  5.69  5.69 12.7  -4.78 -4.78 12.7 ]
         [-2.32 91.23  5.69  5.69 -4.78 12.7  12.7  -4.78]
         [ 5.69  5.69 91.23 -2.32 12.7  -4.78 12.7  -4.78]
         [ 5.69  5.69 -2.32 91.23 -4.78 12.7  -4.78 12.7 ]
         [12.7  -4.78 12.7  -4.78 91.23 -2.32  5.69  5.69]
         [-4.78 12.7  -4.78 12.7  -2.32 91.23  5.69  5.69]
         [-4.78 12.7  12.7  -4.78  5.69  5.69 91.23 -2.32]
         [12.7  -4.78 -4.78 12.7   5.69  5.69 -2.32 91.23]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fftbr(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(ifftbr(fftbr(y)*lam),kmat@y)
        True
        >>> np.allclose(ifftbr(fftbr(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch 
        >>> xtorch = torch.from_numpy(x)
        >>> kernel_torch = KernelShiftInvar(
        ...     d = d, 
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> kernel.single_integral_01d(x)
        array([10., 10., 10., 10., 10., 10., 10., 10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10., 10., 10., 10., 10., 10., 10., 10.], dtype=torch.float64,
               grad_fn=<AddBackward0>)
        
        Batch Params 
        
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelShiftInvar(
        ...     d = 2, 
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,2])
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kfast = kernel(x[:,None,:,None,:],x[None,:,None,:,:])
        >>> kfast.shape
        (4, 3, 6, 6, 5, 5)
        >>> kstable = kernel(x[:,None,:,None,:],x[None,:,None,:,:],stable=True)
        >>> np.abs(kfast-kstable).max()
        np.float64(4.440892098500626e-16)

        Derivatives 

        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> scale = rng.uniform(low=0,high=1,size=(1,))
        >>> lengthscales = rng.uniform(low=0,high=1,size=(3,))
        >>> kernel = KernelShiftInvar(
        ...     d = 3,
        ...     alpha = 3,
        ...     torchify = True,
        ...     scale = torch.from_numpy(scale),
        ...     lengthscales = torch.from_numpy(lengthscales))
        >>> x0 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x1 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x2 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> x = torch.stack([x0,x1,x2],axis=-1)
        >>> z0 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z1 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z2 = torch.from_numpy(rng.uniform(low=0,high=1,size=(4,))).requires_grad_(True)
        >>> z = torch.stack([z0,z1,z2],axis=-1)
        >>> c = torch.from_numpy(rng.uniform(low=0,high=1,size=(2,)))
        >>> beta0 = torch.tensor([
        ...     [1,0,0],
        ...     [0,2,0]])
        >>> beta1 = torch.tensor([
        ...     [0,0,2],
        ...     [2,1,0]])
        >>> with torch.no_grad():
        ...     y = kernel(x,z,beta0,beta1,c)
        >>> with np.printoptions(formatter={"float": lambda x: "%.2f"%x}):
        ...     y.numpy()
        array([1455.14, 9475.57, 7807.08, 2785.47])
        >>> y_no_deriv = kernel(x,z)
        >>> y_first = y_no_deriv.clone()
        >>> y_first = torch.autograd.grad(y_first,x0,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_first = torch.autograd.grad(y_first,z2,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_first = torch.autograd.grad(y_first,z2,grad_outputs=torch.ones_like(y_first,requires_grad=True),create_graph=True)[0]
        >>> y_second = y_no_deriv.clone()
        >>> y_second = torch.autograd.grad(y_second,x1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,x1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z0,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z0,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> y_second = torch.autograd.grad(y_second,z1,grad_outputs=torch.ones_like(y_second,requires_grad=True),create_graph=True)[0]
        >>> yhat = (y_first*c[0]+y_second*c[1]).detach()
        >>> with np.printoptions(formatter={"float": lambda x: "%.3f"%x}):
        ...     yhat.numpy()
        array([1455.140, 9475.570, 7807.076, 2785.473])
        >>> torch.allclose(y,yhat)
        True
        >>> kernel = KernelShiftInvar(
        ...     d = 3,
        ...     alpha = 3,
        ...     scale = scale,
        ...     lengthscales = lengthscales)
        >>> ynp = kernel(x.detach().numpy(),z.detach().numpy(),beta0.numpy(),beta1.numpy(),c.numpy())
        >>> with np.printoptions(formatter={"float": lambda x: "%.2f"%x}):
        ...     ynp
        array([1455.14, 9475.59, 7807.09, 2785.48])
        >>> np.allclose(ynp,y.numpy())
        True

    **References:** 
    
    1.  Kaarnioja, Vesa, Frances Y. Kuo, and Ian H. Sloan.  
        "Lattice-based kernel approximation and serendipitous weights for parametric PDEs in very high dimensions."  
        International Conference on Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing. Cham: Springer International Publishing, 2022.
    """

    def __init__(
        self,
        d,
        scale=1.0,
        lengthscales=None,
        alpha=2,
        shape_scale=None,
        shape_lengthscales=None,
        tfs_scale=None,
        tfs_lengthscales=None,
        torchify=False,
        requires_grad_scale=None,
        requires_grad_lengthscales=None,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
        weights=None,
        shape_weights=None,
        tfs_weights=None,
        requires_grad_weights=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray, torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
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
            weights (Union[np.ndarray, torch.Tensor]): Alias for `lengthscales`.
            shape_weights (list): Alias for `shape_lengthscales`.
            tfs_weights (Tuple[callable,callable]): Alias for `tfs_lengthscales`.
            requires_grad_weights (bool): Alias for `requires_grad_lengthscales`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            alpha=alpha,
            shape_alpha=[d],
            alpha_endsize_ops=[d],
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            tfs_alpha=(tf_identity, tf_identity),
            tfs_scale=tfs_scale,
            requires_grad_alpha=False,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
            weights=weights,
            shape_weights=shape_weights,
            tfs_weights=tfs_weights,
            requires_grad_weights=requires_grad_weights,
        )
        assert self.alpha.shape == (self.d,)
        assert all(int(alphaj) in BERNOULLIPOLYSDICT for alphaj in self.alpha)
        if self.torchify:
            import torch

            self.lgamma = torch.lgamma
        else:
            self.lgamma = scipy.special.loggamma

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        p = len(beta0)
        betasum = beta0 + beta1
        order = 2 * self.alpha - betasum
        assert (
            2 <= order
        ).all(), "order must all be at least 2, but got order = %s" % str(order)
        coeffs = (-1) ** (self.alpha + beta1 + 1) * self.npt.exp(
            2 * self.alpha * np.log(2 * np.pi) - self.lgamma(order + 1)
        )
        delta = (x0 - x1) % 1
        kperdim = coeffs * self.npt.stack(
            [
                self.npt.concatenate(
                    [
                        bernoulli_poly(int(order[l, j].item()), delta[..., j, None])
                        for j in range(self.d)
                    ],
                    -1,
                )
                for l in range(p)
            ],
            -2,
        )
        return kperdim


class KernelShiftInvarCombined(AbstractSIDSIKernel):
    r"""
    Shift invariant kernel with
    combination weights $\boldsymbol{\alpha}_1,\dots,\boldsymbol{\alpha}_d \in \mathbb{R}_{>0}^4$, product weights (lengthscales) $\boldsymbol{\gamma}$, and scale $S$:

    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \left(\sum_{p=1}^4 \alpha_{jp} \tilde{K}_p(x_j \mod 1 z_j)\right)\right)
    \end{aligned}$$

    where, $\tilde{K}_p$ are defined in `KernelShiftInvar` for $p \in \{1,2,3,4\}$

    Examples:
        >>> from qmcpy import Lattice, fftbr, ifftbr
        >>> n = 8
        >>> d = 4
        >>> lat = Lattice(d,seed=11)
        >>> x = lat(n)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('float64')
        >>> kernel = KernelShiftInvarCombined(
        ...     d = d,
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        117.22427096475315
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [ 117.22  153.84   28.59   36.1  -139.83   77.1    -8.51   35.22]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[ 117.22  153.84   36.1    28.59   35.22   -8.51   77.1  -139.83]
         [ 153.84  117.22   28.59   36.1    -8.51   35.22 -139.83   77.1 ]
         [  28.59   36.1   117.22  153.84 -139.83   77.1    35.22   -8.51]
         [  36.1    28.59  153.84  117.22   77.1  -139.83   -8.51   35.22]
         [-139.83   77.1    35.22   -8.51  117.22  153.84   36.1    28.59]
         [  77.1  -139.83   -8.51   35.22  153.84  117.22   28.59   36.1 ]
         [  -8.51   35.22 -139.83   77.1    28.59   36.1   117.22  153.84]
         [  35.22   -8.51   77.1  -139.83   36.1    28.59  153.84  117.22]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fftbr(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(ifftbr(fftbr(y)*lam),kmat@y)
        True
        >>> np.allclose(ifftbr(fftbr(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch
        >>> xtorch = torch.from_numpy(x)
        >>> kernel_torch = KernelShiftInvarCombined(
        ...     d = d,
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> kernel.single_integral_01d(x)
        array([10., 10., 10., 10., 10., 10., 10., 10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10., 10., 10., 10., 10., 10., 10., 10.], dtype=torch.float64,
               grad_fn=<AddBackward0>)

        Batch Params

        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelShiftInvarCombined(
        ...     d = 2,
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,2])
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kfast = kernel(x[:,None,:,None,:],x[None,:,None,:,:])
        >>> kfast.shape
        (4, 3, 6, 6, 5, 5)
        >>> kstable = kernel(x[:,None,:,None,:],x[None,:,None,:,:],stable=True)
        >>> np.abs(kfast-kstable).max()
        np.float64(3.552713678800501e-15)

    **References:**

    1.  Kaarnioja, Vesa, Frances Y. Kuo, and Ian H. Sloan.
        "Lattice-based kernel approximation and serendipitous weights for parametric PDEs in very high dimensions."
        International Conference on Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing. Cham: Springer International Publishing, 2022.
    """

    def __init__(
        self,
        d,
        scale=1.0,
        lengthscales=None,
        alpha=1,
        shape_scale=None,
        shape_lengthscales=None,
        shape_alpha=None,
        tfs_scale=None,
        tfs_lengthscales=None,
        tfs_alpha=None,
        torchify=False,
        requires_grad_scale=None,
        requires_grad_lengthscales=None,
        requires_grad_alpha=None,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
        weights=None,
        shape_weights=None,
        tfs_weights=None,
        requires_grad_weights=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray, torch.Tensor]): Weights $\boldsymbol{\alpha}_1,\dots,\boldsymbol{\alpha}_d \in \mathbb{R}_{>0}^4$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`.
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            shape_alpha (list): Shape of `alpha` when `np.isscalar(alpha)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_alpha (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            requires_grad_alpha (bool): If `True` and `torchify`, set `requires_grad=True` for `alpha`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method.
            compile_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
            weights (Union[np.ndarray, torch.Tensor]): Alias for `lengthscales`.
            shape_weights (list): Alias for `shape_lengthscales`.
            tfs_weights (Tuple[callable,callable]): Alias for `tfs_lengthscales`.
            requires_grad_weights (bool): Alias for `requires_grad_lengthscales`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            alpha=alpha,
            alpha_endsize_ops=[d],
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            shape_alpha=[4, d] if shape_alpha is None else shape_alpha,
            tfs_scale=tfs_scale,
            tfs_alpha=tfs_alpha,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            requires_grad_alpha=requires_grad_alpha,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
            weights=weights,
            shape_weights=shape_weights,
            tfs_weights=tfs_weights,
            requires_grad_weights=requires_grad_weights,
        )
        assert self.alpha.shape[-2:] == (4, d)
        if self.torchify:
            import torch

            self.lgamma = torch.lgamma
        else:
            self.lgamma = scipy.special.loggamma
        a = self.npt.arange(1, 5, **self.nptkwargs)
        self.coeffs = (-1) ** (a + 1) * self.npt.exp(
            2 * a * np.log(2 * np.pi) - self.lgamma(2 * a + 1)
        )

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        p = len(beta0)
        assert (beta0 == 0).all() and (
            beta1 == 0
        ).all(), "KernelDSICombined does not support derivatives"
        delta = (x0 - x1) % 1
        kparts = [None] * 4
        kparts[0] = bernoulli_poly(1, delta)
        kparts[1] = bernoulli_poly(2, delta)
        kparts[2] = bernoulli_poly(3, delta)
        kparts[3] = bernoulli_poly(4, delta)
        kparts = self.coeffs[:, None] * self.npt.stack(kparts, -2)
        kperdim = self.npt.stack([kparts for j in range(p)], -2)
        return kperdim

    def combine_per_dim_components_raw_m1(
        self, kparts, beta0, beta1, c, batch_params, stable
    ):
        kparts = (self.alpha[..., None, :, None, :] * kparts).sum(-3)
        return super().combine_per_dim_components_raw_m1(
            kparts, beta0, beta1, c, batch_params, stable
        )


class KernelDigShiftInvar(AbstractSIDSIKernel):
    r""" 
    Digitally shift invariant kernel in base $b=2$ with 
    smoothness $\boldsymbol{\alpha}$, product weights $\boldsymbol{\gamma}$, and scale $S$: 
    
    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \tilde{K}_{\alpha_j}(x_j \oplus z_j)\right), \qquad\mathrm{where} \\
        \tilde{K}_1(x) &= 6 \left(\frac{1}{6} - 2^{\lfloor \log_2(x) \rfloor -1}\right), \\
        \tilde{K}_2(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_2(k)}} = -\beta(x) x + \frac{5}{2}\left[1-t_1(x)\right]-1, \\
        \tilde{K}_3(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_3(k)}} = \beta(x)x^2-5\left[1-t_1(x)\right]x+\frac{43}{18}\left[1-t_2(x)\right]-1, \\
        \tilde{K}_4(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{\mu_4(k)}} = - \frac{2}{3}\beta(x)x^3+5\left[1-t_1(x)\right]x^2 - \frac{43}{9}\left[1-t_2(x)\right]x +\frac{701}{294}\left[1-t_3(x)\right]+\beta(x)\left[\frac{1}{48}\sum_{a=0}^\infty \frac{\mathrm{wal}_{2^a}(x)}{2^{3a}} - \frac{1}{42}\right] - 1.
    \end{aligned}$$

    where 
    
    - $x \oplus z$ is XOR between bits, 
    - $\mathrm{wal}_k$ is the $k^\text{th}$ Walsh function, 
    - $\beta(x) = - \lfloor \log_2(x) \rfloor$ and $t_\nu(x) = 2^{-\nu \beta(x)}$ where $\beta(0)=t_\nu(0) = 0$, and 
    - and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of $1$ bits in the binary expansion of $k$ 
    e.g. $k=13=1101_2$ has 1-bit indexes $(4,3,1)$ so 
    
    $$\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots.$$

    Examples:
        >>> from qmcpy import DigitalNetB2, fwht
        >>> n = 8
        >>> d = 4
        >>> dnb2 = DigitalNetB2(d,seed=11)
        >>> x = dnb2(n,return_binary=True)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('uint64')
        >>> kernel = KernelDigShiftInvar(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        34.490370029184525
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [34.49  4.15  9.59  4.98 15.42  5.45 11.99  4.51]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[34.49  4.15  9.59  4.98 15.42  5.45 11.99  4.51]
         [ 4.15 34.49  4.98  9.59  5.45 15.42  4.51 11.99]
         [ 9.59  4.98 34.49  4.15 11.99  4.51 15.42  5.45]
         [ 4.98  9.59  4.15 34.49  4.51 11.99  5.45 15.42]
         [15.42  5.45 11.99  4.51 34.49  4.15  9.59  4.98]
         [ 5.45 15.42  4.51 11.99  4.15 34.49  4.98  9.59]
         [11.99  4.51 15.42  5.45  9.59  4.98 34.49  4.15]
         [ 4.51 11.99  5.45 15.42  4.98  9.59  4.15 34.49]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fwht(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(fwht(fwht(y)*lam),kmat@y)
        True
        >>> np.allclose(fwht(fwht(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch 
        >>> xtorch = bin_from_numpy_to_torch(x)
        >>> kernel_torch = KernelDigShiftInvar(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> xf = to_float(x,dnb2.t)
        >>> kmat_from_floats = kernel(xf[:,None,:],xf[None,:,:])
        >>> np.allclose(kmat,kmat_from_floats)
        True
        >>> xftorch = to_float(xtorch,dnb2.t)
        >>> xftorch.dtype
        torch.float32
        >>> kmat_torch_from_floats = kernel_torch(xftorch[:,None,:],xftorch[None,:,:])
        >>> torch.allclose(kmat_torch_from_floats,kmat_torch)
        True
        >>> kernel.single_integral_01d(x)
        array([10., 10., 10., 10., 10., 10., 10., 10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10., 10., 10., 10., 10., 10., 10., 10.], grad_fn=<AddBackward0>)

        Batch Params 
        
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelDigShiftInvar(
        ...     d = 2, 
        ...     t = 10,
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,2])
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kfast = kernel(x[:,None,:,None,:],x[None,:,None,:,:])
        >>> kfast.shape
        (4, 3, 6, 6, 5, 5)
        >>> kstable = kernel(x[:,None,:,None,:],x[None,:,None,:,:],stable=True)
        >>> np.abs(kfast-kstable).max()
        np.float64(4.440892098500626e-16)

    **References:**
        
    1.  Dick, Josef.  
        "Walsh spaces containing smooth functions and quasi-Monte Carlo rules of arbitrary high order."  
        SIAM Journal on Numerical Analysis 46.3 (2008): 1519-1553.

    2.  Dick, Josef.  
        "The decay of the Walsh coefficients of smooth functions."  
        Bulletin of the Australian Mathematical Society 80.3 (2009): 430-453.  

    3.  Jagadeeswaran, Rathinavel, and Fred J. Hickernell.  
        "Fast automatic Bayesian cubature using Sobol' sampling."  
        Advances in Modeling and Simulation: Festschrift for Pierre L'Ecuyer. Cham: Springer International Publishing, 2022. 301-318.

    4.  Rathinavel, Jagadeeswaran.  
        Fast automatic Bayesian cubature using matching kernels and designs.  
        Illinois Institute of Technology, 2019.
    
    5.  Sorokin, Aleksei.  
        "A Unified Implementation of Quasi-Monte Carlo Generators, Randomization Routines, and Fast Kernel Methods."  
        arXiv preprint arXiv:2502.14256 (2025).
    """

    def __init__(
        self,
        d,
        t=None,
        scale=1.0,
        lengthscales=None,
        alpha=2,
        shape_scale=None,
        shape_lengthscales=None,
        tfs_scale=None,
        tfs_lengthscales=None,
        torchify=False,
        requires_grad_scale=None,
        requires_grad_lengthscales=None,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
        weights=None,
        shape_weights=None,
        tfs_weights=None,
        requires_grad_weights=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray, torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
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
            weights (Union[np.ndarray, torch.Tensor]): Alias for `lengthscales`.
            shape_weights (list): Alias for `shape_lengthscales`.
            tfs_weights (Tuple[callable,callable]): Alias for `tfs_lengthscales`.
            requires_grad_weights (bool): Alias for `requires_grad_lengthscales`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            alpha=alpha,
            shape_alpha=[d],
            alpha_endsize_ops=[d],
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            tfs_alpha=(tf_identity, tf_identity),
            tfs_scale=tfs_scale,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_alpha=False,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
            weights=weights,
            shape_weights=shape_weights,
            tfs_weights=tfs_weights,
            requires_grad_weights=requires_grad_weights,
        )
        assert self.alpha.shape == (self.d,)
        self.set_t(t)
        assert all(1 <= int(alphaj) <= 4 for alphaj in self.alpha)

    @property
    def t(self):
        if self._t is None:
            raise ParameterError("please use set_t to set the t value")
        return self._t

    def set_t(self, t):
        if t is None:
            self._t = t
        else:
            assert t % 1 == 0
            if self.torchify:
                assert 0 <= t <= 63  # torch only supports torch.int64
            else:
                assert 0 <= t <= 64  # numpy supports np.uint64
            self._t = t

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        t = self.t
        x0 = to_bin(x0, t)
        x1 = to_bin(x1, t)
        p = len(beta0)
        betasum = beta0 + beta1
        order = self.alpha - betasum
        assert (1 <= order).all() and (order <= 4).all(), (
            "order must all be between 2 and 4, but got order = %s. Try increasing alpha"
            % str(order)
        )
        assert not (
            (order == 1) * (self.alpha > 1)
        ).any(), "taking the derivative of the order 2 digitally shift invariant kernel is not supported"
        ind = 1.0 * (betasum > 0)
        delta = x0 ^ x1
        kparts = [None] * p
        for l in range(p):
            kparts_l = [None] * self.d
            for j in range(self.d):
                deltaj = delta[..., j, None]
                if (
                    order[l, j] == 1
                ):  # order[j]=alpha[j] as we cannot take derivatives WRT the alpha=1 kernel and this cannot be the derivative of any kernels
                    flog2deltaj = -self.npt.inf * self.npt.ones(
                        deltaj.shape, **self.nptkwargs
                    )
                    pos = deltaj > 0
                    flog2deltaj[pos] = self.npt.floor(self.npt.log2(deltaj[pos])) - t
                    kparts_l[j] = 6 * (1 / 6 - 2 ** (flog2deltaj - 1))
                else:
                    kparts_l[j] = (
                        weighted_walsh_funcs(int(order[l, j]), deltaj[..., None], t)[
                            ..., 0
                        ]
                        - 1
                    )
            kparts[l] = self.npt.concatenate(kparts_l, -1)
        kparts = self.npt.stack(kparts, -2)
        kperdim = (-2) ** betasum * (ind + kparts)
        return kperdim


class KernelDigShiftInvarAdaptiveAlpha(AbstractSIDSIKernel):
    r""" 
    Digitally shift invariant kernel in base $b=2$ with 
    smoothness $\boldsymbol{\alpha} \geq \boldsymbol{0}$, product weights $\boldsymbol{\gamma}$, and scale $S$: 
    
    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \tilde{K}_{\alpha_j}(x_j \oplus z_j)\right), \qquad\mathrm{where} \\
        \tilde{K}_\alpha(x) &= \sum_{k \in \mathbb{N}} \frac{\mathrm{wal}_k(x)}{2^{{\alpha+1} (\mu_1(k)-1)}} = \frac{2^{\alpha+1}}{2^{\alpha+1}-2} - \left(\frac{2^{\alpha+1}}{2^{\alpha+1}-2}+1\right) 2^{\alpha(\lfloor \log_2(x) \rfloor+1)}, \\
    \end{aligned}$$

    where 
    
    - $x \oplus z$ is XOR between bits, 
    - $\mathrm{wal}_k$ is the $k^\text{th}$ Walsh function, 
    - $\beta(x) = - \lfloor \log_2(x) \rfloor$ and $t_\nu(x) = 2^{-\nu \beta(x)}$ where $\beta(0)=t_\nu(0) = 0$, and 
    - and $\mu_\alpha$ is the Dick weight function which sums the first $\alpha$ largest indices of $1$ bits in the binary expansion of $k$ 
    e.g. $k=13=1101_2$ has 1-bit indexes $(4,3,1)$ so 
    
    $$\mu_1(k) = 4, \mu_2(k) = 4+3, \mu_3(k) = 4+3+1 = \mu_4(k) = \mu_5(k) = \dots.$$

    Examples:
        >>> from qmcpy import DigitalNetB2, fwht
        >>> n = 8
        >>> d = 4
        >>> dnb2 = DigitalNetB2(d,seed=11)
        >>> x = dnb2(n,return_binary=True)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('uint64')
        >>> kernel = KernelDigShiftInvarAdaptiveAlpha(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        48.084656084656096
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [48.08  0.    9.38  0.   19.74  0.   14.84  0.  ]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[48.08  0.    9.38  0.   19.74  0.   14.84  0.  ]
         [ 0.   48.08  0.    9.38  0.   19.74  0.   14.84]
         [ 9.38  0.   48.08  0.   14.84  0.   19.74  0.  ]
         [ 0.    9.38  0.   48.08  0.   14.84  0.   19.74]
         [19.74  0.   14.84  0.   48.08  0.    9.38  0.  ]
         [ 0.   19.74  0.   14.84  0.   48.08  0.    9.38]
         [14.84  0.   19.74  0.    9.38  0.   48.08  0.  ]
         [ 0.   14.84  0.   19.74  0.    9.38  0.   48.08]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fwht(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(fwht(fwht(y)*lam),kmat@y)
        True
        >>> np.allclose(fwht(fwht(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch 
        >>> xtorch = bin_from_numpy_to_torch(x)
        >>> kernel_torch = KernelDigShiftInvarAdaptiveAlpha(
        ...     d = d, 
        ...     t = dnb2.t,
        ...     alpha = list(range(1,d+1)),
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat,atol=1e-5)
        True
        >>> xf = to_float(x,dnb2.t)
        >>> kmat_from_floats = kernel(xf[:,None,:],xf[None,:,:])
        >>> np.allclose(kmat,kmat_from_floats)
        True
        >>> xftorch = to_float(xtorch,dnb2.t)
        >>> xftorch.dtype
        torch.float32
        >>> kmat_torch_from_floats = kernel_torch(xftorch[:,None,:],xftorch[None,:,:])
        >>> torch.allclose(kmat_torch_from_floats,kmat_torch)
        True
        >>> kernel.single_integral_01d(x)
        array([10., 10., 10., 10., 10., 10., 10., 10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10., 10., 10., 10., 10., 10., 10., 10.], grad_fn=<AddBackward0>)

        Batch Params 
        
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelDigShiftInvarAdaptiveAlpha(
        ...     d = 2, 
        ...     t = 10,
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,2])
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape 
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kfast = kernel(x[:,None,:,None,:],x[None,:,None,:,:])
        >>> kfast.shape
        (4, 3, 6, 6, 5, 5)
        >>> kstable = kernel(x[:,None,:,None,:],x[None,:,None,:,:],stable=True)
        >>> np.abs(kfast-kstable).max()
        np.float64(4.440892098500626e-16)

    **References:**
        
    3.  Dick, Josef, and Friedrich Pillichshammer.  
        "Multivariate integration in weighted Hilbert spaces based on Walsh functions and weighted Sobolev spaces."  
        Journal of Complexity 21.2 (2005): 149-195.
    """

    def __init__(
        self,
        d,
        t=None,
        scale=1.0,
        lengthscales=None,
        alpha=1,
        shape_scale=None,
        shape_lengthscales=None,
        shape_alpha=None,
        tfs_scale=None,
        tfs_lengthscales=None,
        tfs_alpha=None,
        torchify=False,
        requires_grad_scale=None,
        requires_grad_lengthscales=None,
        requires_grad_alpha=None,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
        weights=None,
        shape_weights=None,
        tfs_weights=None,
        requires_grad_weights=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray, torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
            shape_alpha (list): Shape of `alpha` when `np.isscalar(alpha)`
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`.
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_alpha (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            requires_grad_alpha (bool): If `True` and `torchify`, set `requires_grad=True` for `alpha`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method.
            compile_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
            weights (Union[np.ndarray, torch.Tensor]): Alias for `lengthscales`.
            shape_weights (list): Alias for `shape_lengthscales`.
            tfs_weights (Tuple[callable,callable]): Alias for `tfs_lengthscales`.
            requires_grad_weights (bool): Alias for `requires_grad_lengthscales`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            alpha=alpha,
            shape_alpha=[d] if shape_alpha is None else shape_alpha,
            alpha_endsize_ops=[d],
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            tfs_alpha=tfs_alpha,
            tfs_scale=tfs_scale,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_alpha=requires_grad_alpha,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
            weights=weights,
            shape_weights=shape_weights,
            tfs_weights=tfs_weights,
            requires_grad_weights=requires_grad_weights,
        )
        self.set_t(t)
        self.batch_param_names.append("alpha")

    @property
    def t(self):
        if self._t is None:
            raise ParameterError("please use set_t to set the t value")
        return self._t

    def set_t(self, t):
        if t is None:
            self._t = t
        else:
            assert t % 1 == 0
            if self.torchify:
                assert 0 <= t <= 63  # torch only supports torch.int64
            else:
                assert 0 <= t <= 64  # numpy supports np.uint64
            self._t = t

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        t = self.t
        x0 = to_bin(x0, t)
        x1 = to_bin(x1, t)
        assert (beta0 == 0).all() and (
            beta1 == 0
        ).all(), "KernelDigShiftInvarAdaptiveAlpha does not support taking derivatives"
        p = len(beta0)
        delta = x0 ^ x1
        flog2delta = self.npt.zeros(delta.shape, **self.nptkwargs)  # should be -inf
        pos = delta > 0
        flog2delta[pos] = self.npt.floor(self.npt.log2(delta[pos])) - t
        flog2deltas = self.npt.stack([flog2delta for j in range(p)], -2)
        return flog2deltas

    def combine_per_dim_components_raw_m1(
        self, flog2deltas, beta0, beta1, c, batch_params, stable
    ):
        alpha = batch_params["alpha"]
        p2alphap1 = 2 ** (alpha + 1)
        nu = p2alphap1 / (p2alphap1 - 2)
        neginfs = flog2deltas == 0
        s = (nu + 1) * 2 ** (alpha * (flog2deltas + 1))
        kparts = nu - s
        kparts[neginfs] = (nu - 0 * s)[neginfs]
        return super().combine_per_dim_components_raw_m1(
            kparts, beta0, beta1, c, batch_params, stable
        )


class KernelDigShiftInvarCombined(AbstractSIDSIKernel):
    r"""
    Digitally shift invariant kernel in base $b=2$ with
    combination weights $\boldsymbol{\alpha}_1,\dots,\boldsymbol{\alpha}_d \in \mathbb{R}_{>0}^4$, smoothness $\boldsymbol{\alpha}$, product weights $\boldsymbol{\gamma}$, and scale $S$:

    $$\begin{aligned}
        K(\boldsymbol{x},\boldsymbol{z}) &= S \prod_{j=1}^d \left(1+ \gamma_j \left(\sum_{p=1}^4 \alpha_{jp} \tilde{K}_p(x_j \oplus z_j)\right)\right)
    \end{aligned}$$

    where, $\oplus$ is defined in the docs for `KernelDigShiftInvar` and so are $\tilde{K}_p$ for $p \in \{1,2,3,4\}$

    Examples:
        >>> from qmcpy import DigitalNetB2, fwht
        >>> n = 8
        >>> d = 4
        >>> dnb2 = DigitalNetB2(d,seed=11)
        >>> x = dnb2(n,return_binary=True)
        >>> x.shape
        (8, 4)
        >>> x.dtype
        dtype('uint64')
        >>> kernel = KernelDigShiftInvarCombined(
        ...     d = d,
        ...     t = dnb2.t,
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)])
        >>> k00 = kernel(x[0],x[0])
        >>> k00.item()
        306.66030731930863
        >>> k0 = kernel(x,x[0])
        >>> with np.printoptions(precision=2):
        ...     print(k0)
        [306.66  -1.65   6.88 -13.5   22.57  -9.34   7.99  -7.54]
        >>> assert k0[0]==k00
        >>> kmat = kernel(x[:,None,:],x[None,:,:])
        >>> with np.printoptions(precision=2):
        ...     print(kmat)
        [[306.66  -1.65   6.88 -13.5   22.57  -9.34   7.99  -7.54]
         [ -1.65 306.66 -13.5    6.88  -9.34  22.57  -7.54   7.99]
         [  6.88 -13.5  306.66  -1.65   7.99  -7.54  22.57  -9.34]
         [-13.5    6.88  -1.65 306.66  -7.54   7.99  -9.34  22.57]
         [ 22.57  -9.34   7.99  -7.54 306.66  -1.65   6.88 -13.5 ]
         [ -9.34  22.57  -7.54   7.99  -1.65 306.66 -13.5    6.88]
         [  7.99  -7.54  22.57  -9.34   6.88 -13.5  306.66  -1.65]
         [ -7.54   7.99  -9.34  22.57 -13.5    6.88  -1.65 306.66]]
        >>> assert (kmat[:,0]==k0).all()
        >>> lam = np.sqrt(n)*fwht(k0)
        >>> y = np.random.Generator(np.random.PCG64(7)).uniform(low=0,high=1,size=(n))
        >>> np.allclose(fwht(fwht(y)*lam),kmat@y)
        True
        >>> np.allclose(fwht(fwht(y)/lam),np.linalg.solve(kmat,y))
        True
        >>> import torch
        >>> xtorch = bin_from_numpy_to_torch(x)
        >>> kernel_torch = KernelDigShiftInvarCombined(
        ...     d = d,
        ...     t = dnb2.t,
        ...     scale = 10,
        ...     lengthscales = [1/j**2 for j in range(1,d+1)],
        ...     torchify = True)
        >>> kmat_torch = kernel_torch(xtorch[:,None,:],xtorch[None,:,:])
        >>> np.allclose(kmat_torch.detach().numpy(),kmat)
        True
        >>> xf = to_float(x,dnb2.t)
        >>> kmat_from_floats = kernel(xf[:,None,:],xf[None,:,:])
        >>> np.allclose(kmat,kmat_from_floats)
        True
        >>> xftorch = to_float(xtorch,dnb2.t)
        >>> xftorch.dtype
        torch.float32
        >>> kmat_torch_from_floats = kernel_torch(xftorch[:,None,:],xftorch[None,:,:])
        >>> torch.allclose(kmat_torch_from_floats,kmat_torch)
        True
        >>> kernel.single_integral_01d(x)
        array([10., 10., 10., 10., 10., 10., 10., 10.])
        >>> kernel_torch.single_integral_01d(xtorch)
        tensor([10., 10., 10., 10., 10., 10., 10., 10.], grad_fn=<AddBackward0>)

        Batch Params

        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kernel = KernelDigShiftInvarCombined(
        ...     d = 2,
        ...     t = 10,
        ...     shape_scale = [4,3,1],
        ...     shape_lengthscales = [3,2])
        >>> x = rng.uniform(low=0,high=1,size=(6,5,2))
        >>> kernel(x,x).shape
        (4, 3, 6, 5)
        >>> kernel(x[:,:,None,:],x[:,None,:,:]).shape
        (4, 3, 6, 5, 5)
        >>> kfast = kernel(x[:,None,:,None,:],x[None,:,None,:,:])
        >>> kfast.shape
        (4, 3, 6, 6, 5, 5)
        >>> kstable = kernel(x[:,None,:,None,:],x[None,:,None,:,:],stable=True)
        >>> np.abs(kfast-kstable).max()
        np.float64(8.881784197001252e-16)
    """

    def __init__(
        self,
        d,
        t=None,
        scale=1.0,
        lengthscales=None,
        alpha=1.0,
        shape_scale=None,
        shape_lengthscales=None,
        shape_alpha=None,
        tfs_scale=None,
        tfs_lengthscales=None,
        tfs_alpha=None,
        torchify=False,
        requires_grad_scale=None,
        requires_grad_lengthscales=None,
        requires_grad_alpha=None,
        device="cpu",
        compile_call=False,
        compile_call_kwargs=None,
        weights=None,
        shape_weights=None,
        tfs_weights=None,
        requires_grad_weights=None,
    ):
        r"""
        Args:
            d (int): Dimension.
            t (int): number of bits in binary represtnations. Typically `dnb2.t` where `isinstance(dnb2,DigitalNetB2)`.
            scale (Union[np.ndarray, torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray, torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray, torch.Tensor]): Weights $\boldsymbol{\alpha}_1,\dots,\boldsymbol{\alpha}_d \in \mathbb{R}_{>0}^4$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`.
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            shape_alpha (list): Shape of `alpha` when `np.isscalar(alpha)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            requires_grad_alpha (bool): If `True` and `torchify`, set `requires_grad=True` for `alpha`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method.
            compile_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
            weights (Union[np.ndarray, torch.Tensor]): Alias for `lengthscales`.
            shape_weights (list): Alias for `shape_lengthscales`.
            tfs_weights (Tuple[callable,callable]): Alias for `tfs_lengthscales`.
            requires_grad_weights (bool): Alias for `requires_grad_lengthscales`.
        """
        if shape_scale is None:
            shape_scale = [1]
        if compile_call_kwargs is None:
            compile_call_kwargs = {}
        super().__init__(
            d=d,
            scale=scale,
            lengthscales=lengthscales,
            alpha=alpha,
            shape_alpha=[4, d] if shape_alpha is None else shape_alpha,
            alpha_endsize_ops=[d],
            shape_scale=shape_scale,
            shape_lengthscales=shape_lengthscales,
            tfs_alpha=tfs_alpha,
            tfs_scale=tfs_scale,
            tfs_lengthscales=tfs_lengthscales,
            torchify=torchify,
            requires_grad_alpha=requires_grad_alpha,
            requires_grad_scale=requires_grad_scale,
            requires_grad_lengthscales=requires_grad_lengthscales,
            device=device,
            compile_call=compile_call,
            compile_call_kwargs=compile_call_kwargs,
            weights=weights,
            shape_weights=shape_weights,
            tfs_weights=tfs_weights,
            requires_grad_weights=requires_grad_weights,
        )
        self.set_t(t)
        assert self.alpha.shape[-2:] == (4, d)

    @property
    def t(self):
        if self._t is None:
            raise ParameterError("please use set_t to set the t value")
        return self._t

    def set_t(self, t):
        if t is None:
            self._t = t
        else:
            assert t % 1 == 0
            if self.torchify:
                assert 0 <= t <= 63  # torch only supports torch.int64
            else:
                assert 0 <= t <= 64  # numpy supports np.uint64
            self._t = t

    def get_per_dim_components(self, x0, x1, beta0, beta1):
        t = self.t
        x0 = to_bin(x0, t)
        x1 = to_bin(x1, t)
        p = len(beta0)
        assert (beta0 == 0).all() and (
            beta1 == 0
        ).all(), "KernelDSICombined does not support derivatives"
        delta = x0 ^ x1
        kparts = [None] * 4
        flog2deltaj = -self.npt.inf * self.npt.ones(delta.shape, **self.nptkwargs)
        pos = delta > 0
        flog2deltaj[pos] = self.npt.floor(self.npt.log2(delta[pos])) - t
        kparts[0] = 6 * (1 / 6 - 2 ** (flog2deltaj - 1))
        kparts[1] = weighted_walsh_funcs(2, delta, t) - 1
        kparts[2] = weighted_walsh_funcs(3, delta, t) - 1
        kparts[3] = weighted_walsh_funcs(4, delta, t) - 1
        kparts = self.npt.stack(kparts, -2)
        kperdim = self.npt.stack([kparts for j in range(p)], -2)
        return kperdim

    def combine_per_dim_components_raw_m1(
        self, kparts, beta0, beta1, c, batch_params, stable
    ):
        kparts = (self.alpha[..., None, :, None, :] * kparts).sum(-3)
        return super().combine_per_dim_components_raw_m1(
            kparts, beta0, beta1, c, batch_params, stable
        )
