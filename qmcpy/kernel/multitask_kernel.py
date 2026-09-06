from .abstract_kernel import AbstractKernel
from .common_kernels import KernelGaussian
from ..util.transforms import tf_identity, tf_exp_eps, tf_exp_eps_inv, insert_batch_dims
import numpy as np


class KernelMultiTask(AbstractKernel):
    r"""Multi-task kernel

    $$K((i,\boldsymbol{x}),(j,\boldsymbol{z})) = K_{\mathrm{task}}(i,j)
    K_{\mathrm{base}}(\boldsymbol{x},\boldsymbol{z})$$

    parameterized for $T$ tasks by a factor $\mathsf{F} \in \mathbb{R}^{T
    \times r}$ and a diagonal $\boldsymbol{v} \in \mathbb{R}^T$ so that

    $$\left[K_{\mathrm{task}}(i,j)\right]_{i,j=1}^T = \mathsf{F} \mathsf{F}^T +
    \mathrm{diag}(\boldsymbol{v}).$$

    Examples:
        >>> kmt = KernelMultiTask(KernelGaussian(d=2),num_tasks=3,diag=[1,2,3])
        >>> x = np.random.rand(4,2)
        >>> task = np.arange(3)
        >>> kmt(task[1],task[1],x[0],x[0]).item()
        3.0
        >>> kmt(task,task,x[0],x[0])
        array([2., 3., 4.])
        >>> kmt(task[:,None],task[None,:],x[0],x[0]).shape
        (3, 3)
        >>> kmt(task[1],task[1],x,x,)
        array([3., 3., 3., 3.])
        >>> kmt(task,task,x[:3],x[:3])
        array([2., 3., 4.])
        >>> kmt(task[:,None],task[:,None],x,x).shape
        (3, 4)
        >>> v = kmt(task,task,x[:3,None,:],x[None,:3,:])
        >>> v.shape
        (3, 3)
        >>> np.allclose(v,kmt.base_kernel(x[:3,None,:],x[None,:3,:])*kmt.taskmat[task,task])
        True
        >>> kmt(task[:,None,None],task[:,None,None],x[:,None,:],x[None,:,:]).shape
        (3, 4, 4)
        >>> kmt(task[:,None,None,None],task[None,:,None,None],x[:,None,:],x[None,:,:]).shape
        (3, 3, 4, 4)

        Batched inference

        >>> kernel_base = KernelGaussian(d=10,shape_lengthscales=(5,1),shape_scale=(3,5,1))
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kmt = KernelMultiTask(kernel_base,
        ...     num_tasks=4,
        ...     shape_factor=(6,3,5,4,2),
        ...     diag = rng.uniform(low=0,high=1,size=(3,5,4)))
        >>> x = np.random.rand(8,10)
        >>> task = np.arange(4)
        >>> kmt(task[0],task[0],x[0],x[0]).shape
        (6, 3, 5)
        >>> kmt(task,task,x[0],x[0]).shape
        (6, 3, 5, 4)
        >>> kmt(task[0],task[0],x,x).shape
        (6, 3, 5, 8)
        >>> v = kmt(task,task,x[:4,None,:],x[None,:4,:])
        >>> v.shape
        (6, 3, 5, 4, 4)
        >>> kmat_x = kmt.base_kernel(x[:4,None,:],x[None,:4,:])
        >>> kmat_x.shape
        (3, 5, 4, 4)
        >>> kmat_tasks = kmt.taskmat[...,task,task]
        >>> kmat_tasks.shape
        (6, 3, 5, 4)
        >>> np.allclose(v,kmat_tasks[...,None,:]*kmat_x)
        True
        >>> np.allclose(v,kmat_tasks[...,:,None]*kmat_x)
        False
        >>> kmt(task[:,None,None,None],task[None,:,None,None],x[:,None,:],x[None,:,:]).shape
        (6, 3, 5, 4, 4, 8, 8)
        >>> kmt(task[:,None],task[None,:],x[:,None,None,None,:],x[None,:,None,None,:]).shape
        (6, 3, 5, 8, 8, 4, 4)

        Integrals

        >>> kernel = KernelGaussian(
        ...     d = 2,
        ...     scale = rng.uniform(low=0,high=1,size=(3,1)),
        ...     lengthscales = rng.uniform(low=0,high=1,size=(1,2)))
        >>> kmt = KernelMultiTask(
        ...     base_kernel=kernel,
        ...     num_tasks=5,
        ...     diag = rng.uniform(low=0,high=1,size=(6,3,5)),
        ...     factor = rng.uniform(low=0,high=1,size=(3,5,2)))
        >>> task = np.arange(5)
        >>> kmt.double_integral_01d(task0=task[0],task1=task[1]).shape
        (6, 3)
        >>> kmt.double_integral_01d(task0=task,task1=task).shape
        (6, 3, 5)
        >>> kmt.double_integral_01d(task0=task[:,None],task1=task[None,:]).shape
        (6, 3, 5, 5)
        >>> x = rng.uniform(low=0,high=1,size=(7,4,2))
        >>> kmt.single_integral_01d(task0=task[0],task1=task[1],x=x).shape
        (6, 3, 7, 4)
        >>> kmt.single_integral_01d(task0=task[:,None,None],task1=task[:,None,None],x=x).shape
        (6, 3, 5, 7, 4)
        >>> kmt.single_integral_01d(task0=task[:,None,None,None],task1=task[None,:,None,None],x=x).shape
        (6, 3, 5, 5, 7, 4)

        Cholesky Construction

        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     )
        >>> kmt.taskmat
        array([[ 2.25,  3.  ,  3.  ],
               [ 3.  ,  6.25,  7.  ],
               [ 3.  ,  7.  , 10.25]])
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [2,3*(1+3)//2-3],
        ...     shape_diag = [3],
        ...     )
        >>> kmt.taskmat
        array([[[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]],
        <BLANKLINE>
               [[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]]])
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [3*(1+3)//2-3],
        ...     shape_diag = [2,3],
        ...     )
        >>> kmt.taskmat
        array([[[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]],
        <BLANKLINE>
               [[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]]])
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [2,3*(1+3)//2-3],
        ...     shape_diag = [2,3],
        ...     )
        >>> kmt.taskmat
        array([[[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]],
        <BLANKLINE>
               [[ 2.25,  3.  ,  3.  ],
                [ 3.  ,  6.25,  7.  ],
                [ 3.  ,  7.  , 10.25]]])

        PyTorch

        >>> import torch
        >>> kmt = KernelMultiTask(KernelGaussian(d=2,torchify=True),num_tasks=3,diag=[1,2,3])
        >>> x = torch.from_numpy(np.random.rand(4,2))
        >>> task = np.arange(3)
        >>> kmt(task[1],task[1],x[0],x[0]).item()
        3.0
        >>> kmt(task,task,x[0],x[0])
        tensor([2., 3., 4.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(task[:,None],task[None,:],x[0],x[0]).shape
        torch.Size([3, 3])
        >>> kmt(task[1],task[1],x,x)
        tensor([3., 3., 3., 3.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(task,task,x[:3],x[:3])
        tensor([2., 3., 4.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(task[:,None],task[:,None],x,x).shape
        torch.Size([3, 4])
        >>> v = kmt(task,task,x[:3,None,:],x[None,:3,:])
        >>> v.shape
        torch.Size([3, 3])
        >>> torch.allclose(v,kmt.base_kernel(x[:3,None,:],x[None,:3,:])*kmt.taskmat[task,task])
        True
        >>> kmt(task[:,None,None],task[:,None,None],x[:,None,:],x[None,:,:]).shape
        torch.Size([3, 4, 4])
        >>> kmt(task[:,None,None,None],task[None,:,None,None],x[:,None,:],x[None,:,:]).shape
        torch.Size([3, 3, 4, 4])

        Batched inference

        >>> kernel_base = KernelGaussian(d=10,shape_lengthscales=(5,1),shape_scale=(3,5,1),torchify=True)
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kmt = KernelMultiTask(kernel_base,
        ...     num_tasks=4,
        ...     shape_factor=(6,3,5,4,2),
        ...     diag = rng.uniform(low=0,high=1,size=(3,5,4)))
        >>> x = torch.from_numpy(np.random.rand(8,10))
        >>> task = torch.arange(4)
        >>> kmt(task[0],task[0],x[0],x[0]).shape
        torch.Size([6, 3, 5])
        >>> kmt(task,task,x[0],x[0]).shape
        torch.Size([6, 3, 5, 4])
        >>> kmt(task[0],task[0],x,x).shape
        torch.Size([6, 3, 5, 8])
        >>> v = kmt(task,task,x[:4,None,:],x[None,:4,:])
        >>> v.shape
        torch.Size([6, 3, 5, 4, 4])
        >>> kmat_x = kmt.base_kernel(x[:4,None,:],x[None,:4,:])
        >>> kmat_x.shape
        torch.Size([3, 5, 4, 4])
        >>> kmat_tasks = kmt.taskmat[...,task,task]
        >>> kmat_tasks.shape
        torch.Size([6, 3, 5, 4])
        >>> torch.allclose(v,kmat_tasks[...,None,:]*kmat_x)
        True
        >>> torch.allclose(v,kmat_tasks[...,:,None]*kmat_x)
        False
        >>> kmt(task[:,None,None,None],task[None,:,None,None],x[:,None,:],x[None,:,:]).shape
        torch.Size([6, 3, 5, 4, 4, 8, 8])
        >>> kmt(task[:,None],task[None,:],x[:,None,None,None,:],x[None,:,None,None,:]).shape
        torch.Size([6, 3, 5, 8, 8, 4, 4])
        >>> kmt.factor.dtype
        torch.float32
        >>> kmt.diag.dtype
        torch.float64

        Integrals

        >>> kernel = KernelGaussian(
        ...     d = 2,
        ...     torchify = True,
        ...     scale = rng.uniform(low=0,high=1,size=(3,1)),
        ...     lengthscales = rng.uniform(low=0,high=1,size=(1,2)))
        >>> kmt = KernelMultiTask(
        ...     base_kernel=kernel,
        ...     num_tasks=5,
        ...     diag = rng.uniform(low=0,high=1,size=(6,3,5)),
        ...     factor = rng.uniform(low=0,high=1,size=(3,5,2)))
        >>> task = torch.arange(5)
        >>> kmt.double_integral_01d(task0=task[0],task1=task[1]).shape
        torch.Size([6, 3])
        >>> kmt.double_integral_01d(task0=task,task1=task).shape
        torch.Size([6, 3, 5])
        >>> kmt.double_integral_01d(task0=task[:,None],task1=task[None,:]).shape
        torch.Size([6, 3, 5, 5])
        >>> x = torch.from_numpy(rng.uniform(low=0,high=1,size=(7,4,2)))
        >>> kmt.single_integral_01d(task0=task[0],task1=task[1],x=x).shape
        torch.Size([6, 3, 7, 4])
        >>> kmt.single_integral_01d(task0=task[:,None,None],task1=task[:,None,None],x=x).shape
        torch.Size([6, 3, 5, 7, 4])
        >>> kmt.single_integral_01d(task0=task[:,None,None,None],task1=task[None,:,None,None],x=x).shape
        torch.Size([6, 3, 5, 5, 7, 4])

        Cholesky Construction

        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5,torchify=True),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     )
        >>> kmt.taskmat
        tensor([[ 2.2500,  3.0000,  3.0000],
                [ 3.0000,  6.2500,  7.0000],
                [ 3.0000,  7.0000, 10.2500]], grad_fn=<ViewBackward0>)
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5,torchify=True),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [2,3*(1+3)//2-3],
        ...     shape_diag = [3],
        ...     )
        >>> kmt.taskmat
        tensor([[[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]],
        <BLANKLINE>
                [[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]]], grad_fn=<ViewBackward0>)
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5,torchify=True),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [3*(1+3)//2-3],
        ...     shape_diag = [2,3],
        ...     )
        >>> kmt.taskmat
        tensor([[[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]],
        <BLANKLINE>
                [[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]]], grad_fn=<ViewBackward0>)
        >>> kmt = KernelMultiTask(
        ...     KernelGaussian(5,torchify=True),
        ...     num_tasks = 3,
        ...     method = "CHOLESKY",
        ...     factor = 2,
        ...     diag = 1.5,
        ...     shape_factor = [2,3*(1+3)//2-3],
        ...     shape_diag = [2,3],
        ...     )
        >>> kmt.taskmat
        tensor([[[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]],
        <BLANKLINE>
                [[ 2.2500,  3.0000,  3.0000],
                 [ 3.0000,  6.2500,  7.0000],
                 [ 3.0000,  7.0000, 10.2500]]], grad_fn=<ViewBackward0>)
    """

    def __init__(
        self,
        base_kernel: AbstractKernel,
        num_tasks: int,
        factor=1.0,
        diag=1.0,
        shape_factor: list = None,
        shape_diag: list = None,
        tfs_factor=(tf_identity, tf_identity),
        tfs_diag=(tf_exp_eps_inv, tf_exp_eps),
        requires_grad_factor: bool = True,
        requires_grad_diag: bool = True,
        rank_factor=1,
        method: str = "LOW RANK",
    ) -> None:
        r"""
        Args:
            base_kernel (AbstractKernel): $K_{\mathrm{base}}$.
            num_tasks (int): Number of tasks $T>1$.
            factor (Union[np.ndarray, torch.Tensor]): Factor $\mathsf{F}$.
            diag (Union[np.ndarray, torch.Tensor]): Diagonal parameter
                $\boldsymbol{v}$.
            shape_factor (list): Shape of `factor` when `np.isscalar(factor)`.
            shape_diag (list): Shape of `diag` when `np.isscalar(diag)`.
            tfs_factor (Tuple[callable,callable]): The first argument
                transforms to the raw value to be optimized; the second applies
                the inverse transform.
            tfs_diag (Tuple[callable,callable]): The first argument transforms
                to the raw value to be optimized; the second applies the
                inverse transform.
            requires_grad_factor (bool): If `True` and `torchify`, set
                `requires_grad=True` for `factor`.
            requires_grad_diag (bool): If `True` and `torchify`, set
                `requires_grad=True` for `diag`.
            method (str): `"LOW RANK"` or "CHOLESKY"
        """
        if not (isinstance(base_kernel, AbstractKernel)):
            raise AssertionError
        super().__init__(
            d=base_kernel.d,
            torchify=base_kernel.torchify,
            device=base_kernel.device,
            compile_call=False,
            compile_call_kwargs={},
        )
        self.base_kernel = base_kernel
        self.AUTOGRADKERNEL = base_kernel.AUTOGRADKERNEL
        if not (np.isscalar(num_tasks) and num_tasks % 1 == 0):
            raise AssertionError
        self.num_tasks = num_tasks
        if not (
            np.isscalar(rank_factor)
            and rank_factor % 1 == 0
            and 0 <= rank_factor <= self.num_tasks
        ):
            raise AssertionError
        self.method = str(method).upper().replace("_", " ").strip()
        if self.method == "LOW RANK":
            if shape_factor is None:
                shape_factor = [self.num_tasks, rank_factor]
            factor_endsize_ops = list(range(self.num_tasks + 1))
            diag_endsize_ops = [1, self.num_tasks]
        elif self.method == "CHOLESKY":
            if self.torchify:
                self.lti0, self.lti1 = self.npt.tril_indices(num_tasks, num_tasks, -1)
            else:
                self.lti0, self.lti1 = self.npt.tril_indices(num_tasks, -1)
            if shape_factor is None:
                shape_factor = [len(self.lti0)]
            factor_endsize_ops = [len(self.lti0)]
            diag_endsize_ops = [self.num_tasks]
        else:
            raise ValueError(
                "invalid method = %s, must be in ['LOW RANK','CHOLESKY']" % self.method
            )
        self.raw_factor = self.parse_assign_param(
            pname="factor",
            param=factor,
            shape_param=shape_factor,
            requires_grad_param=requires_grad_factor,
            tfs_param=tfs_factor,
            endsize_ops=factor_endsize_ops,
            constraints=[],
        )
        self.tfs_factor = tfs_factor
        if self.method == "LOW RANK":
            if not (self.raw_factor.shape[-2] == self.num_tasks):
                raise AssertionError
        self.raw_diag = self.parse_assign_param(
            pname="diag",
            param=diag,
            shape_param=[self.num_tasks] if shape_diag is None else shape_diag,
            requires_grad_param=requires_grad_diag,
            tfs_param=tfs_diag,
            endsize_ops=diag_endsize_ops,
            constraints=["NON-NEGATIVE"],
        )
        self.tfs_diag = tfs_diag
        self.eye_num_tasks = self.npt.eye(self.num_tasks, **self.nptkwargs)
        self.batch_param_names.append("taskmat")
        self._nbdim_base = None

    @property
    def nbdim_base(self):
        if self._nbdim_base is None:
            self._nbdim_base = self.base_kernel.nbdim
        return self._nbdim_base

    @property
    def factor(self):
        return self.tfs_factor[1](self.raw_factor)

    @property
    def diag(self):
        return self.tfs_diag[1](self.raw_diag)

    @property
    def taskmat(self):
        factor = self.factor
        diag = self.diag
        if self.method == "LOW RANK":
            taskmat = (
                self.npt.einsum("...ij,...kj->...ik", factor, factor)
                + diag[..., None] * self.eye_num_tasks
            )
        elif self.method == "CHOLESKY":
            L = diag[..., None] * self.eye_num_tasks + self.npt.zeros(
                list(self.factor.shape[:-1]) + [1, 1], **self.nptkwargs
            )
            L[..., self.lti0, self.lti1] = self.factor
            taskmat = self.npt.einsum("...ij,...kj->...ik", L, L)
        else:
            raise ValueError(
                "invalid method = %s, must be in ['LOW RANK','CHOLESKY']" % self.method
            )
        return taskmat

    def _parsed__call__(self, task0, task1, k_x):
        k_x = k_x[..., None]
        nnbdim_v = k_x.ndim - self.nbdim_base - 1
        kmat_tasks = self.taskmat
        nbdim = kmat_tasks.ndim - 2
        kmat_tasks = kmat_tasks[..., task0, task1, None]
        nnbdim_kmat = kmat_tasks.ndim - nbdim - 1
        commondim = max(nnbdim_v, nnbdim_kmat)
        k_x = insert_batch_dims(k_x, commondim - nnbdim_v, self.nbdim_base)
        kmat_tasks = insert_batch_dims(kmat_tasks, commondim - nnbdim_kmat, nbdim)
        kmat = k_x * kmat_tasks
        return kmat[..., 0]

    def __call__(self, task0, task1, x0, x1, beta0=None, beta1=None, c=None):
        r"""Evaluate the kernel with (optional) partial derivatives

        $$\sum_{\ell=1}^p c_\ell
        \partial_{\boldsymbol{x}_0}^{\boldsymbol{\beta}_{\ell,0}}
        \partial_{\boldsymbol{x}_1}^{\boldsymbol{\beta}_{\ell,1}}
        K((i_0,\boldsymbol{x}_0),(i_1,\boldsymbol{x}_1)).$$

        Args:
            task0 (Union[int, np.ndarray, torch.Tensor]): First task indices
                $i_0$.
            task1 (Union[int, np.ndarray, torch.Tensor]): Second task indices
                $i_1$.
            x0 (Union[np.ndarray, torch.Tensor]): Shape `x0.shape=(...,d)`
                first input to kernel.
            x1 (Union[np.ndarray, torch.Tensor]): Shape `x1.shape=(...,d)`
                second input to kernel.
            beta0 (Union[np.ndarray, torch.Tensor]): Shape `beta0.shape=(p,d)`
                derivative orders with respect to first inputs,
                $\boldsymbol{\beta}_0$.
            beta1 (Union[np.ndarray, torch.Tensor]): Shape `beta1.shape=(p,d)`
                derivative orders with respect to first inputs,
                $\boldsymbol{\beta}_1$.
            c (Union[np.ndarray, torch.Tensor]): Shape `c.shape=(p,)`
                coefficients of derivatives.

        Returns:
            Kernel evaluations with batched shape, see the doctests for
            examples.
        """
        kmat_x = self.base_kernel.__call__(x0, x1, beta0, beta1, c)
        return self._parsed__call__(task0, task1, kmat_x)

    def single_integral_01d(self, task0, task1, x):
        r"""Evaluate the integral of the kernel over the unit cube

        $$\tilde{K}((i_0,\boldsymbol{x}),i_1) = \int_{[0,1]^d}
        K((i_0,\boldsymbol{x}),(i_1,\boldsymbol{z}) \; \mathrm{d}
        \boldsymbol{z}.$$

        Args:
            task0 (Union[int, np.ndarray, torch.Tensor]): First task indices
                $i_0$.
            task1 (Union[int, np.ndarray, torch.Tensor]): Second task indices
                $i_1$.
            x (Union[np.ndarray, torch.Tensor]): Shape `x0.shape=(...,d)` first
                input to kernel with

        Returns:
            Shape `y.shape=x.shape[:-1]` integral kernel evaluations.
        """
        kint_x = self.base_kernel.single_integral_01d(x)
        return self._parsed__call__(task0, task1, kint_x)

    def double_integral_01d(self, task0, task1):
        r"""Evaluate the integral of the kernel over the unit cube

        $$\tilde{K}(i_0,i_1) = \int_{[0,1]^d} \int_{[0,1]^d}
        K((i_0,\boldsymbol{x}),(i_1,\boldsymbol{z})) \; \mathrm{d}
        \boldsymbol{x} \; \mathrm{d} \boldsymbol{z}.$$

        Args:
            task0 (Union[int, np.ndarray, torch.Tensor]): First task indices
                $i_0$.
            task1 (Union[int, np.ndarray, torch.Tensor]): Second task indices
                $i_1$.

        Returns:
            Double integral kernel evaluations.
        """
        kint_x = self.base_kernel.double_integral_01d()
        return self._parsed__call__(task0, task1, kint_x)


class KernelMultiTaskDerivs(KernelMultiTask):
    def __init__(
        self,
        base_kernel,
        num_tasks,
    ) -> None:
        super().__init__(
            base_kernel=base_kernel,
            num_tasks=num_tasks,
            factor=1.0,
            diag=0.0,
            requires_grad_factor=False,
            requires_grad_diag=False,
            tfs_factor=(tf_identity, tf_identity),
            tfs_diag=(tf_identity, tf_identity),
            rank_factor=1,
        )
