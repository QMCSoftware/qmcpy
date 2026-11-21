# My review and proposed next steps for PR #647

**Laasya (`Laasya-73`)**  
**Related:** [Issue #646](https://github.com/QMCSoftware/qmcpy/issues/646) · [Draft PR #647](https://github.com/QMCSoftware/qmcpy/pull/647)

I've reviewed the [design](kernel-discrepancy-design.md) and [audit](kernel-discrepancy-audit.md), along with the kernel and transform code that they refer to. I wanted to put my findings in one place so Fred and I can discuss the first implementation without losing track of the numerical issues already identified in the audit.

This note is meant to **support PR #647**, not introduce a different design. The existing design document should remain the place where we record agreed decisions, and the audit remains the baseline for the findings at `3c89e7fc`. I'm using the [reference guide](references/README.md) for the paper context. Nothing below is an approved API or an implemented change.

My main preference is to build on QMCPy's existing classes and numerical routines. I don't want this work to disturb the kernel hierarchy, public behavior, or the current GP and Bayesian cubature paths. If a proposed feature needs a broader architectural change, I'd rather discuss that separately before touching the implementation.

## Where I think we should start

I would start with one SI/lattice combination and one DSI/digital-net combination, keeping the first eligibility rules narrow enough that we can test them independently.

| Pair | Initial scope I'd suggest | Existing operation to reuse |
| --- | --- | --- |
| `KernelShiftInvar(alpha=1)` + `Lattice` | One complete rank-1 prefix with `n=2^m` points, one replication, `RADICAL INVERSE` order, and either `FALSE` or a common `SHIFT`. Use the continuous unit-cube Uniform measure and equal quadrature weights. | `fftbr` / `ifftbr` for Gram operations in bit-reversed order. |
| `KernelDigShiftInvar(alpha=1)` + `DigitalNetB2` | One complete compatible dyadic prefix in `RADICAL INVERSE` order, one replication, interlacing factor 1, `return_binary=True`, and matching binary precision `t`. Consider `FALSE`, `DS`, `LMS`, and `LMS DS` initially. | `fwht` for a Gram matrix whose XOR structure has been verified. Discrepancy also needs the correct target moments. |

### SI kernel with a lattice

For the SI case, the kernel evaluates modular differences. If all lattice points have the same shift, that shift disappears in `k(X, X[0])`. In linear group order the Gram matrix is circulant; radical-inverse order gives the bit-reversed indexing expected by the existing FFT code.

I wouldn't enable the same FFT indexing for `LINEAR` or `GRAY` yet. They represent valid orderings of a complete lattice, but the transform needs a corresponding permutation or a separately verified indexing rule. That restriction is about the **FFT route**, not the scalar discrepancy: reordering the same complete point set does not change the equal-weight discrepancy or the one-column identity.

For this SI kernel and the continuous Uniform target, the single and double kernel moments are both the scale $S$. With equal quadrature weights and diagonals included, the squared discrepancy is the mean of one **directly evaluated remainder** column. That calculation takes $O(nd)$ work at fixed coordinate cost and does not need an FFT. The FFT is useful when we need structured Gram products and solves.

Sources: [lattice sampling](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/discrete_distribution/lattice/lattice.py#L311) · [SI kernel evaluation](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L480)

### DSI kernel with a digital net

I think `KernelDigShiftInvar(alpha=1)` and a compatible `DigitalNetB2` prefix are the natural second pair. For an affine digital construction $x_i=L(i)\oplus\Delta$, we have $x_i\oplus x_j=L(i\oplus j)$. This is why a column formed using binary XOR differences can be used with the existing FWHT, provided the point ordering and precision match the assumed structure.

The audit includes passing cases for Gray order and interlacing factor 2. Those are worth following up on, but I wouldn't add them to the first supported contract without targeted tests. I'd also exclude NUS initially because it can break the XOR identity even when a particular kernel example happens to give the right matvec.

For either structured path, we still need to compare against a dense reference: symmetry, spectral normalization, matvecs, solves, and positive eigenvalues where a nonsingular covariance solve is required. If the design doesn't match the kernel, or if a prefix is incomplete or replications are pooled, I would use the planned dense or blocked path **only where the kernel–measure moments are supported**. Otherwise, the API should report that the pair is unsupported.

Sources: [digital-net settings](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/discrete_distribution/digital_net_b2/digital_net_b2.py#L218) · [DSI evaluation](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L1007) · [audit probes](kernel-discrepancy-audit.md#compatibility-probes) · [transforms](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/fast_transform/ft.py#L6)

## One DSI question I think we should resolve first

The finite-bit DSI moments deserve a decision before we turn on a fast discrepancy calculation. The implemented `to_bin` operation maps a continuous Uniform coordinate onto equally likely binary cells. For the order-1 coordinate term, the average over all $2^t$ XOR values is $4^{-t}$. The zero XOR contributes 1; for each highest set bit $h=0,\ldots,t-1$, there are $2^h$ values contributing $1-3\,2^{h-t}$.

This gives the following single and double moments for the quantized product kernel under independent Uniform coordinates:

$$
M_t=S\prod_j\left(1+\gamma_j4^{-t}\right),
$$

where $\gamma_j$ is the coordinate weight. For example, with one dimension, `t=2`, scale 1, and weight 1, the enumerated mean is **1.0625**, but the current moment hooks return **1.0**.

For the implemented kernel, the remainder centered at the finite-bit Uniform moment is $k-M_t$. I'd like us to decide which kernel to use for discrepancy. We could use the implemented finite-bit kernel with its induced Uniform grid measure, or an idealized Walsh kernel with its own evaluation rule and Uniform moments. For the finite-bit interpretation, I'd check the moments by exhaustive small-grid enumeration, require zero squared discrepancy for a full grid against its induced target, and test the FWHT constant mode separately from the Gram matvec.

Sources: [binary quantization](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/util/dig_shift_invar_ops.py#L164) · [order-1 term](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L1032) · [moment hooks](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L140)

## Checks included with this review

I kept two small, reproducible examples here so the discussion isn't based only on a proposed API. The SI example builds an independent dense matrix using the Bernoulli polynomial and compares it with the current FFT route for three point orderings. The DSI example enumerates the four binary cells and compares their mean with the reported moments.

These are **focused checks**, not a full test-suite result or a claim that the outstanding audit defects are fixed.

```python
import numpy as np
import qmcpy as qp

n = 16
v = np.arange(1, n + 1, dtype=float) / n
for order in ("RADICAL INVERSE", "LINEAR", "GRAY"):
    x = qp.Lattice(2, generating_vector=np.array([1, 3], dtype=np.uint64),
                   m_max=4,
                   order=order, randomize=False)(n, warn=False)
    u = (x[:, None, :] - x[None, :, :]) % 1
    dense = np.prod(1 + (np.pi**2 / 4) * (u*u - u + 1/6), axis=-1)
    k = qp.KernelShiftInvar(2, alpha=1, weights=1/8)
    column = k(x, x[0])
    np.testing.assert_allclose(column, dense[:, 0], rtol=1e-13, atol=1e-14)
    fast = qp.ifftbr(qp.fftbr(v) * (np.sqrt(n) * qp.fftbr(column)))
    error = np.max(np.abs(fast - dense @ v))
    if order == "RADICAL INVERSE":
        assert error < 1e-12
    else:
        assert error > 1e-3

grid = np.arange(4)[:, None] / 4
dsi = qp.KernelDigShiftInvar(1, alpha=1, t=2, scale=1, weights=1)
gram = dsi(grid[:, None, :], grid[None, :, :])
np.testing.assert_allclose(gram.mean(axis=1), 17/16, rtol=0, atol=1e-14)
np.testing.assert_allclose(gram.mean(), 17/16, rtol=0, atol=1e-14)
np.testing.assert_allclose(dsi.single_integral_01d(grid), 1, rtol=0, atol=1e-14)
np.testing.assert_allclose(dsi.double_integral_01d(), 1, rtol=0, atol=1e-14)
```

In the reported run, the SI matvec errors were about `3.6e-15` for radical-inverse order, `0.47` for linear order, and `0.073` for Gray order. This is a useful warning against treating transform ordering as interchangeable. The DSI check makes the finite-bit moment difference visible with just four points.

## A first patch I'd be interested in taking

My suggestion is to start with **direct remainder evaluation for ordinary `KernelShiftInvar` values**, rather than implementing discrepancy dispatch and structured Gram operators together.

The reason is numerical cancellation. For a product kernel

$$
k=S\prod_j(1+a_j),
$$

`AbstractSIDSIKernel.combine_per_dim_components_raw_m1` already has a stable recurrence for the part beyond the constant:

$$
r_1=a_1,\qquad r_j=r_{j-1}(1+a_j)+a_j.
$$

It computes $Sr_d$ directly. The current full-kernel call then adds $S$, which can round away a very small remainder. Subtracting $S$ afterward cannot recover it. I'd like to expose that existing computation through a small method such as `KernelShiftInvar.kernel_minus_constant(x0, x1)`, returning the remainder only. For the matching SI/Uniform case, the caller could obtain $S$ through the existing `double_integral_01d()` hook.

I would keep this **additive**: use the existing validation, broadcasting, parameter-batching, and backend conventions without changing `__call__`, supported derivative calls, or the class hierarchy. The likely future code changes would be confined to `qmcpy/kernel/si_dsi_kernels.py` and focused tests in `test/test_kn_kernels.py`. The method name, supported smoothness values, and backend scope still need Fred's agreement.

If we later use the remainder in Bayesian calculations, it would also help preserve the constant-mode remainder eigenvalue $\rho_0$ directly, so a factor of the form $\rho_0/(nS+\rho_0)$ does not have to be obtained through cancellation-prone subtraction. I see that as a later integration step, not part of this first patch.

Sources: [existing stable recurrence](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L154) · [kernel call path](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/abstract_kernel.py#L110) · [lattice paper, section 4.4](references/README.md)

### How I'd validate that first patch

| Check | What I'd compare against |
| --- | --- |
| Ordinary values and scaling | For `alpha=1`, use $R=2\pi^2S\gamma B_2(u)$, with $u=(x-y)\bmod1$. At $S=2.5$, $\gamma=1/8$, the values at $u=0$ and $u=1/2$ are $5\pi^2/48$ and $-5\pi^2/96$. In two dimensions, compare with $S(a_1+a_2+a_1a_2)$. Start with `rtol=1e-13`, `atol=2e-14` for float64. |
| Small positive discrepancy | For a one-dimensional grid with `n=1024`, $S=1$, and $\gamma=10^{-12}$, the expected column mean is $\gamma\pi^2/(3n^2)\approx3.13746274347\times10^{-18}$. Require a positive value with `rtol=1e-8`, `atol=0`. The [audit example](kernel-discrepancy-audit.md#cancellation-sensitive-discrepancy) shows that averaging the full kernel and then subtracting its constant can return zero. |
| Shapes and existing behavior | Check `(2,1,1)` by `(1,3,1)` inputs with two scale/weight batches, expecting `(2,2,3)`. Confirm ordinary and `stable=True` full-kernel results remain unchanged, and run existing derivative regressions. |
| Backend behavior | Verify NumPy first. If Torch is included in the approved scope, also test values, shapes, dtype, device, and scale/weight gradients. Include compiled calls if Fred wants those covered in the first patch. |

I would leave DSI centering, the general discrepancy API, structured Gram operators, and Bayesian integration for their own issues and implementation PRs. The combined-SI symmetry, shifted spectra, signed-integral, and GCV problems already recorded in the [audit](kernel-discrepancy-audit.md#confirmed-baseline-problems-to-triage) also deserve separate regression and fix work, rather than being bundled into this feature.

## What I'd like to agree on with Fred

Before starting an implementation branch, I'd like us to settle a few things:

1. **Initial pair eligibility:** Are the narrow SI/lattice and DSI/digital-net settings above the right place to begin? For DSI, which kernel and target-moment interpretation should we use?
2. **Remainder API:** Would a remainder-only `kernel_minus_constant(x0, x1)` on `KernelShiftInvar` fit the current object model? Which smoothness values should the first version support?
3. **Backend scope:** Should NumPy and Torch, including autograd and compiled calls, all be part of the first PR, or should we phase that validation?
4. **Ownership:** If the scope looks right, can I take the SI remainder method as my first focused task?

Once we've agreed, the active editor can add this note as supporting material for PR #647 and record only the accepted decisions in the existing design document. I'd track the implementation ownership, checks, and follow-ups in [Issue #646](https://github.com/QMCSoftware/qmcpy/issues/646), with the implementation on a separate issue-linked branch from the current `develop`.
