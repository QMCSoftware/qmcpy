# Kernel/discrepancy audit of develop

**Baseline:** [`3c89e7fc`](https://github.com/QMCSoftware/qmcpy/tree/3c89e7fc),
audited November 3, 2026. The audit used source inspection, small deterministic
NumPy/PyTorch probes, and existing focused tests. This is evidence for
[issue #646](https://github.com/QMCSoftware/qmcpy/issues/646), not a report
that the defects below have been fixed. No production code changed.

The [design plan](kernel-discrepancy-design.md) separates the proposed
interfaces, staged work, and required human decisions. Its
[paper copies](references/README.md) supply the mathematical starting point.

## What already exists

| Area | Evidence in this revision | Consequence |
| --- | --- | --- |
| Public SI/DSI kernels | [si_dsi_kernels.py](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py), including FFT matvec/solve examples near lines 257–279 and FWHT examples near 812–819, 1117–1124, and 1395–1402. | Reuse the existing classes and transforms; do not rebuild the old Classlib kernel hierarchy. |
| Unit-cube integrals | `AbstractSIDSIKernel` methods near lines 140–152 return the constant `scale` for ordinary kernel values. | Equal-weight matching designs admit a one-column discrepancy reduction; arbitrary measures require additional supported integral pairs. |
| Transforms | [ft.py](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/fast_transform/ft.py): orthonormal `fftbr`/`ifftbr`, `fwht`, and Torch variants. | Preserve bit-reversal, normalization, last-axis batching, binary precision, and supported parameter gradients. |
| Bayesian fast paths | [Lattice spectrum](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/cub_qmc_bayes_lattice_g.py#L251), [digital-net spectrum](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/cub_qmc_bayes_net_g.py#L251), and [spectral objective/credible bounds](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/abstract_cub_bayes_ld_g.py#L113). | First-column FFT/FWHT and spectral likelihood work already avoid dense inverses/determinants. Private kernel formulas need reconciliation with public kernels and independent correctness checks. |
| Cancellation protection | Public `combine_per_dim_components_raw_m1(..., stable=True)` near lines 154–189; Bayesian `kernel_t`, remainder spectra, and `avoid_cancel_error` paths. | Retain and expose the remainder through public discrepancy/Gram operations instead of subtracting the constant after full evaluation. |
| General discrepancy and reusable Gram operators | No general public discrepancy package or reusable structured Gram/operator class was found in the inspected tree. Specialized Kronecker/MPMC calculations exist. | The main architectural work is a common capability/operation contract and independently validated consumers. |

## Compatibility probes

These small checks establish starting cases, not a proof for every permitted
parameter, device, precision, or design.

- SI/lattice, `n=16`: the documented bit-reversed FFT product agreed with
  dense multiplication to about `3.6e-15` in radical-inverse order. Applying
  that same indexing unchanged to LINEAR or GRAY order produced errors about
  `9.1` and `8.9`. Support another order through an explicit verified
  permutation/transform, or reject/fall back.
- DSI/digital net, `n=8`, `d=2`, seed 11, binary precision 32: dense/FWHT
  products agreed within `3.2e-15` for FALSE, DS, LMS, and LMS DS in radical
  inverse and Gray order. NUS failed the same simple group-based product,
  with errors about `0.0922` and `0.1039` respectively.
- The three DSI families passed small products with two independent sampler
  replications and two kernel parameter batches. Interlacing factors 1 and 2
  passed; the sampler's interlacing factor and kernel's smoothness/combination
  parameters remain distinct.
- CPU Torch float64 products and trainable-parameter gradients agreed with
  dense calculations within approximately `7.2e-15` and `1.1e-14` in those
  small cases. This does not establish every input-derivative contract.
- Whole-design common modular/digital shifts cancel in the correct pairwise
  kernel column. Independent replications should be processed independently;
  their cross-replication Gram blocks do not inherit a single transform.

Full dyadic prefixes are the initial fast-path contract. A sampler accepting
an interval, or a kernel being stationary, does not by itself prove the
matrix has the required group structure.

## Confirmed baseline problems to triage

Record focused regression/fix tasks before reusing the affected code as an
oracle. These findings belong to the existing revision; the planning branch
contains no attempted numerical fixes.

| Finding | Observed evidence | Required check |
| --- | --- | --- |
| Combined SI kernel can be nonsymmetric | `KernelShiftInvarCombined.get_per_dim_components` near [line 725](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py#L725) evaluates Bernoulli degrees 1–4 with coefficients associated with even-degree kernels. The `n=8`, `d=3` example below has `max(abs(K-K.T)) = 35.3878784`. | Check the intended kernel against its formula; test symmetry/PSD as well as fast/dense operation parity. |
| Bayesian lattice spectrum uses shifted coordinates as a column | The integration loop stores randomized points in `data.xfull` and passes them through the objective to the private kernel. Its direct polynomial evaluation does not subtract the first point. For `n=32`, `d=3`, seed 11, order 1, fixed `theta=.5`, the restored spectrum differed from the correct dense spectrum by `0.956173`, with minimum eigenvalue `-0.226934`. Using `(X-X[0]) % 1` reduced error to `1.9e-15`. | Verify the actual integration call path and construct `K(X,X[0])` or equivalent differences; preserve returned spectrum normalization. |
| Analogous digital-shift spectrum mismatch | Private order 1, parameter `.5`, matches public DSI order 1 with coordinate weight `.05`. At `n=8`, `d=2`, seed 11, LMS DS, the restored spectrum error was `0.4220068`; XOR differences reduced it to `1.8e-15`. | Correct first-column construction and separately validate the design's group structure. XOR-correcting NUS's first column does not make its Gram FWHT-diagonalizable. |
| Bayesian integral sign is lost | [AbstractCubBayesLDG near line 187](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/abstract_cub_bayes_ld_g.py#L187) applies `abs` to the estimate. A negative linear integrand with target integral `-1.5` produces approximately `+1.5` for both samplers in MLE and FULL modes. | Signed-integrand regression independent of credible-width checks. |
| GCV raises a runtime error | Near [line 171](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/abstract_cub_bayes_ld_g.py#L171), `temp(temp != 0)` calls a NumPy array. Both GCV probes below raise `TypeError`. | Explicit GCV execution and formula checks, not only the default MLE path. |
| Public stable evaluation does not preserve a separately accessible tiny remainder | Ordinary SI/DSI evaluation adds the stable remainder back to the constant. In the `n=1024`, weight `1e-12` example below, subsequent subtraction gives `0.0`; the known squared discrepancy is about `3.13746274347e-18`. | Direct public remainder access and relative/high-precision checks that cannot pass by returning zero. |

The objective currently applies absolute values to eigenvalues. A new
operator must distinguish roundoff from a materially invalid covariance;
absolute values must not conceal a broken kernel or first-column assumption.

## Reproducing the key findings

Run the snippets from the QMCPy checkout using its development environment.
Warnings about deliberately fixed small sample limits can be ignored for
these probes; they are not convergence/coverage experiments.

### Integral sign, GCV, and combined-kernel symmetry

```python
import warnings
import numpy as np
import qmcpy as qp

warnings.filterwarnings("ignore")
for criterion, sampler in [
    (qp.CubBayesLatticeG, qp.Lattice),
    (qp.CubBayesNetG, qp.DigitalNetB2),
]:
    for mode in ["MLE", "FULL", "GCV"]:
        f = qp.CustomFun(
            qp.Uniform(sampler(dimension=1, seed=7, order="RADICAL INVERSE")),
            lambda x: -1 - x[:, 0],
        )
        try:
            value, data = criterion(
                f, n_init=32, n_limit=32, abs_tol=10, errbd_type=mode
            ).integrate()
            print(criterion.__name__, mode, value, data.bounds_half_width)
        except Exception as exc:
            print(criterion.__name__, mode, type(exc).__name__, str(exc))

x = qp.Lattice(3, seed=11)(8)
k = qp.KernelShiftInvarCombined(3, scale=1, lengthscales=[1, .5, .25])
K = k(x[:, None, :], x[None, :, :])
print("Combined symmetry error:", np.max(np.abs(K - K.T)))
```

Observed MLE/FULL estimates were `+1.500004667` for the lattice and
`+1.503910065` for the net; both GCV modes raised
`TypeError: 'numpy.ndarray' object is not callable`.

### Digital shifts versus the correct first column

```python
import warnings
import numpy as np
import qmcpy as qp

warnings.filterwarnings("ignore")
for mode in ["LMS", "DS", "LMS DS"]:
    design = qp.DigitalNetB2(2, seed=11, randomize=mode, t=32)
    criterion = qp.CubQMCBayesNetG(qp.Keister(design), n_init=8, n_limit=16)
    xb = design(8, return_binary=True, warn=False)
    x = xb * 2.0**(-design.t)
    differences = (xb ^ xb[0]) * 2.0**(-design.t)
    kernel = qp.KernelDigShiftInvar(d=2, t=design.t, alpha=1, weights=.05)
    truth = np.sqrt(8) * qp.fwht(kernel(xb, xb[0]))
    for label, points in [("raw", x), ("XOR differences", differences)]:
        lam, _, factor = criterion._shift_inv_kernel_digital(
            points, 1, .5, True, 1, False
        )
        print(mode, label, np.max(np.abs(factor * lam - truth)))
```

This restores the private method's `lambda_factor` before comparing. It
checks a first column; it does not by itself test the entire Gram matrix.

### Cancellation-sensitive discrepancy

```python
import numpy as np
import qmcpy as qp

n, weight = 1024, 1e-12
x = (np.arange(n) / n)[:, None]
kernel = qp.KernelShiftInvar(d=1, alpha=1, scale=1, weights=weight)
full = kernel(x, x[0], stable=True)
t = x[:, 0]
remainder = weight * 2 * np.pi**2 * (t*t - t + 1/6)
print("Full subtraction:", full.mean() - 1)
print("Direct remainder:", remainder.mean())
print("Known value:", weight * np.pi**2 / (3*n*n))
```

Observed: `0.0`, `3.137462743292149e-18`, and
`3.13746274346967e-18`. The product-remainder construction and stable
credible factors are developed in the lattice paper's sec. 4.4; the
[design plan](kernel-discrepancy-design.md) specifies how to preserve them
through the proposed public interface.

## Existing tests and remaining validation

This command passed **8 tests** on the audited code:

```bash
python -m pytest -q test/test_kn_kernels.py \
    test/test_ft_fast_transform_fallbacks.py test/test_sc_cubbayes_vec.py
```

Their success does not cover the failures reproduced above. The implementation
work needs direct structured/dense parity, symmetry/PSD, ordering and shift
invariants, incompatible scrambling, independent replications, supported
gradients, analytical moments, signed estimates, every supported credible
mode, cancellation stress cases, and explicit cost measurements. Full test
and notebook suites were not run for this documentation-only audit.
