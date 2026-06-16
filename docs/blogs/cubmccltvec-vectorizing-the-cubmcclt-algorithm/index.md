<!--
Source WordPress URL: https://qmcpy.org/2026/02/25/cubmccltvec-vectorizing-the-cubmcclt-algorithm/
Source TeX file: docs/blogs/cubmccltvec.tex
Original metadata: Posted by Aadit Jain; February 25, 2026.
Image handling: no local images were supplied or required.
-->

# CubMCCLTVec: Vectorizing the CubMCCLT Algorithm

--8<-- "snippets/blog-authors/cubmccltvec-vectorizing-the-cubmcclt-algorithm.md"

February 25, 2026

This post introduces `CubMCCLTVec`, a vectorized extension of `CubMCCLT` for confidence intervals on vector-valued quantities of interest.

Recent work by Aleksei G. Sorokin and Jagadeeswaran Rathinavel [1]
discuss extending stopping criterion for a scalar mean to stopping
criterion for vector quantities of interest formulated as functions of
multiple means. One such scalar stopping criterion is `CubMCCLT` that
calculates a confidence interval for \(\mu\) by using the Central Limit
Theorem. When \(\{x_i\}_{i=1}^{n}\) are IID and \(f\) has a finite
variance, the confidence interval for \(\mu\) can be calculated as:

$$
\mu^{\pm} = \hat{\mu}\; \pm\; C Z_{\alpha^{(\mu)}/2} \hat{\sigma}/\sqrt{n}
$$

Here, \(\hat{\mu}\) is the sample average of function evaluations,
\(Z_{\alpha^{(\mu)}/2}\) is the inverse CDF of a standard normal
distribution at \(1 - \alpha^{(\mu)}/2\), the variance of \(f(X)\) can
be approximated by
\(\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^{n} (f(\boldsymbol{x}_i) - \hat{\mu})^2\),
and \(C^2\) is an inflation factor greater than 1 for a more
conservative estimate.

Building on the `CubMCCLT` algorithm, we have developed a vectorized
version of it known as `CubMCCLTVec`.

## What Does the CubMCCLTVec Class Do?

The `CubMCCLTVec` class, which is a stopping criterion object,
calculates a confidence interval for functions with multiple outputs
based on the user-specified confidence level (default is 99%). Given an
initial and maximum sample size and an absolute tolerance, we keep on
doubling the sample size and recomputing the confidence interval until
half the confidence interval width is less than the absolute tolerance
or the double of the current sample size exceeds the maximum sample
size.

Like the other stopping criterion objects, `CubMCCLTVec` too utilizes an
accumulate data object to recompute the confidence interval known as
`MeanVarDataVec`.

## Some Examples Utilizing the CubMCCLTVec Class

The following code illustrates three examples that are being solved using
`CubMCCLTVec`:

1. Covariance [2]: \(T \sim \mathcal{N}(1,I_d)\) and the covariance of
   \(P = T_0\cdots T_{d-1}\) and \(S = T_0+\dots+T_{d-1}\) is:

   $$
   \mathrm{Cov}[P,S] = \mathbb{E}[PS]-\mathbb{E}[P]\mathbb{E}[S] = \mu_0-\mu_1\mu_2
   $$

   Theoretically, \(\mathrm{Cov}[P,S] = 2d-(1)(d) = d\).

2. Box Integral [3]: \(B_s(x) = (\sum_{j=1}^{d} x^2_j)^{s/2}\) where
   \(x_1,\dots,x_d \sim \mathcal{U}[0,1]\) and the box integral is
   computed for \(s=-1\) and \(s=1\) (the two outputs).

3. Custom Fun: \(x_j \sim \mathcal{U}[0,2j]\) for \(j=1,\dots,6\).

## Python Implementation

```python
import qmcpy as qp
import numpy as np

# Example 1: Covariance
dimension = 4
true_measure = qp.IIDStdUniform(dimension)

class Covariance(qp.integrand.Integrand):
    def __init__(self, true_measure):
        super().__init__(true_measure)

    def g(self, x):
        P = np.prod(x, axis=1)
        S = np.sum(x, axis=1)
        return np.vstack((P*S, P, S)).T

integrand = Covariance(true_measure)
solution = qp.CubMCCLTVec(integrand, abs_tol=1e-2)
print("Covariance:", solution.integrate())


# Example 2: Box Integral
dimension = 5
true_measure = qp.IIDStdUniform(dimension)

class BoxIntegral(qp.integrand.Integrand):
    def __init__(self, true_measure):
        super().__init__(true_measure)

    def g(self, x):
        norm_sq = np.sum(x**2, axis=1)
        return np.vstack((norm_sq**(-0.5), norm_sq**(0.5))).T

integrand = BoxIntegral(true_measure)
solution = qp.CubMCCLTVec(integrand, abs_tol=1e-2)
print("Box Integral:", solution.integrate())


# Example 3: Custom Function
dimension = 6
true_measure = qp.IIDStdUniform(dimension)

class CustomFun(qp.integrand.Integrand):
    def __init__(self, true_measure):
        super().__init__(true_measure)

    def g(self, x):
        weights = np.arange(1, dimension+1)
        scaled = x * (2*weights)
        f1 = np.sum(scaled, axis=1)
        f2 = np.prod(scaled, axis=1)
        return np.vstack((f1, f2)).T

integrand = CustomFun(true_measure)
solution = qp.CubMCCLTVec(integrand, abs_tol=1e-2)
print("Custom Fun:", solution.integrate())
```

**Example 1 Output: Covariance**

```text
Solution: 3.998147791818197
Data:
MeanVarDataVec (AccumulateData Object)
    solution        3.998
    comb_bound_low  3.977
    comb_bound_high 4.019
    comb_flags      1
    n_total         2^(26)
    n               [67108864. 67108864. 67108864.]
    time_integrate  152.068
CubMCCLTVec (StoppingCriterion Object)
    inflate         1.200
    alpha           0.010
    abs_tol         0.025
    rel_tol         0
    n_init          2^(8)
    n_max           2^(30)
CovIntegrand (Integrand Object)
Gaussian (TrueMeasure Object)
    mean            1
    covariance      1
    decomp_type     PCA
IIDStdUniform (DiscreteDistribution Object)
    d               2^(2)
    entropy         7
    spawn_key       ()
```

**Example 2 Output: Box Integral**

```text
Solution: [1.1853359  0.95670595]
Data:
MeanVarDataVec (AccumulateData Object)
    solution        [1.185 0.957]
    comb_bound_low  [1.139 0.918]
    comb_bound_high [1.232 0.995]
    comb_flags      [ True  True]
    n_total         2^(11)
    n               [2048.  512.]
    time_integrate  0
CubMCCLTVec (StoppingCriterion Object)
    inflate         1.200
    alpha           0.010
    abs_tol         0.050
    rel_tol         0
    n_init          2^(8)
    n_max           2^(30)
BoxIntegral (Integrand Object)
    s               [-1  1]
Uniform (TrueMeasure Object)
    lower_bound     0
    upper_bound     1
IIDStdUniform (DiscreteDistribution Object)
    d               3
    entropy         7
    spawn_key       ()
```

**Example 3 Output: Custom Fun**

```text
Solution: [[1.00001237 1.99884832 2.99870039]
           [4.00056276 4.99789845 6.00074045]]
Data:
MeanVarDataVec (AccumulateData Object)
    solution            [[1.    1.999 2.999]
                        [4.001 4.998 6.001]]
    comb_bound_low      [[0.99  1.989 2.991]
                        [3.991 4.989 5.993]]
    comb_bound_high     [[1.01  2.009 3.006]
                        [4.01  5.007 6.008]]
    comb_flags          [[ True  True  True]
                        [ True  True  True]]
    n_total             2^(21)
    n                   [[ 32768. 131072.  524288.]
                        [ 524288. 1048576. 2097152.]]
    time_integrate      0.650
CubMCCLTVec (StoppingCriterion Object)
    inflate             1.200
    alpha               0.010
    abs_tol             0.010
    rel_tol             0
    n_init              2^(8)
    n_max               2^(30)
CustomFun (Integrand Object)
Uniform (TrueMeasure Object)
    lower_bound         0
    upper_bound         1
IIDStdUniform (DiscreteDistribution Object)
    d                   6
    entropy             7
    spawn_key           ()
```

## Benefits of Developing the CubMCCLTVec Class

This class gives us a new and different option to find when the
user-specified error tolerance has been satisfied and its generalization
to functions with multiple outputs allows us to utilize the existing
`CubMCCLT` algorithm and extend it to such functions.

## References

1. Sorokin, A. G., & Rathinavel, J. *On Bounding and Approximating
   Functions of Multiple Expectations using Quasi-Monte Carlo*. To
   appear in the *Monte Carlo and Quasi-Monte Carlo Methods in
   Scientific Computing Proceedings 2022* (2022).
2. Sorokin, A. *Monte Carlo for Vector Functions of Integrals*. Jupyter
   Notebook. QMCPy: A quasi-Monte Carlo Python Library, 2023.
   [https://github.com/QMCSoftware/QMCSoftware/blob/master/demos/pydata.chi.2023.ipynb](https://github.com/QMCSoftware/QMCSoftware/blob/master/demos/pydata.chi.2023.ipynb).
3. Bailey, D., Borwein, J., & Crandall, R. *Box integrals*. *Journal of
   Computational and Applied Mathematics* **206**, 196-208. ISSN:
   0377-0427.
   [https://www.sciencedirect.com/science/article/pii/S0377042706004250](https://www.sciencedirect.com/science/article/pii/S0377042706004250)
   (2007).
