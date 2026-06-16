<!--
Source WordPress URL: https://qmcpy.org/2022/05/19/bayesian-stopping-criteria/
Original metadata: Posted by Jagadeeswaran Rathinavel and Fred J. Hickernell; May 19, 2022; updated February 6, 2023.
Image handling: no content images were present in the original post.
-->

# Bayesian Stopping Criteria

--8<-- "snippets/blog-authors/bayesian-stopping-criteria.md"

May 19, 2022

This post explains Bayesian stopping criteria for QMC integration and how matching kernels with lattice or digital net designs reduces credible-interval costs.

The blog [Why Add Q to MC?](../why-add-q-to-mc/index.md) explained the
advantages of carefully chosen, low discrepancy sampling sites for
approximating multivariate integrals, or equivalently, expectations of
functions of multivariate random variables. This blog post explains a
Bayesian approach to determining the sample size required to satisfy the
user's error tolerance.

Recall that the problem of interest takes the following form:

$$
\begin{aligned}
\mu
&:= \int_{[0,1]^d} f(\boldsymbol{x}) \,
\mathrm{d}\boldsymbol{x}
= \mathbb{E}[f(\boldsymbol{X})],
\qquad
\boldsymbol{X} \sim \mathcal{U}[0,1]^d, \\
\mu \approx \widehat{\mu}_n
&:= \frac{1}{n}
\sum_{i=1}^n f(\boldsymbol{x}_i),
\qquad
\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots
\text{ are the sampling sites or nodes.}
\end{aligned}
$$

The question of when enough samples have been taken to satisfy the
specified error criterion,
$\left|\mu - \widehat{\mu}_n\right| \le \varepsilon$, is an important
one. Stopping criteria based on Bayesian credible intervals are one good
way of answering this question.

Traditionally, Bayesian credible intervals take $\mathcal{O}(n^3)$
operations to construct, where $n$ is the sample size. This is due to
the need to invert an $n \times n$ covariance matrix. In contrast,
computing $\widehat{\mu}$ requires only $\mathcal{O}(n)$ operations.
Fortuitously, there is a way to reduce the computational cost of
constructing the Bayesian credible interval to
$\mathcal{O}(n \log(n))$ when using lattice or digital net low
discrepancy samples. These Bayesian stopping criteria are implemented as
[`CubBayesLatticeG`](https://qmcsoftware.github.io/QMCSoftware/) and
[`CubBayesNetG`](https://qmcsoftware.github.io/QMCSoftware/). This blog
explains the key points.

## The Bayesian Approach

The Bayesian approach to approximating integrals assumes that the
integrand, $f:[0,1]^d \to \mathbb{R}$, is drawn from a Gaussian
stochastic process parameterized by a constant mean and a covariance
function defined by scale and shape parameters. This is denoted
$f \sim \mathcal{GP}(m, s^2 C_{\boldsymbol{\theta}})$, and means that

$$
\mathbb{E}[f(\boldsymbol{x})] = m,
\qquad
\operatorname{cov}
\left(f(\boldsymbol{t}), f(\boldsymbol{x})\right)
=
\mathbb{E}
\left[
\left(f(\boldsymbol{t}) - m\right)
\left(f(\boldsymbol{x}) - m\right)
\right]
=
s^2 C_{\boldsymbol{\theta}}(\boldsymbol{t}, \boldsymbol{x}),
\qquad
\forall \boldsymbol{t}, \boldsymbol{x} \in [0,1]^d.
$$

Here, $m$, $s$, and $C_{\boldsymbol{\theta}}$ must be specified or
estimated. Because $f$ is a Gaussian process, the multivariate integral
of $f$, i.e., $\mu$, has a Gaussian distribution. Automatic Bayesian
cubature uses these assumptions to increase $n$ until we reach

$$
\mathbb{P}_f
\left[
\left| \mu - \widehat{\mu}_n \right|
\le \varepsilon
\right]
\ge 99\%.
$$

The [Bayesian credible interval](https://arxiv.org/abs/1809.09803),
which depends on the sampling nodes and the parameters defining the
Gaussian process, is

$$
\mathbb{P}_f
\left[
\left|\mu - \widehat{\mu}_n\right|
\le \operatorname{err}_{\operatorname{CI}}
\right]
= 99\%,
\qquad
\operatorname{err}_{\operatorname{CI}}
:=
2.58\,s
\sqrt{
c_{0,\boldsymbol{\theta}}
-
\boldsymbol{c}_{\boldsymbol{\theta}}(\mathsf{X})^{\mathsf{T}}
\mathsf{C}_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})^{-1}
\boldsymbol{c}_{\boldsymbol{\theta}}(\mathsf{X})
},
$$

assuming that the mean of the Gaussian process, $m$, is zero. The
formulae for the case of general $m$ are more complicated, but similar.
Here,

$$
\begin{gathered}
\mathsf{X}
=
\left(
\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n
\right)^{\mathsf{T}},
\qquad
c_{0,\boldsymbol{\theta}}
:=
\int_{[0,1]^d \times [0,1]^d}
C_{\boldsymbol{\theta}}(\boldsymbol{t}, \boldsymbol{x})
\, \mathrm{d}\boldsymbol{t}
\, \mathrm{d}\boldsymbol{x}, \\
\boldsymbol{c}_{\boldsymbol{\theta}}(\mathsf{X})
:=
\left(
\int_{[0,1]^d}
C_{\boldsymbol{\theta}}(\boldsymbol{t}, \boldsymbol{x}_i)
\, \mathrm{d}\boldsymbol{t}
\right)_{i=1}^n,
\qquad
\mathsf{C}_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})
=
\left(
C_{\boldsymbol{\theta}}(\boldsymbol{x}_i, \boldsymbol{x}_j)
\right)_{i,j=1}^n,
\end{gathered}
$$

and $2.58$ represents the $99.5\%$ quantile of the standard Gaussian
distribution. Bayesian cubature proceeds by increasing $n$ until
$\operatorname{err}_{\operatorname{CI}}$ is no greater than the error
tolerance.

The Bayesian credible interval as a stopping criterion assumes that the
integrand, $f$, is not an outlier with respect to this Gaussian process.
To improve the chances that this is the case, the hyperparameters $m$,
$s$, and $\boldsymbol{\theta}$ should be tuned using the function data.
In particular, $s$ should depend on the magnitude of the fluctuations of
$f$. The standard approaches for tuning the hyperparameters require
optimization, for which each iteration involves an eigendecomposition of
$C_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})$.

## Fast Bayesian Cubature

The computation of $\operatorname{err}_{\operatorname{CI}}$ for a single
$\mathsf{X}$ and $\boldsymbol{\theta}$ requires inversion of the
covariance matrix
$C_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})$, which typically costs
$\mathcal{O}(n^3)$. An eigendecomposition of
$C_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})$ costs at least this
much. The way to overcome this computational burden is to use families
of covariance kernels, $C_{\boldsymbol{\theta}}$, that match the design,
$\mathsf{X}$, in a way that the eigenvectors and eigenvalues of the
matrix $C_{\boldsymbol{\theta}}(\mathsf{X}, \mathsf{X})$ can be obtained
via fast transforms.

For
[integration lattices](https://global.oup.com/academic/product/lattice-methods-for-multiple-integration-9780198534723?cc=us&lang=en&)
and kernels $C_{\boldsymbol{\theta}}$ such as

$$
C_{\boldsymbol{\theta}}(\boldsymbol{t}, \boldsymbol{x})
=
K_{\boldsymbol{\theta}}
\left(
\boldsymbol{t} - \boldsymbol{x} \bmod \boldsymbol{1}
\right),
$$

where

$$
K_{\boldsymbol{\theta}}(\boldsymbol{x})
=
\prod_{l=1}^d
\left[
1
- (-1)^r \eta B_{2r}(x_l)
\right],
\qquad
\boldsymbol{\theta} = (r,\eta),
\quad
r \in \mathbb{N},
\quad
\eta > 0,
$$

and $B_{2r}$ is an even degree Bernoulli polynomial, the fast transform
corresponds to a fast
[Fourier transform](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/S0025-5718-1965-0178586-1.pdf).
For digital nets, see the blog
[Digital Sequences, the Niederreiter Construction](https://qmcpy.org/2021/06/04/digital-sequences-the-niederreiter-construction/).
For $C_{\boldsymbol{\theta}}$ that are
[digitally-shift invariant](https://repository.iit.edu/islandora/object/islandora%3A1009768),
the fast transform corresponds to a
[fast Walsh transform](https://dl.acm.org/doi/abs/10.1109/TC.1976.1674569).
In both cases, the computational burden attributable to tuning the
hyperparameters and computing the width of the credible interval is a
reasonable $\mathcal{O}(n \log(n))$.

The following code shows how to use a Bayesian stopping criterion for
Keister's example.

```python
import qmcpy as qp

tol = 0.005

integrand = qp.Keister(qp.Lattice(dimension=2, order="RADICAL INVERSE"))
keister_2d_exact = integrand.exact_integ(2)
solution, data = qp.CubBayesLatticeG(
    integrand,
    abs_tol=tol,
    n_init=2**5,
).integrate()
print("Integration error: ", abs(solution - keister_2d_exact) < tol)
```

Listing 1: Example usage of the lattice Bayesian cubature algorithm.

This example can be run in Google Colab without any installation using
this
[notebook](https://github.com/QMCSoftware/QMCSoftware/blob/develop/demos/integration_examples.ipynb).

## References

1. Hickernell, F. J. Blog: Why Add Q to MC?
   [https://qmcpy.org/2020/06/25/why_add_q_to_mc/](https://qmcpy.org/2020/06/25/why_add_q_to_mc/).
   2020.
2. Choi, S.-C. T., Hickernell, F., McCourt, M., & Sorokin, A. QMCPy: A
   quasi-Monte Carlo Python Library.
   [https://qmcsoftware.github.io/QMCSoftware/](https://qmcsoftware.github.io/QMCSoftware/).
   2020.
3. Rathinavel, J., & Hickernell, F. Fast automatic Bayesian cubature
   using lattice sampling. *Statistics and Computing*, 29, 1215-1229
   (2019).
4. Sloan, I. H., & Joe, S. *Lattice Methods for Multiple Integration*.
   Oxford University Press, Oxford (1994).
5. Hickernell, F., & Niederreiter, H. The existence of good extensible
   rank-1 lattices. *Journal of Complexity*, 19, 286-300 (2003).
6. Cooley, J. W., & Tukey, J. W. An algorithm for the machine
   calculation of complex Fourier series. *Mathematics of Computation*,
   19, 297-301 (1965).
7. Ebert, A. Blog: Digital Sequences, the Niederreiter Construction.
   [https://qmcpy.org/2021/06/04/digital-sequences-the-niederreiter-construction/](https://qmcpy.org/2021/06/04/digital-sequences-the-niederreiter-construction/).
   2021.
8. Rathinavel, J. *Fast Automatic Bayesian Cubature Using Matching
   Kernels and Designs*. PhD thesis, Illinois Institute of Technology
   (2019).
9. Fino, B. J., & Algazi, V. R. Unified matrix treatment of the fast
   Walsh-Hadamard transform. *IEEE Transactions on Computers*, C-25,
   1142-1146 (1976).
