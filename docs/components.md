# Components of QMCPy: Quasi-Monte Carlo Software in Python

Monte Carlo (MC) methods approximate the true mean (expectation) $\mu$ of a random variable $g(\boldsymbol{T})$ by the sample mean $\hat{\mu}_n = \frac{1}{n} \sum_{i=0}^{n-1} g(\boldsymbol{T}_i)$ for some samples $\boldsymbol{T}_0,\dots,\boldsymbol{T}_{n-1}$. We call the $d$-dimensional vector random variable $\boldsymbol{T}$ the **true measure** and we call $g$ the **integrand**. As most computer-generated random numbers are uniformly distributed, we use a transform $\boldsymbol{\psi}$ to write $\boldsymbol{T} \sim \boldsymbol{\psi}(\boldsymbol{X})$ where $\boldsymbol{X} \sim \mathcal{U}[0,1]^d$. The resulting Monte Carlo approximation is written in terms of the transformed integrand $f(\boldsymbol{x}) = g(\boldsymbol{\psi}(\boldsymbol{x}))$ as

$$\mu = \mathbb{E}[f(\boldsymbol{X})] = \int_{[0,1]^d} f(\boldsymbol{x}) \mathrm{d} \boldsymbol{x} \approx \int_{[0,1]^d} f(\boldsymbol{x}) \hat{\lambda}_n(\mathrm{d} \boldsymbol{x}) = \frac{1}{n} \sum_{i=0}^{n-1} f(\boldsymbol{x}_i) = \hat{\mu}, \qquad \boldsymbol{X} \sim \mathcal{U}[0,1]^d$$

for some **discrete distribution** $\hat{\lambda}_n$ defined by samples $\boldsymbol{x}_0,\dots,\boldsymbol{x}_{n-1} \in [0,1]^d$ (formally $\hat{\lambda}_n(A)$ measures the proportion of points $(\boldsymbol{x}_i)_{i=0}^{n-1}$ which lie in some set $A$). The *error* of this approximation is

$$E_n = \lvert \mu - \hat{\mu}_n \rvert.$$

Classic **Monte Carlo** methods choose IID (independent and identically distributed) samples $\boldsymbol{x}_0,\dots,\boldsymbol{x}_{n-1} \overset{\mathrm{IID}}{\sim} \mathcal{U}[0,1]^d$ and have error $E_n$ like $\mathcal{O}(n^{-1/2})$. **Quasi-Monte Carlo (QMC)** methods achieve a significantly better error rate of $\mathcal{O}(n^{-1})$ by using low discrepancy (LD) sequences for $(\boldsymbol{x}_i)_{i=0}^{n-1}$ which more evenly fill the unit cube than IID points.

| <img src="https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/refs/heads/gh-pages/assets/points.svg" style="width:100%; height:auto;"> |
|:--|
| The first $32$ points of each sequence are shown as purple stars, the next $32$ points are shown as green triangles, and the $64$ points after that are shown as blue circles. Notice the gaps and clusters of IID points compared to the more uniform coverage of LD sequences. |

Often practitioners would like to run their (Quasi-)Monte Carlo method until the error $E_n$ is below a desired error tolerance $\varepsilon$ and/or until they have expired their sample budget $B$. For example, one may wish to estimate the expected discounted payoff of a financial option to within a tolerance of one penny, $\varepsilon = 0.01$, or until $1$ million option paths have been simulated, $B=10^6$. **Stopping criteria** deploy (Quasi-)Monte Carlo methods under such constraints by utilizing adaptive sampling schemes and efficient error estimation procedures.  

`QMCPy` is organized into the four main components below. Details for each of these classes are available in the linked guides and API docs.

## Discrete Distributions

These generate IID or LD points $\boldsymbol{x}_0,\boldsymbol{x}_1,\dots$. Supported LD sequences include

- **Lattices** with
    - extensible constructions
    - random shifts
- **Digital Nets** in base $b=2$ with
    - extensible constructions
    - digital shifts
    - linear matrix scrambling
    - nested uniform scrambling (also called Owen scrambling)
    - higher order constructions via digital interlacing
- **Halton** point sets with
    - extensible constructions
    - digital shifts
    - permutation scrambling
    - linear matrix scrambling
    - nested uniform scrambling

## True Measures

These define $\boldsymbol{T}$, for which `QMCPy` will automatically choose an appropriate transform $\boldsymbol{\psi}$ so that $\boldsymbol{T} \sim \boldsymbol{\psi}(\boldsymbol{X})$ with $\boldsymbol{X} \sim \mathcal{U}[0,1]^d$. Some popular true measures are

- **Uniform** $\boldsymbol{T} \sim \mathcal{U}[\boldsymbol{l},\boldsymbol{u}]$ with elementwise $\boldsymbol{l} \leq \boldsymbol{u}$ for which $\boldsymbol{\psi}(\boldsymbol{x}) = \boldsymbol{l}+(\boldsymbol{u}-\boldsymbol{l}) \odot \boldsymbol{x}$ with $\odot$ the Hadamard (elementwise) product.
- **Gaussian** $\boldsymbol{T} \sim \mathcal{N}(\boldsymbol{m},\mathsf{\Sigma})$ for which $\boldsymbol{\psi}(\boldsymbol{x}) = \boldsymbol{m}+\mathsf{A}\Phi^{-1}(\boldsymbol{x})$ where $\Phi^{-1}$ is the inverse CDF of the standard Gaussian distribution applied elementwise and the covariance $\mathsf{\Sigma} = \mathsf{A} \mathsf{A}^T$ may be decomposed using either
    - the Cholesky decomposition or
    - the eigendecomposition.
- **Brownian Motion** observed with an initial value $B_0$, drift $\gamma$, and diffusion $\sigma^2$ at times $\boldsymbol{t} := (t_1,\dots,t_d)^T$ satisfying $0 \leq t_1 < t_2 < \dots < t_d$ is a Gaussian with mean and covariance

$$\boldsymbol{m} = B_0 + \gamma \boldsymbol{t}$$

$$\mathsf{\Sigma} = \sigma^2 \left(\min\{t_i,t_{i'}\}\right)_{i,i'=1}^{d}$$

- **Independent Marginals** have $\boldsymbol{T} = (T_1,\dots,T_d)^T$ with $T_1,\dots,T_d$ independent. We support (continuous) marginal distributions from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions).

## Integrands

These define $g$, which `QMCPy` will use to define $f = g \circ \boldsymbol{\psi}$. Some popular integrands are

- **User Defined Integrands**, where the user provides a function handle for $g$
- **Financial Options**, including the *European option*, *Asian option*, and *Barrier option*
- **[`UM-Bridge`](https://um-bridge-benchmarks.readthedocs.io/en/docs/) Functions**. From their docs, `UM-Bridge` is a universal interface that makes any numerical model accessible from any programming language or higher-level software through the use of containerized environments. `UM-Bridge` also enables simulations to scale to supercomputers or the cloud with minimal effort.

## Stopping Criteria

| <img src="https://raw.githubusercontent.com/QMCSoftware/QMCSoftware/refs/heads/gh-pages/assets/stopping_crit.svg" style="width:100%; height:auto;"> |
|:--|
| The cost of IID-Monte Carlo algorithms is $n = \mathcal{O}(1/\varepsilon^2)$ in the number of samples $n$ and error tolerance $\varepsilon$ while Quasi-Monte Carlo algorithms only cost around $n=\mathcal{O}(1/\varepsilon)$. Both IID-Monte Carlo and Quasi-Monte Carlo stopping criteria consistently determine approximations which meet the desired error tolerance. |

These deploy (Quasi-)Monte Carlo methods under error tolerance and budgetary constraints by utilizing adaptive sampling schemes and efficient error estimation procedures. Common stopping criteria include

- **Quasi-Monte Carlo** via tracking the decay of coefficients in an orthogonal basis expansion. These methods are *guaranteed* for cones of functions whose coefficients decay in a regular manner.  Efficient procedures exist to estimate coefficients when
    - pairing lattices with the Fourier expansion or
    - pairing digital nets with the Walsh expansion.
- **Quasi-Monte Carlo** via efficient Bayesian cubature methods which assume $f$ is a draw from a Gaussian process so the posterior expectation has an analytic expression. While classic Bayesian cubature would require $\mathcal{O}(n^2)$ storage and $\mathcal{O}(n^3)$ computations, when matching certain LD sequences to special kernels the Gram matrices become nicely structured to permit Bayesian cubature with only $\mathcal{O}(n)$ storage and $\mathcal{O}(n \log n)$ computations. Specifically,
    - pairing lattices with shift-invariant kernels gives circulant Gram matrices which are diagonalizable by the [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform), and
    - pairing digital nets with digitally-shift-invariant kernels gives Gram matrices which are diagonalizable by the [Fast Walsh-Hadamard Transform (FWHT)](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform).
- **Quasi-Monte Carlo** via multiple independent randomizations of an LD point set and Student's $t$ confidence intervals.
- **IID Monte Carlo** via a two step procedure using the Central Limit Theorem (CLT). Error estimates are *not guaranteed* as CLT is asymptotic in $n$ and the variance must be estimated.
- **IID Monte Carlo** via a two step procedure using Berry-Esseen inequalities to account for finite sample sizes. Error estimates are *guaranteed* for functions with bounded Kurtosis.
- **Multilevel IID Monte Carlo and Quasi-Monte Carlo** which more efficiently integrate expensive functions by exploiting a telescoping sum over lower fidelity models.
