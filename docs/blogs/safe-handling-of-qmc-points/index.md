<!--
Source WordPress URL: https://qmcpy.org/2020/08/31/safe-handling-of-qmc-points/
Original metadata: Posted by Art B. Owen; August 31, 2020.
Image handling: no content images were present in the original post.
-->

# Safe Handling of QMC Points

--8<-- "snippets/blog-authors/safe-handling-of-qmc-points.md"

August 31, 2020

This post explains why QMC points should not be handled like IID samples and warns against unsafe practices such as skipping, thinning, and arbitrary sample sizes.

A Quasi-Monte Carlo construction of $n$ points in $d$ dimensions may
look like IID points, but they must be used with a bit more care.
Because QMC can give errors that are $o(1/n)$ as $n \to \infty$,
changing or ignoring even one point can change the estimate by an amount
much larger than the error would have been and worsen the convergence
rate. As a result, certain practices that fit quite naturally and
intuitively with MC points are very detrimental to QMC performance.
Operations like burn-in, thinning, and even using a round number sample
size, like a power of ten, can degrade QMC effectiveness or even make it
converge to the wrong answer. The safe way to use QMC points is to take
all $n$ points produced, after applying a randomization to avoid
singularities and to support uncertainty quantification.

## Introduction

This note arose from a discussion of quasi-Monte Carlo (QMC) and
randomized quasi-Monte Carlo (RQMC) software during and following the
plenary tutorial at MCQMC 2020 by Fred Hickernell. Common ways of
handling IID points can fail to work for (R)QMC points. A longer
discussion of this point is available at
[arXiv:2008.08051](https://arxiv.org/abs/2008.08051).

QMC sampling methods provide a set of $n$ points in $[0,1]^d$ that we
can use instead of a sample of $\mathcal{U}[0,1]^d$ points. We can apply
transformations to them to simulate non-uniform distributions and
domains other than the unit cube. Then the resulting points can be used
to estimate an expectation or just to explore the input to a function.

If the points are
$\boldsymbol{x}_1, \dots, \boldsymbol{x}_n \in [0,1]^d$, we may
estimate
$\mu = \int_{[0,1]^d} f(\boldsymbol{x}) \, \mathrm{d}\boldsymbol{x}$
by

$$
\hat{\mu}
= \frac{1}{n} \sum_{i=1}^n f(\boldsymbol{x}_i),
$$

just as we would have done with
$\boldsymbol{x}_i \overset{\text{iid}}{\sim} \mathcal{U}[0,1]^d$. The
function $f(\cdot)$ subsumes transformations as well as the integrand of
interest in the transformed space.

Plain QMC points are deterministic. Randomizing them in one of several
possible ways makes them individually uniformly distributed while
preserving the low discrepancy structure that makes them valuable for
integration. The resulting RQMC methods allow uncertainty quantification
via replication. If it is important to be accurate, then it must also be
important to know that you were accurate and to show that you were
accurate. A plain $t$-test based confidence interval, or better yet a
bootstrap-$t$ confidence interval for $\mu$, then lets one estimate
accuracy.

Bootstrap-$t$ works very well even with a modest number of replicates.
We might want a modest number $R$ of replicates because the root mean
squared error (RMSE) decreases proportionally to $1/\sqrt{R}$ as the
number of replicates increases but often faster than $1/\sqrt{n}$ as
the number of sample points increases. The work involved is proportional
to $nR$.

A second reason to randomize is that QMC points are really designed for
Riemann integrable functions. Those are necessarily bounded. If
$\hat{\mu} \to \mu$ whenever the star discrepancy of
$\boldsymbol{x}_1, \dots, \boldsymbol{x}_n$ converges to zero, then it
must hold that $f$ is Riemann integrable. That is, if $f$ is not Riemann
integrable, as for instance it would be if it were unbounded, then there
are sequences of inputs with vanishing star discrepancy for which
$\hat{\mu} - \mu$ does not converge to zero.

It is safer to randomize. Nested uniform scrambles ensure that
$\hat{\mu} \to \mu$ with probability one under the weak condition that
$f \in L^{1+\epsilon}[0,1]^d$ for some $\epsilon > 0$. That is,

$$
\int_{[0,1]^d}
\left|f(\boldsymbol{x})\right|^{1+\epsilon}
\, \mathrm{d}\boldsymbol{x}
< \infty,
$$

and $f$ is measurable.

Because (R)QMC points look so similar to plain IID points, many users
and software implementations handle (R)QMC points in inefficient or even
unsafe ways that would be no problem for IID points.

## Sample Sizes

(R)QMC points are usually constructed as a finite sequence of points for
a specific sample size $n$, such as $n=2^m$ or $n=p$ for a large prime
number $p$. If one uses only a round number such as $1000$ of them, then
the result will ordinarily be much less effective than using them all
and can possibly even fail to sample a portion of $[0,1]^d$. Those
$1000$ points might easily be less effective than using a smaller
sequence of $512$ points. As for antibiotics, one should use the whole
sequence.

## Skipping or Burn-In

For IID points, we do as well with
$\boldsymbol{x}_{B+1}, \dots, \boldsymbol{x}_{B+n}$ for any $B \ge 0$.
Taking $B > 0$ is a kind of burn-in that actually has an advantage in
Markov chain Monte Carlo, where the points may only approach their
desired distribution. For RQMC points, skipping even one observation can
make the rate of convergence worse. In the case of scrambled nets,
taking $B=1$ can turn the RMSE from approximately
$\mathcal{O}(n^{-3/2})$ to approximately $\mathcal{O}(n^{-1})$.

The reason that people often skip the first point is that this first
point is often equal to $(0,0,\dots,0)$. Such a point is then
problematic when $f$ maps $[0,1]^d$ onto $\mathbb{R}^d$, as it would
when using a transformation to induce a Gaussian distribution before
evaluating the quantity of interest. The point at the origin can map to
an infinite point or even result in `not a number`.

If one uses RQMC, then that first point ends up with the
$\mathcal{U}[0,1]^d$ distribution, as do all the others. That avoids the
problem of singularities at least mathematically. One might still hit a
singularity in a floating point representation if one is extremely
unfortunate. That possibility is also there with QMC, and plain QMC does
not have the same assurance of avoiding singularities that RQMC has.

## Thinning

In MCMC one often takes every $k$-th point for reasons of storage or
computational efficiency. In IID sampling taking every $k$-th point
would be statistically equivalent to taking an equal number of
consecutive points. If we use $\boldsymbol{x}_{ki}$ for integer $k > 1$
and $i=1, \dots, n$ in (R)QMC, the result can be disastrously bad. For
instance, the van der Corput sequence in $[0,1]$ alternates between
values in $[0,1/2)$ and values in $[1/2,1)$. Taking every second point
would ignore half of the domain. The first component of a Sobol'
sequence is ordinarily the van der Corput sequence.

Thinning (R)QMC points can be extremely dangerous. It should not be done
without some very careful mathematical explanation of why it might be ok
in some special setting.

## van der Corput Sequences

These are for $d=1$, so $x_i \in [0,1]$. Any consecutive $2^m$ points
of the van der Corput sequence are a digital net and hence have some
good discrepancy properties. The same holds for generalizations of van
der Corput to bases $b > 2$. There any $b^m$ consecutive points are a
digital net. So van der Corput points are an exception. If we use
burn-in, we still get a digital net and thus still get low discrepancy.
We should take care of the chosen sample size, preferring $n$ to be a
power of $b$. If the powers of $b$ are too far apart for our purposes,
then an integer multiple of a power of $b$ is next best. That only makes
a difference when $b > 2$.

We should not thin van der Corput sequences.

## Halton Sequences

Halton sequences are somewhat robust to burn-in and using round numbers.
Each of the $d$ component variables of a Halton sequence is a van der
Corput sequence in a different base. Usually the first $d$ prime numbers
are used.

For modestly large $d$, the special values of $n$ are so large and so
far apart that we can consider that there simply are no specially good
sample sizes. Think of making $n$ divisible by a power of the product of
the first $d$ prime numbers. Even the first such value may be too large
to use. When no feasible sample sizes are very good, then maybe there is
no particular harm from using a power of ten.

Halton sequences start at the origin, which is problematic as described
above. We can easily skip that point in Halton sequences because there
are no especially good ranges. It may even be advantageous to use a very
large burn-in for the Halton sequence because the initial points for
large $d$ have unpleasant striping artifacts.

It is however safer to randomize the Halton sequence. Scrambling the
Halton sequence counters those striping artifacts more surely than a
burn-in would. It also moves the point at the origin to a uniformly
distributed random point. This is another instance where RQMC is safer
and more effective than plain QMC.
