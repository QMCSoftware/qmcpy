# SciPyWrapper dependence and custom distributions demo

This folder contains the demo notebook and helper files for my project on extending `qmcpy.true_measure.SciPyWrapper` to handle

1. Joint dependent distributions, and  
2. User defined distributions that follow a SciPy like interface, with basic sanity checks.

The goal is to show that we can move beyond independent marginals from `scipy.stats` and still plug everything into the usual QMCPy workflow.

---

## Folder structure

From the top level of the QMCSoftware repo, the pieces for this project are:

- `qmcpy/true_measure/scipy_wrapper.py`  
  Updated `SciPyWrapper` with:
  - support for a joint distribution object with a `transform` and optional `logpdf` or `pdf`,
  - support for user defined univariate distributions that look like frozen SciPy objects,
  - light weight validation that warns when a custom distribution looks suspicious.

- `demos/scipywrapper_dependence_custom/scipywrapper_demo.ipynb`  
  Jupyter notebook that produces all figures and printed statistics used in the blog:
  - Example 1: independent vs dependent normals,
  - Example 2: zero inflated exponential plus uniform,
  - Example 3: acceptance rejection target with iid MC and with QMC,
  - Example 4: custom triangular marginal,
  - Example 5: intentionally broken custom distribution to show the warnings,
  - Example 6: multivariate Student t (joint, dependent) using SciPy’s multivariate t.

- `demos/scipywrapper_dependence_custom/README.md`  
  This file.

- [SciPyWrapper article on the QMCSoftware Blog](https://qmcsoftware.org/blogs/scipywrapper/) Blog describing the project at a higher level.

- `test/test_scipy_wrapper_custom.py`  
  Unit tests that:
  - check joint sampling and correlations for the multivariate normal example,
  - verify the zero inflated exponential marginal mass at zero,
  - exercise the validation logic for user defined distributions.

---

## How to run the demo

These commands are all intended to be run from the root of the QMCSoftware repo, inside a Python environment where you plan to develop.

1. Install QMCPy from source in editable mode

   ```bash
   pip install -e .
   ```
This makes sure Python will use your updated `SciPyWrapper` instead of any pip installed version.

2. Run the custom tests

    ```bash
    pytest test/test_scipy_wrapper_custom.py
    ```
You should see all tests pass. A warning about the zero inflated joint distribution not having logpdf is expected. That example is only used for sampling and visualisation so weights are intentionally treated as 1.

3. Open the notebook

    ```bash
    cd demos/scipywrapper_dependence_custom
    jupyter notebook scipywrapper_demo.ipynb
    ```
Run all cells from top to bottom. You will see:
- a side by side scatter plot for independent normals and a correlated multivariate normal,
- histograms and scatter plots for the zero inflated exponential plus uniform joint,
- acceptance rejection clouds for iid MC and QMC on the same triangular target,
- a histogram of the custom triangular marginal overlaid with its analytic pdf,
- console output with warnings for the intentionally broken custom distribution,
- a marginal histogram and joint scatter plot for a dependent multivariate Student t, along with printed checks for correlation and the theoretical covariance relationship.

These are the same figures that the blog refers to.

---

## How to use the new features in your own code

Here is the minimal idea for each feature.

### 1. Joint dependent distributions

Wrap any joint distribution that can map `[0, 1]^d` to your target space:

```python
joint = MyJointDistribution(dim=2, some_param=...)
sampler = DigitalNetB2(2, seed=17)
tm = SciPyWrapper(sampler, joint)
x = tm(1000)       # shape (1000, 2)
```
`MyJointDistribution` needs:
- a `dim` attribute, and
- a `transform(u)` method that accepts an array in `[0, 1]^dim` and returns samples.

If you also define `logpdf(x)` or `pdf(x)`, SciPyWrapper will use that for weights when required.

### 2. User defined univariate distributions

Any univariate distribution with a SciPy-like interface can be used as a marginal:

```python
tri = TriangularUserDistribution(c=0.3, loc=-1.0, scale=2.0)
sampler = DigitalNetB2(1, seed=31)
tm = SciPyWrapper(sampler, tri)
x = tm(4096)        # samples from the triangular distribution
```

The wrapper will run a quick sanity check on your `ppf` and `pdf`. If it sees that the CDF is not increasing or the density does not look normalised, it will raise a `UserWarning` instead of failing silently.

---

## Notes
- The behavior for the original use case (lists of SciPy univariate frozen distributions) is unchanged.
- Dependence is introduced only when the user passes a joint distribution object.
- Validation is kept intentionally light so that advanced users can still experiment, but beginners are warned when their custom distributions are clearly broken.
