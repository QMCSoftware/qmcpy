from typing import cast, Sequence, Mapping

from .abstract_true_measure import AbstractTrueMeasure, _clip_unit_interval
from ..util import DimensionError, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..discrete_distribution import DigitalNetB2

import numpy as np
import warnings
from pytensor.tensor.utils import get_static_shape_from_size_variables

import pymc
import pytensor
import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.basic import Variable, Constant
from pytensor.graph.rewriting.basic import GraphRewriter, NodeRewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableOp, _icdf
from pymc.logprob.basic import _find_unallowed_rvs_in_graph
from pymc.logprob.utils import get_related_valued_nodes
from pymc.pytensorf import PointFunc, replace_vars_in_graphs, Function, toposort_replace
from pymc.model.fgraph import (
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
)
from pymc import Model

if not (int(pymc.__version__.split('.')[0]) == 6): # backends matter here
    raise AssertionError(f"PyMC{pymc.__version__} unsupported. PyMCWrapper supports running models in PyMC6")
###### based on `observe` from pymc model/transform/conditioning

def hide( ## tongue in cheek, opposite of `observe`
    model: Model,
) -> Model:
    """Convert ALL observed RVs to free RVs.

    Parameters:
        model (Model): PyMC Model

    Returns:
        new_model (Model): PyMC model
            A distinct PyMC model with all variables free.

    Examples:
        .. code-block:: python

        import pymc as pm

        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal("y", x)
            z = pm.Normal("z", y, observed=1.0)

        m_new = pm.hide(m)

    """
    fgraph, memo = fgraph_from_model(model)

    replacements = {}
    for rv in model.basic_RVs:
        # NOTE: this seems to convert rv -> whatever is actually in the fgraph??
        model_var = memo[rv] # was like this in `observe`, not sure of exact meaning...

        # `model_free_rv` had a breaking change on version 6.1.0
        # https://github.com/pymc-devs/pymc/pull/8340
        # call the correct version accoring to installed PyMC version
        from packaging.version import Version
        PYMC_VERSION_INTERNAL_CHANGE = Version("6.1.0")
        if Version(pymc.__version__) < PYMC_VERSION_INTERNAL_CHANGE:
            free_rv = model_free_rv(rv, rv.type(), model.rvs_to_transforms[rv], *model.named_vars_to_dims.get(rv.name, ()))
        else:
            free_rv = model_free_rv(rv, rv.type(), model.rvs_to_transforms[rv], rv.name, *model.named_vars_to_dims.get(rv.name, ()))
        replacements[model_var] = free_rv

    fgraph.replace_all(tuple(replacements.items()), import_missing=True)

    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    # toposort the model to undo any possible reordering that took place
    fgraph, memo = fgraph_from_model(model)
    toposort_replace(fgraph, {})
    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return model

###### based on `conditional_logp` and `transformed_conditional_logp` from pymc logprob/basic.py
def conditional_icdf(
    rv_values: dict[Variable, Variable],
    warn_rvs: bool=True,
    ir_rewriter: GraphRewriter | None = None,
    extra_rewrites: GraphRewriter | NodeRewriter | None = None,
    **kwargs: Mapping,
) -> dict[Variable, Variable]:
    r"""Create a map between variables and conditional icdfs

    The `rv_values` dictionary specifies a joint probability graph defined by
    pairs of random variables and respective measure-space input parameters

    For example, consider the following

    .. code-block:: python

        import pytensor.tensor as pt

        sigma2_rv = pt.random.invgamma(0.5, 0.5)
        Y_rv = pt.random.normal(0, pt.sqrt(sigma2_rv))

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \sigma^2)

    If we create a value variable for ``u_rv``, i.e. ``u_vv = pt.scalar("u")``,
    the graph of ``conditional_icdf({Y_rv: u_vv})`` is equivalent to the
    conditional icdf :math:`p^{-1}_{Y \mid \sigma^2}(u \mid s^2)`, with a stochastic
    ``sigma2_rv``.

    If we specify a value variable for ``sigma2_rv``, i.e.
    ``u2_vv = pt.scalar("u2")``, then ``conditional_icdf({Y_rv: u_vv, sigma2_rv: u2_vv})``
    yields the conditional icdf of the two variables.

    Parameters:
        rv_values (dict[Variable, Variable]):
            A ``dict`` of variables that maps stochastic elements
            (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing conditional probability
            values in a joint icdf
        warn_rvs (bool): default True
            When ``True``, issue a warning when a `RandomVariable` is found in
            the logp graph and doesn't have a corresponding value variable specified in
            `rv_values`.
        ir_rewriter (GraphRewriter | None):
            Rewriter that produces the intermediate representation of Measurable Variables.
        extra_rewrites (GraphRewriter | NodeRewriter | None):
            Extra rewrites to be applied (e.g. reparameterizations, transforms,
            etc.),
        **kwargs (Mapping): other compilation related arguments

    Returns:
        values_to_icdf (dict[Variable, Variable]):
            A ``dict`` that maps each value variable to the conditional icdf term derived
            from the respective `RandomVariable`.

    """
    fgraph: FunctionGraph = pymc.logprob.rewriting.construct_ir_fgraph(rv_values, ir_rewriter=ir_rewriter)

    if extra_rewrites is not None:
        extra_rewrites.rewrite(fgraph)

    # Walk the graph from its inputs to its outputs and construct the
    # icdf
    replacements = {}

    # To avoid cloning the value variables (or ancestors of value variables),
    # we map them to themselves in the `replacements` `dict`
    # (i.e. entries already existing in `replacements` aren't cloned)
    replacements.update(
        {v: v for v in ancestors(rv_values.values()) if not isinstance(v, Constant)}
    )

    # Walk the graph from its inputs to its outputs and construct the
    # icdf
    values_to_icdf = {}
    original_values = tuple(rv_values.values())

    for node in fgraph.toposort():
        if not isinstance(node.op, MeasurableOp):
            continue

        valued_nodes = get_related_valued_nodes(fgraph, node)

        if not valued_nodes:
            continue

        node_rvs = [valued_var.inputs[0] for valued_var in valued_nodes]
        node_values = [valued_var.inputs[1] for valued_var in valued_nodes]
        node_output_idxs = [
            fgraph.outputs.index(valued_var.outputs[0]) for valued_var in valued_nodes
        ]

        # Replace `RandomVariable`s in the inputs with value variables.
        # Also, store the results in the `replacements` map for the nodes that follow.
        for node_rv, node_value in zip(node_rvs, node_values):
            replacements[node_rv] = node_value

        remapped_vars = replace_vars_in_graphs(
            graphs=node_values + list(node.inputs),
            replacements=replacements,
        )
        node_values = remapped_vars[: len(node_values)]
        node_inputs = remapped_vars[len(node_values) :]

        node_icdfs = [_icdf(
            node.op,
            node_value,
            *node_inputs,
            **kwargs,
        ) for node_value in node_values]

        # add the result of the icdf (sample value) to the replacements
        # for later (dependent) nodes
        for node_rv, node_icdf in zip(node_rvs, node_icdfs):
            replacements[node_rv] = node_icdf

        for node_output_idx, node_value, node_icdf in zip(
            node_output_idxs, node_values, node_icdfs
        ):
            original_value = original_values[node_output_idx]

            if original_value.name:
                node_icdf.name = f"{original_value.name}_icdf"

            if original_value in values_to_icdf:
                raise ValueError(
                    f"More than one icdf term was assigned to the value var {original_value}"
                )

            values_to_icdf[original_value] = node_icdf

    missing_value_terms = set(original_values) - set(values_to_icdf)
    if missing_value_terms:
        raise RuntimeError(
            f"The icdf terms of the following value variables could not be derived: {missing_value_terms}"
        )

    # Ensure same order as input
    icdfs = pymc.logprob.rewriting.cleanup_ir(tuple(values_to_icdf[v] for v in original_values))

    if warn_rvs:
        rvs_in_icdf_expressions = _find_unallowed_rvs_in_graph(icdfs)
        if rvs_in_icdf_expressions:
            warnings.warn(
                f"Random variables detected in the icdf graph: {rvs_in_icdf_expressions}.\n"
                "This can happen when not all random variables have a corresponding value variable.",
                UserWarning,
            )

    return dict(zip(original_values, icdfs))

def transformed_conditional_icdf(
    rvs: Sequence[Variable],
    *,
    rvs_to_values: dict[Variable, Variable],
    **kwargs,
) -> list[Variable]:
    """Thin wrapper around conditional_, which creates a value transform rewrite.

    This helper will only return the subset of logprob terms corresponding to `rvs`.
    All rvs_to_values and rvs_to_transforms mappings are required.
    """
    kwargs.setdefault("warn_rvs", False)
    temp_icdf_terms = conditional_icdf(
        rvs_to_values,
        **kwargs,
    )

    # The function returns the icdf for every single value term we provided to it.
    # This includes the extra values we plugged in above, so we filter those we
    # actually wanted in the same order they were given in.
    icdf_terms = {}
    for rv in rvs:
        value_var = rvs_to_values[rv]
        icdf_terms[value_var] = temp_icdf_terms[value_var]

    icdf_terms_list = list(icdf_terms.values())

    rvs_in_icdf_expressions = _find_unallowed_rvs_in_graph(icdf_terms_list)
    if rvs_in_icdf_expressions:
        raise ValueError(
            f"Random variables detected in the icdf graph: {rvs_in_icdf_expressions}.\n"
            "This can happen when mixing variables from different models, "
            "or when CustomDist logp or Interval transform functions reference nonlocal variables."
        )

    return icdf_terms_list

######## based on functions from pymc model/core.py
def icdf(
        model: Model,
        vars: Variable | Sequence[Variable] | None = None,
    ) -> Variable | list[Variable]:
    """Elemwise log-probability of the model.

    On frozen models the returned graph is memoized (keyed on the arguments), so
    repeated calls return the same object and ``compile_fn`` can reuse a compiled
    function. Do not mutate it in place.

    Parameters:
        model (Model): pymc Model
        vars (Variable | Sequence[Variable] | None): list of random variables, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables in model.

    Returns:
        icdf graph(s)
    """
    varlist: list[TensorVariable]
    if vars is None:
        varlist = model.free_RVs + model.observed_RVs
    elif not isinstance(vars, list | tuple):
        varlist = [vars]
    else:
        varlist = cast(list[TensorVariable], vars)

    # We need to remember rv original order so that we can merge them together in the same order at the end
    rvs = []
    rv_order = []
    for i, var in enumerate(varlist):
        rv = model.values_to_rvs.get(var, var)
        if rv in model.basic_RVs:
            rvs.append(rv)
            rv_order.append(i)
        else:
            raise ValueError(
                f"Requested variable {var} not found among the model variables"
            )

    rv_icdfs: list[TensorVariable] = []
    if rvs:
        rv_icdfs = transformed_conditional_icdf(
            rvs=rvs,
            rvs_to_values=model.rvs_to_values,
        )
        if not (isinstance(rv_icdfs, list)):
            raise AssertionError

    icdf_factors = [None] * len(varlist)
    for icdf_order, _icdf in zip(rv_order, rv_icdfs):
        icdf_factors[icdf_order] = _icdf.flatten()

    return pt.concatenate(icdf_factors)

def compile_icdf(
        model: Model,
        vars: Variable | Sequence[Variable] | None = None,
        point_fn: bool = False,
        **compile_kwargs: Mapping,
    ) -> PointFunc | Function:
    """Compiled log probability density function.

    The function expects as input a dictionary with the same structure as self.initial_point()

    Parameters:
        model (Model): pymc Model
        vars (Variable | Sequence[Variable] | None): list of random variables or potential terms, optional
            Compute the gradient with respect to those variables. If None, use all
            free and observed random variables in model.
        point_fn (bool): controls return type PointFun or Function
        **compile_kwargs (Mapping): other compilation related options

    Returns:
        icdf (PointFunc | Function): compiled icdf function
    """
    compile_kwargs.setdefault("on_unused_input", "ignore")
    icdf_graph = icdf(model, vars=vars)
    # use `vectorize_graph` to convert the input shapes to (None,*old_shape) to automatically vectorize the calculations
    var_shapes = [pt.utils.get_static_shape_from_size_variables([x.size]) for x in model.value_vars]
    # if right shape is None, it indicates already a batch shape, so delete it
    var_shapes = [shape[:-1] if shape[-1] is None else shape for shape in var_shapes]
    replacements = {var: pt.TensorType(dtype="float64", shape=(None,*size))(var.name) for (var, size) in
                    zip(model.value_vars, var_shapes)}
    icdf_graph = pytensor.graph.vectorize_graph(icdf_graph, replacements)
    inputs = list(replacements.values())
    return model.compile_fn(
        inputs=inputs,
        outs=icdf_graph,
        point_fn=point_fn,
        **compile_kwargs,
    )

def compile_vectorized_logp(
        model: Model,
        vars: Variable | Sequence[Variable] | None = None,
        jacobian: bool = True,
        sum: bool = True,
        **compile_kwargs: Mapping,
    ) -> PointFunc | Function:
        """Compiled log probability density function.

        Parameters:
            model (Model): PyMC Model
            vars (Variable | Sequence[Variable] | None): list of random variables or potential terms, optional
                Compute the gradient with respect to those variables. If None, use all
                free and observed random variables, as well as potential terms in model.
            jacobian (bool):
                Whether to include jacobian terms in logprob graph. Defaults to True.
            sum (bool): 
                Whether to sum all logp terms or return elemwise logp for each variable.
                Defaults to True.
            **compile_kwargs (Mapping): compilation related options

        Returns:
            logp (PointFunc | Function): compiled vectorized logp function
        """
        pytensor.config.on_unused_input = 'warn'
        compile_kwargs.setdefault("on_unused_input", "ignore")
        logp_graph = model.logp(vars=model.value_vars, jacobian=jacobian, sum=sum)
        # use `vectorize_graph` to convert the input shapes to (None,*old_shape) to automatically vectorize the calculations
        var_shapes = [pt.utils.get_static_shape_from_size_variables([x.size]) for x in model.value_vars]
        # if right shape is None, it indicates already a batch shape, so delete it
        var_shapes = [shape[:-1] if shape[-1] is None else shape for shape in var_shapes]
        replacements = {var: pt.TensorType(dtype="float64", shape=(None,*size))(var.name) for (var, size) in
                    zip(model.value_vars, var_shapes)}
        logp_graph = pytensor.graph.vectorize_graph(logp_graph, replacements)
        inputs = list(replacements.values())
        return model.compile_fn(
            inputs=inputs,
            outs=logp_graph,
            **compile_kwargs,
        )

############# wrapper

# def _detect_issues(icdf_f, dims, split_indices, n_grid=3):
#     issues = []

#     eps = np.finfo(float).eps
#     us = np.meshgrid(*[np.linspace(eps, 1.0 - eps, n_grid) for _ in range(dims)], indexing='ij')
#     us = [u.ravel() for u in us]
#     us = np.stack(us, axis=-1)
#     us = np.split(us, split_indices, axis=-1)
#     us = [_x.ravel() for _x in us]
#     if np.prod(us[-1].shape) == 0:
#         us = us[:-1]
#     try:
#         x = icdf_f(*us)
#     except Exception as e:
#         return [f"icdf raised {type(e).__name__}: {e}"]

#     if np.any(~np.isfinite(x)):
#         issues.append("ppf() returned non-finite values")

#     # TODO: can we generalize this for dims>1?
#     # dx = np.diff(x)
#     # if np.any(dx < -1e-12):
#     #     issues.append("ppf() is not nondecreasing (looks non-monotone)")

#     # Light-touch pdf/logpdf checks on quantiles (works for bounded/unbounded support)
#     # uq = np.linspace(0.01, 0.99, 9)
#     # try:
#     #     xq = np.asarray(icdf_f(uq), dtype=float)
#     # except Exception:
#     #     xq = None

#     # if xq is not None and hasattr(dist, "pdf"):
#     #     try:
#     #         p = np.asarray(dist.pdf(xq), dtype=float)
#     #         if np.any(~np.isfinite(p)):
#     #             issues.append("pdf() returned non-finite values on quantiles")
#     #         if np.any(p < -1e-12):
#     #             issues.append("pdf() returned negative values on quantiles")
#     #     except Exception as e:
#     #         issues.append(f"pdf() raised {type(e).__name__}: {e}")

#     # if xq is not None and hasattr(dist, "logpdf"):
#     #     try:
#     #         lp = np.asarray(dist.logpdf(xq), dtype=float)
#     #         if np.any(~np.isfinite(lp)):
#     #             issues.append("logpdf() returned non-finite values on quantiles")
#     #     except Exception as e:
#     #         issues.append(f"logpdf() raised {type(e).__name__}: {e}")

#     return issues

class PyMCWrapper(AbstractTrueMeasure):
    r"""True measure that wraps PyMC Model (i.e. distribution).

    PyMC is based on pytensor, which is a "symbolic" computation paradigm.
    A `Model` is a (collection of) pytensor "graphs" for each random variable, that are later compiled
    into c/c++/numba/jax functions called from python. Some scipy code is also used.

    PyMC focuses on MCMC methods and compiles the model to a `logpdf` function and its derivatives for use MCMC.

    This class compiles:
     - a transform based on the icdf for the distributions,
     - a vectorized version of the PyMC logpdf - without any potentials or variable transforms 

    Multivariate distributions are not supported currently due to missing icdf for these in PyTensor.
    The suggested approach to construct these is to transform a set of conditionaly independent
    distributions. See example below.

    Limitations:
     - The samples correspond strictly to the random variables, not to any transformation of them or any Deterministics in the model.
     - Multivariate distributions are not supported (see workaround below).
     - Only sampling from the prior distribution is supported. Posterior sampling takes extra steps shown in the demo notebook.
     - Discrete and some other distributions are currently not supported as their icdf is not currently defined by PyMC.

    Examples:
        A simple single distribution model and basic usage:

        >>> import pymc, qmcpy
        >>> import numpy as np
        >>> import pytensor.tensor as pt
        >>> from qmcpy import PyMCWrapper, DigitalNetB2
        >>> with pymc.Model() as model:
        ...    x = pymc.Normal("x", mu=0.0, sigma=1.0)

        >>> tm_normal = PyMCWrapper(DigitalNetB2(1, seed=7), model)
        >>> samples = tm_normal(4)
        >>> np.round(samples, 7)
        array([[-0.337347 ],
               [ 0.8774886],
               [-1.271268 ],
               [ 0.1124088]])

        With independent replications:

        >>> tm_rep = PyMCWrapper(DigitalNetB2(1, seed=7, replications=2), model)
        >>> samples = tm_rep(4)
        >>> np.round(samples, 7)
        array([[[ 0.5876714],
                [-0.9803547],
                [ 2.2191739],
                [-0.1774778]],
        <BLANKLINE>
               [[ 1.3719147],
                [-0.177269 ],
                [ 0.1478348],
                [-1.8202058]]])

        T distribution specified by the hierarchical InverseGamma-Normal construnction:

        >>> with pymc.Model() as model:
        ...     s = pymc.InverseGamma("sigma", alpha=10.0)
        ...     x = pymc.Normal("x", mu=0.0, sigma=pt.sqrt(s))
        >>> tm = PyMCWrapper(DigitalNetB2(2, seed=7), model)
        >>> x = tm(2)
        >>> np.round(x, 7)
        array([[ 0.1256344,  0.4862743],
               [ 0.0767114, -0.0490979]])
        >>> x.shape
        (2, 2)

        With replications:

        >>> tm = PyMCWrapper(DigitalNetB2(2, seed=7,replications=3), model)
        >>> x = tm(4)
        >>> x.shape
        (3, 4, 2)
        >>> np.round(x, 7)
        array([[[ 0.0836618, -0.2623627],
                [ 0.1207955,  0.144963 ],
                [ 0.1018947,  0.2665866],
                [ 0.1664003, -0.2195593]],
        <BLANKLINE>
               [[ 0.1290611, -0.0462615],
                [ 0.0977059,  0.0285278],
                [ 0.1331088,  0.2494873],
                [ 0.0682587, -0.1911961]],
        <BLANKLINE>
               [[ 0.1491155,  0.0053208],
                [ 0.0625432, -0.1061282],
                [ 0.1125356, -0.4990805],
                [ 0.0872658,  0.2179012]]])

        Multivariate normal not directly supported:

        >>> with pymc.Model() as model:
        ...     s = pymc.MvNormal("x", mu=np.zeros(2), cov=np.array([[1, 0.5], [0.5, 1]]))
        >>> # model created normaly, but:
        >>> try:
        ...     tm_joint = PyMCWrapper(DigitalNetB2(2, seed=7), model) # raises exception
        ... except NotImplementedError as err:
        ...     print(f"caught {str(err)[:70]}...")
        caught Inverse CDF method not implemented for MvNormalRV{name='multivariate_n...

        Multivariate normal as transformed i.i.d:

        >>> with pymc.Model() as model:
        ...    cov = np.array([[1, 0.5], [0.5, 1]])
        ...    mu = np.array([-0.2, 0.2])
        ...    x = pymc.Normal("x_diag", mu=0.0, sigma=np.ones_like(np.diag(cov)))
        ...    mv_x = pymc.Deterministic("x", x @ np.linalg.cholesky(cov).T + mu) # can be used in other distributions, but not directly sampled
        >>> tm_joint_iid = PyMCWrapper(DigitalNetB2(2, seed=7), model)
        >>> samples = tm_joint_iid(2)
        >>> # equivalent to sampling from mv_x
        >>> mv_normal_samples = samples @ np.linalg.cholesky(cov).T + mu
        >>> mv_normal_samples
        array([[ 0.3876714 ,  1.68194866],
               [-1.18035468, -0.44369678]])
    """

    def __init__(self, sampler: AbstractDiscreteDistribution, model: Model) -> None:
        r"""Wrap a PyMC Model as a QMCPy true measure.

        Args:
            sampler (AbstractDiscreteDistribution): Low discrepancy or iid
                sampler in dimension d, living on $[0,1)^d$.
            model (Model): A single PyMC model.
        """
        self.domain = np.tile([0, 1], (sampler.d, 1))

        if not isinstance(sampler, AbstractDiscreteDistribution):
            if not (
                hasattr(sampler, "d")
                and hasattr(sampler, "gen_samples")
                and callable(sampler.gen_samples)
            ):
                raise ParameterError(
                    "PyMCWrapper requires sampler to be an AbstractDiscreteDistribution."
                )
        if not isinstance(model, pymc.Model):
            raise ParameterError(
                "model parameter must be a pymc.Model object."
            )
        self._parse_sampler(sampler)
        # Remember what the user originally passed in so that _spawn can reuse it.
        self.model = hide(model)
        self.icdf = compile_icdf(self.model)
        self.logpdf = compile_vectorized_logp(self.model, point_fn=False)
        # determine shapes of random variables and prepare their indices in a flat vector
        static_shapes = [pt.utils.get_static_shape_from_size_variables([x.size]) for x in self.model.value_vars]
        # if right shape is None, it indicates already a batch shape, so delete it.
        self.shapes = [shape[:-1] if shape[-1] is None else shape for shape in static_shapes]
        shaped_arrays = [np.atleast_1d(np.ones(shape)) for shape in self.shapes]
        lens_at_least_1d = [np.prod(arr.shape) for arr in shaped_arrays]
        self.array_shapes = [arr.shape for arr in shaped_arrays]
        self.dimension = sum(lens_at_least_1d)
        if self.dimension != sampler.d:
            raise DimensionError(
                f"sampler dimension {sampler.d} must match the model joint samples dimension {self.dimension}."
            )
        self.split_indices = np.cumsum(lens_at_least_1d)

        # TODO: we have some checks commented here, because not really sure they are needed
        # try:
        #     # do some sampling and check no errors
        #     bs = 10
        #     inputs = np.ones((bs, self.d))*0.5
        #     # split to right array lengths and remove trailing empty array if present
        #     if self.d > 1:
        #         inputs_list = np.split(inputs, self.split_indices, axis=-1)
        #     else:
        #         inputs_list = [inputs.ravel()]
        #     inputs_list = [_x.ravel() for _x in inputs_list]
        #     if np.prod(inputs[-1].shape) == 0:
        #         inputs_list = inputs_list[:-1]
        #     # reshape to correct shapes
        #     print(self.shapes)
        #     print(inputs_list)
        #     inputs_list = [np.reshape(_x, bs, *shape) for _x, shape in zip(inputs_list, self.shapes)]
        #     _ = self.icdf(*inputs_list)
        # except:
        #     raise DimensionError(
        #         f"sampler dimension {sampler.d} must match the model joint samples dimension {self.dimension}."
        #     )

        # issues = _detect_issues(self.icdf, sampler.d, self.split_indices)
        # if issues:
        #     warnings.warn(
        #         "PyMCWrapper received a PyMC Model "
        #         "and it failed sanity checks:\n  - " + "\n  - ".join(issues),
        #         UserWarning,
        #         stacklevel=2,
        #     )

        # There does not seem to be a reliable way to determine the range, so we set it to [-inf, inf]
        lo = -np.inf
        hi = np.inf
        self.range = np.array([[lo, hi]*sampler.d])

        super(PyMCWrapper, self).__init__()

    # ------------------------------------------------------------------
    # Core AbstractTrueMeasure interface
    # ------------------------------------------------------------------

    def _transform(self, x):
        """Map unit cube samples to the physical space.
        """
        x = np.asarray(x, dtype=float)
        x = _clip_unit_interval(x)
        batch_shape = x.shape[:-1]
        flat_batch = np.prod(batch_shape)
        if self.d > 1:
            inputs_list = np.split(x, self.split_indices, axis=-1)
        else:
            inputs_list = [x.ravel()]
        inputs_list = [_x.ravel() for _x in inputs_list]
        if np.prod(inputs_list[-1].shape) == 0:
            inputs_list = inputs_list[:-1]
        inputs_list = [np.reshape(_x, (flat_batch, *shape)) for _x, shape in zip(inputs_list, self.shapes)]
        samples = self.icdf(*inputs_list)
        samples = samples.reshape(*batch_shape, -1)
        return samples

    def _weight(self, x):
        """
        Compute density weights, pdf(x)
        """
        x = np.asarray(x, dtype=float)
        batch_shape = x.shape[:-1]
        flat_batch = np.prod(batch_shape)
        if self.d > 1:
            inputs_list = np.split(x, self.split_indices, axis=-1)
        else:
            inputs_list = [x.ravel()]
        inputs_list = [_x.ravel() for _x in inputs_list]
        if np.prod(inputs_list[-1].shape) == 0:
            inputs_list = inputs_list[:-1]
        inputs_list = [np.reshape(_x, (flat_batch, *shape)) for _x, shape in zip(inputs_list, self.shapes)]

        return np.exp(self.logpdf(*inputs_list)).reshape(*batch_shape)

    def _spawn(self, sampler, dimension):
        """Create a child true measure that shares the same distribution
        configuration but uses a new sampler.

        We simply reuse the original ``model`` argument so the
        behavior matches the parent.
        """
        # TODO: check if we can reuse the compiled functions and only change sampler
        # this will involve adding a constructor
        return PyMCWrapper(sampler, self.model)
