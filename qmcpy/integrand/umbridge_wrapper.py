from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError
import numpy as np
import os


class UMBridgeWrapper(AbstractIntegrand):
    """Wrapper around a
    [`UM-Bridge`](https://um-bridge-benchmarks.readthedocs.io/en/docs/index.html)
    model. See also the [`UM-Bridge` documentation for the QMCPy
    client](https://um-bridge-benchmarks.readthedocs.io/en/docs/umbridge/clients.html).
    Requires [Docker](https://www.docker.com/) is installed.

    Examples:
        >>> _ = os.system('docker run --name muqbppytest -dit -p 4243:4243 linusseelinger/benchmark-muq-beam-propagation:latest > /dev/null')
        >>> import umbridge
        >>> dnb2 = DigitalNetB2(dimension=3,seed=7)
        >>> true_measure = Uniform(dnb2,lower_bound=1,upper_bound=1.05)
        >>> um_bridge_model = umbridge.HTTPModel('http://localhost:4243','forward')
        >>> um_bridge_config = {"d": dnb2.d}
        >>> integrand = UMBridgeWrapper(true_measure,um_bridge_model,um_bridge_config,parallel=False)
        >>> y = integrand(2**10)
        >>> with np.printoptions(formatter={"float":lambda x: "%.1e"%x}):
        ...     y.mean(-1)
        array([0.0e+00, 3.9e+00, 1.5e+01, 3.2e+01, 5.5e+01, 8.3e+01, 1.2e+02,
               1.5e+02, 2.0e+02, 2.4e+02, 2.9e+02, 3.4e+02, 3.9e+02, 4.3e+02,
               4.7e+02, 5.0e+02, 5.3e+02, 5.6e+02, 5.9e+02, 6.2e+02, 6.4e+02,
               6.6e+02, 6.9e+02, 7.2e+02, 7.6e+02, 7.9e+02, 8.3e+02, 8.6e+02,
               9.0e+02, 9.4e+02, 9.7e+02])
        >>> _ = os.system('docker rm -f muqbppytest > /dev/null')

        Custom model with independent replications

        >>> class TestModel(umbridge.Model):
        ...     def __init__(self):
        ...         super().__init__("forward")
        ...     def get_input_sizes(self, config):
        ...         return [1,2,3]
        ...     def get_output_sizes(self, config):
        ...         return [3,2,1]
        ...     def __call__(self, parameters, config):
        ...         out0 = [parameters[2][0],sum(parameters[2][:2]),sum(parameters[2])]
        ...         out1 = [parameters[1][0],sum(parameters[1])]
        ...         out2 = [parameters[0]]
        ...         return [out0,out1,out2]
        ...     def supports_evaluate(self):
        ...         return True
        >>> um_bridge_model = TestModel()
        >>> um_bridge_config = {}
        >>> d = sum(um_bridge_model.get_input_sizes(config=um_bridge_config))
        >>> true_measure = Uniform(DigitalNetB2(dimension=d,seed=7,replications=15),lower_bound=-1,upper_bound=1)
        >>> integrand = UMBridgeWrapper(true_measure,um_bridge_model)
        >>> y = integrand(2**6)
        >>> y.shape
        (6, 15, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (6, 15)
        >>> muhats_aggregate = muhats.mean(-1)
        >>> muhats_aggregate.shape
        (6,)
        >>> muhats_agg_list_of_lists = integrand.to_umbridge_out_sizes(muhats_aggregate)
        >>> [["%.2e"%ii for ii in i] for i in muhats_agg_list_of_lists]
        [['-1.59e-08', '1.49e-04', '1.49e-04'], ['8.20e-06', '-1.38e-04'], ['-8.14e-06']]
    """

    def __init__(self, true_measure, model, config=None, parallel=False):
        """
        Args:
            true_measure (AbstractTrueMeasure): The true measure.
            model (umbridge.HTTPModel): A `UM-Bridge` model.
            config (dict): Configuration keyword argument to
                `umbridge.HTTPModel(url,name).__call__`.
            parallel (int): Parallelization flag.

                - When `parallel = 0` or `parallel = 1` then function evaluation is done in serial fashion.
                - `parallel > 1` specifies the number of processes used by `multiprocessing.Pool` or `multiprocessing.pool.ThreadPool`.

                Setting `parallel=True` is equivalent to `parallel =
                os.cpu_count()`.
        """
        if config is None:
            config = {}
        import umbridge

        self.parameters = []
        self.true_measure = true_measure
        self.sampler = self.true_measure
        self.model = model
        if not self.model.supports_evaluate():
            raise ParameterError("UMBridgeWrapper requires model supports evaluation.")
        self.config = config
        self.parallel = parallel
        self.d_in_umbridge = np.append(
            0, np.cumsum(self.model.get_input_sizes(self.config))
        )
        self.n_d_in_umbridge = len(self.d_in_umbridge) - 1
        if self.true_measure.d != self.d_in_umbridge[-1]:
            raise ParameterError(
                "sampler dimension (%d) must equal the sum of UMBridgeWrapper input sizes (%d)."
                % (self.true_measure.d, self.d_in_umbridge[-1])
            )
        self.d_out_umbridge = np.append(
            0, np.cumsum(self.model.get_output_sizes(self.config))
        )
        self.n_d_out_umbridge = len(self.d_out_umbridge) - 1
        self.total_out_elements = int(self.d_out_umbridge[-1])
        super(UMBridgeWrapper, self).__init__(
            dimension_indv=(
                () if self.total_out_elements == 1 else (self.total_out_elements,)
            ),
            dimension_comb=(
                () if self.total_out_elements == 1 else (self.total_out_elements,)
            ),
            parallel=self.parallel,
            threadpool=True,
        )

    def g(self, t, **kwargs):
        y = np.zeros((self.total_out_elements,) + tuple(t.shape[:-1]), dtype=float)
        idxiterator = np.ndindex(t.shape[:-1])
        for i in idxiterator:
            ti_ll = [
                t[
                    i + (slice(self.d_in_umbridge[j], self.d_in_umbridge[j + 1]),)
                ].tolist()
                for j in range(self.n_d_in_umbridge)
            ]
            yi_ll = self.model.__call__(ti_ll, self.config)
            for j, yi_l in enumerate(yi_ll):
                y[(slice(self.d_out_umbridge[j], self.d_out_umbridge[j + 1]),) + i] = (
                    yi_l if len(yi_l) > 1 else yi_l[0]
                )
        return y[0] if self.total_out_elements == 1 else y

    def _spawn(self, _level, _sampler):
        return UMBridgeWrapper(
            true_measure=self.true_measure,
            model=self.model,
            config=self.config,
            parallel=self.parallel,
        )

    def to_umbridge_out_sizes(self, x):
        """Convert a data attribute to `UM-Bridge` output sized list of
        lists.

        Args:
            x (np.ndarray): Array of length
                `sum(model.get_output_sizes(self.config))` where `model` is a
                `umbridge.HTTPModel`.

        Returns:
            List of lists with sub-list lengths specified by
            `model.get_output_sizes(self.config)`.
        """
        return [
            x[..., self.d_out_umbridge[j] : self.d_out_umbridge[j + 1]].tolist()
            for j in range(self.n_d_out_umbridge)
        ]