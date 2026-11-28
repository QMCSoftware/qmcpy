from .abstract_discrete_distribution import AbstractDiscreteDistribution
from .iid_std_uniform import IIDStdUniform
from .lattice import Lattice
from .digital_net_b2 import DigitalNetB2
from .digital_net_any_bases import DigitalNetAnyBases,Halton,Faure,Hammersley
from .mpmc import MPMC
from .kronecker import Kronecker
from .korobov import KorobovLattice
from .dummy_sampler import DummySampler
from .latin_hypercube import LatinHypercube

try:
    _mpmc_utils_available = False
    # Keep the Torch-only utilities available when the heavier PyG import fails.
    from .mpmc import utils as mpmc_utils

    _mpmc_utils_available = True
    from .mpmc.models import MPMC_net
except ImportError as error:
    _missing_module = getattr(error, "name", None)
    if _missing_module is None or _missing_module.split(".", 1)[0] not in {
        "pyg_lib",
        "torch",
        "torch_geometric",
        "torch_cluster",
        "torch_scatter",
        "torch_sparse",
        "torch_spline_conv",
    }:
        raise

    def _raise_missing_mpmc_dependency(component):
        raise ModuleNotFoundError(
            f"{component} requires optional MPMC dependencies; missing module "
            f"'{_missing_module}'. Install the 'qmcpy[mpmc]' extra, then run "
            f"'qmcpy-install-mpmc'.",
            name=_missing_module,
        )

    if not _mpmc_utils_available:
        class _MissingMPMCUtils(object):
            def __getattr__(self, _name):
                _raise_missing_mpmc_dependency("mpmc_utils")

        mpmc_utils = _MissingMPMCUtils()

    class MPMC_net(object):
        def __init__(self, *args, **kwargs):
            _raise_missing_mpmc_dependency("MPMC_net")

DiscreteDistribution = AbstractDiscreteDistribution
_DiscreteDistribution = AbstractDiscreteDistribution
Sobol = DigitalNetB2
DigitalNet = DigitalNetB2
Net = DigitalNetB2
NetB2 = DigitalNetB2

