from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .kernel import (
    KernelGaussian,
    KernelSquaredExponential,
    KernelRationalQuadratic,
    KernelMatern12,
    KernelMatern32,
    KernelMatern52,
    KernelShiftInvar,
    KernelSI,
    KernelShiftInvarCombined,
    KernelSICombined,
    KernelDigShiftInvar,
    KernelDSI,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDSIAA,
    KernelDigShiftInvarCombined,
    KernelDSICombined,
    KernelMultiTask,
    KernelMultiTaskDerivs,
)
from .fast_transform import (
    fftbr,
    ifftbr,
    fwht,
    omega_fftbr,
    omega_fwht,
    fftbr_torch,
    ifftbr_torch,
    fwht_torch,
    omega_fftbr_torch,
    omega_fwht_torch,
)
from .util import plot_proj, mlmc_test

try:
    _mpmc_utils_available = False
    # Keep the Torch-only utilities available when the heavier PyG import fails.
    from .discrete_distribution.mpmc import utils as mpmc_utils

    _mpmc_utils_available = True
    from .discrete_distribution.mpmc.models import MPMC_net
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

name = "qmcpy"
__version__ = "2.4"
