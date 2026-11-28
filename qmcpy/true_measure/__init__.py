from .abstract_true_measure import AbstractTrueMeasure
from .brownian_motion import BrownianMotion
from .copula import AbstractCopula
from .clayton_copula import ClaytonCopula
from .frank_copula import FrankCopula
from .geometric_brownian_motion import GeometricBrownianMotion
from .gaussian import Gaussian
from .gaussian_copula import GaussianCopula
from .gumbel_copula import GumbelCopula
from .lebesgue import Lebesgue
from .uniform import Uniform
from .kumaraswamy import Kumaraswamy
from .bernoulli_cont import BernoulliCont
from .johnsons_su import JohnsonsSU
from .scipy_wrapper import SciPyWrapper
from .matern_gp import MaternGP
from .mixture import Mixture
from .student_t import StudentT
from .student_t_copula import StudentTCopula
from .uniform_triangle import UniformTriangle
from .simplex_uniform import SimplexUniform
from .zero_inflated_exp_uniform import ZeroInflatedExpUniform
from .triangular import Triangular
from .acceptance_rejection import AcceptanceRejection, AcceptanceRejectionReal
from .product_measure import ProductMeasure
from .importance_sampling import ImportanceSampling

try:
    pymc_available = False
    from .pymc_wrapper import PyMCWrapper
    pymc_available = True
except ImportError as error:
    _missing_module = getattr(error, "name", None)
    if _missing_module is None or _missing_module.split(".", 1)[0] not in {
        "pymc",
        "pytensor",
    }:
        raise

    def _raise_missing_pymc_dependency(component):
        raise ModuleNotFoundError(
            f"{component} requires optional PyMC dependencies; missing module "
            f"'{_missing_module}'. Install the pymc >= 6.0.1"
        )

    class PyMCWrapper(AbstractTrueMeasure):
        def __init__(self, *args, **kwargs):
            _raise_missing_pymc_dependency("PyMCWrapper")

TrueMeasure = AbstractTrueMeasure
_TrueMeasure = AbstractTrueMeasure
Normal = Gaussian
Matern = MaternGP
