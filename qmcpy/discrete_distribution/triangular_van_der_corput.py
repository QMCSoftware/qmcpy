from typing import Union
import numpy as np

from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from qmcpy.util import ParameterError


class TriangularVanDerCorput(AbstractLDDiscreteDistribution):
    r"""
    Basu-Owen triangular van der Corput sequence.

    Generates low-discrepancy points directly in the 
    triangle 
    This is a two-dimensional construction.

    References:
        K. Basu and A. B. Owen (2015),
        Low Discrepancy Constructions in the Triangle.

    Examples:
        >>> discrete_distrib = TriangularVanDerCorput()
        >>> discrete_distrib(4)
        array([[0.33333333, 0.33333333],
               [0.16666667, 0.16666667],
               [0.16666667, 0.66666667],
               [0.66666667, 0.16666667]])
    """

    def __init__(
        self,
        dimension: int = 2,
        replications: Union[None, int] = None,
        seed: Union[None, int, np.random.SeedSequence] = None,
    ) -> None:
        r"""Initialize a trianglular van Der Corput discrete distribution.

        Args:
            dimension (int = 2): Dimension of the generator. 
                Note that this construction does not support an arbitrary dimension
            replications (Union[None, int]): Number of independent randomizations.
            seed (Union[None, int, np.random.SeedSequence]): Seed the random
                number generator for reproducibility.
    """
        if dimension != 2:
            raise ParameterError("TriangularVanDerCorput requires dimension = 2")

        super().__init__(dimension=2, replications=replications, seed=seed, d_limit=2, n_limit=np.inf)


        # base triangle vertices. note: PIllards and Cools use ((0,0)^T,(0,1)^T,(1,1)^T))
        self.A = np.array([0.0, 0.0])
        self.B = np.array([0.0, 1.0])
        self.C = np.array([1.0, 0.0]) 

    
    @staticmethod
    def _subdivide(A, B, C, digit):
        """Return the subtriangle corresponding to a base-4 digit."""

        AB = (A + B) / 2
        AC = (A + C) / 2
        BC = (B + C) / 2

        if digit == 0:
            return BC, AC, AB
        elif digit == 1:
            return A, AB, AC
        elif digit == 2:
            return AB, B, BC
        elif digit == 3:
            return AC, BC, C

        raise ValueError("digit must be in {0, 1, 2, 3}")

    def _point(self, i):
        """Return the i-th (0-indexed) triangular van der Corput point."""

        A = self.A.copy()
        B = self.B.copy()
        C = self.C.copy()

        if i == 0: # center of the original triangle
            return (A + B + C) / 3

        while i > 0:
            digit = i % 4 # least significant base-4 digit of i
            i //= 4
            A, B, C = self._subdivide(A, B, C, digit)

        return (A + B + C) / 3

    def _gen_samples(self, n_min, n_max, return_binary=False, warn=True):
        if return_binary:
            raise ParameterError("TriangularVanDerCorput does not support return_binary")
            #after adding randomization add a warning about origin?
        
        n = n_max - n_min
        points = np.array([self._point(i)for i in range(n_min, n_max)])

        # identical deterministic replcations (will change after randomization)
        return np.broadcast_to(points,(self.replications, n, 2),).copy()

    def _spawn(self, child_seed, dimension):
        if dimension != 2:
            raise ParameterError("TriangularVanDerCorput requires dimension = 2")

        return TriangularVanDerCorput(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            # add randomize later
        )
