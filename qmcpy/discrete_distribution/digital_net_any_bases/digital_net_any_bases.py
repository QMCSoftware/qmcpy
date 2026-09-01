import warnings
from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError,ParameterWarning
from ..digital_net_b2 import DigitalNetB2
import qmctoolscl
import numpy as np
from math import comb
from copy import deepcopy


class DigitalNetAnyBases(AbstractLDDiscreteDistribution):
    r"""
    Low discrepancy digital net with arbitrary bases for each dimension. 
    
    Note: 
        - Digital net samples sizes should be products of powers of bases, 
        i.e., a digital net with bases $(b_1,\dots,b_d)$ 
        will prefer sample sizes $n = b_1^{p_1} \cdots b_d^{p_d}$ for some $p_1,\dots,p_d \in \mathbb{N}_0$.
        - The first point of an unrandomized digital net is the origin. 
        - The construction of higher order digital nets requires the same base for each dimension. 
        To construct higher order digital nets, either: 
        
            - Pass in `generating_matrices` *without* interlacing and supply `alpha>1` to apply interlacing, or 
            - Pass in `generating_matrices` *with* interlacing and set `alpha=1` to avoid additional interlacing. 
            
            i.e. do *not* pass in interlaced `generating_matrices` and set `alpha>1`, this will apply additional interlacing. 
    
    A few examples below showcase how to pass in custom bases and generating matrices. Many other examples can be found in the Halton and Faure implementations
    
    Examples:
        >>> bases = 3 
        >>> generating_matrices = np.array(
        ...     [
        ...         [[1, 0, 0],
        ...          [0, 1, 0],
        ...          [1, 2, 1]],
        ...         [[2, 0, 0],
        ...          [1, 2, 0],
        ...          [0, 2, 2]]
        ...     ],
        ...     dtype=np.uint64)
        >>> DigitalNetAnyBases(2,randomize="False",bases_generating_matrices=(bases,generating_matrices))(9,warn=False)
        array([[0.        , 0.        ],
               [0.33333333, 0.66666667],
               [0.66666667, 0.33333333],
               [0.11111111, 0.55555556],
               [0.44444444, 0.22222222],
               [0.77777778, 0.88888889],
               [0.22222222, 0.77777778],
               [0.55555556, 0.44444444],
               [0.88888889, 0.11111111]])
        >>> DigitalNetAnyBases(1,randomize="False",alpha=2,bases_generating_matrices=(bases,generating_matrices))(9,warn=False)
        array([[0.        ],
               [0.55555556],
               [0.77777778],
               [0.17283951],
               [0.39506173],
               [0.95061728],
               [0.30864198],
               [0.5308642 ],
               [0.75308642]])
    
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> bases = np.array(
        ...     [[2,5,7,23],
        ...      [3,11,13,17]],
        ...     dtype=np.uint64)
        >>> generating_matrices = np.stack(
        ...     [
        ...         np.stack(
        ...             [   np.tril(rng.integers(0,2,(10,10))),
        ...                 np.tril(rng.integers(0,5,(10,10))),
        ...                 np.tril(rng.integers(0,7,(10,10))),
        ...                 np.tril(rng.integers(0,23,(10,10))),],axis=0),
        ...         np.stack(
        ...             [   np.tril(rng.integers(0,3,(10,10))),
        ...                 np.tril(rng.integers(0,11,(10,10))),
        ...                 np.tril(rng.integers(0,13,(10,10))),
        ...                 np.tril(rng.integers(0,17,(10,10))),],axis=0),
        ...     ],axis=0).astype(np.uint64)
        >>> discrete_distrib = DigitalNetAnyBases(
        ...     dimension = 4,
        ...     randomize = "NUS",
        ...     seed = 7,
        ...     replications = 2,
        ...     bases_generating_matrices=(bases,generating_matrices),
        ...     )
        >>> discrete_distrib(10,warn=False)
        array([[[0.141964  , 0.50711584, 0.75766621, 0.43359373],
                [0.65536579, 0.66327673, 0.21825902, 0.93901529],
                [0.46955206, 0.21554445, 0.54265048, 0.17421341],
                [0.78505432, 0.99432677, 0.08408121, 0.8317411 ],
                [0.65536579, 0.05422203, 0.37171211, 0.05616883],
                [0.141964  , 0.21554445, 0.94198662, 0.89162246],
                [0.78505432, 0.99432677, 0.62395426, 0.70351649],
                [0.46955206, 0.05422203, 0.51786701, 0.75215221],
                [0.98479561, 0.50711584, 0.06267682, 0.59165814],
                [0.31524867, 0.66327673, 0.29204202, 0.29200658]],
        <BLANKLINE>
               [[0.16158904, 0.00911626, 0.99631587, 0.7984386 ],
                [0.33578478, 0.98734618, 0.48165571, 0.84189861],
                [0.84596814, 0.67093277, 0.72532656, 0.49339621],
                [0.33578478, 0.54281197, 0.26069214, 0.43832903],
                [0.84596814, 0.58529648, 0.80521628, 0.15910404],
                [0.16158904, 0.4486066 , 0.84973668, 0.6750655 ],
                [0.84596814, 0.21294277, 0.40387763, 0.19059409],
                [0.16158904, 0.74257456, 0.19604142, 0.98366484],
                [0.33578478, 0.31746094, 0.35948446, 0.75911922],
                [0.64593022, 0.11007165, 0.63174328, 0.55910368]]])
        
        >>> bases = 2
        >>> generating_matrices = np.array([
        ...     [[1, 0, 0],
        ...      [0, 1, 0],
        ...      [1, 0, 1]],
        ...     [[1, 0, 0],
        ...      [1, 1, 0],
        ...      [0, 1, 1]]],dtype=np.uint64)
        >>> x = DigitalNetAnyBases(2,randomize="False",bases_generating_matrices=(bases,generating_matrices),warn=False)(8,warn=False)
        >>> x
        array([[0.   , 0.   ],
               [0.5  , 0.5  ],
               [0.25 , 0.75 ],
               [0.75 , 0.25 ],
               [0.625, 0.375],
               [0.125, 0.875],
               [0.875, 0.625],
               [0.375, 0.125]])
        >>> generating_matrices_b2 = (generating_matrices*2**np.array([2,1,0])).sum(-1).astype(np.uint64)
        >>> x_b2 = DigitalNetB2(2,randomize="False",order="radical inverse",generating_matrices=generating_matrices_b2,t=3,msb=True)(8,warn=False)
        >>> x_b2
        array([[0.   , 0.   ],
               [0.5  , 0.5  ],
               [0.25 , 0.75 ],
               [0.75 , 0.25 ],
               [0.625, 0.375],
               [0.125, 0.875],
               [0.875, 0.625],
               [0.375, 0.125]])
        >>> bool((x==x_b2).all())
        True
    
    **References:**

    1.  Dick, Josef, and Friedrich Pillichshammer.  
        Digital nets and sequences: discrepancy theory and quasi–Monte Carlo integration.  
        Cambridge University Press, 2010.
    
    2.  Sorokin, Aleksei.  
        "QMCPy: A Python Software for Randomized Low-Discrepancy Sequences, Quasi-Monte Carlo, and Fast Kernel Methods"  
        arXiv preprint arXiv:2502.14256 (2025).
    """

    DEFAULT_GENERATING_MATRICES = None

    def __init__(self,
                 dimension = 1,
                 replications = None,
                 seed = None,
                 randomize = 'LMS DP',
                 bases_generating_matrices = None,
                 t = None,
                 alpha = 1,
                 n_lim = 2**32,
                 warn = True):
        r"""
        Args:
            dimension (Union[int,np.ndarray]): Dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1.
                - If an `np.ndarray` is passed in, use generating vector components at these indices.
            
            replications (int): Number of independent randomizations of a pointset.
            seed (Union[None,int,np.random.SeedSeq]): Seed the random number generator for reproducibility.
            randomize (str): Options are
                
                - `'LMS DP'`: Linear matrix scramble with digital permutation.
                - `'LMS DS'`: Linear matrix scramble with digital shift.
                - `'LMS'`: Linear matrix scramble only.
                - `'DP'`: Digital permutation scramble only.
                - `'DS'`: Digital shift only.
                - `'NUS'`: Nested uniform scrambling.
                - `'QRNG'`: Deterministic permutation scramble and random digital shift from QRNG [1] (with `generalize=True`). Does *not* support replications>1.
                - `None`: No randomization. In this case the first point will be the origin. 
            
            bases_generating_matrices (Union[str, tuple]): Specify the bases and the generating matrices.
                
                - `"HALTON"` will use Halton generating matrices.
                - `"FAURE"` will use Faure generating matrices .
                - `bases,generating_matrices` requires 
                    
                    - `bases` is an `np.ndarray` of integers with shape $(,d)$ or $(r,d)$ where $d$ is the number of dimensions and $r$ is the number of replications.
                    - `generating_matrices` is an `np.ndarray` of integers with shape $(d,m_\mathrm{max},t_\mathrm{max})$ or $(r,d,m_\mathrm{max},t_\mathrm{max})$ where $d$ is the number of dimensions, $r$ is the number of replications, and $2^{m_\mathrm{max}}$ is the maximum number of supported points.
            
            t (int): Number of digits *after* randomization. The number of digits in the generating matrices is inferred.
            alpha (int): Interlacing factor for higher order nets.  
                When `alpha`>1, interlacing is performed regardless of the generating matrices,  
                i.e., for `alpha`>1 do *not* pass in generating matrices which are already interlaced.  
                The Note for this class contains more info.  
            n_lim (int): Maximum number of compatible points, determines the number of rows in the generating matrices. 
            warn (bool): If `False`, suppress warnings in construction 
        """
        self.parameters = ['randomize','t','n_limit']
        self.all_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919],dtype=np.uint64)
        if bases_generating_matrices is None:
            if self.DEFAULT_GENERATING_MATRICES=="HALTON":
                self.type_bases_generating_matrices = "HALTON"
                d_limit = len(self.all_primes)
            elif self.DEFAULT_GENERATING_MATRICES=="FAURE":
                self.type_bases_generating_matrices = "FAURE"
                d_limit = int(self.all_primes[-1])
            else:
                raise ParameterError("must supply bases_generating_matrices")
        else:
            self.type_bases_generating_matrices = "CUSTOM"
            assert len(bases_generating_matrices)==2
            bases,generating_matrices = bases_generating_matrices
            assert isinstance(generating_matrices,np.ndarray)
            assert generating_matrices.ndim==3 or generating_matrices.ndim==4
            d_limit = generating_matrices.shape[1]
            if np.isscalar(bases):
                assert bases>0
                assert bases%1==0
                bases = int(bases)*np.ones(d_limit,dtype=int)
            assert bases.ndim==1 or bases.ndim==2
        self.input_t = deepcopy(t)
        self.input_bases_generating_matrices = deepcopy(bases_generating_matrices)
        super(DigitalNetAnyBases,self).__init__(dimension,replications,seed,d_limit,n_lim)
        self.randomize = str(randomize).upper().strip().replace("_"," ")
        if self.randomize=="TRUE": self.randomize = "LMS DP"
        if self.randomize=="LMS PERM": self.randomize = "LMS DP"
        if self.randomize=="PERM": self.randomize = "DP"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS DP","LMS DS","LMS","DP","DS","NUS","QRNG","FALSE"]
        if self.randomize=="QRNG":
            assert self.type_bases_generating_matrices=="HALTON", "QRNG randomization is only applicable for the Halton generator."
            assert self.replications==1, "QRNG requires replications=1"
            self.randu_d_32 = self.rng.uniform(size=(self.d,32))
        self.alpha = alpha
        assert self.alpha>=1
        assert self.alpha%1==0
        if self.alpha>1:
            assert (self.dvec==np.arange(self.d)).all(), "digital interlacing requires dimension is an int"
        self.dtalpha = self.alpha*self.d
        if self.type_bases_generating_matrices=="HALTON":
            self.bases = self.all_primes[self.dvec][None,:]
            self.m_max = int(np.ceil(np.log(self.n_limit)/np.log(self.bases.min())))
            self._t_curr = self.m_max
            if t is None: t = int(np.ceil(-np.log(2**(-63))/np.log(self.bases.min())))
            self.t = self.m_max if self.m_max>t else t
            self.C = qmctoolscl.gdn_get_halton_generating_matrix(np.uint64(1),np.uint64(self.d),np.uint64(self._t_curr))
        elif self.type_bases_generating_matrices=="FAURE":
            assert (self.dvec==np.arange(self.d)).all(), "Faure requires dimension is an int"
            p = self.all_primes[np.argmax(self.all_primes>=self.d)]
            self.bases = p*np.ones((1,self.dtalpha),dtype=np.uint64)
            self.m_max = int(np.ceil(np.log(self.n_limit)/np.log(p)))
            if t is None: t = int(np.ceil(-np.log(2**(-63))/np.log(p)))
            self._t_curr = self.m_max
            self.t = self.m_max if self.m_max>t else t
            self.C = np.ones((self.dtalpha,1,1),dtype=np.uint64)*np.eye(self._t_curr,dtype=np.uint64)
            if self.dtalpha>1:
                for a in range(self._t_curr):
                    for b in range(a+1):
                        self.C[1,a,b] = comb(a,b)%p
            if self.dtalpha>2:
                for k in range(2,self.dtalpha):
                    for a in range(self._t_curr):
                        for b in range(a+1):
                            self.C[k,a,b] = (int(self.C[1,a,b])*((k**(a-b))%p))%p
            self.C = self.C[None,:,:,:]
            self.faure_base = p
        else:
            self.bases = bases.astype(np.uint64)
            if self.bases.ndim==1: self.bases = self.bases[None,:]
            assert self.bases.shape[1]>=self.dtalpha
            if self.alpha==1:
                self.bases = self.bases[:,self.dvec]
            else:
                self.bases = self.bases[:,:self.dtalpha]
            self.C = generating_matrices.astype(np.uint64)
            if self.C.ndim==3: self.C = self.C[None,:,:,:]
            assert self.C.shape[1]>=self.dtalpha
            if self.alpha==1:
                self.C = self.C[:,self.dvec,:,:]
            else:
                self.C = self.C[:,:self.dtalpha,:,:]
            self.m_max,self._t_curr = self.C.shape[-2:]
            if t is None: t = int(np.ceil(-np.log(2**(-63))/np.log(self.bases.min())))
            self.t = self.m_max if self.m_max>t else t
            assert (0<=self.C).all()
            assert (self.C<self.bases[:,:,None,None]).all()
            self.C = np.ascontiguousarray(self.C)
            self.bases = np.ascontiguousarray(self.bases)
        if self.alpha>1:
            assert (self.bases==self.bases[0,0]).all(), "alpha>1 performs digital interlacing which requires the same base across dimensions and replications."
            if warn and self.m_max!=self._t_curr:
                warnings.warn("Digital interlacing is often performed on generating matrices with the number of columns (m_max = %d) equal to the number of rows (_t_curr = %d), but this is not the case. Ensure you are NOT setting alpha>1 when generating matrices are already interlaced."%(self.m_max,self._t_curr),ParameterWarning)
        assert self.bases.ndim==2
        assert self.bases.shape[-1]==self.dtalpha
        assert self.bases.shape[0]==1 or self.bases.shape[0]==self.replications
        assert self.C.ndim==4
        assert self.C.shape[-3:]==(self.dtalpha,self.m_max,self._t_curr)
        assert self.C.shape[0]==1 or self.C.shape[0]==self.replications
        r_b = self.bases.shape[0]
        r_C = self.C.shape[0]
        if self.randomize=="FALSE":
            if self.alpha>1:
                t_alpha = min(self.t,self._t_curr*self.alpha)
                C_ho = np.empty((self.replications,self.d,self.m_max,t_alpha),dtype=np.uint64)
                qmctoolscl.gdn_interlace(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(self.dtalpha),np.uint64(self._t_curr),np.uint64(t_alpha),np.uint64(self.alpha),self.C,C_ho)
                self.C = C_ho
                self._t_curr = t_alpha
                self.t = t_alpha
                self.bases = self.bases[:,:self.d]
        elif self.randomize=="DP":
            self.perms = qmctoolscl.gdn_get_digital_permutations(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(r_b),self.bases)
        elif self.randomize=="DS":
            self.rshift = qmctoolscl.gdn_get_digital_shifts(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(r_b),self.bases)
        elif self.randomize in ["LMS","LMS DS","LMS DP"]:
            if self.alpha==1:
                S = qmctoolscl.gdn_get_linear_scramble_matrix(self.rng,np.uint64(self.replications),np.uint64(self.d),np.uint64(self._t_curr),np.uint64(self.t),np.uint64(r_b),self.bases)
                C_lms = np.empty((self.replications,self.d,self.m_max,self.t),dtype=np.uint64)
                qmctoolscl.gdn_linear_matrix_scramble(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(r_C),np.uint64(r_b),np.uint64(self._t_curr),np.uint64(self.t),self.bases,S,self.C,C_lms,backend="c")
                self.C = C_lms
                self._t_curr = self.t
            else:
                t_dig = np.ceil(max(self.t/self.alpha,self._t_curr))
                S = qmctoolscl.gdn_get_linear_scramble_matrix(self.rng,np.uint64(self.replications),np.uint64(self.dtalpha),np.uint64(self._t_curr),np.uint64(t_dig),np.uint64(r_b),self.bases)
                C_lms = np.empty((self.replications,self.dtalpha,self.m_max,self.t),dtype=np.uint64)
                qmctoolscl.gdn_linear_matrix_scramble(np.uint64(self.replications),np.uint64(self.dtalpha),np.uint64(self.m_max),np.uint64(r_C),np.uint64(r_b),np.uint64(self._t_curr),np.uint64(t_dig),self.bases,S,self.C,C_lms,backend="c")
                C_lms_ho = np.empty((self.replications,self.d,self.m_max,self.t),dtype=np.uint64)
                qmctoolscl.gdn_interlace(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(self.dtalpha),np.uint64(t_dig),np.uint64(self.t),np.uint64(self.alpha),C_lms,C_lms_ho)
                self.C = C_lms_ho
                self._t_curr = self.t
                self.t = self._t_curr
                self.bases = self.bases[:,:self.d]
            if self.randomize=="LMS DP":
                self.perms = qmctoolscl.gdn_get_digital_permutations(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(r_b),self.bases)
            elif self.randomize=="LMS DS":
                self.rshift = qmctoolscl.gdn_get_digital_shifts(self.rng,np.uint64(self.replications),np.uint64(self.d),self.t,np.uint64(r_b),self.bases)
        elif self.randomize=="NUS":
            if self.alpha==1:
                new_seeds = self._base_seed.spawn(self.replications*self.d)
                self.rngs = np.array([np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(self.replications*self.d)]).reshape(self.replications,self.d)
                self.root_nodes = np.array([qmctoolscl.NUSNode_gdn() for i in range(self.replications*self.d)]).reshape(self.replications,self.d)
            else:
                new_seeds = self._base_seed.spawn(self.replications*self.dtalpha)
                self.rngs = np.array([np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(self.replications*self.dtalpha)]).reshape(self.replications,self.dtalpha)
                self.root_nodes = np.array([qmctoolscl.NUSNode_gdn() for i in range(self.replications*self.dtalpha)]).reshape(self.replications,self.dtalpha)
        assert self.C.ndim==4 and (self.C.shape[0]==1 or self.C.shape[0]==self.replications) and self.C.shape[1]==(self.dtalpha if self.randomize=="NUS" else self.d) and self.C.shape[2]==self.m_max and self.C.shape[3]==self._t_curr
        assert self.bases.ndim==2 and (self.bases.shape[0]==1 or self.bases.shape[0]==self.replications) and self.bases.shape[1]==(self.dtalpha if self.randomize=="NUS" else self.d)
        assert 0<self._t_curr<=self.t<=64
        if self.randomize=="FALSE": assert self.C.shape[0]==self.replications, "randomize='FALSE' but replications = %d does not equal the number of sets of generating vectors %d"%(self.replications,self.C.shape[0])
        if warn and (self.bases==2).all():
            warnings.warn("It is more efficient to use DigitalNetB2 instead of DigitalNetAnyBases when all bases are 2")
        self.warn = warn

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if n_min == 0 and self.randomize in ["FALSE","LMS"] and warn:
            warnings.warn("Without randomization, the first DigitalNetAnyBases point is the origin")
        if warn and (self.bases==self.bases[0,0]).all() and not ( (n_min == 0 or n_min == self.bases[0,0]**np.round(np.log(n_min)/np.log(self.bases[0,0]))) and (n_max == 0 or n_max == self.bases[0,0]**np.round(np.log(n_max)/np.log(self.bases[0,0])))):
            warnings.warn("DigitalNetAnyBases recommends n_min and n_max be 0 or powers of the common base %d at which the digital net achieves superior uniformity properties"%self.bases[0,0])
        r_b = np.uint64(self.bases.shape[0])
        r_C = np.uint64(self.C.shape[0])
        n = np.uint64(n_max-n_min)
        d = np.uint64(self.dtalpha) if self.randomize=="NUS" and self.alpha>1 else np.uint64(self.d)
        n_start = np.uint64(n_min)
        mmax = np.uint64(self.m_max)
        _t_curr = np.uint64(self._t_curr)
        t = np.uint64(self.t)
        bmax = np.uint64(self.bases.max())
        xdig = np.empty((r_C,n,d,_t_curr),dtype=np.uint64)
        qmctoolscl.gdn_gen_natural(r_C,n,d,r_b,mmax,_t_curr,n_start,self.bases,self.C,xdig,backend="c")
        if self.randomize in ["FALSE","LMS"]:
            x = np.empty((r_C,n,d),dtype=np.float64)
            qmctoolscl.gdn_integer_to_float(r_C,n,d,r_b,_t_curr,self.bases,xdig,x,backend="c")
            return x
        if self.randomize=="QRNG": # no replications
            x = np.zeros((self.d,n),dtype=np.double)
            qmctoolscl.util.halton_qrng_c(n,self.d,int(n_min),True,x,self.randu_d_32,np.int32(self.dvec))
            return x.T[None,:,:]
        r = np.uint64(self.replications)
        xdig_new = np.empty((r,n,d,t),dtype=np.uint64)
        if "DP" in self.randomize:
            qmctoolscl.gdn_digital_permutation(r,n,d,r_C,r_b,_t_curr,t,bmax,self.perms,xdig,xdig_new,backend="c")
        if "DS" in self.randomize:
            qmctoolscl.gdn_digital_shift(r,n,d,r_C,r_b,_t_curr,t,self.bases,self.rshift,xdig,xdig_new,backend="c")
        if "NUS" in self.randomize:
            if self.alpha==1:
                qmctoolscl.gdn_nested_uniform_scramble(r,n,d,r_C,r_b,_t_curr,t,self.rngs,self.root_nodes,self.bases,xdig,xdig_new)
            else:
                d = np.uint64(self.d)
                dtalpha = np.uint64(self.dtalpha)
                alpha = np.uint64(self.alpha)
                t_alpha = np.uint64(max(_t_curr,np.ceil(t/self.alpha)))
                xdig_new = np.empty((r,n,dtalpha,t_alpha),dtype=np.uint64)
                qmctoolscl.gdn_nested_uniform_scramble(r,n,dtalpha,r_C,r_b,_t_curr,t_alpha,self.rngs,self.root_nodes,self.bases,xdig,xdig_new)
                xdig_new_reord = np.moveaxis(xdig_new,[1,2],[2,1]).copy()
                xdig_new_new_reord = np.empty((r,d,n,t),dtype=np.uint64)
                qmctoolscl.gdn_interlace(np.uint64(self.replications),d,np.uint64(n),dtalpha,t_alpha,t,alpha,xdig_new_reord,xdig_new_new_reord)
                xdig_new = np.moveaxis(xdig_new_new_reord,[1,2],[2,1]).copy()
        x = np.empty((r,n,d),dtype=np.float64)
        qmctoolscl.gdn_integer_to_float(r,n,d,r_b,t,self.bases,xdig_new,x,backend="c")
        return x

    def _spawn(self, child_seed, dimension):
        return type(self)(
            dimension = dimension,
            replications = None if self.no_replications else self.replications,
            seed = child_seed,
            randomize = self.randomize,
            bases_generating_matrices = self.input_bases_generating_matrices,
            t = self.input_t,
            alpha = self.alpha,
            n_lim = self.n_limit,
            warn = self.warn)
