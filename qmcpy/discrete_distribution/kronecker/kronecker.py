from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError
import numpy as np
import warnings

# # For generating more primes if needed for higher dimensions
# def _is_prime(n):
#     if n < 2:
#         return False
#     if n == 2:
#         return True
#     if n % 2 == 0:
#         return False
#     k = 3
#     while k * k <= n:
#         if n % k == 0:
#             return False
#         k += 2
#     return True


# def _next_prime(n):
#     candidate = max(2, int(n) + 1)
#     if candidate == 2:
#         return 2
#     if candidate % 2 == 0:
#         candidate += 1
#     while not _is_prime(candidate):
#         candidate += 2
#     return candidate


# def _get_primes(dimension):
#     PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919])
#     if dimension <= len(PRIMES):
#         return PRIMES[:dimension]
#     primes = list(PRIMES)
#     current = int(primes[-1])
#     while len(primes) < dimension:
#         current = _next_prime(current)
#         primes.append(current)
#     return np.array(primes, dtype=np.int64)

# def _richtmyer_alpha(dimension):
#     if dimension <= len(PRIMES):
#         return np.sqrt(PRIMES[:dimension]) % 1
#     return np.sqrt(_get_primes(dimension)) % 1

def _richtmyer_generating_vector(dimension):
    PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919])
    assert dimension<len(PRIMES)
    return np.sqrt(PRIMES[:dimension]) % 1

def _suzuki_generating_vector(dimension):
    return 2 ** (np.arange(1, dimension + 1) / (dimension + 1))

class Kronecker(AbstractLDDiscreteDistribution):
    r"""
    Kronecker sequence (additive recurrence sequence) for quasi-Monte Carlo.

    A Kronecker sequence is defined by
    $$
    \boldsymbol{x}_i =  i \boldsymbol{\alpha} + \boldsymbol{\delta} \bmod \boldsymbol{1} \in [0,1)^d, \quad i = 0,1,2,\dots,
    $$
    where $\boldsymbol{\alpha} \in \mathbb{R}^d$ is a generating vector and $\boldsymbol{\delta} \in [0,1)^d$
    is an optional shift. The fractional part is taken componentwise.

    These sequences are simple, extensible low-discrepancy sequences when
    $\boldsymbol{\alpha}$ has components that are irrational and well-distributed.

    Notes:
        - The Kronecker sequence is fully extensible in $n$ (no restriction to powers of 2).
        - Quality depends strongly on the choice of $\boldsymbol{\alpha}$.
        - Random shifting preserves unbiasedness for integration.

    Examples:
        >>> discrete_distrib = Kronecker(2,seed=7)
        >>> discrete_distrib(4)
        array([[0.04386058, 0.58727432],
               [0.46629777, 0.94787084],
               [0.88873496, 0.30846736],
               [0.31117214, 0.66906388]])
        >>> discrete_distrib(1) # first point in the sequence
        array([[0.04386058, 0.58727432]])
        >>> discrete_distrib
        Kronecker (AbstractLDDiscreteDistribution)
            d               2^(1)
            replications    1
            randomize       SHIFT
            gen_vec_source  CBC
            entropy         7
        
        Replications of independent randomizations

        >>> x = Kronecker(3,seed=7,replications=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.46629777, 0.94787084, 0.71785454],
                [0.88873496, 0.30846736, 0.06652667],
                [0.31117214, 0.66906388, 0.41519881]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.07456704, 0.0572962 , 0.45472566],
                [0.49700422, 0.41789272, 0.80339779],
                [0.91944141, 0.77848924, 0.15206993]]])
        >>> Kronecker(3,seed=7,replications=2)(2,4)
        array([[[0.88873496, 0.30846736, 0.06652667],
                [0.31117214, 0.66906388, 0.41519881]],
        <BLANKLINE>
               [[0.49700422, 0.41789272, 0.80339779],
                [0.91944141, 0.77848924, 0.15206993]]])
        
        Switch from CBC to Richtmyer generating vector when the dimension is too large.

        >>> Kronecker(15,seed=7,warn=False)(4).shape
        (4, 15)
        >>> Kronecker(15,replications=2,seed=7,warn=False)(4).shape
        (2, 4, 15)

        CBC unrandomized 
        
        >>> Kronecker(3,generating_vector="CBC",randomize=False)(4)
        array([[0.        , 0.        , 0.        ],
               [0.42243719, 0.36059652, 0.34867214],
               [0.84487437, 0.72119304, 0.69734427],
               [0.26731156, 0.08178956, 0.04601641]])
        
        Richtmyer construction 

        >>> Kronecker(3,generating_vector="RICHTMYER",randomize=False)(4)
        array([[0.        , 0.        , 0.        ],
               [0.41421356, 0.73205081, 0.23606798],
               [0.82842712, 0.46410162, 0.47213595],
               [0.24264069, 0.19615242, 0.70820393]])
        >>> Kronecker(3,replications=2,seed=7,generating_vector="RICHTMYER")(4)
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.45807414, 0.31932513, 0.60525038],
                [0.87228771, 0.05137594, 0.84131836],
                [0.28650127, 0.78342675, 0.07738633]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.06634341, 0.42875049, 0.3421215 ],
                [0.48055697, 0.16080129, 0.57818947],
                [0.89477054, 0.8928521 , 0.81425745]]])

        Suzuki construction 

        >>> Kronecker(3,generating_vector="SUZUKI",randomize=False)(4)
        array([[0.        , 0.        , 0.        ],
               [0.18920712, 0.41421356, 0.68179283],
               [0.37841423, 0.82842712, 0.36358566],
               [0.56762135, 0.24264069, 0.04537849]])
        >>> Kronecker(3,replications=2,seed=7,generating_vector="SUZUKI")(4)
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.2330677 , 0.00148789, 0.05097523],
                [0.42227481, 0.41570145, 0.73276806],
                [0.61148193, 0.82991501, 0.41456089]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.84133696, 0.11091324, 0.78784635],
                [0.03054408, 0.5251268 , 0.46963918],
                [0.21975119, 0.93934037, 0.15143201]]])

        Custom shifts

        >>> Kronecker(3,generating_vector="SUZUKI",shift=[0.1,0.2,0.3])(4)
        array([[0.1       , 0.2       , 0.3       ],
               [0.28920712, 0.61421356, 0.98179283],
               [0.47841423, 0.02842712, 0.66358566],
               [0.66762135, 0.44264069, 0.34537849]])
        >>> Kronecker(3,generating_vector="SUZUKI",replications=2,shift=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]]))(4)
        array([[[0.1       , 0.2       , 0.3       ],
                [0.28920712, 0.61421356, 0.98179283],
                [0.47841423, 0.02842712, 0.66358566],
                [0.66762135, 0.44264069, 0.34537849]],
        <BLANKLINE>
               [[0.4       , 0.5       , 0.6       ],
                [0.58920712, 0.91421356, 0.28179283],
                [0.77841423, 0.32842712, 0.96358566],
                [0.96762135, 0.74264069, 0.64537849]]])

        Custom generating vectors 

        >>> Kronecker(3,generating_vector=2**(np.arange(1,4)/(3 + 1)),randomize=False)(4)
        array([[0.        , 0.        , 0.        ],
               [0.18920712, 0.41421356, 0.68179283],
               [0.37841423, 0.82842712, 0.36358566],
               [0.56762135, 0.24264069, 0.04537849]])

        >>> Kronecker(3,generating_vector=2**(np.arange(1,4)/(3 + 1)),randomize="SHIFT",replications=2,seed=7)(4)
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.2330677 , 0.00148789, 0.05097523],
                [0.42227481, 0.41570145, 0.73276806],
                [0.61148193, 0.82991501, 0.41456089]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.84133696, 0.11091324, 0.78784635],
                [0.03054408, 0.5251268 , 0.46963918],
                [0.21975119, 0.93934037, 0.15143201]]])
                
        Subset dimensions 

        >>> Kronecker([0,2],generating_vector=2**(np.arange(1,4)/(3 + 1)),randomize=False)(4)
        array([[0.        , 0.        ],
               [0.18920712, 0.68179283],
               [0.37841423, 0.36358566],
               [0.56762135, 0.04537849]])

    **References**

    1.  Richtmyer, R. D. (1951). "The evaluation of definite integrals and a quasi-Monte Carlo method."
    
    2.  Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*.
    """

    def __init__(self,
        dimension=1,
        replications=None,
        seed=None,
        randomize="SHIFT",
        generating_vector="CBC",
        shift=None,
        warn=True,
    ):
        r"""
        Args:
            dimension (Union[int, np.ndarray]): Dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1.
                - If an `np.ndarray` is passed in, use generating vector components at these indices.
            
            replications (int): Number of independent randomizations.
            seed (Union[None, int, np.random.SeedSeq): Seed the random number generator for reproducibility.
            randomize (str): Options are

                - `'SHIFT'`: use `shift` if supplied, otherwise use a random shift $\boldsymbol{\delta} \sim \mathrm{Uniform}([0,1)^d)$.
                - `'FALSE'`: zero shift.
            
            generating_vector (Union[str,np.ndarray]): Generating vector $\boldsymbol{\alpha}$.
                
                - `"CBC"`: uses the first $d$ components of a known good Component-by-Component (CBC) generating vector.
                - `"RICHTMYER"`: uses $\boldsymbol{\alpha}_j = \sqrt{p_j} \bmod 1$, where $p_j$ are primes. This is the classical Richtmyer construction.
                - `"SUZUKI"`: uses a deterministic construction $\boldsymbol{\alpha}_j = 2^{j/(d+1)}$.
                - `"CBC_MT"`: uses the first $d$ components of a known good CBC generating vector obtained using the Mobius transformation method, which can be found in kronecker_search_methods.py.
                - np.array: user-specified generating vector.

            shift (np.ndarray): Shift vector $\boldsymbol{\delta}$. If `randomize=True`, this is ignored and a random shift is generated. Otherwise, a fixed shift is used.
            warn (bool): If False, suppress warnings during construction 
        """
        self.parameters = ["randomize", "gen_vec_source"]
        self.input_generating_vector = generating_vector
        self.input_shift = shift
        self.mimics = "StdUniform"
        self.randomize = randomize
        super(Kronecker, self).__init__(dimension, replications, seed, d_limit=np.inf, n_limit=np.inf)
        if isinstance(generating_vector, str) and generating_vector.lower() == 'cbc':
            self.gen_vec_source = "CBC"
            CBC = np.array([
                4.224371872086318813e-01,
                3.605965189622313272e-01,
                3.486721371284548510e-01,
                4.520388055082059653e-01,
                2.550750763977845947e-01,
                2.205289926147350477e-01,
                2.071242872822959824e-01,
                3.049354991086913325e-01,
                3.872168854974577523e-01,
                2.275808872220986823e-01,
                1.773740893160189180e-01,
                1.958399682530986008e-01,
                3.216346950830996643e-01], dtype=np.float64)
            gen_vec = CBC
            if not (self.dvec.max() < len(gen_vec)):
                if warn:
                    warnings.warn(
                        f"CBC generating vector only supports dimension <= {len(CBC)}; falling back to Richtmyer.",
                        RuntimeWarning,
                    )
                self.gen_vec_source = "RICHTMYER"
                gen_vec = _richtmyer_generating_vector(self.dvec.max()+1)
        elif isinstance(generating_vector, str) and generating_vector.lower() == 'richtmyer':
            self.gen_vec_source = "RICHTMYER"
            gen_vec = _richtmyer_generating_vector(self.dvec.max()+1)
        elif isinstance(generating_vector, str) and generating_vector.lower() == "suzuki":
            self.gen_vec_source = "SUZUKI"
            gen_vec = _suzuki_generating_vector(self.dvec.max()+1)
        elif isinstance(generating_vector, str) and generating_vector.lower() == "cbc_mt":
            self.gen_vec_source = "CBC_MT"
            CBC_MT = np.array([0.618033988749895,
                0.3173225474723,
                0.59332263014446,
                0.20776441643926,
                0.27373719258623,
                0.649734278361753,
                0.478954018631769,
                0.86866022435182,
                0.22845082022244,
                0.581365429377986,
                0.282365231829842,
                0.0822850909119904,
                0.223849641007295,
                0.5770772201756,
                0.51769659336634,
                0.568025390904592,
                0.156782234569368,
                0.82246227056154,
                0.805675312097409,
                0.63877102813393,
                0.358300563495856,
                0.241741343018598,
                0.705003192174204,
                0.1931911954956,
                0.261022001488623,
                0.897938992038015,
                0.46839743115877,
                0.884022067965329,
                0.752352896871505,
                0.1601583600427,
                0.10727599509739,
                0.151478435512877,
                0.163863657127101,
                0.948303450359399,
                0.80350943597439,
                0.426371623468333,
                0.435930910910882,
                0.21329852459791,
                0.661698149534002,
                0.900679822160453,
                0.122436710671457,
                0.483663584095611,
                0.928181067731583,
                0.443143014606576,
                0.74491332336194,
                0.87948409225588,
                0.0428242449803,
                0.534576896789579,
                0.24340042100879,
                0.30424418245585,
                0.574003104342617,
                0.897289023268963,
                0.541424476559586,
                0.356895660350464,
                0.507567280910795,
                0.513983550428507,
                0.0610821922457415,
                0.183871471606587,
                0.446015178033969,
                0.455684287415085,
                0.280817534817491,
                0.115220095666085,
                0.433740673279323,
                0.515605957977756,
                0.113076735656464,
                0.733928297688305,
                0.0597515651584137,
                0.422268695684775,
                0.0979181139173599,
                0.213699261322352,
                0.866811679881922,
                0.0878569329036737,
                0.678412735893121,
                0.181093969536107,
                0.128913741473518,
                0.109341703717108,
                0.289067270578427,
                0.352218331663839,
                0.303605902333137,
                0.0613899204730832,
                0.959535877660851,
                0.475508309069064,
                0.688698902674194,
                0.657037932118495,
                0.645555897563869,
                0.720658665263604,
                0.914423387894897,
                0.425763295044487,
                0.328825255006553,
                0.892452975558004,
                0.16973367306396,
                0.912292406867098,
                0.0923260018966512,
                0.216301713289429,
                0.147861410064151,
                0.8600781655845,
                0.752129792595509,
                0.337431120990153,
                0.542476014178907,
                0.307279789725491], dtype=np.float64)
            gen_vec = CBC_MT
            if not (self.dvec.max() < len(gen_vec)):
                if warn:
                    warnings.warn(
                        f"CBC_MT generating vector only supports dimension <= {len(CBC_MT)}; falling back to Richtmyer.",
                        RuntimeWarning,
                    )
                self.gen_vec_source = "RICHTMYER"
                gen_vec = _richtmyer_generating_vector(self.dvec.max()+1)
        else:
            self.gen_vec_source = "CUSTOM"
            gen_vec = np.asarray(generating_vector, dtype=float)
            if gen_vec.ndim >2:
                raise ParameterError("generating_vector must be a 1D or 2D np.ndarray")
        gen_vec = np.atleast_2d(gen_vec).astype(float)
        assert gen_vec.ndim==2, "gen_vec must be a 2D array"
        assert gen_vec.shape[1]>=self.d
        assert (gen_vec.shape[0] == 1 or gen_vec.shape[0] == self.replications)
        assert gen_vec.shape[1]>self.dvec.max()
        self.gen_vec = gen_vec[:,self.dvec].copy()
        self.randomize = str(randomize).upper()
        if self.randomize == "TRUE":
            self.randomize = "SHIFT"
        if self.randomize == "NONE":
            self.randomize = "FALSE"
        if self.randomize == "NO":
            self.randomize = "FALSE"
        assert self.randomize in ["SHIFT", "FALSE"]
        if shift is not None: assert self.randomize=="SHIFT", "require randomize='SHIFT' when shift is not None"
        if self.randomize=="SHIFT":
            if shift is not None:
                self.shift = np.atleast_2d(shift).astype(float)
            else:
                self.shift = self.rng.uniform(size=(self.replications, self.d))
        else: # self.randomize=="FALSE":
            self.shift = np.zeros((self.replications, self.d))
        assert self.shift.ndim==2
        assert self.shift.shape[1]==self.d
        assert (self.shift.shape[0] == 1 or self.shift.shape[0] == self.replications)

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if return_binary:
            raise ParameterError("Kronecker does not support return_binary=True")
        i = np.arange(n_min,n_max)
        points = ((i[:,None] * self.gen_vec[:,None,:]) + self.shift[:, None, :]) % 1
        return points

    def periodic_discrepancy(self, n, k_tilde=None, gamma=None):
        # """
        # Calculates the discrepancy for a periodic kernel.

        # Args:
        #     n (int): the number of sample points
        #     k_tilde (Tuple[function, float]): the function takes in 2 arguments: the sample points and the coordinate weights.
        #         The float is the integral over the unit hypercube.
        #     gamma (np.ndarray): shape (1xd)

        # Returns:
        #     discrep (np.ndarray): discrepancy

        # Notes:
        #     - If k_tilde is not specified, the second Bernoulli polynomial is used.
        #     - If gamma is not specified, the coordinate weights will be just all ones.
        # """
        if gamma is None:
            gamma = np.ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: np.prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        return np.sqrt(self._square_periodic_discrepancies(n, k_tilde, gamma))

    def wssd_discrepancy(self, n, sample_weights, k_tilde = None, gamma = None):
        # calculates the weighted sum of square discrepancy
        if gamma is None:
            gamma = np.ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: np.prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        discrepancies = self._square_periodic_discrepancies(n, k_tilde, gamma)
        return np.sum(sample_weights * discrepancies, axis=-1)


    def _square_periodic_discrepancies(self, n, k_tilde, gamma):
        n_array = np.arange(1, n + 1)
        # we need the points without a random shift for the calculation, so we can't use self._gen_samples
        i = np.arange(0, n)
        points = (i[:,None] * self.gen_vec[:,None,:]) % 1
        k_tilde_terms = k_tilde[0](points, gamma)

        left_sum = np.cumsum(k_tilde_terms[...,1:], axis=-1) * n_array[1:]
        right_sum = np.cumsum(n_array[:-1] * k_tilde_terms[...,1:], axis=-1)

        k_tilde_zero_terms = k_tilde_terms[...,0] * n_array
        summation = np.zeros_like(k_tilde_terms)
        summation[...,1:] = left_sum - right_sum
        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - k_tilde[1]


    def _spawn(self, child_seed, dimension):
        assert self.input_shift is None, "spawn requires shift=None"
        return Kronecker(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            randomize=self.randomize,
            generating_vector=self.input_generating_vector,
            shift=None,
        )
