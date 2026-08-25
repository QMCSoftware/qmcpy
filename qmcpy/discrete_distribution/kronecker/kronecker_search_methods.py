import numpy as np

def kronecker_vector_search_mobius_transform(n_max, d_max, searchsize, kernel=None, coord_weights=None, gen_vec_init=None):
    """
    Note that the sympy package is highly recommended for this search method, though not required.

    A deterministic CBC search method for finding a generating vector for a Kronecker sequence, minimizing the weighted sum of squared discrepancies (WSSD).
        - The first component is gen_vec_init, defaults to the golden ratio.
        - We use a modified mobius transformation f(x) = (a*x + b)/(c*x + d) where a, c are distinct primes and b, d are the two pairs of the smallest positive integers such that |a*d - b*c| = 1.
        - Each subsequent component is found by performing the mobius transformation on the previous component, searching over all pairs of distinct primes from the first searchsize many primes.
    Args:
        n_max (int): The maximum sample size to be searched over.
        d_max (int): The maximum dimension for which to find the generating vector.
        kernel (callable): The kernel function to use in the search.
        searchsize (int): The number of primes to search over for each component of the generating vector.
        coord_weights (array-like, optional): An array of coordinate weights to use in the search. If None, weights are set to j^(-2).
        gen_vec_init (array-like, optional): The initial value for the generating vector. If None, the golden ratio is used for the first component. Note that gen_vec_init is taken mod 1.
    Returns:
        generating_vector, wssd, discrepancies, coeff (tuple):
        - generating_vector (numpy array): The generating vector found by the search.
        - wssd (float): The weighted sum of squared discrepancies for n = 1,...,n_max, for the generating vector found.
        - discrepancies (numpy array): The discrepancies for n = 1,...,n_max.
        - coeff (numpy array): The coefficients of the linear transformation used in the search. A description of the coeff array is found below.
    Time cost:
        The time cost of the search is O(searchsize^2 * d_max * n_max).
    Approach:
        Conducts a deterministic CBC search for a generating vector, minimizing the weighted sum of squared discrepancies (wssd) with sample weights w_n = n.
    Details on coeff array:
        The coeff array is a (d_max-1) x 4 array where each row corresponds to a dimension from 2 to d_max. The columns correspond to the coefficients of the linear transformation used to compute the gen_vec component for that dimension. Specifically,
        - gen_vec[dim+1] = (coeff[dim, 0] * gen_vec[dim] + coeff[dim, 1]) / (coeff[dim, 2] * gen_vec[dim] + coeff[dim, 3])
    """

    if searchsize < 2:
        raise ValueError("searchsize must be at least 2.")
    if n_max < 2:
        raise ValueError("n_max must be at least 2.")
    if d_max < 1:
        raise ValueError("d_max must be at least 1.")
    if coord_weights is not None and len(coord_weights) < d_max:
        raise ValueError("Length of coord_weights must be greater than or equal to d_max.")
    

    # the quadratic Bernoulli polynomial
    if kernel is None:
        kernel = lambda t: t * (t - 1) + 1/6
    
    # define coordinate weights if not provided, default to j^(-2)
    if coord_weights is None:
        coord_weights = np.array([j**(-2) for j in range(1, d_max + 1)], dtype=np.float64)
    else:
        coord_weights = np.asarray(coord_weights, dtype=np.float64)

    # use sympy if it's already installed, otherwise uses slower and recursive direct implementation
    try:
        import sympy
    except ImportError:
        print("While not required, installing the sympy package is recommended for this search method. It is used to compute the Bezout coefficients for the linear transformation used in the search. If sympy is not installed, the search will use a recursive and likely slower implementation of the Euclidean algorithm instead.")
        response = input("Do you want to continue without sympy? (y/n): ")
        if response.lower() != 'y':
            raise ImportError("Please install sympy and try again.")
        else:
            has_sympy = False
            print("Continuing without sympy. This may take longer.")
    else:
        has_sympy = True

    if has_sympy:
        # search over the first n primes, n = searchsize
        searchspace = np.array(list(sympy.primerange(1, sympy.prime(searchsize)+1)), dtype=np.float64)
    else:
        def get_primes(n):
            primes = []
            num = 2
            while len(primes) < n:
                is_prime = True
                for p in primes:
                    if p * p > num:
                        break
                    if num % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(num)
                num += 1
            return primes
        
        # search over the first n primes, n = searchsize
        searchspace = np.array(get_primes(searchsize), dtype=np.float64)

        # we define this method here for convenience, to use in computing Bezout coefficients if necessary
        def recursive_euclidean_algorithm(a, b):
            if b == 0:
                return 1, 0, a
            x1, y1, gcd = recursive_euclidean_algorithm(b, a % b)
            x = y1
            y = x1 - (a // b) * y1
            return x, y, gcd
    
    # gen_vec is our generating vector, will be found cbc
    gen_vec = np.zeros(d_max, dtype=np.float64)

    # we pick the golden ratio as the first component of gen_vec, or let the user specify
    if gen_vec_init is None:
        gen_vec[0] = np.float64((np.sqrt(5) - 1) / 2)
    else:
        gen_vec[0] = np.mod(gen_vec_init, 1,dtype=np.float64)

    # precompute several constants for the wssd calculation
    diff = np.cumsum(1.0 / np.arange(n_max, 1, -1,dtype=np.float64))
    freq = np.cumsum(diff)
    freq = np.flip(freq)

    num = n_max * (n_max + 1) / 2

    nK0 = (1 + coord_weights/6)
    nK0 = n_max * np.cumprod(nK0)

    # precompute Bezout coefficients for all pairs of primes in the search space
    bezoutCoeffs = np.zeros((searchsize, searchsize))
    if has_sympy:
        from sympy.core.intfunc import igcdex
        for i in range(searchsize - 1):
            a = searchspace[i]
            for j in range(i + 1, searchsize):
                c = searchspace[j]
                # Use sympy.igcdex to get Bezout coefficients
                d_coeff, b_coeff, _ = igcdex(int(a), int(c))
                bezoutCoeffs[i, j] = np.float64(b_coeff)
                bezoutCoeffs[j, i] = np.float64(d_coeff)
    else:
        for i in range(searchsize - 1):
            a = searchspace[i]
            for j in range(i + 1, searchsize):
                c = searchspace[j]
                # Use the recursive Euclidean algorithm to get Bezout coefficients
                d_coeff, b_coeff, _ = recursive_euclidean_algorithm(int(a), int(c))
                bezoutCoeffs[i, j] = np.float64(b_coeff)
                bezoutCoeffs[j, i] = np.float64(d_coeff)


    # setting up some useful variables for the search
    coeff = np.zeros((d_max - 1, 4)) # stores the coefficients of the linear transformation at each dimension
    t = gen_vec[0] * np.arange(1, n_max) % 1 # t vector is the vector of coordinates generated for the first dimension
    kPrev = 1 + coord_weights[0] * kernel(t) # gets the k vector for the first dimension, which is used in the wssd calculation and updated each dimension of the search.
    # The k vector is Ktilde(x_i) for i = 1,...,n_max-1, where Ktilde is the kernel and x_i are the points generated by the gen_vec vector, up to the current dimension.
    
    # the main search loop
    for dim in range(1, d_max):
        best_wssd = np.inf # stores the current wssd found for each dimension, initialized to infinity
        best_gen_vec = 0 # stores the current best gen_vec component found for this dimension, initialized to 0
        best_k = None # stores the k vector for the current best gen_vec, used to update the k vector for the next dimension after the search is done for this dimension
        for i in range(searchsize):
            p1 = searchspace[i] 
            for j in range(searchsize):
                if j == i: # the two primes have to be distinct, so we skip this case
                    continue

                p2 = searchspace[j]

                b = bezoutCoeffs[i, j]
                d = bezoutCoeffs[j, i]

                if b < 0: # we search over both minimal Bezout coefficients
                    b1 = -b
                    d1 = d
                    b2 = np.abs(b + p1)
                    d2 = np.abs(d -p2)
                else:
                    d1 = -d
                    b1 = b
                    d2 = np.abs(d + p2)
                    b2 = np.abs(b - p1)
                
                gen_vec_dim1 = (p1 * gen_vec[dim - 1] + b1) /  (p2 * gen_vec[dim - 1] + d1) # the linear transformation to get the next gen_vec_dim candidate to test
                gen_vec_dim2 = (p1 * gen_vec[dim - 1] + b2) / (p2 * gen_vec[dim - 1] + d2) # the other candidate from the linear transformation
                t1 = (gen_vec_dim1 * np.arange(1, n_max)) - np.floor(gen_vec_dim1 * np.arange(1, n_max)) # vector of coordinates generated by this candidate component
                t2 = (gen_vec_dim2 * np.arange(1, n_max)) - np.floor(gen_vec_dim2 * np.arange(1, n_max)) 
                k_vector1 = kPrev * (1 + kernel(t1) * coord_weights[dim]) # get the k vector for this candidate component, used in the wssd calculation
                k_vector2 = kPrev * (1 + kernel(t2) * coord_weights[dim]) 

                wssd1 = np.dot(freq, k_vector1)
                wssd2 = np.dot(freq, k_vector2)

                if wssd1 < wssd2:
                    b = b1
                    d = d1
                    wssd = wssd1
                    k_vector = k_vector1
                    gen_vec_dim = gen_vec_dim1
                else:
                    b = b2
                    d = d2
                    wssd = wssd2
                    k_vector = k_vector2
                    gen_vec_dim = gen_vec_dim2
                
                if wssd < best_wssd: # if this candidate has a better wssd than the best found so far, we update the best coefficients and wssd
                    coeff[dim-1, 0] = p1
                    coeff[dim-1, 1] = b
                    coeff[dim-1, 2] = p2
                    coeff[dim-1, 3] = d
                    best_wssd = wssd
                    best_gen_vec = gen_vec_dim % 1
                    best_k = k_vector
        gen_vec[dim] = best_gen_vec # update the gen_vec vector with the best candidate found for this dimension

        kPrev = best_k # update the k vector for the next dimension with the k vector of the best candidate found for this dimension
        best_wssd = nK0[dim] - num + 2 * best_wssd # calculate the best wssd for this dimension using the formula from the paper, which involves the nK0 constants precomputed at the beginning of the function. This is used for debugging and to check the wssd at each dimension of the search.

        # print(coeff[dim - 1, :], (nK0[dim] - num + 2 * best_wssd)) # debugging line to check the coefficients and wssd at each dimension
    
    # Adapted from Jimmy's code for calculating the discrepancies for n = 1,...,n_max from SURE 2025
    n_array = np.arange(1, n_max + 1)
    k_tilde = lambda x, coord_weight: np.prod(1 + kernel(x) * coord_weight, axis=1)
    k_tilde_terms = k_tilde(gen_vec * np.arange(n_max).reshape((n_max, 1)) - np.floor(gen_vec * np.arange(n_max).reshape((n_max, 1))), coord_weights)

    left_sum = np.cumsum(k_tilde_terms[1:]) * n_array[1:]
    right_sum = np.cumsum(n_array[:-1] * k_tilde_terms[1:])
        
    k_tilde_zero_terms = k_tilde_terms[0] * n_array
    summation = np.zeros(n_max)
    summation[1:] = left_sum - right_sum
    discrepancies = (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - 1

    return gen_vec, best_wssd, discrepancies, coeff