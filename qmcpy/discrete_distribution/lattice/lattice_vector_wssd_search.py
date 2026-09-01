import numpy as np

def lattice_vector_wssd_search(n_max, d_max, coord_weights=None, kernel=None):
    """
    CBC search method for finding a lattice rule minimizing the WSSD.

    Args:
        n_max (int): The maximum number of points the lattice rule is optimized for.
        d_max (int): The dimension of the lattice rule.
        coord_weights (array-like, optional): The coordinate weights used to compute the discrepancy. Defaults to j^(-2) for j=1,...,d_max.
        kernel (callable, optional): The kernel used to compute the discrepancy. Should accept a single argument and return a scalar. Defaults to the second Bernoulli polynomial.
    Returns:
        gen_vec (array-like): The generating vector of the lattice that minimizes the WSSD.

    Time cost:
        The time cost of the search is O(d_max * n_max * log(n_max)), though the contribution of d_max is smaller until around d_max = 100.
    Note:
        Uses sample weights of w_n = n for n = 1,...,n_max when calculating the WSSD.
    
    Examples:
        >>> lattice_vector_wssd_search(n_max=2**10, d_max=5)
        array([  1, 403, 361, 281, 421])
        >>> lattice_vector_wssd_search(n_max=2**15, d_max=10)
        array([    1,  4825, 13541, 15249, 15405,  9909,  7493, 11407, 14819,
               10089])
        
        Custom coordinate weights

        >>> lattice_vector_wssd_search(n_max=2**15, d_max=10, coord_weights=[j**(-1) for j in range(1, 11)])
        array([    1,  4825, 13541, 15249,  7311, 10339,  5933,  6307, 14729,
               13037])

        Custom kernels

        >>> bernoulli6 = lambda x: x * (x * (-1/2 + x * (x * (5/2 + x * (-3 + x))))) + 1/42
        >>> lattice_vector_wssd_search(n_max=2**15, d_max=10, coord_weights=None, kernel=bernoulli6) # doctest: +ELLIPSIS
        array([    1, 12589, ...]...)

        The algorithm in its current form is sensitive to differences in floating point precision across platforms, hence the nondeterministic nature of the example above. This can cause differences in generator quality, though in my ad hoc testing it is usually not catastrophic. It was originally built on a Windows machine.

    """
    np.seterr(all='warn')
    if kernel is None:
        kernel = lambda x: x * (x - 1) + 1 / 6 # default kernel is the second Bernoulli polynomial
    if coord_weights is None:
        coord_weights = np.array([j**(-2) for j in range(1, d_max + 1)], dtype=np.float64) # default coordinate weights are j^(-2)

    if not callable(kernel):
        raise ValueError("kernel must be a callable function")
    if not isinstance(coord_weights, (list, np.ndarray)):
        raise ValueError("coord_weights must be array-like")
    if not isinstance(n_max, int) or not isinstance(d_max, int):
        raise ValueError("n_max and d_max must be integers")
    
    if len(coord_weights) < d_max:
        raise ValueError("coord_weights must have length at least d_max")
    if n_max < 3:
        raise ValueError("n_max must be at least 3")
    if d_max < 1:
        raise ValueError("d_max must be at least 1")

    m = np.ceil(np.log2(n_max)).astype(int)

    # ----------------------------------------------------------------------
    # Set up rhovector - how often each value appears
    # ----------------------------------------------------------------------
    bits = np.zeros((n_max, m), dtype=np.uint64)
    for i in range(n_max):
        bits[i, :] = 2 * np.array([((i >> j) & 1) for j in range(m)], dtype=np.uint64)

    cumsumbits = np.cumsum(bits, axis=0)        # n_max x m
    rhovector = np.dot((1.0 / np.arange(1, n_max + 1)), cumsumbits)  # 1 x m

    rhovectorNx1 = np.zeros((2**m - 1, 1))
    rIdx1 = 0
    for r in range(m, 0, -1):
        rIdx2 = rIdx1 + 2**(r - 1) - 1
        rhovectorNx1[rIdx1:rIdx2 + 1, 0] = rhovector[r - 1]
        rIdx1 = rIdx2 + 1

    # ----------------------------------------------------------------------
    # Get ordering of the search space - needed for circulant matrix
    # ----------------------------------------------------------------------
    gR = np.ones(2**(m - 2), dtype=int) # gR determines ordering of rows to have circulant matrix
    intMod = 2**m
    for idx in range(1, 2**(m - 2)):
        temp = (gR[idx - 1] * 5) % intMod
        gR[idx] = min(intMod - temp, temp)

    gRows = np.ones(2**(m - 1), dtype=int) # gRows determines* ordering of cols to have circulant matrix
    gRows[-1] = 0
    rowVects = np.ones(2**m - 1, dtype=int) # rowVects  
    gStrtIdx = 0
    vStrtIdx = 0

    for l in range(m, 1, -1):
        gEndIdx = gStrtIdx + 2**(l - 2) - 1
        vEndIdx = vStrtIdx + 2**(l - 1) - 1

        gRow = np.ones(2**(l - 2), dtype=int)
        intMod = 2**l
        for idx in range(1, 2**(l - 2)):
            temp = (gRow[idx - 1] * 5) % intMod
            gRow[idx] = min(intMod - temp, temp)

        gRows[gStrtIdx:gEndIdx + 1] = gRow
        rowV = np.concatenate(([1], np.flip(gRow[1:])))
        doubled = np.concatenate((rowV, rowV))
        rowVects[vStrtIdx:vEndIdx + 1] = 2**(m - l) * doubled

        gStrtIdx = gEndIdx + 1
        vStrtIdx = vEndIdx + 1

    rowVects[-1] = 2**(m - 1)

    # ----------------------------------------------------------------------
    # Set up prodV - where we store information about previous components 
    # ----------------------------------------------------------------------
    prodV = np.ones((2**m - 1, 1))
    prodV = prodV * rhovectorNx1

    # Initial 1D case
    rowV = rowVects / 2**m
    rowV = 1 + coord_weights[0] * kernel(rowV)
    prodV = prodV * rowV[:, None]

    # Set up k0
    k0 = 1 + coord_weights[0] * kernel(0)

    # ----------------------------------------------------------------------
    # Begin search
    # ----------------------------------------------------------------------
    gen_vec = np.ones(d_max, dtype=int)

    for hComp in range(2, d_max + 1):
        wssd = np.zeros(2**(m - 2), dtype=np.float64)

        gamma = coord_weights[hComp - 1]
        omega = lambda x: 1 + gamma * kernel(x)
        k0 = k0 * (1 + gamma * kernel(0))

        curIdx2 = 0
        prodIdx1 = 0
        for l in range(m, 1, -1): # we iterate over decreasing size blocks of powers of two=
            nextIdx2 = curIdx2 + 2**(l - 2)
            prodIdx2 = prodIdx1 + 2**(l - 2)

            curRow = gRows[curIdx2:nextIdx2]
            col = curRow / 2**l 
            fftCol = omega(col).astype(np.complex128) # first column of this circulant matrix block

            pCol = prodV[prodIdx1:prodIdx2, 0].astype(np.complex128) # corresponding section of prodV

            wVector = 2 * np.fft.ifft(np.fft.fft(fftCol) * np.fft.fft(pCol)).real # matrix vector product as fft
            numrep = 2**(m - l)
            wssd = wssd + np.tile(wVector, numrep)

            curIdx2 = nextIdx2
            prodIdx1 = prodIdx2 + 2**(l - 2)

        wssd = wssd + omega(1 / 2) * prodV[-1, 0] # not actually wssd; we avoid subtracting a constant to save precision

        bestIdx = np.uint64(np.argmin(wssd))
        newH = np.uint64(gR[bestIdx])

        # Avoid duplicates
        while newH in gen_vec:
            wssd[bestIdx] = np.inf
            bestIdx = np.uint64(np.argmin(wssd))
            newH = int(gR[bestIdx])

        gen_vec[hComp - 1] = newH

        # set up prodV for next iteration
        rowV = (newH * rowVects) % 2**m
        rowV = rowV / 2**m
        rowV = omega(rowV)
        prodV = prodV * rowV[:, None]

    return gen_vec