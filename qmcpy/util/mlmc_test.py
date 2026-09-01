import qmcpy as qp
import numpy as np

def mlmc_test(
    integrand,
    n = 20000,
    l = 8,
    n_init = 200,
    rmse_tols = np.array([.005, 0.01, 0.02, 0.05, 0.1]),
    levels_min = 2,
    levels_max = 10,
    ):
    r"""
    Multilevel Monte Carlo test routine.

    Examples:
        >>> fo = qp.FinancialOption(
        ...     sampler=qp.IIDStdUniform(seed=7),
        ...     option = "ASIAN",
        ...     asian_mean = "GEOMETRIC",
        ...     volatility = 0.2, 
        ...     start_price = 100, 
        ...     strike_price = 100, 
        ...     interest_rate = 0.05, 
        ...     t_final = 1)
        >>> print('Exact Value: %s'%fo.get_exact_value_inf_dim())
        Exact Value: 5.546818633789201
        >>> mlmc_test(fo)
        Convergence tests, kurtosis, telescoping sum check using N =  20000 samples
            l              ave(Pf-Pc)     ave(Pf)        var(Pf-Pc)     var(Pf)        kurtosis       check          cost
            0              5.4486e+00     5.4486e+00     5.673e+01      5.673e+01      0.00e+00       0.00e+00       2.00e+00
            1              1.4925e-01     5.5937e+00     3.839e+00      5.838e+01      5.48e+00       1.14e-02       4.00e+00
            2              3.5921e-02     5.6024e+00     9.585e-01      5.990e+01      5.59e+00       7.86e-02       8.00e+00
            3              8.7217e-03     5.5128e+00     2.332e-01      5.828e+01      5.36e+00       2.92e-01       1.60e+01
            4              1.9773e-03     5.6850e+00     6.021e-02      6.081e+01      5.46e+00       5.12e-01       3.20e+01
            5              9.5925e-04     5.5628e+00     1.512e-02      5.939e+01      5.37e+00       3.71e-01       6.40e+01
            6              8.5998e-04     5.5706e+00     3.773e-03      5.995e+01      5.48e+00       2.10e-02       1.28e+02
            7              1.3592e-04     5.4359e+00     9.285e-04      5.808e+01      5.51e+00       4.13e-01       2.56e+02
            8              3.4520e-05     5.5322e+00     2.313e-04      5.881e+01      5.57e+00       2.96e-01       5.12e+02
        Linear regression estimates of MLMC parameters
            alpha = 1.617207  (exponent for MLMC weak convergence)
            beta  = 2.000355  (exponent for MLMC variance)
            gamma = 1.000000  (exponent for MLMC cost)
        MLMC complexity tests
            rmse_tol       value          mlmc_cost      std_cost       savings        N_l
            5.000e-03      5.545e+00      3.339e+07      1.038e+08      3.11           8605392      1566846      559701       198886       70359        
            1.000e-02      5.539e+00      7.272e+06      1.243e+07      1.71           2009192      365451       130781       46623        
            2.000e-02      5.549e+00      1.827e+06      3.108e+06      1.70           503397       91821        33196        11736        
            5.000e-02      5.474e+00      2.324e+05      2.556e+05      1.10           71432        13143        4617         
            1.000e-01      5.466e+00      6.220e+04      6.389e+04      1.03           19477        3361         1225         
        
    Args:
        integrand (AbstractIntegrand): multilevel integrand
        n (int): number of samples for convergence tests
        l (int): number of levels for convergence tests
        n_init (int): initial number of samples for MLMC calcs
        rmse_tols (np.ndarray): desired accuracy array for MLMC calcs
        levels_min (int): minimum number of levels for MLMC calcs
        levels_max (int): maximum number of levels for MLMC calcs
    """
    # first, convergence tests
    n = 100*np.ceil(n/100) # make N a multiple of 100
    print('Convergence tests, kurtosis, telescoping sum check using N =%7d samples'%n)
    print('    %-15s%-15s%-15s%-15s%-15s%-15s%-15s%s'\
        %('l','ave(Pf-Pc)','ave(Pf)','var(Pf-Pc)','var(Pf)','kurtosis','check','cost'))
    del1 = np.array([])
    del2 = np.array([])
    var1 = np.array([])
    var2 = np.array([])
    kur1 = np.array([])
    chk1 = np.array([])
    cost = np.array([])
    integrand_spawns = integrand.spawn(levels=np.arange(l+1))
    for ll in range(l+1):
        sums = np.zeros(6)
        cst = 0
        integrand_spawn = integrand_spawns[ll]
        for j in range(1,101):
            # evaluate integral at sampleing points samples
            samples = integrand_spawn.discrete_distrib.gen_samples(n=n/100)
            Pc,Pf = integrand_spawn.f(samples)
            dP = Pf-Pc
            sums_j = np.array([
                np.sum(dP),
                np.sum(dP**2),
                np.sum(dP**3),
                np.sum(dP**4),
                np.sum(Pf),
                np.sum(Pf**2),
            ])
            cst_j = integrand_spawn.cost*(n/100)
            sums = sums + sums_j/n
            cst = cst + cst_j/n
        if ll == 0:
            kurt = 0.
        else:
            kurt = ( sums[3] - 4*sums[2]*sums[0] + 6*sums[1]*sums[0]**2 -
                     3*sums[0]*sums[0]**3 ) /  (sums[1]-sums[0]**2)**2.
        cost = np.hstack((cost, cst))
        del1 = np.hstack((del1, sums[0]))
        del2 = np.hstack((del2, sums[4]))
        var1 = np.hstack((var1, sums[1]-sums[0]**2))
        var2 = np.hstack((var2, sums[5]-sums[4]**2))
        var2 = np.maximum(var2, 1e-10) # fix for cases with var=0
        kur1 = np.hstack((kur1, kurt))
        if ll == 0:
            check = 0
        else:
            check = abs( del1[ll] + del2[ll-1] - del2[ll]) / \
                    ( 3.*( np.sqrt(var1[ll]) + np.sqrt(var2[ll-1]) + np.sqrt(var2[ll]) ) / np.sqrt(n))
        chk1 = np.hstack((chk1, check))

        print('    %-15d%-15.4e%-15.4e%-15.3e%-15.3e%-15.2e%-15.2e%.2e'\
              %(ll,del1[ll],del2[ll],var1[ll],var2[ll],kur1[ll],chk1[ll],cst))
    # print out a warning if kurtosis or consistency check looks bad
    if kur1[-1] > 100.:
        print('WARNING: kurtosis on finest level = %f'%kur1[-1])
        print(' indicates MLMC correction dominated by a few rare paths;')
        print(' for information on the connection to variance of sample variances,')
        print(' see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n')
    if np.max(chk1) > 1.:
        print('WARNING: maximum consistency error = %f'%max(chk1))
        print(' indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied;')
        print(' to be more certain, re-run mlmc_test with larger N\n')
    # use linear regression to estimate alpha, beta and gamma
    l1 = 2
    l2 = l+1
    x = np.ones((l2+1-l1,2))
    x[:,1] = np.arange(l1,l2+1)
    pa = np.linalg.lstsq(x,np.log2(np.absolute(del1[(l1-1):l2])),rcond=None)[0]
    alpha = -pa[1]
    pb = np.linalg.lstsq(x,np.log2(np.absolute(var1[(l1-1):l2])),rcond=None)[0]
    beta = -pb[1]
    pg = np.linalg.lstsq(x,np.log2(np.absolute(cost[(l1-1):l2])),rcond=None)[0]
    gamma = pg[1]
    print('Linear regression estimates of MLMC parameters')
    print('    alpha = %f  (exponent for MLMC weak convergence)'%alpha)
    print('    beta  = %f  (exponent for MLMC variance)'%beta)
    print('    gamma = %f  (exponent for MLMC cost)'%gamma)
    #second, mlmc complexity tests
    print('MLMC complexity tests')
    print('    %-15s%-15s%-15s%-15s%-15s%s'\
        %('rmse_tol','value','mlmc_cost','std_cost','savings','N_l'))
    alpha = np.maximum(alpha,0.5)
    beta  = np.maximum(beta,0.5)
    theta = 0.25
    for i in range(len(rmse_tols)):
        mlmc_qmcpy = qp.CubMLMC(integrand,
            rmse_tol = rmse_tols[i],
            n_init = n_init,
            levels_min = levels_min,
            levels_max = levels_max,
            alpha0 = alpha,
            beta0 = beta,
            gamma0 = gamma)
        sol,data = mlmc_qmcpy.integrate()
        p = data.solution
        nl = data.n_level
        cl = data.cost_per_sample
        mlmc_cost = sum(nl*cl)
        idx = np.minimum(len(var2),len(nl))-1
        std_cost = var2[idx]*cl[-1] / ((1.-theta)*rmse_tols[i]**2)
        print('    %-15.3e%-15.3e%-15.3e%-15.3e%-15.2f%s'\
            %(rmse_tols[i], p, mlmc_cost, std_cost, std_cost/mlmc_cost,''.join('%-13d'%nli for nli in nl)))
