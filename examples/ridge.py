import numpy as np
import vegas

np.random.seed((1,3))   # causes reproducible random numbers

# compile ridge_integrand.pyx, if not already compiled when imported
import pyximport
pyximport.install(inplace=True)

import ridge_integrand

def main():
    integ = vegas.Integrator(4 * [[0, 1]], sync_ran=True)
    # adapt
    # f = vegas.batchintegrand(ridge_integrand.fbatch)
    f = ridge_integrand.f
    integ(f, nitn=10, neval=1e4)
    # final results
    result = integ(f, nitn=10, neval=1e4)
    if integ.mpi_rank == 0:
        # result should be approximately 0.851
        print('result = %s    Q = %.2f' % (result, result.Q))

if __name__ == '__main__':
    main()
