import multiprocessing
import numpy as np
import vegas

class parallelintegrand(vegas.BatchIntegrand):
    """ Convert (batch) integrand into multiprocessor integrand.

    Integrand should return a numpy array.
    """
    def __init__(self, fcn, nproc=4):
        " Save integrand; create pool of nproc processes. "
        self.fcn = fcn
        self.nproc = nproc
        self.pool = multiprocessing.Pool(processes=nproc)
    def __del__(self):
        " Standard cleanup. "
        self.pool.close()
        self.pool.join()
    def __call__(self, x):
        " Divide x into self.nproc chunks, feeding one to each process. "
        nx = x.shape[0] // self.nproc + 1
        # launch evaluation of self.fcn for each chunk, in parallel
        po = self.pool.map_async(
            self.fcn,
            [x[i*nx : (i+1)*nx] for i in range(self.nproc)],
            1,
            )
        # harvest the results
        results = po.get()
        # convert list of results into a single numpy array
        return np.concatenate(results)

def f(x):
    dim = 4
    N = 10
    ans = np.zeros(x.shape[0], np.double)
    for i in range(x.shape[0]):
        for j in range(N):
            x0 = j / (N - 1.)
            dx2 = 0.0
            for d in range(dim):
                dx2 += (x[i, d] - x0) ** 2
            ans[i] += np.exp(-100. * dx2)
        ans[i] *= (100. / np.pi) ** 2 / N
    return ans

def main():
    # seed the random number generator so results reproducible
    np.random.seed((1, 2, 3))
    integ = vegas.Integrator(4 * [[0, 1]])
    # adapt
    fparallel = parallelintegrand(f, 4)
    integ(fparallel, nitn=10, neval=1e3)
    # final results
    result = integ(fparallel, nitn=10, neval=1e3)
    # result should be approximately 0.851
    print('result = %s    Q = %.2f' % (result, result.Q))

if __name__ == '__main__':
    main()

# Copyright (c) 2014 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
