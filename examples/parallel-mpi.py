import numpy as np
import vegas

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
    # convert to MPI integrand
    fparallel = vegas.MPIintegrand(f)
    # adapt
    integ(fparallel, nitn=10, neval=1e3)
    # final results
    result = integ(fparallel, nitn=10, neval=1e3)
    if fparallel.rank == 0:
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
