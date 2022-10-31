import numpy as np
import vegas
import sys

if sys.argv[1:]:
    NPROC = eval(sys.argv[1])   # number of processors
else:
    NPROC = 1

np.random.seed((1,3))   # causes reproducible random numbers

def ridge(x):
    N = 1000
    x0 = np.linspace(0.4, 0.6, N)
    dx2 = 0.0
    for xd in x:
        dx2 += (xd - x0) ** 2
    return np.average(np.exp(-100. * dx2)) *  (100. / np.pi) ** (len(x) / 2.)


def main():
    integ = vegas.Integrator(4 * [[0, 1]], sync_ran=False, nproc=NPROC)
    # adapt
    f = ridge
    integ(f, nitn=10, neval=1e4)
    # final results
    result = integ(f, nitn=10, neval=1e4)
    # print from only one process if using MPI
    if integ.mpi_rank == 0:
        print('result =', result, '    Q = {:.2f}'.format(result.Q))

if __name__ == '__main__':
    main()


# Copyright (c) 2022 G. Peter Lepage.
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