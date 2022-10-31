"""
Three Gaussians spread along the diagonal of a  six-dimensional hypercube.

This coding style for the integrand is the  simplest but also gives the
slowest runtime. It is made 3x faster by using nproc=8 processors, but 
is still slower than faster.py and fastest.py (using nproc=1 processor).
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import math
import numpy as np

np.random.seed((1,2, 3))   # causes reproducible random numbers

DIM = 6
NORM_AC = 1. / 0.17720931990702889842 ** DIM
NORM_B = 1. / 0.17724538509027909508 ** DIM

def f(x):
    dx2a = 0
    for d in range(DIM):
        dx2a += (x[d] - 0.25) ** 2
    dx2b = 0
    for d in range(DIM):
        dx2b += (x[d] - 0.5) ** 2
    dx2c = 0
    for d in range(DIM):
        dx2c += (x[d] - 0.75) ** 2
    return (
        math.exp(- 100. * dx2a) * NORM_AC
        + math.exp(-100. * dx2b) * NORM_B
        + math.exp(-100. * dx2c) * NORM_AC
        ) / 3.

def main():
    integ = vegas.Integrator(DIM * [[0, 1]], nproc=8)  # 8 processors

    # adapt the grid; discard these results
    integ(f, neval=25000, nitn=10)

    # final result; slow down adaptation because
    # already adapted, so increases stability
    result = integ(f, neval=25000, nitn=10, alpha=0.1)

    print(result.summary())

if __name__ == '__main__':
    main()




# Copyright (c) 2013-22 G. Peter Lepage.
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
