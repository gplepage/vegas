"""
Three Gaussians spread along the diagonal of a  six-dimensional hypercube.

This coding style for the integrand is more complicated than slow.py but
leads to run times that are 20x shorter because the integrand is expressed
in terms of numpy arrays and whole-array operations, thereby greatly reducing
the overhead from Python.

Compare performance with slow.py and fastest.py.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import math
import numpy as np

np.random.seed((1,2,3))   # causes reproducible random numbers

@vegas.batchintegrand
class f_batch:
    def __init__(self, dim):
        self.dim = dim
        self.norm_ac = 1. / 0.17720931990702889842 ** dim
        self.norm_b = 1. / 0.17724538509027909508 ** dim
        
    def __call__(self, x):
        dx2a = 0
        for d in range(self.dim):
            dx2a += (x[:, d] - 0.25) ** 2
        dx2b = 0
        for d in range(self.dim):
            dx2b += (x[:, d] - 0.5) ** 2
        dx2c = 0
        for d in range(self.dim):
            dx2c += (x[:, d] - 0.75) ** 2
        return (
            np.exp(- 100. * dx2a) * self.norm_ac
            + np.exp(-100. * dx2b) * self.norm_b
            + np.exp(-100. * dx2c) * self.norm_ac
            ) / 3.

def main():
    # create integrand
    f = f_batch(dim=6)

    integ = vegas.Integrator(f.dim * [[0, 1]])

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
