"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import math
import numpy
import sys

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True

def f(x):
    dx2 = 0
    for d in range(4):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088

def main():
    # seed the random number generator so results reproducible
    numpy.random.seed((1, 2, 3))

    # assign integration volume to integrator
    integ = vegas.Integrator([[-1., 1.], [0., 1.], [0., 1.], [0., 1.]])

    # adapt to the integrand; discard results
    integ(f, nitn=5, neval=1000)

    # do the final integral
    result = integ(f, nitn=10, neval=1000)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))
    if SHOW_GRID:
        integ.map.show_grid(20)

if __name__ == '__main__':
    main()


# Copyright (c) 2013-14 G. Peter Lepage.
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
