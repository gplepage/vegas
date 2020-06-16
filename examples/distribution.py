"""
Illustrates how to calculate a distribution dI with vegas. 
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import numpy as np

np.random.seed((1,2,3))   # causes reproducible random numbers

RMAX = (2 * 0.5**2) ** 0.5

def fcn(x):
    dx2 = 0.0
    for d in range(2):
        dx2 += (x[d] - 0.5) ** 2
    I = np.exp(-dx2)
    # add I to appropriate bin in dI
    dI = np.zeros(5, dtype=float)
    dr = RMAX / len(dI)
    j = int(dx2 ** 0.5 / dr)
    dI[j] = I
    return dict(I=I, dI=dI)

def main():
    integ = vegas.Integrator(2 * [(0,1)])

    # results returned in a dictionary
    result = integ(fcn)
    print(result.summary())
    print('   I =', result['I'])
    print('dI/I =', result['dI'] / result['I'])
    print('sum(dI/I) =', sum(result['dI']) / result['I'])

if __name__ == "__main__":
    main()


# Copyright (c) 2020 G. Peter Lepage.
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
