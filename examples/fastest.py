"""
Three Gaussians spread along the diagonal of a  six-dimensional hypercube.

This coding style for the integrand is almost as simple as slow.py
but much faster because the integrand is coded in Cython and compiled.
Run times are 20x shorter than slow.py.

Runtime is about the same for faster.py and fastest.py. This is because
the cost of evaluating the integrand is almost negligible in both
cases; most of the time is taken by the Integrator as it generates
random points and adapts the grid. The integrand by itself is more than
twice as fast in Cython compared with the numpy-vectorized integrand
(in faster.py), and also much more flexible.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import math
import numpy as np

# compile fastest_integrand.pyx, if necessary
import pyximport
pyximport.install(
    inplace=True,
    setup_args=dict(include_dirs=[np.get_include()])
    )

from fastest_integrand import f_cython

np.random.seed((1,2, 3))   # causes reproducible random numbers

def main():
    # create integrand and integrator
    f = f_cython(dim=6)
    integ = vegas.Integrator(f.dim * [[0, 1]], sync_ran=False)

    # adapt the grid; discard these results
    integ(f, neval=25000, nitn=10)

    # final result; slow down adaptation because
    # already adapted, so increases stability
    result = integ(f, neval=25000, nitn=10, alpha=0.1)

    print(result.summary())


if __name__ == '__main__':
    if True:
        main()
    else:
        import hotshot, hotshot.stats
        prof = hotshot.Profile("vegas.prof")
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load("vegas.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(40)




# Copyright (c) 2013-16 G. Peter Lepage.
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
