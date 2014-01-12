""" Introduction
--------------------
This package provides tools for estimating multidimensional 
integrals numerically using an enhanced version of 
the adaptive Monte Carlo |vegas| algorithm (G. P. Lepage, 
J. Comput. Phys. 27(1978) 192).

A |vegas| code generally involves two objects, one representing
the integrand and the other representing an integration 
operator for a particular multidimensional volume. A typical
code sequence for a D-dimensional integral has the structure::

    # create the integrand
    def f(x):
        ... compute the integrand at point x[d] d=0,1...D-1 
        ...

    # create an integrator for volume with 
    # xl0 <= x[0] <= xu0, xl1 <= x[1] <= xu1 ...
    integration_region = [[xl0, xu0], [xl1, xu1], ...]
    integrator = vegas.Integrator(integration_region)

    # do the integral and print out the result
    result = integrator(f, nitn=10, neval=10000)
    print(result)

The algorithm iteratively adapts to the integrand over
``nitn`` iterations, each of which uses at most ``neval``
integrand samples to generate a Monte Carlo estimate 
of the integral. The final result is the weighted 
average of the results fom all iterations.

The integrator remembers how it adapted to ``f(x)``
and uses this information as its starting point if it is reapplied 
to ``f(x)`` or applied to some other function ``g(x)``.
An integrator's state can be archived for future applications
using Python's :mod:`pickle` module.

See the extensive Tutorial in the first section of the |vegas| documentation.
"""

# Created by G. Peter Lepage (Cornell University) in 12/2013.
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

from ._vegas import RAvg, RAvgArray, AdaptiveMap, Integrator, BatchIntegrand
from ._vegas import reporter, gvar, have_gvar, batchintegrand
from ._version import version as __version__
# legacy names:
from ._vegas import vecintegrand, VecIntegrand
