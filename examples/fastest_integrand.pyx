# cython: language_level=3
# c#ython: profile=True
"""
Cython code for the integrator used in fastest.py.

Note that this achieves better performance than the integrand in faster.py
without the complication of using numpy vectorization.
Some problems are not easily coded using whole-array operators.
"""

# import Cython description of vegas (to derive from vegas.BatchIntegrand)
cimport vegas

# import exp() from C
from libc.math cimport exp

import numpy as np
import vegas
import math


cdef class f_cython(vegas.BatchIntegrand):
    cdef readonly int dim
    cdef double norm_ac
    cdef double norm_b
    def __init__(self, dim):
        self.dim = dim
        self.norm_ac = 1. / 0.17720931990702889842 ** dim
        self.norm_b = 1. / 0.17724538509027909508 ** dim

    def __call__(self, double[:, ::1] x):
        cdef int d, j
        cdef double dx2a, dx2b, dx2c, dx
        cdef double[::1] f = np.empty(x.shape[0], float)
        for j in range(len(f)):
            dx2a = 0
            for d in range(self.dim):
                dx = (x[j, d] - 0.25)
                dx2a += dx * dx
            dx2b = 0
            for d in range(self.dim):
                dx = (x[j, d] - 0.5)
                dx2b += dx * dx
            dx2c = 0
            for d in range(self.dim):
                dx = (x[j, d] - 0.75)
                dx2c += dx * dx
            f[j] = (
                exp(- 100. * dx2a) * self.norm_ac
                + exp(-100. * dx2b) * self.norm_b
                + exp(-100. * dx2c) * self.norm_ac
                ) / 3.
        return f


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
