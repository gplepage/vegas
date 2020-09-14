# cython: language_level=3
# c#ython: profile=True
"""
Cython code for the integrand used in path-integral.py.

class PathIntegral is a base class for classes that
do path integrals for 1-d systems. class Oscillator
is derived from it and specifies a specific potential.
"""

# import Cython description of vegas
cimport vegas

# import exp(),  tan() from C
from libc.math cimport exp, tan

import collections
import numpy as np
import vegas


cdef class PathIntegrand(vegas.BatchIntegrand):
    """ Integrand for path integral corresponding to < x0 | exp(-H*T) | x0 >

    This class creates a vegas integrand whose integrals correspond to
    the quantum mechanical expectation values < x0 | exp(-H*T) | x0 >
    where H = p**2/2m + V(x) is a 1-d Hamiltonian for a particle with
    mass m, moving in potential V(x). Values are given for a list of
    x0 values, as well as for the integral over all x0.

    Typical usage is:

        integrand = PathIntegrand(V, x0list=[0.0, 0.2], T=4, ndT=6)
        integ = vegas.Integrator(integrand.region)
        results = integ(integrand, neval=100000, nitn=10)

    Then

        results['exp(-E0*T)']

    is the integral over all values of x0 (and therefore equals exp(-E0*T) if
    T is big enough), while

        results['exp(-E0*T) * psi(x0)**2']

    is the value of the expectation value corresponding to the x0 values in
    x0list (and therefore

        results['exp(-E0*T) * psi(x0)**2'] / results['exp(-E0*T)']

    gives the wavefunction squared at each x0, provided again that T is big
    enough).

    Parameters:
        V ...... potential energy (function of x)
        T ...... (Euclidean) time
        x0list . list of x0 values
        ndT .... total number of time steps
        m ...... mass
        xscale . typical length scale for ground state wavefunction
    """
    cdef readonly object V
    cdef readonly int ndT
    cdef readonly double T
    cdef readonly int neval
    cdef readonly int nitn
    cdef readonly double m
    cdef readonly double xscale
    cdef readonly double norm
    cdef readonly double norm_x0
    cdef readonly double[::1] x0list
    cdef readonly double[::1] Vx0list
    cdef readonly object region

    def __init__(self, V, T, x0list=[], ndT=10, m=1, xscale=1.):
        self.V = V
        self.ndT = ndT
        self.T = T
        self.m = m
        self.xscale = xscale
        self.x0list = np.array(x0list)
        self.Vx0list = self.V(np.asarray(self.x0list))
        self.norm = (self.m * self.ndT / 2. / np.pi / T) ** (self.ndT / 2.)
        self.norm_x0 = self.norm / np.pi
        self.region = self.ndT * [[-np.pi/2, np.pi/2]]

    def __call__(self, theta):
        """ integrand for the path integral """
        cdef int i, j
        cdef double S, Smiddle, jac, jfac, jac_x0
        cdef double a = self.T / self.ndT
        cdef double m_2a = self.m / 2. / a
        cdef double[:, ::1] x = np.empty(theta.shape, float)
        cdef double[:, ::1] Vx
        cdef double[::1] x0list = np.empty(len(self.x0list) + 1, float)
        cdef double[::1] Vx0list = np.empty(len(x0list), float)
        cdef double[:, ::1] f = np.empty((theta.shape[0], len(x0list)), float)

        # set up non-changing part of x0list
        x0list[1:] = self.x0list
        Vx0list[1:] = self.Vx0list

        # map back to range -oo to +oo and compute V(x)
        #
        # create and store all x values at same time
        # so can call self.V with very large number of x values
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                x[i, j] = self.xscale * tan(theta[i, j])
        Vx = self.V(np.asarray(x))

        # loop on integration points
        for i in range(theta.shape[0]):
            jac = self.norm
            jac_x0 = self.norm_x0
            for j in range(self.ndT):
                jfac = (self.xscale + x[i, j] ** 2 / self.xscale)
                jac *= jfac
                if j > 0:
                    jac_x0 *= jfac

            # compute the action for central points
            Smiddle = a * Vx[i, -1]
            for j in range(1, self.ndT-1):
                Smiddle += m_2a * (x[i, j + 1] - x[i, j]) ** 2 + a * Vx[i, j]

            # add in end points, with periodic BCs
            x0list[0] = x[i, 0]
            Vx0list[0] = Vx[i, 0]
            for j in range(len(x0list)):
                S = Smiddle + (
                    m_2a * ( (x[i, 1] - x0list[j])**2 + (x0list[j] - x[i, -1])**2 )
                    + a * Vx0list[j]
                    )
                f[i, j] = (jac if j == 0 else jac_x0) * exp(-S)
        # repackage as dictionary
        ans = collections.OrderedDict()
        ans['exp(-E0*T)'] = f[:, 0]
        ans['exp(-E0*T) * psi(x0)**2'] = f[:,1:]
        return ans



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
