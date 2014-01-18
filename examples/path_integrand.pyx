# c#ython: profile=True 
"""
Cython code for the integrand used in path-integral.py.  

class PathIntegral is a base class for classes that    
do path integrals for 1-d systems. class Oscillator 
is derived from it and specifies a specific potential.  
"""

# import Cython description of vegas
cimport vegas

# import exp(), tan() from C
from libc.math cimport exp, tan

import numpy as np
import vegas
import math

cdef class PathIntegral(vegas.BatchIntegrand): 
    """ Computes < x0 | exp(-H*T) | x0 > 

    Parameters:
        T ...... (Euclidean) time 
        x0 ..... starting and ending position (if None => integrate over x0)
        ndT .... total number of time steps
        m ...... mass
        xscale . typical length scale for ground state wavefunction
        neval .. number of evaluations per iteration (vegas)
        nitn ... number of iterations (vegas)
    """
    cdef readonly int ndT 
    cdef readonly double T
    cdef readonly int neval 
    cdef readonly int nitn
    cdef readonly double m
    cdef readonly double xscale
    cdef readonly double norm
    cdef readonly double[::1] x
    cdef readonly object integ

    def __init__(self, T, ndT=10, m=1, xscale=1., neval=25000, nitn=10):
        self.ndT = ndT
        self.T = T 
        self.neval = neval 
        self.nitn = nitn
        self.m = m
        self.xscale = xscale
        self.x = np.empty(self.ndT + 1, float)
        self.norm = (self.m * self.ndT / 2. / math.pi / T) ** (self.ndT / 2.)

    cdef double V(self, double x):
        """ Derived classes needs to fill this in. """
        raise NotImplementedError('need to define V')

    def __call__(self, theta):
        """ integrand for the path integral """
        cdef int i, j 
        cdef double S, jac
        cdef double a = self.T / self.ndT
        cdef double m_2a = self.m / 2. / a
        cdef double[::1] f = np.empty(theta.shape[0], float)
        for i in range(len(f)):
            # map back to range -oo to +oo; compute Jacobian
            jac = self.norm
            for j in range(theta.shape[1]):
                self.x[j + 1] = self.xscale * tan(theta[i, j])
                jac *= (self.xscale + self.x[j + 1] ** 2 / self.xscale)
            
            # enforce periodic boundary condition
            # N.B. self.x[-1] is integrated over if x0=None
            #      but otherwise is set equal to x0 (by set_region)
            self.x[0] = self.x[-1]
            
            # compute the action
            S = 0.
            for j in range(self.ndT):
                S += m_2a * (self.x[j + 1] - self.x[j]) ** 2 + a * self.V(self.x[j])
            f[i] = exp(-S) * jac
        return f

    def correlator(self, x0=None):
        """ Compute < x0 | exp(-H*T) | x0 > for array of x0 values.

        If x0 is None, integrate over x0.
        """
        if x0 is None or len(x0) == 0:
            # integrate over endpoints -> exp(-E0*T)
            self.integ = vegas.Integrator(self.ndT * [[-math.pi/2, math.pi/2]])
            # train integrator
            self.integ(self, neval=self.neval, nitn=self.nitn)
            # final integral
            return self.integ(self, neval=self.neval, nitn=self.nitn, alpha=0.1)
        else:
            # set endpoints equal to x0[i] -> wavefunction(x0[i]) ** 2 * exp(-E0*T)
            ans = []
            self.integ = vegas.Integrator((self.ndT - 1) * [[-math.pi/2, math.pi/2]])
            # train integrator
            self.x[0] = x0[0]
            self.x[-1] = x0[0]
            self.integ(self, neval=self.neval, nitn=self.nitn)
            # do final integrals
            for x0i in x0:
                self.x[0] = x0i
                self.x[-1] = x0i
                ans.append(
                    self.integ(self, neval=self.neval, nitn=self.nitn, alpha=0.1)
                    )
            return np.array(ans)


cdef class Oscillator(PathIntegral):
    """ V(x) = x**2 / 2. + c * x**4 

    Exact E0 is 0.5 for c=0; 0.602405 for c=0.2.
    """
    cdef double c

    def __init__(self, c=0.0, *args, **kargs):
        super(Oscillator, self).__init__(*args, **kargs)
        self.c = c

    cdef double V(self, double x):
        return x * x / 2. + self.c * x * x * x * x 



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
