"""
The most famous integral in quantum mechanics is the  Feynman path integral.
In principle, the quantum  mechanics of any system (with a classical limit)
can be reduced to a problem in multidimensional integration. Normally there
are too many variables, and the integrand is too peaky to attempt using
something like vegas. For a 1-d system, like the harmonic oscillator, however,
it is feasible.

The path integral has to be discretized (put on time lattice). A description
of how this is done can be found in:

G.P. Lepage, "Lattice QCD for Novices",  
at http://arxiv.org/abs/hep-lat/0506036 (June 2005).

The code here uses a Cython integrand, path_integrand.pyx (for speed), and
extracts the ground state energies and wavefunctions for a regular harmonic
oscillator, and also one with a small amount of anharmonicity --- just for
fun. The results are not exactly  correct because we are using a coarse grid,
and also not taking T to infinity.

Using numerical path integration to solve these  problems has got to be one of
the most inefficient numerical approaches there is. One does it for the fun of
experiencing the path integral as an actual integral (as opposed to a formal
construct). Numerical path integration *is* the method of  choice, however,
for solving really complicated, highly nonlinear problems like QCD; but one
doesn't   use vegas to do this --- see the paper above.
"""
from __future__ import print_function   # makes this work for python2 and 3

import pyximport; pyximport.install()   # compiles path_integrand.pyx

import vegas
import numpy as np 
import math
from path_integrand import Oscillator

DO_WAVEFUNCTIONS = True

SHOW_PLOT = False # True   # plot wavefunction ** 2

def main():
    # seed random numbers so reproducible
    np.random.seed((1, 2))
    
    # initialize path integral
    T = 4.
    ndT = 8
    neval = 25000

    # Harmonic oscillator: V = x ** 2 / 2
    pathint = Oscillator(T=T, ndT=ndT, neval=neval)
    print('Harmonic Oscillator')
    print('===================')

    # compute groundstate energy
    exp_E0T = pathint.correlator()
    # print(exp_E0T.summary())
    E0 = - np.log(exp_E0T) / T
    print('Ground-state energy = %s    Q = %.2f\n' % (E0, exp_E0T.Q))
    
    # pathint.integ.map.show_grid(50)

    if DO_WAVEFUNCTIONS:
        # compute wavefunction ** 2 at points x0
        x0 = np.linspace(0., 2., 6)
        psi2_hosc = pathint.correlator(x0=x0) / exp_E0T.mean
        exact_hosc = np.empty(psi2_hosc.shape, float)

        print('%5s  %-12s %-10s' % ('x', 'psi**2', 'exact'))
        print(30 * '-')
        for i, (x0i, psi2i) in enumerate(zip(x0, psi2_hosc)):
            exact_hosc[i] = np.exp(- x0i ** 2) / np.sqrt(np.pi) #* np.exp(-T / 2.)
            print(
                "%5.1f  %-12s %-10.5f"
                % (x0i, psi2i, exact_hosc[i])
                )
        plot_results(E0, x0, psi2_hosc, T)

    # Anharmonic oscillator: V = x ** 2 / 2 + 0.2 * x ** 4
    pathint = Oscillator(c=0.2, T=T, ndT=ndT, neval=neval)
    print('\n\nAnharmonic Oscillator')
    print(    '=====================')
    
    # compute groundstate energy
    exp_E0T = pathint.correlator()
    E0 = - np.log(exp_E0T) / T
    print('Ground-state energy = %s    Q = %.2f\n' % (E0, exp_E0T.Q))
    

    if DO_WAVEFUNCTIONS:
        # compute wavefunction ** 2 at points x0
        x0 = np.linspace(0., 2., 6)
        psi2_ahosc = pathint.correlator(x0=x0) / exp_E0T.mean

        print('%5s  %-12s' % ('x', 'psi**2'))
        print(19 * '-')
        for x0i, psi2i in zip(x0, psi2_ahosc):
            print("%5.1f  %-12s" % (x0i, psi2i))
        print()

def plot_results(E0, x0, corr, T):
    if not SHOW_PLOT:
        return
    try:
        import matplotlib.pyplot as plt 
    except ImportError:
        return
    def make_plot(x0=x0, E0=E0, corr=corr, T=T):
        corr_mean = np.array([z.mean for z in corr])
        corr_sdev = np.array([z.sdev for z in corr])
        plt.errorbar(x=x0, y=corr_mean, yerr=corr_sdev, fmt='bo', label='path integral')
        x = np.linspace(0,2.,100)
        y = np.exp(-x ** 2) / np.sqrt(np.pi) 
        plt.plot(x, y, 'r:', label='exact')
        plt.legend(('path integral', 'exact'), frameon=False)
        plt.xlabel('$x$')
        plt.ylabel('$|\psi(x)|^2$')
        plt.text(1.4, 0.475, '$E_0 =$ %s' % E0)
        plt.title("Harmonic Oscillator Wavefunction ** 2 (type 'q' to continue)")
        plt.draw()
    def onpress(event):
        if event.key == 'q':
            plt.close()
            return
    plt.gcf().canvas.mpl_connect('key_press_event', onpress)
    make_plot()
    plt.show()


if __name__ == '__main__':
    main()



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
