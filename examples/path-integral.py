"""
The most famous integral in quantum mechanics is the  Feynman path integral.
In principle, the quantum  mechanics of any system (with a classical limit)
can be reduced to a problem in multidimensional integration. Normally there
are too many variables, and the integrand is too peaky to attempt using
something like vegas. For a 1-d system, like the harmonic oscillator, however,
it is feasible.

The path integral has to be discretized (put on time lattice). A description
of how this is done can be found in (try googling hep-lat/0506036):

G.P. Lepage, "Lattice QCD for Novices",
at http://arxiv.org/abs/hep-lat/0506036 (June 2005).

The code here uses a Cython integrand, path_integrand.pyx (for speed), and
extracts the ground state energies and wavefunctions for a regular harmonic
oscillator, and also one with a small amount of anharmonicity.
The results are not exactly  correct because we are using a coarse grid,
and also not taking T to infinity.

Using numerical path integration to solve these  problems has got to be one of
the most inefficient numerical approaches there is. One does it for the fun of
experiencing the path integral as an actual integral (as opposed to a formal
construct). Numerical path integration *is* the method of choice, however,
for solving really complicated, highly nonlinear problems like QCD; but one
doesn't use vegas to do this --- see the paper above.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import numpy as np
import math
import sys
import gvar as gv

# compiles path_integrand.pyx, if needed
import pyximport
pyximport.install(
    inplace=True,
    setup_args=dict(include_dirs=[np.get_include()]),
    )

from path_integrand import PathIntegrand


if sys.argv[1:]:
    SHOW_PLOT = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_PLOT = True


def main():
    # seed random numbers so reproducible
    np.random.seed((1,))

    # Harmonic oscillator: V = x ** 2 / 2
    print('Harmonic Oscillator')
    print('===================')
    E0_sho = analyze_theory(V_sho, x0list=np.linspace(0, 2., 6), plot=True)
    print('\n')

    # Anharmonic oscillator: V = x**2 /2 + 0.2 * x ** 4
    print('Anharmonic Oscillator')
    print('=====================')
    E0_aho = analyze_theory(V_aho, x0list=[], plot=False)

    print(
        'E0(aho)/E0(sho) =', E0_aho / E0_sho,
        '    exact =', 0.602405 / 0.5, '\n'
        )

def V_sho(x):
    """ Harmonic oscillator potential => E0 = 0.5. """
    return x ** 2 / 2.


def V_aho(x):
    """ Slightly anharmonic oscillator potential => E0 =  0.602405. """
    return x ** 2 / 2. + 0.2 * x ** 4


def analyze_theory(V, x0list=[], plot=False):
    """ Extract ground-state energy E0 and psi**2 for potential V. """
    # initialize path integral
    T = 4.
    ndT = 8.         # use larger ndT to reduce discretization error (goes like 1/ndT**2)
    neval = 3e5   # should probably use more evaluations (10x?)
    nitn = 6
    alpha = 0.1     # damp adaptation

    # create integrator and train it (no x0list)
    integrand = PathIntegrand(V=V, T=T, ndT=ndT)
    integ = vegas.Integrator(integrand.region, alpha=alpha)
    integ(integrand, neval=neval, nitn=nitn / 2, alpha=2 * alpha)

    # evaluate path integral with trained integrator and x0list
    integrand = PathIntegrand(V=V, x0list=x0list, T=T, ndT=ndT)
    results = integ(integrand, neval=neval, nitn=nitn, alpha=alpha)
    print(results.summary())
    E0 = -np.log(results['exp(-E0*T)']) / T
    print('Ground-state energy = %s    Q = %.2f\n' % (E0, results.Q))

    if len(x0list) <= 0:
        return E0
    psi2 = results['exp(-E0*T) * psi(x0)**2'] / results['exp(-E0*T)']
    print('%5s  %-12s %-10s' % ('x', 'psi**2', 'sho-exact'))
    print(27 * '-')
    for i, (x0i, psi2i) in enumerate(zip(x0list, psi2)):
        exact = np.exp(- x0i ** 2) / np.sqrt(np.pi) #* np.exp(-T / 2.)
        print(
            "%5.1f  %-12s %-10.5f"
            % (x0i, psi2i, exact)
            )
    if plot:
        plot_results(E0, x0list, psi2, T)
    return E0


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
        plt.plot(x, y, 'r:', label='sho exact')
        plt.legend(frameon=False)
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
