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
from __future__ import print_function   # makes this work for python2 and 3
import vegas
import math
import gvar as gv
import numpy as np
from outputsplitter import log_stdout, unlog_stdout

# def f(x):
#     dx2 = 0.0
#     for d in range(4):
#         dx2 += (x[d] - 0.5) ** 2
#     f = math.exp(-200 * dx2)
#     return [f, f * x[0], f * x[0] ** 2]

class f(vegas.VecIntegrand):
    def __call__(self, x):
        x = np.asarray(x)
        f = np.empty((x.shape[0], 3), float)
        dx2 = 0.0
        for d in range(4):
            dx2 += (x[:, d] - 0.5) ** 2
        f[:, 0] = np.exp(-200. * dx2)
        f[:, 1] = f[:, 0] * x[:, 0]
        f[:, 2] = f[:, 0] * x[:, 0] ** 2
        return f

class fdict(vegas.VecIntegrand):
    def __call__(self, x):
        x = np.asarray(x)
        f = np.empty((x.shape[0], 3), float)
        dx2 = 0.0
        for d in range(4):
            dx2 += (x[:, d] - 0.5) ** 2
        f[:, 0] = np.exp(-200. * dx2)
        f[:, 1] = f[:, 0] * x[:, 0]
        f[:, 2] = f[:, 0] * x[:, 0] ** 2
        return {'1':f[:, 0], 'x':f[:, 1], 'x**2':f[:, 2]}

def main():
    print(gv.ranseed(
        (1814855126, 100213625, 262796317)
        ))

    log_stdout('eg3a.out')
    integ = vegas.Integrator(4 * [[0, 1]])

    # adapt grid
    training = integ(f(), nitn=10, neval=2000)

    # evaluate multi-integrands
    result = integ(f(), nitn=10, neval=10000)
    print('I[0] =', result[0], '  I[1] =', result[1], '  I[2] =', result[2])
    print('Q = %.2f\n' % result.Q)
    print('<x> =', result[1] / result[0])
    print(
        'sigma_x**2 = <x**2> - <x>**2 =',
        result[2] / result[0] - (result[1] / result[0]) ** 2
        )
    print('\ncorrelation matrix:\n', gv.evalcorr(result))
    unlog_stdout()
    r = gv.gvar(gv.mean(result), gv.sdev(result))
    print(r[1] / r[0])
    print((r[1] / r[0]).sdev / (result[1] / result[0]).sdev)
    print(r[2] / r[0] - (r[1] / r[0])**2)
    print((r[2] / r[0] - (r[1] / r[0])**2).sdev / (result[2] / result[0] - (result[1] / result[0]) ** 2).sdev)
    print(result.summary())

    # do it again for a dictionary
    print(gv.ranseed(
        (1814855126, 100213625, 262796317)
        ))
    integ = vegas.Integrator(4 * [[0, 1]])

    # adapt grid
    training = integ(f(), nitn=10, neval=2000)

    # evaluate the integrals
    result = integ(fdict(), nitn=10, neval=10000)
    log_stdout('eg3b.out')
    print(result)
    print('Q = %.2f\n' % result.Q)
    print('<x> =', result['x'] / result['1'])
    print(
        'sigma_x**2 = <x**2> - <x>**2 =',
        result['x**2'] / result['1'] - (result['x'] / result['1']) ** 2
        )
    unlog_stdout()

if __name__ == '__main__':
    main()