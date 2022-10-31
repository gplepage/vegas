"""
Tests nproc>1.

N.B. nproc>1 slows things down here because integrand not sufficiently costly.
"""
import numpy as np
import vegas
numpy = np

DIM = 3
NPROC = 2
N = 10
NEVAL = 3e3
NITN = 5
NS = 1_000

@vegas.lbatchintegrand
def fl(x):
    ans = 0.0
    x0list = np.linspace(0.4, 0.6, N)
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM):
            dx2 += (x[:, d] - x0) ** 2
        ans += np.exp(-100. * dx2)
    ans *= (100. / np.pi) ** (DIM / 2.) / N
    return ans

def fl_samples():
    x0list = np.linspace(0.4, 0.6, N)
    return np.concatenate([                        
        np.random.normal(loc=x0, scale=(2/100.)**0.5, size=(NS, DIM))
        for x0 in x0list
        ])

def _fr(x):
    ans = 0.0
    x0list = np.linspace(0.4, 0.6, N)
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM):
            dx2 += (x[d] - x0) ** 2
        ans += np.exp(-100. * dx2)
    ans *= (100. / np.pi) ** (DIM / 2.) / N
    return ans
fr = vegas.RBatchIntegrand(_fr)

def _fl_a(x):
    norm = fl(x)
    normx = norm * x[:, 0]
    ans = np.zeros((x.shape[0], 2), float)
    ans[:, 0] = fl(x)
    ans[:, 1] = ans[:, 0] * x[:, 0]
    return ans 
fl_a = vegas.LBatchIntegrand(_fl_a)

@vegas.rbatchintegrand
def fr_a(x):
    norm = fr(x)
    return [norm, norm * x[0]]

@vegas.lbatchintegrand 
def fl_d(x):
    norm = fl(x)
    return dict(norm=norm, normx=norm * x[:, 0])

def _fr_d(x):
    norm = fr(x)
    return dict(norm=norm, normx=norm * x[0])
fr_d = vegas.RBatchIntegrand(_fr_d)

@vegas.lbatchintegrand
def fl_jac(x, jac):
    ans0 = 0.0
    x0list = np.linspace(0.4, 0.6, N)
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM):
            dx2 += (x[:, d] - x0) ** 2
        ans0 += np.exp(-100. * dx2)
    ans0 *= (100. / np.pi) ** (DIM / 2.) / N
    ans1 = 0.0
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM-1):
            dx2 += (x[:, d] - x0) ** 2
        ans1 += np.exp(-100. * dx2)
    ans1 *= (100. / np.pi) ** ((DIM - 1) / 2.) / N / jac[:, -1]
    ans = np.zeros((x.shape[0], 2), float)
    ans[:, 0] = ans0
    ans[:, 1] = ans1
    return ans

@vegas.rbatchintegrand
def fr_jac(x, jac):
    ans0 = 0.0
    x0list = np.linspace(0.4, 0.6, N)
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM):
            dx2 += (x[d] - x0) ** 2
        ans0 += np.exp(-100. * dx2)
    ans0 *= (100. / np.pi) ** (DIM / 2.) / N
    ans1 = 0.0
    for x0 in x0list:
        dx2 = 0.0
        for d in range(DIM-1):
            dx2 += (x[d] - x0) ** 2
        ans1 += np.exp(-100. * dx2)
    ans1 *= (100. / np.pi) ** ((DIM - 1) / 2.) / N / jac[-1]    
    return [ans0, ans1]

def main():
    # seed the random number generator so results reproducible
    np.random.seed((1, 2))

    # adapt integrator using adapt_to_samples
    integ = vegas.Integrator(DIM * [[0, 1]], nhcube_batch=10000, nproc=NPROC)
    xs = fl_samples()
    integ.map.adapt_to_samples(xs, fl(xs), nitn=5, nproc=NPROC)
    # integ.map.show_grid(30)

    w = integ(fl, nitn=NITN, neval=NEVAL)
    print(w.summary())

    # final results (adaptation turned off)
    for f in [fl, fr, fl_a, fr_a, fl_d, fr_d]:
        result = integ(f, nitn=NITN, neval=NEVAL, adapt=False)
        print('result = %s    Q = %.2f' % (result, result.Q))
        try:
            print('<x> =', result.flat[1] / result.flat[0])
        except:
            pass
    for f in [fl_jac, fr_jac][:]:
        result = integ(f, nitn=NITN, neval=NEVAL, adapt=False, uses_jac=True)
        print('result = %s    Q = %.2f' % (result, result.Q))

if __name__ == '__main__':
    main()

# Copyright (c) 2022 G. Peter Lepage.
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
