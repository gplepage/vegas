# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013 G. Peter Lepage. 
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
from outputsplitter import log_stdout, unlog_stdout 

SAVE_OUTPUT = True

def f(x): 
    dx2 = 0 
    for i in range(4): 
        dx2 += (x[i] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088

def g(x):
    return x[0] * f(x)

def f_sphere(x):
    dx2 = 0
    for i in range(4): 
        dx2 += (x[i] - 0.5) ** 2
    if dx2 < 0.2 ** 2:
        return math.exp(-dx2 * 100.) * [1027.5938263789689227 ,1115.3539360527281318, 3834.4215518273048636][1]
    else:
        return 0.0

def f2(x): 
    dx2 = 0 
    for i in range(4): 
        dx2 += (x[i] - 1/3.) ** 2
    ans = math.exp(-dx2 * 100.) * 1013.2167575422921535
    dx2 = 0 
    for i in range(4): 
        dx2 += (x[i] - 2/3.) ** 2
    ans +=  math.exp(-dx2 * 100.) * 1013.2167575422921535
    return ans / 2.

def main():
    print(gv.ranseed(
        (5751754790502652836, 7676372623888309739, 7570829798026950508)
        ))

    integ = vegas.Integrator(
        [[-1., 1.], [0., 1.], [0., 1.], [0., 1.]]
        )

    if SAVE_OUTPUT:
        log_stdout('eg1a.out') 
    result = integ(f, nitn=10, neval=1000)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))
    integ.map.plot_grid(30, shrink=False)

    if SAVE_OUTPUT:
        unlog_stdout() 
        log_stdout('eg1b.out')
    result = integ(f, nitn=100, neval=1000, )
    print('larger nitn  => %s    Q = %.2f' % (result, result.Q))
    result = integ(f, nitn=10, neval=1e4)
    print('larger neval => %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        log_stdout('eg1c.out')
    # integ.set(map=[[-2., .4, .6, 2.], [0, .4, .6, 2.], [0,.4, .6, 2.], [0.,.4, .6, 2.]])
    integ.set(map=[[-2., 2.], [0, 2.], [0, 2.], [0., 2.]])
    result = integ(f, nitn=10, neval=1000) 
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        log_stdout('eg1d.out')
    integ(f, nitn=7, neval=1000)
    result = integ(f, nitn=10, neval=1000)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        log_stdout('eg1e.out')
    integ = vegas.Integrator([[-1,1]] + 3 * [[0, 1]])
    integ(f, nitn=10, neval=1000)
    def g(x):
        return x[0] * f(x)
    result = integ(g, nitn=10, neval=1000)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        log_stdout('eg1f.out')
    # integ(f_sphere, nitn=10, neval=400000, alpha=0.25)
    # result = integ(f_sphere, nitn=10, neval=400000, alpha=0.25, beta=0.75)#, analyzer=vegas.reporter(5))
    integ(f_sphere, nitn=10, neval=1000, alpha=0.5)
    result = integ(f_sphere, nitn=10, neval=1000, alpha=0.5)#, analyzer=vegas.reporter(5))
    # print(integ.settings())
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        log_stdout('eg1g.out')
    integ(f_sphere, nitn=10, neval=1000, alpha=0.1)
    result = integ(f_sphere, nitn=10, neval=1000, alpha=0.1)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    if SAVE_OUTPUT:
        unlog_stdout()
        # log_stdout('eg1h.out')
    integ = vegas.Integrator(4 * [[0, 1]])
    integ(f2, nitn=10, neval=4e4)
    result = integ(f2, nitn=10, neval=4e4, beta=0.75) # , analyzer=vegas.reporter())
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))
    integ.map.plot_grid(70)
    print(integ(f2, nitn=10, neval=4e4, beta=0.).summary())


if __name__ == '__main__':
    main()