from __future__ import print_function   # makes this work for python2 and 3

import vegas

# compile cfcn_cython, if needed, at import
import pyximport
pyximport.install(inplace=True)

import cfcn_cython

def main():
    integ = vegas.Integrator(4 *[[0,1]])
    print(integ(cfcn_cython.batch_f, neval=1e5, nitn=10).summary())
    print(integ(cfcn_cython.batch_f, neval=1e5, nitn=10).summary())

if __name__ == '__main__':
    import numpy as np
    import gvar as gv
    gv.ranseed(12)
    main()