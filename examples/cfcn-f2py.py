from __future__ import print_function   # makes this work for python2 and 3

import vegas
import cfcn_f2py

def main():
    integ = vegas.Integrator(4 *[[0,1]])
    print(integ(cfcn_f2py.fcn, neval=1e4, nitn=10).summary())
    print(integ(cfcn_f2py.fcn, neval=1e4, nitn=10).summary())

if __name__ == '__main__':
    import numpy as np
    np.random.seed(9)
    main()