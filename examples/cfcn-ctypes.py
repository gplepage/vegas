from __future__ import print_function   # makes this work for python2 and 3

import vegas

import ctypes

# import library
cfcn = ctypes.CDLL('cfcn_ctypes.so')
# specify argument types and result type for cfcn.fcn
cfcn.fcn.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
cfcn.fcn.restype = ctypes.c_double

# Python wrapper for function cfcn.fcn
def f(x):
    global cfcn
    n = len(x)
    array_type = ctypes.c_double * n
    return cfcn.fcn(array_type(*x), ctypes.c_int(n))

def main():
    integ = vegas.Integrator(4 * [[0., 1.]])
    print(integ(f, neval=1e4, nitn=10).summary())
    print(integ(f, neval=1e4, nitn=10).summary())


if __name__ == '__main__':
    import numpy as np
    np.random.seed(9)
    main()