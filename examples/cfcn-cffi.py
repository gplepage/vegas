import vegas
import numpy as np

from cfcn_cffi import ffi, lib

def f(x):
    _x = ffi.cast('double*', x.ctypes.data)
    return lib.fcn(_x, x.shape[0])

@vegas.batchintegrand       # 5 times faster for neval=1e6
def alt_batch_f(x):
    n, dim = x.shape
    _x = ffi.cast('double*', x.ctypes.data)
    ans = np.empty(n, float)
    for i in range(n):
        ans[i] = lib.fcn(_x + i * dim, dim)
    return ans

@vegas.batchintegrand       # 10 times faster for neval=1e6
def batch_f(x):
    n, dim = x.shape
    ans = np.empty(n, float)
    _x = ffi.cast('double*', x.ctypes.data)
    _ans = ffi.cast('double*', ans.ctypes.data)
    lib.batch_fcn(_ans, _x, n, dim)
    return ans

def main():
    integ = vegas.Integrator(4 * [[0., 1.]])
    f = batch_f
    print(integ(f, neval=1e5, nitn=10).summary())
    print(integ(f, neval=1e5, nitn=10).summary())

if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)
    main()
