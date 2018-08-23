import numpy as np
import vegas

cdef extern double fcn (double[] x, int n)

@vegas.batchintegrand
def batch_f(double[:, ::1] x):
    cdef double[:] ans
    cdef int i, dim=x.shape[1]
    ans = np.empty(x.shape[0], type(x[0,0]))
    for i in range(x.shape[0]):
        ans[i] = fcn(&x[i, 0], dim)
    return ans

