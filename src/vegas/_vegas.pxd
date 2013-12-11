# ctypedef numpy.intp_t INT_TYPE 
ctypedef long INT_TYPE
ctypedef double (*cython_integrand)(double[:]) 
ctypedef double (*cython_integrand_exc)(double[:]) except? 1e111

cdef class VecCythonIntegrand:
    cdef cython_integrand fcn
    cdef readonly object fcntype

cdef object python_wrapper(cython_integrand fcn)

cdef class VecCythonIntegrandExc:
    cdef cython_integrand_exc fcn
    cdef readonly object fcntype

cdef object python_wrapper_exc(cython_integrand_exc fcn)

cdef class VecPythonIntegrand:
    cdef object fcn
    cdef readonly object fcntype

cdef class VecIntegrand:
    cdef readonly object fcntype

cdef class AdaptiveMap:
    # first index is direction, second is increment
    cdef public double alpha
    cdef readonly double[:, ::1] grid
    cdef readonly double[:, ::1] inc
    cdef double[:, ::1] sum_f
    cdef double[:, ::1] n_f

    cpdef map(self, double[:, ::1] y, double[:, ::1] x, double[::1] J, INT_TYPE ny=*)
    cpdef add_training_data(self, double[:, ::1] y, double[::1] f, INT_TYPE ny=*)

cdef class Integrator:
    # inputs
    cdef public object fcntype
    cdef public INT_TYPE neval
    cdef public object neval_hcube_range
    cdef public INT_TYPE nhcube_vec
    cdef public INT_TYPE maxinc_axis
    cdef public INT_TYPE nstrat_crit
    cdef public INT_TYPE nitn
    cdef public double alpha
    cdef public double rtol
    cdef public double atol
    cdef public object mode
    cdef public double beta 
    cdef public object analyzer
    # generated
    cdef readonly AdaptiveMap map 
    cdef double[::1] sigf_list
    cdef INT_TYPE nstrat 
    cdef INT_TYPE neval_hcube 
    cdef INT_TYPE dim 
    cdef readonly INT_TYPE last_neval
