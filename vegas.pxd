import numpy
cimport numpy

ctypedef numpy.intp_t INT_TYPE 
ctypedef void (*cython_vec_integrand)(double[:, ::1], double[::1], INT_TYPE)

cdef class AdaptiveMap:
	# first index is direction, second is increment
	cdef public double alpha
	cdef readonly double[:, ::1] grid
	cdef readonly double[:, ::1] inc
	cdef double[:, ::1] sum_f
	cdef double[:, ::1] n_f

	cpdef map(self, double[:, ::1] y, double[:, ::1] x, double[::1] J, INT_TYPE ny=*)
	cpdef accumulate_training_data(self, double[:, ::1] y, double[::1] f, INT_TYPE ny=*)

cdef class Integrator:
	# inputs
	cdef public INT_TYPE neval
	cdef public object neval_hcube_range
	cdef public INT_TYPE maxvec
	cdef public INT_TYPE maxinc
	cdef public INT_TYPE cstrat
	cdef public INT_TYPE nitn
	cdef public double alpha
	cdef public double rtol
	cdef public double atol
	cdef public object mode
	cdef public bint redistribute  
	cdef public double beta 
	cdef public object analyzer
	# generated
	cdef readonly AdaptiveMap map 
	cdef double[::1] sigf_list
	cdef INT_TYPE nstrat 
	cdef INT_TYPE neval_hcube 
	cdef INT_TYPE dim 
	cdef object _mode
	cdef readonly INT_TYPE last_neval
	cdef cython_vec_integrand vec_integrand
	
	cdef cython_integrate(self, cython_vec_integrand fcn, kargs)
	cdef _integrate(self, kargs)
	
cdef class VegasTest:
	cdef readonly double[::1] sig
	cdef readonly double[::1] x0
	cdef readonly double[::1] ampl
	cdef readonly double exact
	cdef public Integrator I
	cdef void cython_vec_fcn(self, double[:, ::1] x, double[::1] f, INT_TYPE nx)
