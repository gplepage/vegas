Integrands in C or Fortran
=======================================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`vegas.RunningWAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: x
.. |y| replace:: y


Older implementations of the |vegas| algorithm
have been used extensively in C and Fortran codes. The Python
implementation described here uses a more powerful algorithm.
It is relatively straightforward to combine this version with integrands
coded in C or Fortran. Such integrands are usually substantially
faster than integrands coded directly in Python; they are similar in
speed to optimized Cython code.
There are
many ways to access C and Fortran integrands from Python. Here we
review a few of the options.

:mod:`ctypes` for C
....................
The simplest way to access an integrand coded in C is to use the
Python :mod:`ctypes` module. To illustrate, consider the following
integrand, written in C and stored in file ``cfcn.c``:

.. code-block:: C

    // file cfcn.c
    #include <math.h>

    double fcn(double x[], int dim)
    {
          int i;
          double xsq = 0.0;
          for(i=0; i<dim; i++)
                xsq += x[i] * x[i] ;
          return exp(-100. * sqrt(xsq)) * pow(100.,dim);
    }

This file needs to be compiled into a shared library using something
like::

    cc -fPIC -shared -o cfcn.so cfcn.c

The exact compilation command depends on the operating system and compiler
being used. The function in this library is then wrapped in
Python function ``f``, and integrated using |vegas|::

    import vegas
    import numpy as np

    import ctypes

    # import cfcn.so
    cfcn = ctypes.CDLL('cfcn.so')
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
        main()

The output shows 10 iterations that are used to adapt |vegas| to the
integrand, and then an additional 10 iterations to generate the
final result::

    itn   integral        wgt average     chi2/dof        Q
    -------------------------------------------------------
      1   8.6(7.1)        8.6(7.1)            0.00     1.00
      2   8.2(1.7)        8.2(1.7)            0.00     0.96
      3   7.14(76)        7.32(69)            0.18     0.84
      4   7.88(38)        7.75(33)            0.29     0.84
      5   7.39(13)        7.44(12)            0.47     0.76
      6   7.359(81)       7.383(68)           0.43     0.82
      7   7.400(55)       7.393(43)           0.37     0.90
      8   7.392(51)       7.393(33)           0.32     0.95
      9   7.427(48)       7.404(27)           0.32     0.96
     10   7.388(41)       7.399(23)           0.30     0.98

    itn   integral        wgt average     chi2/dof        Q
    -------------------------------------------------------
      1   7.429(34)       7.429(34)           0.00     1.00
      2   7.412(32)       7.420(24)           0.13     0.72
      3   7.413(28)       7.417(18)           0.08     0.92
      4   7.366(25)       7.400(15)           0.96     0.41
      5   7.366(23)       7.390(12)           1.12     0.34
      6   7.410(22)       7.395(11)           1.02     0.40
      7   7.395(20)       7.3951(95)          0.85     0.53
      8   7.425(19)       7.4011(85)          1.02     0.42
      9   7.394(19)       7.3998(77)          0.91     0.51
     10   7.386(17)       7.3976(71)          0.86     0.56

The final estimate for the integral is ``7.3976(71)``
(1000 times more accurate than the very first iteration).

The :mod:`ctypes` implementation is probably the slowest of the
implementations shown here.

Cython for C
.....................
A more flexible (and often faster) interface to a C integrand can be
created using Cython. To increase efficiency (slightly, in this case),
we use Cython code in file ``cfcn.pyx`` to convert the orginal
function (in ``cfcn.c``) into a batch integral:

.. code-block:: Cython

    # file cfcn.pyx
    import numpy as np
    import vegas

    cdef extern double fcn (double[] x, int n)

    @vegas.batchintegrand
    def f(double[:, ::1] x):
        cdef double[:] ans
        cdef int i, dim=x.shape[1]
        ans = np.empty(x.shape[0], type(x[0,0]))
        for i in range(x.shape[0]):
            ans[i] = fcn(&x[i, 0], dim)
        return ans

We also have to tell Cython how to construct the ``cfcn`` Python
module since that module needs to include compiled code
from ``cfcn.c``. This is done with a `.pyxbld` file::

    # file cfcn.pyxbld
    import numpy as np

    def make_ext(modname, pyxfilename):
        from distutils.extension import Extension
        return Extension(name = modname,
                         sources=[pyxfilename, 'cfcn.c'],
                         libraries=[],
                         include_dirs=[np.get_include()],
                         )

    def make_setup_args():
        return dict()

Finally the integral is evaluated using the Python code ::

    import vegas

    # compile cfcn, if needed, at import
    import pyximport
    pyximport.install(inplace=True)

    import cfcn

    def main():
        integ = vegas.Integrator(4 *[[0,1]])
        print(integ(cfcn.f, neval=1e4, nitn=10).summary())
        print(integ(cfcn.f, neval=1e4, nitn=10).summary())

    if __name__ == '__main__':
        main()

where, again, :mod:`pyximport` guarantees that the ``cfcn`` module
is compiled the first time the code is run.

This implementation is probably the fastest of those presented here.
Cython also works with C++.


:mod:`f2py` for Fortran
.........................
The :mod:`f2py` package, which is distributed with :mod:`numpy`,
makes it relatively easy to compile Fortran
code directly into Python modules. Consider a Fortran implementation of
integrand discussed above, stored in file ``ffcn.f``:

.. code-block:: Fortran

    c file ffcn.f
    c
          function fcn(x, dim)
          integer i, dim
          real*8 x(dim), x2, fcn
          x2 = 0.0
          do i=1,dim
            x2 = x2 + x(i) ** 2
          end do
          fcn = exp(-100. * sqrt(x2)) * 100. ** dim
          return
          end

This code is compiled into a Python module using ::

    f2py -m ffcn -c ffcn.f

and the resulting module provides access to the
integrand from Python::

    import vegas
    import ffcn

    def main():
        integ = vegas.Integrator(4 *[[0,1]])
        print(integ(ffcn.fcn, neval=1e4, nitn=10).summary())
        print(integ(ffcn.fcn, neval=1e4, nitn=10).summary())

    if __name__ == '__main__':
        main()

Again you can make the code somewhat faster by converting the integrand
into a batch integrand inside the Fortran module. Adding the following
function to the end of file ``ffcn.f`` above :

.. code-block:: Fortran

    c part 2 of file ffcn.f --- batch form of integrand

          subroutine batch_fcn(ans, x, dim, nbatch)
          integer dim, nbatch, i, j
          real*8 x(nbatch, dim), xi(dim), ans(nbatch), fcn
    cf2py intent(out) ans
          do i=1,nbatch
                do j=1,dim
                      xi(j) = x(i, j)
                end do
                ans(i) = fcn(xi, dim)
          end do
          end

results in a second Python function ``ffcn.batch_fcn(x)`` that takes the
integration points ``x[i,d]`` as input and returns an array of
integrand values ``ans[i]``. (The second Fortran comment tells ``f2py``
that array ``ans`` should be returned by the correponding Python
function; ``f2py`` also has the function automatically deduce ``dim`` and
``nbatch`` from the shape of ``x``.)
The correponding Python script for doing the integral
is then::

    import vegas
    import ffcn_f2py
    import numpy as np

    def main():
        integ = vegas.Integrator(4 *[[0,1]])
        batch_fcn = vegas.batchintegrand(ffcn_f2py.batch_fcn)
        print(integ(batch_fcn, neval=1e4, nitn=10).summary())
        print(integ(batch_fcn, neval=1e4, nitn=10).summary())

    if __name__ == '__main__':
        main()

This runs roughly twice as fast as the original
when ``neval`` is large (eg, 1e6).

:mod:`f2py` for C
..................
:mod:`f2py` can also be used to compile C code directly into Python
modules, but usually needs an interface file to specify how the
C is turned into Python. The interface file ``cfcn.pyf`` for
the C file ``cfcn.c`` (above) is::

    python module cfcn
    interface
      real*8 function fcn(x, n)
        intent(c) fcn                 ! fcn is a C function
        intent(c)                     ! all fcn arguments are
                                      ! considered as C based
        integer intent(hide), depend(x) :: n=len(x)  ! n is the length
                                                     ! of input array x
        real*8 intent(in) :: x(n)     ! x is input array
      end function fcn
    end interface
    end python module cfcn

More information is available in the documentation for :mod:`f2py`.
The module is created using ::

    f2py -m cfcn -c cfcn.pyf cfcn.c

and the integral evaluated using Python code::

    import vegas
    import cfcn

    def main():
        integ = vegas.Integrator(4 *[[0,1]])
        print(integ(cfcn.fcn, neval=1e4, nitn=10).summary())
        print(integ(cfcn.fcn, neval=1e4, nitn=10).summary())

    if __name__ == '__main__':
        main()
