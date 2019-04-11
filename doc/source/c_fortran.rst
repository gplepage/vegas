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


:mod:`cffi` for C
...................
The simplest way to access an integrand coded in C is to use the
:mod:`cffi` module in Python. To illustrate, consider the following
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

Running the following Python code creates (in file ``cfcn_cffi.c``) and
compiles code for a Python module called ``cfcn_cffi`` that provides
access to ``fcn(x,dim)``::

  # file cfcn_cffi-builder.py
  from cffi import FFI
  ffibuilder = FFI()

  # specify functions, etc to be made available to Python
  ffibuilder.cdef('double fcn(double x[], int dim);')

  # specify code needed to build the Python module
  ffibuilder.set_source(
      module_name='cfcn_cffi',
      source='double fcn(double x[], int dim);',
      sources=['cfcn.c'],     # other sources -- file containing fcn(x, dim)
      libraries=['m'],        # may need to specify the math library (-lm)
      )

  if __name__ == '__main__':
      # create C code for module and compile it
      ffibuilder.compile(verbose=True)

We integrate the function in this module by wrapping it in a
Python function (here ``f(x)``), which is integrated using |vegas|::

  import vegas

  from cfcn_cffi import ffi, lib

  def f(x):
      _x = ffi.cast('double*', x.ctypes.data) # pointer to x's data
      return lib.fcn(_x, 4)

  def main():
      integ = vegas.Integrator(4 * [[0., 1.]])
      print(integ(f, neval=1e6, nitn=10).summary())
      print(integ(f, neval=1e6, nitn=10).summary())

  if __name__ == '__main__':
      main()

The output shows 10 iterations that are used to adapt |vegas| to the
integrand, and then an additional 10 iterations to generate the
final result::

  itn   integral        wgt average     chi2/dof        Q
  -------------------------------------------------------
    1   4.1(2.1)        4.1(2.1)            0.00     1.00
    2   7.403(42)       7.401(42)           2.52     0.11
    3   7.366(27)       7.376(23)           1.51     0.22
    4   7.4041(73)      7.4014(70)          1.48     0.22
    5   7.4046(36)      7.4039(32)          1.15     0.33
    6   7.4003(23)      7.4015(19)          1.09     0.37
    7   7.4036(20)      7.4025(14)          1.01     0.42
    8   7.4017(16)      7.4022(10)          0.89     0.52
    9   7.4010(14)      7.40174(83)         0.84     0.57
   10   7.4017(13)      7.40174(70)         0.74     0.67

  itn   integral        wgt average     chi2/dof        Q
  -------------------------------------------------------
    1   7.4016(12)      7.4016(12)          0.00     1.00
    2   7.4030(11)      7.40239(81)         0.79     0.37
    3   7.4020(10)      7.40224(63)         0.44     0.64
    4   7.40249(92)     7.40232(52)         0.31     0.82
    5   7.40258(86)     7.40239(44)         0.25     0.91
    6   7.40093(81)     7.40205(39)         0.70     0.62
    7   7.40228(76)     7.40210(35)         0.60     0.73
    8   7.40276(72)     7.40222(31)         0.61     0.75
    9   7.40181(71)     7.40216(29)         0.57     0.80
   10   7.40178(70)     7.40210(27)         0.53     0.85

The final estimate for the integral is ``7.40210(27)`` (about 10,000
times more accurate than the first iteration).

This code can be made substantially faster by converting the
integrand into a batch integrand. We do this by adding a batch
version of the integrand function to the ``cfcn_cffi`` module::

  # file cfcn_cffi-builder.py
  from cffi import FFI
  ffibuilder = FFI()

  # specify functions, etc made available to Python
  ffibuilder.cdef("""
      void batch_fcn(double ans[], double x[], int n, int dim);
      """)

  # specify code needed to build the module
  ffibuilder.set_source(
      module_name='cfcn_cffi',
      source="""
      // code for module
      double fcn(double x[], int dim);

      void batch_fcn(double ans[], double x[], int n, int dim)
      {
          int i;
          for(i=0; i<n; i++)
              ans[i] = fcn(&x[i * dim], dim);
      }
      """,
      sources=['cfcn.c'],     # other sources -- file containing fcn(x, dim)
      libraries=['m'],        # may need to specify the math library (-lm)
      )

  if __name__ == '__main__':
      # create C code for module and compile it
      ffibuilder.compile(verbose=True)

The resulting module is then used by the following integration code::

  import vegas
  import numpy as np

  from cfcn_cffi import ffi, lib

  @vegas.batchintegrand
  def batch_f(x):
      n, dim = x.shape
      ans = np.empty(n, float)
      _x = ffi.cast('double*', x.ctypes.data)
      _ans = ffi.cast('double*', ans.ctypes.data)
      lib.batch_fcn(_ans, _x, n, dim)
      return ans

  def main():
      integ = vegas.Integrator(4 * [[0., 1.]])
      print(integ(batch_f, neval=1e6, nitn=10).summary())
      print(integ(batch_f, neval=1e6, nitn=10).summary())

  if __name__ == '__main__':
      main()

Running this code gives identical results to those above, but in about 1/10
the time.

Obviously :mod:`cffi` can also be used to access C++ functions by
creating C wrappers for those functions.
Another option for accessing C code is
the :mod:`ctypes` module, but it is more complicated to use and typically
gives slower code.


:mod:`cffi` for Fortran
.........................
Module :mod:`cffi` can be used for Fortran integrands by calling the
Fortran function from C code. Consider a Fortran implementation of
the integrand discussed above, stored in file ``ffcn.f``:

.. code-block:: Fortran

    c file ffcn.f
    c
          function fcn(x, dim)
          integer i, dim
          real*8 x(dim), xsq, fcn
          xsq = 0.0
          do i=1,dim
            xsq = xsq + x(i) ** 2
          end do
          fcn = exp(-100. * sqrt(xsq)) * 100. ** dim
          return
          end

This file is compiled into ``ffcn.o`` using something like ::

  gfortran -c ffcn.c -o ffcn.o

The :mod:`cffi` build script (batch version) from the previous section can be
used here with only three modifications: 1) the Fortran function must be
compiled separately and so is included in a list of object files
(``extra_objects``); 2) the name of the Fortran function may have an extra
underscore (or other modification, depending on the compilers used; on UNIX
systems ``nm ffcn.o`` lists the names in the file); and 3) the Fortran
function's arguments are passed by address and so are pointers in C. The
modified script is::

  # file ffcn_cffi-builder.py
  from cffi import FFI
  ffibuilder = FFI()

  # specify functions, etc needed by Python
  ffibuilder.cdef("""
      void batch_fcn(double ans[], double x[], int n, int dim);
      """)

  # specify code needed to build the module
  ffibuilder.set_source(
      module_name='ffcn_cffi',
      source="""
      // code for module
      double fcn_(double* x, int* dim);   // Fortran function in ffcn.o

      void batch_fcn(double ans[], double x[], int n, int dim)
      {
          int i;
          for(i=0; i<n; i++)
              ans[i] = fcn_(&x[i * dim], &dim);
      }
      """,
      extra_objects=['ffcn.o'],   # compiled Fortran
      libraries=['m'],            # may need to specify the math library (-lm)
      )

  if __name__ == "__main__":
      # create C code for module and compile it
      ffibuilder.compile(verbose=True)

The new Python module is used exactly
the same way as for C code, with module ``ffcn_cffi`` replacing
module ``cfcn_ffi`` (see the previous section)::

  import vegas
  import numpy as np

  from ffcn_cffi import ffi, lib

  @vegas.batchintegrand
  def batch_f(x):
      n, dim = x.shape
      ans = np.empty(n, float)
      _x = ffi.cast('double*', x.ctypes.data)
      _ans = ffi.cast('double*', ans.ctypes.data)
      lib.batch_fcn(_ans, _x, n, dim)
      return ans

  def main():
      integ = vegas.Integrator(4 * [[0., 1.]])
      print(integ(batch_f, neval=1e6, nitn=10).summary())
      print(integ(batch_f, neval=1e6, nitn=10).summary())

  if __name__ == '__main__':
      import numpy as np
      np.random.seed(12)

This code runs about as fast as the corresponding C code in the previous
section.


Cython for C (or C++ or Fortran)
................................
A more flexible interface to a C integrand can be
created using Cython. To increase efficiency,
we use Cython code in file ``cfcn.pyx`` to convert the original
function (in ``cfcn.c``) into a batch integral (again):

.. code-block:: Cython

    # file cfcn.pyx
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
        print(integ(cfcn.batch_f, neval=1e4, nitn=10).summary())
        print(integ(cfcn.batch_f, neval=1e4, nitn=10).summary())

    if __name__ == '__main__':
        main()

where, again, :mod:`pyximport` guarantees that the ``cfcn`` module
is compiled the first time the code is run.

This implementation is as fast as the other batch implementations, above.
Cython also works with C++. It can also work with Fortran code, analogously
to :mod:`cffi`.


:mod:`f2py` for Fortran
.........................
The :mod:`f2py` package, which is distributed with :mod:`numpy`,
makes it relatively easy to compile Fortran
code directly into Python modules. Using the same Fortran code as
above (in ``ffcn.f``), the code is compiled into a Python module using ::

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
        print(integ(batch_fcn, neval=1e6, nitn=10).summary())
        print(integ(batch_fcn, neval=1e6, nitn=10).summary())

    if __name__ == '__main__':
        main()

This runs roughly twice as fast as the original, and about the same
speed as the batch versions of the C code, above.
