# c#ython: profile=True

# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-14 G. Peter Lepage. 
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

cimport cython
cimport numpy
from libc.math cimport floor, log, abs, tanh, erf, exp, sqrt, lgamma

import sys
import numpy 
import math
import warnings

cdef double TINY = 1e-308                            # smallest and biggest
cdef double HUGE = 1e308

# following two functions are here in case gvar (from lsqfit package)
# is not available --- see _gvar_standin

cdef double gammaP_ser(double a, double x, double rtol, int itmax):
    """ Power series expansion for P(a, x) (for x < a+1).

    P(a, x) = 1/Gamma(a) * \int_0^x dt exp(-t) t ** (a-1) = 1 - Q(a, x)
    """
    cdef int n
    cdef double ans, term 
    if x == 0:
        return 0.
    ans = 0.
    term = 1. / x
    for n in range(itmax):
        term *= x / float(a + n)
        ans += term
        if abs(term) < rtol * abs(ans):
            break
    else:
        warnings.warn(
            'gammaP convergence not complete -- want: %.3g << %.3g' 
            % (abs(term), rtol * abs(ans))
            )
    log_ans = math.log(ans) - x + a * math.log(x) - math.lgamma(a)
    return math.exp(log_ans)

cdef double gammaQ_cf(double a, double x, double rtol, int itmax):
    """ Continuing fraction expansion for Q(a, x) (for x > a+1).

    Q(a, x) = 1/Gamma(a) * \int_x^\infty dt exp(-t) t ** (a-1) = 1 - P(a, x)
    Uses Lentz's algorithm for continued fractions.
    """
    cdef double tiny = 1e-30 
    cdef double den, Cj, Dj, fj
    cdef int j
    den = x + 1. - a
    if abs(den) < tiny:
        den = tiny
    Cj = x + 1. - a + 1. / tiny
    Dj = 1 / den 
    fj = Cj * Dj * tiny
    for j in range(1, itmax):
        aj = - j * (j - a) 
        bj = x + 2 * j + 1. - a
        Dj = bj + aj * Dj
        if abs(Dj) < tiny:
            Dj = tiny
        Dj = 1. / Dj
        Cj = bj + aj / Cj
        if abs(Cj) < tiny:
            Cj = tiny
        fac = Cj * Dj
        fj = fac * fj
        if abs(fac-1) < rtol:
            break
    else:
        warnings.warn(
            'gammaQ convergence not complete -- want: %.3g << %.3g' 
            % (abs(fac-1), rtol)
            )
    return math.exp(math.log(fj) - x + a * math.log(x) - math.lgamma(a))


have_gvar = False
try:
    import gvar
    have_gvar = True
except ImportError:
    # fake version of gvar.gvar
    # for use if lsqfit module not available
    class GVar(object):
        """ Poor substitute for GVar in the lsqfit package.

        This supports arithmetic involving GVars and numbers 
        but not arithmetic involving GVars and GVars. For
        the latter, you need to install the lsqfit
        package (whose gvar module provides this functionality).

        This also supports log, sqrt, and exp, but not
        trig functions etc --- again install lsqfit if 
        these are needed.
        """
        def __init__(self, mean, sdev):
            self.mean = float(mean)
            self.sdev = abs(float(sdev))
            self.internaldata = (self.mean, self.sdev)
        def __add__(self, double a):
            return GVar(a + self.mean, self.sdev)

        def __radd__(self, double a):
            return GVar(a + self.mean, self.sdev)
            
        def __sub__(self, double a):
            return GVar(self.mean - a, self.sdev)

        def __rsub__(self, double a):
            return GVar(a - self.mean, self.sdev)
            
        def __mul__(self, double a):
            return GVar(self.mean * a, self.sdev * a)

        def __rmul__(self, double a):
            return GVar(self.mean * a, self.sdev * a)

        def __div__(self, double a):
            return GVar(self.mean / a, self.sdev / a)

        def __truediv__(self, double a):  # for python3
            return GVar(self.mean / a, self.sdev / a)

        def __rdiv__(self, double a):
            return (a / self.mean) * GVar(1., self.sdev / self.mean)
    
        def __rtruediv__(self, double a):
            return (a / self.mean) * GVar(1., self.sdev / self.mean)
    

        def __neg__(self):
            return GVar(-self.mean, self.sdev)

        def __pos__(self):
            return self

        def __pow__(self, double a):
            return (self.mean ** a) * GVar(1, a * self.sdev / self.mean)

        def __rpow__(self, double a):
            return (a ** self.mean) * GVar(1., self.sdev * math.log(a))

        def log(self):
            return GVar(math.log(self.mean), self.sdev / self.mean)

        def exp(self):
            return math.exp(self.mean) * GVar(1., self.sdev)

        def sqrt(self):
            return math.sqrt(self.mean) * GVar(1., self.sdev / 2. / self.mean)

        def __str__(self):
            """ Return string representation of ``self``.

            The representation is designed to show at least
            one digit of the mean and two digits of the standard deviation. 
            For cases where mean and standard deviation are not 
            too different in magnitude, the representation is of the
            form ``'mean(sdev)'``. When this is not possible, the string
            has the form ``'mean +- sdev'``.
            """
            # taken from gvar.GVar in lsqfit package.
            def ndec(x, offset=2):
                ans = offset - numpy.log10(x)
                ans = int(ans)
                if ans > 0 and x * 10. ** ans >= [0.5, 9.5, 99.5][offset]:
                    ans -= 1
                return 0 if ans < 0 else ans
            dv = abs(self.sdev)
            v = self.mean
            
            # special cases 
            if dv == float('inf'):
                return '%g +- inf' % v
            elif v == 0 and (dv >= 1e5 or dv < 1e-4):
                if dv == 0:
                    return '0(0)'
                else:
                    ans = ("%.1e" % dv).split('e')
                    return "0.0(" + ans[0] + ")e" + ans[1]
            elif v == 0:
                if dv >= 9.95:
                    return '0(%.0f)' % dv
                elif dv >= 0.995:
                    return '0.0(%.1f)' % dv
                else:
                    ndecimal = ndec(dv)
                    return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)
            elif dv == 0:
                ans = ('%g' % v).split('e')
                if len(ans) == 2:
                    return ans[0] + "(0)e" + ans[1]
                else:
                    return ans[0] + "(0)"
            elif dv < 1e-6 * abs(v) or dv > 1e4 * abs(v):
                return '%g +- %.2g' % (v, dv)
            elif abs(v) >= 1e6 or abs(v) < 1e-5:
                # exponential notation for large |self.mean| 
                exponent = numpy.floor(numpy.log10(abs(v)))
                fac = 10.**exponent
                mantissa = str(self/fac)
                exponent = "e" + ("%.0e" % fac).split("e")[-1]
                return mantissa + exponent

            # normal cases
            if dv >= 9.95:
                if abs(v) >= 9.5:
                    return '%.0f(%.0f)' % (v, dv)
                else:
                    ndecimal = ndec(abs(v), offset=1)
                    return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv)
            if dv >= 0.995:
                if abs(v) >= 0.95:
                    return '%.1f(%.1f)' % (v, dv)
                else:
                    ndecimal = ndec(abs(v), offset=1)
                    return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv)
            else:
                ndecimal = max(ndec(abs(v), offset=1), ndec(dv))
                return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)

    class _gvar_standin:
        def __init__(self):
            pass
        def gvar(self, mean, sdev):
            return GVar(mean, sdev)
        def mean(self, glist):
            return numpy.array([g.mean for g in glist])
        def var(self, glist):
            return numpy.array([g.sdev ** 2 for g in glist])
        def gammaQ(self, double a, double x, double rtol=1e-5, int itmax=10000):
            " complement of normalized incomplete gamma function: Q(a,x) "
            if x < 0 or a < 0:
                raise ValueError('negative argument: %g, %g' % (a, x))
            if x == 0:
                return 1.
            elif a == 0:
                return 0.
            if x < a + 1.:
                return 1. - gammaP_ser(a, x, rtol=rtol, itmax=itmax)
            else:
                return gammaQ_cf(a, x, rtol=rtol, itmax=itmax)
        def gammaP(self, double a, double x, double rtol=1e-5, itmax=10000):
            " normalized incomplete gamma function: P(a,x) "
            if x < 0 or a < 0:
                raise ValueError('negative argument: %g, %g' % (a, x))
            if x == 0:
                return 0.
            elif a == 0:
                return 1.
            if x < a + 1.:
                return gammaP_ser(a, x, rtol=rtol, itmax=itmax)
            else:
                return 1. - gammaQ_cf(a, x, rtol=rtol, itmax=itmax)
    gvar = _gvar_standin()
    gvar.GVar = GVar

class RunningWAvg(gvar.GVar):
    """ Running weighted average of Monte Carlo estimates.

    This class accumulates independent Monte Carlo 
    estimates (e.g., of an integral) and combines 
    them into a single weighted average. It 
    is derived from :class:`gvar.GVar` (from 
    the :mod:`lsqfit` module if it is present) and 
    represents a Gaussian random variable.
    """
    def __init__(self, gvar_list=None):
        if gvar_list is not None:
            self.__init__()
            for g in gvar_list:
                self.add(g)
        else:
            self._v_s2 = 0.
            self._v2_s2 = 0.
            self._1_s2 = 0.
            self.itn_results = []
            super(RunningWAvg, self).__init__(
                *gvar.gvar(0., 0.).internaldata,
               )

    def _chi2(self):
        return self._v2_s2 - self._v_s2 ** 2 / self._1_s2
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        return len(self.itn_results) - 1
    dof = property(
        _dof, 
        None, 
        None, 
        "Number of degrees of freedom in weighted average."
        )

    def _Q(self):
        return (
            gvar.gammaQ(self.dof / 2., self.chi2 / 2.)
            if self.dof > 0 and self.chi2 > 0
            else 1
            )
    Q = property(
        _Q, 
        None, 
        None, 
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )

    def add(self, g):
        """ Add estimate ``g`` to the running average. """
        self.itn_results.append(g)
        var = g.sdev ** 2
        assert var > 0.0, 'zero variance not allowed'          
        self._v_s2 = self._v_s2 + g.mean / var
        self._v2_s2 = self._v2_s2 + g.mean ** 2 / var
        self._1_s2 = self._1_s2 + 1. / var
        super(RunningWAvg, self).__init__(*gvar.gvar(
            self._v_s2 / self._1_s2,
            sqrt(1. / self._1_s2),
            ).internaldata)
    def summary(self):
        """ Assemble summary of independent results into a string. """
        acc = RunningWAvg()
        linedata = []
        for i, res in enumerate(self.itn_results):
            acc.add(res)
            if i > 0:
                chi2_dof = acc.chi2 / acc.dof 
                Q = acc.Q
            else:
                chi2_dof = 0.0
                Q = 1.0
            itn = '%3d' % (i + 1)
            integral = '%-15s' % res
            wgtavg = '%-15s' % acc
            chi2dof = '%8.2f' % (acc.chi2 / acc.dof if i != 0 else 0.0)
            Q = '%8.2f' % (acc.Q if i != 0 else 1.0)
            linedata.append((itn, integral, wgtavg, chi2dof, Q))
        nchar = 5 * [0]
        for data in linedata:
            for i, d in enumerate(data):
                if len(d) > nchar[i]:
                    nchar[i] = len(d)
        fmt = '%%%ds   %%-%ds %%-%ds %%%ds %%%ds\n' % tuple(nchar)
        ans = fmt % ('itn', 'integral', 'wgt average', 'chi2/dof', 'Q')
        ans += len(ans[:-1]) * '-' + '\n'
        for data in linedata:
            ans += fmt % data
        return ans

# AdaptiveMap is used by Integrator 
cdef class AdaptiveMap:
    """ Adaptive map ``y->x(y)`` for multidimensional ``y`` and ``x``.

    An :class:`AdaptiveMap` defines a multidimensional map ``y -> x(y)`` 
    from the unit hypercube, with ``0 <= y[d] <= 1``, to an arbitrary
    hypercube in ``x`` space. Each direction is mapped independently 
    with a Jacobian that is tunable (i.e., "adaptive").

    The map is specified by a grid in ``x``-space that, by definition, 
    maps into a uniformly spaced grid in ``y``-space. The nodes of 
    the grid are specified by ``grid[d, i]`` where d is the 
    direction (``d=0,1...dim-1``) and ``i`` labels the grid point
    (``i=0,1...N``). The mapping for a specific point ``y`` into
    ``x`` space is:: 

        y[d] -> x[d] = grid[d, i(y[d])] + inc[d, i(y[d])] * delta(y[d])

    where ``i(y)=floor(y*N``), ``delta(y)=y*N - i(y)``, and
    ``inc[d, i] = grid[d, i+1] - grid[d, i]``. The Jacobian for this map, :: 

        dx[d]/dy[d] = inc[d, i(y[d])] * N,

    is piece-wise constant and proportional to the ``x``-space grid 
    spacing. Each increment in the ``x``-space grid maps into an increment of 
    size ``1/N`` in the corresponding ``y`` space. So regions in 
    ``x`` space where ``inc[d, i]`` is small are stretched out
    in ``y`` space, while larger increments are compressed.

    The ``x`` grid for an :class:`AdaptiveMap` can be specified explicitly
    when the map is created: for example, ::

        m = AdaptiveMap([[0, 0.1, 1], [-1, 0, 1]])

    creates a two-dimensional map where the ``x[0]`` interval ``(0,0.1)``
    and ``(0.1,1)`` map into the ``y[0]`` intervals ``(0,0.5)`` and 
    ``(0.5,1)`` respectively, while ``x[1]`` intervals ``(-1,0)`` 
    and ``(0,1)`` map into ``y[1]`` intervals ``(0,0.5)`` and  ``(0.5,1)``.

    More typically an initially uniform map is trained with data 
    ``f[j]`` corresponding to ``ny`` points ``y[j, d]``,
    with ``j=0...ny-1``, uniformly distributed in |y| space: 
    for example, ::

        m.add_training_data(y, f)
        m.adapt(alpha=1.5)

    ``m.adapt(alpha=1.5)`` shrinks grid increments where ``f[j]``
    is large, and expands them where ``f[j]`` is small. Typically 
    one has to iterate over several sets of ``y``\s and ``f``\s 
    before the grid has fully adapted. 

    The speed with which the grid adapts is determined by parameter ``alpha``.
    Large (positive) values imply rapid adaptation, while small values (much
    less than one) imply slow adaptation. As in any iterative process, it is
    usually a good idea to slow adaptation down in order to avoid
    instabilities.

    :param grid: Initial ``x`` grid, where ``grid[d, i]`` is the ``i``-th 
        node in direction ``d``.
    :type x: 2-d array of floats
    :param ninc: Number of increments along each axis of the ``x`` grid. 
        A new grid is generated if ``ninc`` differs from ``grid.shape[1]``.
        The new grid is designed to give the same Jacobian ``dx(y)/dy``
        as the original grid. The default value, ``ninc=None``,  leaves 
        the grid unchanged.
    :type ninc: ``int`` or ``None``
    """
    def __init__(self, grid, ninc=None):
        cdef INT_TYPE i, d
        if isinstance(grid, AdaptiveMap):
            self._init_grid(grid.grid, initinc=True)
        else:
            grid = numpy.array(grid, float)
            if grid.ndim != 2:
                raise ValueError('grid must be 2-d array not %d-d' % grid.ndim)
            grid.sort(axis=1)
            if grid.shape[1] < 2: 
                raise ValueError("grid.shape[1] smaller than 2: " % grid.shape[1])
            self._init_grid(grid, initinc=True)
        self.sum_f = None
        self.n_f = None
        if ninc is not None and ninc != self.inc.shape[1]:
            if self.inc.shape[1] == 1:
                self.make_uniform(ninc=ninc)
            else:
                self.adapt(ninc=ninc)

    property ninc:
        " Number of increments along each grid axis."
        def __get__(self):
            return self.inc.shape[1] 
    property  dim:
        " Number of dimensions."
        def __get__(self):
            return self.grid.shape[0]
    def region(self, INT_TYPE d=-1):
        """ x-space region.

        ``region(d)`` returns a tuple ``(xl,xu)`` specifying the ``x``-space
        interval covered by the map in direction ``d``. A list containing
        the intervals for each direction is returned if ``d`` is omitted.
        """
        if d < 0:
            return [self.region(d) for d in range(self.dim)]
        else:
            return (self.grid[d, 0], self.grid[d, -1])

    def __reduce__(self):
        """ Capture state for pickling. """
        return (AdaptiveMap, (numpy.asarray(self.grid),))

    def settings(self, ngrid=5):
        """ Create string with information about grid nodes.

        Creates a string containing the locations of the nodes
        in the map grid for each direction. Parameter 
        ``ngrid`` specifies the maximum number of nodes to print
        (spread evenly over the grid).
        """
        ans = []
        if ngrid > 0:
            grid = numpy.array(self.grid)
            nskip = int(self.ninc // ngrid)
            if nskip<1:
                nskip = 1
            start = nskip // 2
            for d in range(self.dim):
                ans += [
                    "    grid[%2d] = %s" 
                    % (
                        d, 
                        numpy.array2string(
                            grid[d, start::nskip],precision=3,
                            prefix='    grid[xx] = ')
                          )
                    ]
        return '\n'.join(ans) + '\n'

    def random(self, n=None):
        " Create ``n`` random points in |x| space. "
        if n is None:
            y = numpy.random.uniform(0., 1., self.dim)
        else:
            y = numpy.random.uniform(0., 1., (n, self.dim))
        return self(y)

    def make_uniform(self, ninc=None):
        """ Replace the grid with a uniform grid.

        The new grid has ``ninc`` increments along each direction if 
        ``ninc`` is specified. Otherwise it has the same number of 
        increments as the old grid.
        """
        cdef INT_TYPE i, d
        cdef INT_TYPE dim = self.grid.shape[0]
        cdef double[:] tmp
        cdef double[:, ::1] new_grid
        if ninc is None:
            ninc = self.inc.shape[1]
        ninc = int(ninc)
        if ninc < 1:
            raise ValueError(
                "no of increments < 1 in AdaptiveMap -- %d" 
                % ninc
                )
        new_grid = numpy.empty((dim, ninc + 1), float)
        for d in range(dim):
            tmp = numpy.linspace(self.grid[d, 0], self.grid[d, -1], ninc + 1)
            for i in range(ninc + 1):
                new_grid[d, i] = tmp[i]
        self._init_grid(new_grid)

    def _init_grid(self, new_grid, initinc=False):
        " Set the grid equal to new_grid. "
        cdef INT_TYPE dim = new_grid.shape[0]
        cdef INT_TYPE ninc = new_grid.shape[1] - 1 
        cdef INT_TYPE d, i
        self.grid = new_grid
        if initinc or self.inc.shape[0] != dim or self.inc.shape[1] != ninc:
            self.inc = numpy.empty((dim, ninc), float)
        for d in range(dim):
            for i in range(ninc):
                self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]

    def __call__(self, y):
        """ Return ``x`` values corresponding to ``y``. 

        ``y`` can be a single ``dim``-dimensional point, or it 
        can be an array ``y[i,j, ..., d]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y, float)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], float)
        self.map(y, x, jac)
        x.shape = y_shape
        return x

    def jac(self, y):
        """ Return the map's Jacobian at ``y``. 

        ``y`` can be a single ``dim``-dimensional point, or it 
        can be an array ``y[d,i,j,...]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], float)
        self.map(y, x, jac)
        return jac

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef map(
        self, 
        double[:, ::1] y, 
        double[:, ::1] x, 
        double[::1] jac, 
        INT_TYPE ny=-1
        ):
        """ Map y to x, where jac is the Jacobian.

        ``y[j, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
        ``x[j, d]`` is filled with the corresponding ``x`` values,
        and ``jac[j]`` is filled with the corresponding Jacobian 
        values. ``x`` and ``jac`` must be preallocated: for example, ::

            x = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[0], float)

        :param y: ``y`` values to be mapped. ``y`` is a contiguous 2-d array,
            where ``y[j, d]`` contains values for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param x: Container for ``x`` values corresponding to ``y``.
        :type x: contiguous 2-d array of floats
        :param jac: Container for Jacobian values corresponding to ``y``.
        :type jac: contiguous 1-d array of floats
        :param ny: Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
            and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
            omitted (or negative).
        :type ny: int
        """
        cdef INT_TYPE ninc = self.inc.shape[1]
        cdef INT_TYPE dim = self.inc.shape[0]
        cdef INT_TYPE i, iy, d  
        cdef double y_ninc, dy_ninc, tmp_jac
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for i in range(ny):
            jac[i] = 1.
            for d in range(dim):
                y_ninc = y[i, d] * ninc
                iy = <int>floor(y_ninc)
                dy_ninc = y_ninc  -  iy
                if iy < ninc:
                    x[i, d] = self.grid[d, iy] + self.inc[d, iy] * dy_ninc
                    jac[i] *= self.inc[d, iy] * ninc
                else:
                    x[i, d] = self.grid[d, ninc]
                    jac[i] *= self.inc[d, ninc - 1] * ninc
        return

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef add_training_data(
        self, 
        double[:, ::1] y, 
        double[::1] f, 
        INT_TYPE ny=-1,
        ):
        """ Add training data ``f`` for ``y``-space points ``y``.

        Accumulates training data for later use by ``self.adapt()``.
        Grid increments will be made smaller in regions where
        ``f`` is larger than average, and larger where ``f`` 
        is smaller than average. The grid is unchanged (converged?)
        when ``f`` is constant across the grid.

        :param y: ``y`` values corresponding to the training data. 
            ``y`` is a contiguous 2-d array, where ``y[j, d]`` 
            is for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param f: Training function values. ``f[j]`` corresponds to 
            point ``y[j, d]`` in ``y``-space.
        :type f: contiguous 2-d array of floats
        :param ny: Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
            and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
            omitted (or negative).
        :type ny: int
        """
        cdef INT_TYPE ninc = self.inc.shape[1]
        cdef INT_TYPE dim = self.inc.shape[0]
        cdef INT_TYPE iy 
        cdef INT_TYPE i, d
        if self.sum_f is None:
            self.sum_f = numpy.zeros((dim, ninc), float)
            self.n_f = numpy.zeros((dim, ninc), float) + TINY
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for d in range(dim):
            for i in range(ny):
                iy = <int> floor(y[i, d] * ninc)
                self.sum_f[d, iy] += abs(f[i])
                self.n_f[d, iy] += 1
        return
        
    # @cython.boundscheck(False)
    def adapt(self, double alpha=0.0, ninc=None):
        """ Adapt grid to accumulated training data.

        ``self.adapt(...)`` projects the training data onto
        each axis independently and maps it into ``x`` space.
        It shrinks ``x``-grid increments in regions where the
        projected training data is large, and grows increments
        where the projected data is small. The grid along 
        any direction is unchanged if the training data 
        is constant along that direction.

        The number of increments along a direction can be 
        changed by setting parameter ``ninc``. 

        The grid does not change if no training data has 
        been accumulated, unless ``ninc`` is specified, in 
        which case the number of increments is adjusted 
        while preserving the relative density of increments 
        at different values of ``x``.

        :parameter alpha: Determines the speed with which the grid adapts to 
            training data. Large (postive) values imply rapid evolution; 
            small values (much less than one) imply slow evolution. Typical 
            values are of order one. Choosing ``alpha<0`` causes adaptation
            to the unmodified training data (usually not a good idea).
        :type alpha: double or None
        :parameter ninc: Number of increments along each direction in the 
            new grid. The number is unchanged from the old grid if ``ninc``
            is omitted (or equals ``None``).
        :type ninc: int or None
        """
        cdef double[:, ::1] new_grid
        cdef double[::1] avg_f, tmp_f
        cdef double sum_f, acc_f, f_ninc
        cdef INT_TYPE old_ninc = self.grid.shape[1] - 1
        cdef INT_TYPE dim = self.grid.shape[0]
        cdef INT_TYPE i, j, new_ninc
        
        # initialization
        if ninc is None:
            new_ninc = old_ninc
        else:
            new_ninc = ninc
        if new_ninc < 1:
            raise ValueError('ninc < 1: ' + str(new_ninc))
        if new_ninc == 1:
            new_grid = numpy.empty((dim, 2), float)
            for d in range(dim):
                new_grid[d, 0] = self.grid[d, 0]
                new_grid[d, 1] = self.grid[d, -1]
            self._init_grid(new_grid)
            return
        
        # smoothing
        new_grid = numpy.empty((dim, new_ninc + 1), float)
        avg_f = numpy.ones(old_ninc, float) # default = uniform
        if alpha > 0 and old_ninc > 1:
            tmp_f = numpy.empty(old_ninc, float)
        for d in range(dim):
            if self.sum_f is not None and alpha != 0:
                for i in range(old_ninc):
                    avg_f[i] = self.sum_f[d, i] / self.n_f[d, i]
            if alpha > 0 and old_ninc > 1:
                tmp_f[0] = (3. * avg_f[0] + avg_f[1]) / 4.
                tmp_f[-1] = (3. * avg_f[-1] + avg_f[-2]) / 4.
                sum_f = tmp_f[0] + tmp_f[-1]
                for i in range(1, old_ninc - 1):
                    tmp_f[i] = (6. * avg_f[i] + avg_f[i-1] + avg_f[i+1]) / 8.
                    sum_f += tmp_f[i]
                if sum_f > 0:
                    for i in range(old_ninc):
                        avg_f[i] = tmp_f[i] / sum_f + TINY
                else:
                    for i in range(old_ninc):
                        avg_f[i] = TINY
                for i in range(old_ninc):
                    avg_f[i] = (-(1 - avg_f[i]) / log(avg_f[i])) ** alpha
            
            # regrid
            new_grid[d, 0] = self.grid[d, 0]
            new_grid[d, -1] = self.grid[d, -1]
            i = 0        # new_x index
            j = -1         # self_x index
            acc_f = 0   # sum(avg_f) accumulated
            f_ninc = 0.
            for i in range(old_ninc):
                f_ninc += avg_f[i]
            f_ninc /= new_ninc     # amount of acc_f per new increment
            for i in range(1, new_ninc):
                while acc_f < f_ninc:
                    j += 1
                    if j < old_ninc:
                        acc_f += avg_f[j]
                    else:
                        break
                else:
                    acc_f -= f_ninc
                    new_grid[d, i] = (
                        self.grid[d, j+1] 
                        - (acc_f / avg_f[j]) * self.inc[d, j]
                        )
                    continue
                break
        self._init_grid(new_grid)
        self.sum_f = None
        self.n_f = None

    def show_grid(self, ngrid=40, axes=None, shrink=False):
        """ Display plots showing the current grid. 

        :param ngrid: The number of grid nodes in each 
            direction to include in the plot. The default is 40.
        :type ngrid: int 
        :nparam axes: List of pairs of directions to use in 
            different views of the grid. Using ``None`` in 
            place of a direction plots the grid for only one
            direction. Omitting ``axes`` causes a default 
            set of pairings to be used.
        :param shrink: Display entire range of each axis
            if ``False``; otherwise shrink range to include
            just the nodes being displayed. The default is
            ``False``. 
        """
        try:
            import matplotlib.pyplot as plt 
        except ImportError:
            warnings.warn('matplotlib not installed; cannot show_grid')
            return
        dim = self.dim
        if axes is None:
            axes = []
            if dim == 1:
                axes = [(0, None)]
            for d in range(dim):
                axes.append((d, (d + 1) % dim))
        else:
            if len(axes) <= 0:
                return
            for dx,dy in axes:
                if dx is not None and (dx < 0 or dx >= dim):
                    raise ValueError('bad directions: %s' % str((dx, dy)))
                if dy is not None and (dy < 0 or dy >= dim):
                    raise ValueError('bad directions: %s' % str((dx, dy)))
        fig = plt.figure()
        nskip = int(self.ninc // ngrid)
        if nskip < 1:
            nskip = 1
        start = nskip // 2
        def plotdata(idx, grid=numpy.asarray(self.grid)):
            dx, dy = axes[idx[0]]
            nnode = 0
            if dx is not None:
                xrange = [self.grid[dx, 0], self.grid[dx, -1]]
                xgrid = grid[dx, start::nskip]
                nnode = len(xgrid)
                xlabel = 'x[%d]' % dx 
            else:
                xrange = [0., 1.]
                xgrid = None
                xlabel = ''
            if dy is not None:
                yrange = [self.grid[dy, 0], self.grid[dy, -1]]
                ygrid = grid[dy, start::nskip]
                nnode = len(ygrid)
                ylabel = 'x[%d]' % dy
            else:
                yrange = [0., 1.]
                ygrid = None
                ylabel = ''
            if shrink:
                if xgrid is not None:
                    xrange = [min(xgrid), max(xgrid)]
                if ygrid is not None:
                    yrange = [min(ygrid), max(ygrid)]
            if None not in [dx, dy]:
                fig_caption = 'axes %d, %d' % (dx, dy)
            elif dx is None and dy is not None:
                fig_caption = 'axis %d' % dy 
            elif dx is not None and dy is None:
                fig_caption = 'axis %d' % dx 
            else:
                return
            fig.clear()
            plt.title(
                "%s   (press 'n', 'p', 'q' or a digit)"
                % fig_caption
                )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            for i in range(nnode):
                if xgrid is not None:
                    plt.plot([xgrid[i], xgrid[i]], yrange, 'k-')
                if ygrid is not None:
                    plt.plot(xrange, [ygrid[i], ygrid[i]], 'k-')
            plt.xlim(*xrange)
            plt.ylim(*yrange)

            plt.draw()

        idx = [0]        
        def onpress(event, idx=idx):
            try:    # digit?
                idx[0] = int(event.key)
            except ValueError:
                if event.key == 'n':
                    idx[0] += 1
                    if idx[0] >= len(axes):
                        idx[0] = len(axes) - 1
                elif event.key == 'p':
                    idx[0] -= 1
                    if idx[0] < 0:
                        idx[0] = 0
                elif event.key == 'q':
                    plt.close()
                    return
                else:
                    return
            plotdata(idx)
        
        fig.canvas.mpl_connect('key_press_event', onpress)
        plotdata(idx)
        plt.show()       


cdef class Integrator(object):
    """ Adaptive multidimensional Monte Carlo integration.

    :class:`vegas.Integrator` objects make Monte Carlo 
    estimates of multidimensional functions ``f(x)``
    where ``x[d]`` is a point in the integration volume::

        integ = vegas.Integrator(integration_region)

        result = integ(f, nitn=10, neval=10000)

    The integator makes ``nitn`` estimates of the integral,  each
    using at most ``neval`` samples of the integrand, as it adapts to
    the specific features of the integrand. Successive estimates
    typically improve in accuracy until the integrator has fully
    adapted. The integrator returns the weighted average of all
    ``nitn`` estimates, together with an estimate of the statistical
    (Monte Carlo) uncertainty in that estimate of the integral. The
    result is an object of type :class:`RunningWAvg` (which is derived
    from :class:`gvar.GVar`).

    |Integrator|\s have a large number of parameters but the 
    only ones that most people will care about are: the
    number ``nitn`` of iterations of the |vegas| algorithm;
    the maximum number ``neval`` of integrand evaluations per
    iteration; and the damping parameter ``alpha``, which is used
    to slow down the adaptive algorithms when they would otherwise
    be unstable (e.g., very peaky integrands). Setting parameter
    ``analyzer=vegas.reporter()`` is sometimes useful, as well,
    since it causes |vegas| to print (on ``sys.stdout``) 
    intermediate results from each iteration, as they are 
    produced. This helps when each iteration takes a long time 
    to complete (e.g., an hour) because it allows you to 
    monitor progress as it is being made (or not).

    :param map: The integration region as specified by 
        an array ``map[d, i]`` where ``d`` is the 
        direction and ``i=0,1`` specify the lower
        and upper limits of integration in direction ``d``.

        ``map`` could also be the integration map from 
        another |Integrator|, or that |Integrator|
        itself. In this case the grid is copied from the 
        existing integrator.
    :type map: array or :class:`vegas.AdaptiveMap` 
        or :class:`vegas.Integrator`
    :param nitn: The maximum number of iterations used to 
        adapt to the integrand and estimate its value. The
        default value is 10; typical values range from 10
        to 20.
    :type nitn: positive int
    :param neval: The maximum number of integrand evaluations
        in each iteration of the |vegas| algorithm. Increasing
        ``neval`` increases the precision: statistical errors should
        fall at least as fast as ``sqrt(1./neval)`` and often
        fall much faster. The default value is 1000; real
        problems often require 10--100 times more evaluations
        than this.
    :type neval: positive int 
    :param alpha: Damping parameter controlling the remapping
        of the integration variables as |vegas| adapts to the
        integrand. Smaller values slow adaptation, which may be
        desirable for difficult integrands. Small or zero ``alpha``\s 
        are also sometimes useful after the grid has adapted,
        to minimize fluctuations away from the optimal grid.
        The default value is 0.5.
    :type alpha: float 
    :param beta: Damping parameter controlling the redistribution
        of integrand evaluations across hypercubes in the 
        stratified sampling of the integrand (over transformed
        variables). Smaller values limit the amount of 
        redistribution. The theoretically optimal value is 1;
        setting ``beta=0`` prevents any redistribution of 
        evaluations. The default value is 0.75.
    :type beta: float 
    :param nhcube_vec: The number of hypercubes (in |y| space)
        whose integration points are combined into a single
        vector to be passed to the integrand, all together,
        when using |vegas| in vector mode (see ``fcntype='vector'``
        below). The default value is 100. Larger values may be
        lead to faster evaluations, but at the cost of 
        more memory for internal work arrays.
    :type nhcube_vec: positive int 
    :param minimize_mem: When ``True``, |vegas| minimizes 
        internal workspace at the cost of extra evaluations of
        the integrand. This can increase execution time by 
        50--100% but might be desirable when the number of 
        evaluations is very large (e.g., ``neval=1e9``). Normally
        |vegas| uses internal work space that grows in 
        proportion to ``neval``. If that work space exceeds
        the size of the RAM available to the processor,
        |vegas| runs much more slowly. Setting ``minimize_mem=True``
        greatly reduces the internal storage used by |vegas|; in 
        particular memory becomes independent of ``neval``. The default
        setting (``minimize_mem=False``), however, is much superior 
        unless memory becomes a problem. (The large memory is needed 
        for adaptive stratified sampling, so memory is not 
        an issue if ``beta=0``.)
    :type minimize_mem: bool
    :param adapt_to_errors: ``adapt_to_errors=False`` causes 
        |vegas| to remap the integration variables to emphasize
        regions where ``|f(x)|`` is largest. This is 
        the default mode.

        ``adapt_to_errors=True`` causes |vegas| to remap 
        variables to emphasize regions where the Monte Carlo
        error is largest. This might be superior when 
        the number of the number of stratifications in 
        the |y| grid is large (> 50?). It is typically 
        useful only in one or two dimensions.
    :type adapt_to_errors: bool
    :param maxinc_axis: The maximum number of increments
        per axis allowed for the |x|-space grid. The default 
        value is 1000; there is probably little need to use
        other values.
    :type maxinc_axis: positive int 
    :param max_nhcube: Maximum number of |y|-space hypercubes 
        used for stratified sampling. Setting ``max_nhcube=1``
        turns stratified sampling off, which is probably never 
        a good idea. The default setting (5e8) was chosen to 
        correspond to the point where internal work arrays 
        become comparable in size to the typical amount of RAM 
        available to a processor (in a laptop in 2014). 
        Internal memory usage is large only when ``beta>0``
        and ``minimize_mem=False``; therefore ``max_nhcube`` is 
        ignored if ``beta=0`` or ``minimize_mem=True``.
    :type max_nhcube: positive int 
    :param max_neval_hcube: Maximum number of integrand evaluations 
        per hypercube in the stratification. The default value 
        is 1e7. Larger values might allow for more adaptation
        (when ``neval`` is larger than ``2 * max_neval_hcube``),
        but also can result in very large internal work arrays.
    :type max_neval_hcube: positive int
    :param fcntype: Specifies the default type of integrand.

        ``fcntype='scalar'`` imples that the integrand
        is a function ``f(x)`` of a single integration
        point ``x[d]``. 

        ``fcntype='vector'`` implies that 
        the integrand function takes three arguments:
        a list of integration points ``x[i, d]``,
        where ``i=0...nx-1`` labels the integration point
        and ``d`` labels the direction; a buffer
        ``f[i]`` into which the corresponding 
        integrand values are written; and the number
        ``nx`` of integration points provided. 

        The default is ``fcntype=scalar``, but this is 
        overridden if the integrand has a ``fcntype`` 
        attribute. It is also overridden for classes
        derived from :class:`vegas.VecIntegrand`, which are
        treated as ``fcntype='vector'`` integrands.
    :type fcntype: str
    :param rtol: Relative error in the integral estimate 
        at which point the integrator can stop. The default
        value is 0.0 which means that the integrator will
        complete all iterations specified by ``nitn``.
    :type rtol: float less than 1
    :param atol: Absolute error in the integral estimate 
        at which point the integrator can stop. The default
        value is 0.0 which means that the integrator will
        complete all iterations specified by ``nitn``.
    :type atol: float 
    :param analyzer: An object with methods 

            ``analyzer.begin(itn, integrator)``

            ``analyzer.end(itn_result, result)``

        where: ``begin(itn, integrator)`` is called at the start
        of each |vegas| iteration with ``itn`` equal to the 
        iteration number and ``integrator`` equal to the 
        integrator itself; and ``end(itn_result, result)``
        is called at the end of each iteration with 
        ``itn_result`` equal to the result for that 
        iteration and ``result`` equal to the cummulative
        result of all iterations so far. 
        Setting ``analyzer=vegas.reporter()``, for 
        example, causes vegas to print out a running report
        of its results as they are produced. The default 
        is ``analyzer=None``.
    """

    # Settings accessible via the constructor and Integrator.set
    defaults = dict(
        map=None,
        fcntype='scalar', # default integrand type
        neval=1000,       # number of evaluations per iteration
        maxinc_axis=1000,  # number of adaptive-map increments per axis
        nhcube_vec=100,    # number of h-cubes per vector
        max_nhcube=5e8,    # max number of h-cubes
        max_neval_hcube=1e7, # max number of evaluations per h-cube
        nitn=10,           # number of iterations
        alpha=0.5,
        beta=0.75,
        adapt=True,
        minimize_mem=False,
        adapt_to_errors=False,
        rtol=0,
        atol=0,
        analyzer=None,
        )

    def __init__(Integrator self not None, map, **kargs):
        # N.B. All attributes initialized automatically by cython.
        #      This is why self.set() works here.
        if isinstance(map, Integrator):
            args = {}
            for k in Integrator.defaults:
                args[k] = getattr(map, k)
            args.update(kargs)
            self.set(args)
        else:
            args = dict(Integrator.defaults)
            del args['map']
            args.update(kargs)
            self.set(args)
            self.map = AdaptiveMap(map)
        self.sigf = numpy.array([], float) # dummy
        self.neval_hcube_range = None
        self.actual_neval = 0

    def __reduce__(Integrator self not None):
        """ Capture state for pickling. """
        odict = dict()
        for k in Integrator.defaults:
            if k in ['map']:
                continue
            odict[k] = getattr(self, k)
        return (Integrator, (self.map,), odict)

    def __setstate__(Integrator self not None, odict):
        """ Set state for unpickling. """
        self.set(odict)

    def set(Integrator self not None, ka={}, **kargs):
        """ Reset default parameters in integrator.

        Usage is analogous to the constructor
        for |Integrator|: for example, :: 
        
            old_defaults = integ.set(neval=1e6, nitn=20)

        resets the default values for ``neval`` and ``nitn``
        in |Integrator| ``integ``. A dictionary, here
        ``old_defaults``, is returned. It can be used 
        to restore the old defaults using, for example::

            integ.set(old_defaults)
        """
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_val = dict() 
        for k in kargs:
            if k == 'map':
                old_val[k] = self.map
                self.map = AdaptiveMap(kargs[k])
            elif k == 'fcntype':
                old_val[k] = self.fcntype
                self.fcntype = kargs[k]
            elif k == 'neval':
                old_val[k] = self.neval
                self.neval = kargs[k]
            elif k == 'maxinc_axis':
                old_val[k] = self.maxinc_axis
                self.maxinc_axis = kargs[k]
            elif k == 'nhcube_vec':
                old_val[k] = self.nhcube_vec
                self.nhcube_vec = kargs[k]
            elif k == 'max_nhcube':
                old_val[k] = self.max_nhcube
                self.max_nhcube = kargs[k]
            elif k == 'max_neval_hcube':
                old_val[k] = self.max_neval_hcube
                self.max_neval_hcube = kargs[k]
            elif k == 'nitn':
                old_val[k] = self.nitn
                self.nitn = kargs[k]
            elif k == 'alpha':
                old_val[k] = self.alpha
                self.alpha = kargs[k]
            elif k == 'adapt_to_errors':
                old_val[k] = self.adapt_to_errors
                self.adapt_to_errors = kargs[k]
            elif k == 'minimize_mem':
                old_val[k] = self.minimize_mem
                self.minimize_mem = kargs[k]
            elif k == 'beta':
                old_val[k] = self.beta
                self.beta = kargs[k]
            elif k == 'adapt':
                old_val[k] = self.adapt
                self.adapt = kargs[k]
            elif k == 'analyzer':
                old_val[k] = self.analyzer
                self.analyzer = kargs[k]
            elif k == 'rtol':
                old_val[k] = self.rtol
                self.rtol = kargs[k]
            elif k == 'atol':
                old_val[k] = self.atol
                self.atol = kargs[k]
            else:
                raise AttributeError('no attribute named "%s"' % str(k))
        return old_val

    def settings(Integrator self not None, ngrid=0):
        """ Assemble summary of integrator settings into string.

        :param ngrid: Number of grid nodes in each direction 
            to include in summary.
            The default is 0.
        :type ngrid: int
        :returns: String containing the settings.
        """
        cdef INT_TYPE d
        self._prepare_integrator()
        nhcube = self.nstrat ** self.dim
        neval = nhcube * self.min_neval_hcube if self.beta <= 0 else self.neval
        ans = ""
        ans = "Integrator Settings:\n"
        if self.beta > 0:
            ans = ans + (
                "    %d (max) integrand evaluations in each of %d iterations\n"
                % (self.neval, self.nitn)
                )
        else:
            ans = ans + (
                "    %d integrand evaluations in each of %d iterations\n"
                % (neval, self.nitn)
                )
        ans = ans + (
            "    number of:  strata/axis = %d  increments/axis = %d\n"
            % (self.nstrat, self.map.ninc)
            )
        if self.beta > 0:
            ans = ans + (
                "                h-cubes = %d  evaluations/h-cube = %d (min)\n"
                % (nhcube, self.min_neval_hcube)
                )
        else:
            ans = ans + (
                    "                h-cubes = %d  evaluations/h-cube = %d\n"
                    % (nhcube, self.min_neval_hcube)
                    )
        ans += "                h-cubes/vector = %d\n" % self.nhcube_vec
        ans = ans + (
            "    minimize_mem = %s\n" 
            % ('True' if self.minimize_mem else 'False') 
            )
        ans = ans + (
            "    adapt_to_errors = %s\n" 
            % ('True' if self.adapt_to_errors else 'False') 
            )
        ans = ans + (
            "    damping parameters: alpha = %g  beta= %g\n" 
            % (self.alpha, self.beta)
            )
        ans += (
            "    limits: h-cubes < %.2g  evaluations/h-cube < %.2g\n"
            % (float(self.max_nhcube), float(self.max_neval_hcube))
            )
        ans = ans + ("    accuracy: relative = %g" % self.rtol)
        ans = ans + ("  absolute accuracy = %g\n\n" % self.atol)
        for d in range(self.dim):
            ans = ans +(
                "    axis %d covers %s\n" % (d, str(self.map.region(d)))
                )
        if ngrid > 0:
            ans += '\n' + self.map.settings(ngrid=ngrid)
        return ans

    def __call__(Integrator self not None, fcn, **kargs):
        """ Integrate ``fcn``, possibly changing default parameters.
        
        This is the main driver for the integration. The integration
        itself is broken up into a whole bunch of different methods.
        This was done to isolate various independent components of 
        the algorithm, so modification and maintenance would be
        simpler. These different methods share a set of work arrays
        that are stored in the class itself in order to avoid 
        having to constantly reallocate space for them; these 
        include: self.x[i,d] which holds integration points;
        self.y[i,d] which holds the points in y-space; self.jac[i]
        which holds the Jacobian; and self.sigf[i] which holds 
        values of sigma_f from different y-space h-cubes.

        Integration points are collected into batches called 
        vectors to be sent to the integrand. The vectorization 
        greatly reduces the Python overheads.

        The main components, most of which are Cython functions, are:

        self._prep_integrator(..) --- determines how many increments 
            and stratifications to use, and initializes work arrays.

        self._calculate_neval_hcube(..) --- determines how many 
            integrand evaluations to use in each y-space h-cube.

        self._resize_workareas(..) --- resizes work arrays if 
            needed. This isn't called often because the arrays
            get big enough pretty quickly.

        self._generate_random_y_x_jac(..) --- generates random 
            points in the subset of hypercubes comprising the 
            current vector.

        self._integrate_vec(..) --- evaluates the integrand at the 
            points created by _generate_random_y_x_jac(..), and 
            evaluates the integral for each h-cube in the current
            vector.

        self._integrate(..) --- runs the script for a single 
            iteration of the vegas algorithm.
        """
        cdef INT_TYPE itn

        # save old settings so they can be restored at end
        if kargs:
            old_kargs = self.set(kargs)
        else:
            old_kargs = {}

        fcntype = getattr(fcn, 'fcntype', self.fcntype)
        if fcntype == 'scalar':
            fcn = VecPythonIntegrand(fcn)

        # main iteration loop
        self._prepare_integrator()
        if (
            (not self.adapt) 
            and (self.beta > 0) 
            and (self.sum_sigf == len(self.sigf))
            ):
            # extra iteration to fill sigf array for adapt. strat. sampling
            self.adapt = True
            self._integrate(fcn)
            self.adapt = False
        for itn in range(self.nitn):    # iterate
            if self.analyzer != None:
                self.analyzer.begin(itn, self)
            self.result.add(self._integrate(fcn))
            if self.analyzer != None:
                self.analyzer.end(self.result.itn_results[-1], self.result)
            if self.adapt:
                self.map.adapt(alpha=self.alpha)
            if (self.rtol * abs(self.result.mean) + self.atol) > self.result.sdev:
                break
        return self.result

    def _prepare_integrator(Integrator self not None):
        """ Prep the integrator before integrating.

        Decide how many increments and strata to use
        and allocate space to store the sigf values.
        """
        # determine # of strata, # of increments
        self.dim = self.map.dim
        neval_eff = (self.neval / 2.0) if self.beta > 0 else self.neval
        ns = int((neval_eff / 2.) ** (1. / self.dim))# stratifications/axis
        ni = int(self.neval / 10.)              # increments/axis
        if ns < 1:
            ns = 1
        elif (
            self.beta > 0 
            and ns ** self.dim > self.max_nhcube
            and not self.minimize_mem
            ):
            ns = int(self.max_nhcube ** (1. / self.dim))
        if ni < 1:
            ni = 1
        elif ni  > self.maxinc_axis:
            ni = self.maxinc_axis
        # want even number increments in each stratification 
        # or vise versa
        if ns > ni:
            if ns < self.maxinc_axis:
                ni = ns
            else:
                ns = int(ns // ni) * ni
        else:
            ni = int(ni // ns) * ns
        if self.adapt_to_errors:
            # ni > ns makes no sense with this mode
            if ni > ns:
                ni = ns

        # rebuild map with correct number of increments
        self.map.adapt(ninc=ni)    

        # determine min number of evaluations per h-cube
        self.nstrat = ns
        nhcube = self.nstrat ** self.dim
        self.min_neval_hcube = int(neval_eff // nhcube)
        if self.min_neval_hcube < 2:
            self.min_neval_hcube = 2

        self._init_workareas()

    def _init_workareas(Integrator self not None):
        """ Allocate space for and initialize work arrays. 

        This method is called once, at the beginning before the 
        first iteration of vegas, whenever the Integrator is 
        applied to an integrand. This means that the work arrays
        are not resized much after the first iteration --- they
        are as big as they need to be. Note that sum_sigf and 
        sigf are reinitialized only if sigf is the wrong size.
        """
        cdef INT_TYPE neval_vec = self.nhcube_vec * self.min_neval_hcube
        cdef INT_TYPE nsigf = (
            self.nhcube_vec if self.minimize_mem else
            self.nstrat ** self.dim
            )
        if self.beta > 0 and len(self.sigf) != nsigf:
            if self.minimize_mem:
                self.sigf = numpy.empty(nsigf, float)
                self.sum_sigf = HUGE
            else:
                self.sigf = numpy.ones(nsigf, float)
                self.sum_sigf = nsigf
        self.neval_hcube = (
            numpy.zeros(self.nhcube_vec, int) + self.min_neval_hcube 
            )
        self.y = numpy.empty((neval_vec, self.dim), float)
        self.x = numpy.empty((neval_vec, self.dim), float)
        self.jac = numpy.empty(neval_vec, float)
        self.f = numpy.empty(neval_vec, float)
        self.fdv2 = numpy.empty(neval_vec, float)
        self.result = RunningWAvg()

    def _resize_workareas(Integrator self not None, INT_TYPE neval_vec):
        " Check that work arrays are adequately large; resize if necessary. "
        if self.y.shape[0] >= neval_vec:
            return
        self.y = numpy.empty((neval_vec, self.dim), float)
        self.x = numpy.empty((neval_vec, self.dim), float)
        self.jac = numpy.empty(neval_vec, float)
        self.f = numpy.empty(neval_vec, float)
        self.fdv2 = numpy.empty(neval_vec, float)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def _calculate_neval_hcube(
        Integrator self not None, 
        INT_TYPE hcube_base,
        INT_TYPE nhcube_vec,
        fcn = None,
        ):
        " Determine the number of integrand evaluations for each h-cube. "
        cdef INT_TYPE[::1] neval_hcube = self.neval_hcube
        cdef INT_TYPE neval_vec
        cdef INT_TYPE ihcube
        cdef double[::1] sigf
        cdef double neval_sigf
        cdef double dummy, vec_sigf = 0.0
        if self.beta > 0:
            if self.minimize_mem:
                sigf = self.sigf
                # fill sigf by taking samples
                neval_hcube[:] = self.min_neval_hcube
                neval_vec = nhcube_vec * self.min_neval_hcube
                self._generate_random_y_x_jac(
                        neval_vec=neval_vec,
                        hcube_base=hcube_base, 
                        nhcube_vec=nhcube_vec,
                        )
                vec_sigf = self._integrate_vec(
                    fcn=fcn, 
                    neval_vec=neval_vec, 
                    hcube_base=hcube_base, 
                    nhcube_vec=nhcube_vec,
                    )[2]
            else:
                # sigf filled by last iteration
                sigf = self.sigf[hcube_base:]
            neval_sigf = self.neval / 2. / self.sum_sigf
            neval_vec = 0
            for ihcube in range(nhcube_vec):
                neval_hcube[ihcube] = <int> (sigf[ihcube] * neval_sigf)
                if neval_hcube[ihcube] < self.min_neval_hcube:
                    neval_hcube[ihcube] = self.min_neval_hcube
                if neval_hcube[ihcube] > self.max_neval_hcube:
                    neval_hcube[ihcube] = self.max_neval_hcube
                if neval_hcube[ihcube] < self.neval_hcube_range[0]:
                    self.neval_hcube_range[0] = neval_hcube[ihcube]
                elif neval_hcube[ihcube] > self.neval_hcube_range[1]:
                    self.neval_hcube_range[1] = neval_hcube[ihcube]
                neval_vec += neval_hcube[ihcube]
        else:
            neval_hcube[:] = self.min_neval_hcube
            neval_vec = nhcube_vec * self.min_neval_hcube
        return (neval_vec, vec_sigf)
    
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def _generate_random_y_x_jac(
        Integrator self not None, 
        INT_TYPE neval_vec,
        INT_TYPE hcube_base, 
        INT_TYPE nhcube_vec,
        ):
        " Generate random integration points for h-cubes in current vector. "
        cdef double[:, ::1] y = self.y
        cdef double[:, ::1] x = self.x
        cdef double[::1] jac = self.jac
        cdef INT_TYPE[::1] neval_hcube = self.neval_hcube
        cdef INT_TYPE ihcube, hcube, tmp_hcube
        cdef INT_TYPE i_start=0
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef INT_TYPE i, d
        cdef double[:, ::1] yran = numpy.random.uniform(
            0., 1., (neval_vec, self.dim)
            )
        for ihcube in range(nhcube_vec):
            hcube = hcube_base + ihcube
            tmp_hcube = hcube
            for d in range(self.dim):
                y0[d] = tmp_hcube % self.nstrat
                tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
            for d in range(self.dim):
                for i in range(i_start, i_start + neval_hcube[ihcube]):
                    y[i, d] = (y0[d] + yran[i, d]) / self.nstrat
            i_start += neval_hcube[ihcube]
        self.map.map(y, x, jac, neval_vec)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def _integrate_vec(
        Integrator self not None, 
        fcn,
        INT_TYPE neval_vec,
        INT_TYPE hcube_base, 
        INT_TYPE nhcube_vec,
        ):
        " Do integral for h-cubes in current vector. "
        cdef INT_TYPE i_start, i, tmp_hcube, ihcube #, hcube
        cdef double sum_fdv, sum_fdv2, mean, var, sigf2, fdv
        cdef INT_TYPE y0_d
        cdef double vec_mean = 0.0 
        cdef double vec_var = 0.0
        cdef double vec_sigf = 0.0
        cdef double dv_y = (1./self.nstrat) ** self.dim
        cdef INT_TYPE neval_hcube
        cdef double[::1] sigf
        if self.beta > 0:
            if self.minimize_mem:
                sigf = self.sigf
            else:
                sigf = self.sigf[hcube_base:]
        fcn(self.x, self.f, neval_vec)
        
        # compute integral h-cube by h-cube
        i_start = 0
        for ihcube in range(nhcube_vec):
            sum_fdv = 0.0
            sum_fdv2 = 0.0
            neval_hcube = self.neval_hcube[ihcube]
            for i in range(i_start, i_start + neval_hcube):
                fdv = self.f[i] * self.jac[i] * dv_y
                self.fdv2[i] = fdv ** 2
                sum_fdv += fdv
                sum_fdv2 += self.fdv2[i]
            mean = sum_fdv / neval_hcube
            sigf2 = abs(sum_fdv2 / neval_hcube - mean * mean)
            if self.beta > 0 and self.adapt:
                sigf[ihcube] = sigf2 ** (self.beta / 2.)
                vec_sigf += sigf[ihcube]
            var = sigf2 / (neval_hcube - 1.)
            vec_mean += mean
            vec_var += var
            if self.adapt and self.adapt_to_errors:
                self.fdv2[i_start] = var
                self.map.add_training_data(self.y[i_start:,:], self.fdv2[i_start:], 1)
            i_start += neval_hcube
        if self.adapt and not self.adapt_to_errors:
            self.map.add_training_data(self.y, self.fdv2, neval_vec)
        return (vec_mean, vec_var, vec_sigf)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def _integrate(Integrator self not None, fcn not None):
        """ Do integral for one iteration. """
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef INT_TYPE nhcube_vec = min(self.nhcube_vec, nhcube)
        cdef double ans_mean = 0.
        cdef double ans_var = 0.
        cdef double sum_sigf = 0.
        cdef double vec_mean, vec_var
        cdef double vec_sigf
        cdef INT_TYPE neval_vec
        cdef INT_TYPE hcube_base 

        # iterate over h-cubes in batches of nhcube_vec h-cubes
        # this allows for vectorization, to reduce python overhead
        self.actual_neval = 0
        self.neval_hcube_range = numpy.zeros(2, int) + self.min_neval_hcube        
        for hcube_base in range(0, nhcube, nhcube_vec):
            if (hcube_base + nhcube_vec) > nhcube:
                nhcube_vec = nhcube - hcube_base 
            neval_vec, vec_sigf = self._calculate_neval_hcube(
                    hcube_base=hcube_base,
                    nhcube_vec=nhcube_vec,
                    fcn=fcn,
                    )
            if self.minimize_mem:
                sum_sigf += vec_sigf
            self.actual_neval += neval_vec
            self._resize_workareas(neval_vec)
            self._generate_random_y_x_jac(
                    neval_vec=neval_vec,
                    hcube_base=hcube_base, 
                    nhcube_vec=nhcube_vec,
                    )
            vec_mean, vec_var, vec_sigf = self._integrate_vec(
                fcn=fcn, 
                neval_vec=neval_vec, 
                hcube_base=hcube_base, 
                nhcube_vec=nhcube_vec,
                )
            ans_mean += vec_mean 
            ans_var += vec_var
            if not self.minimize_mem:
                sum_sigf += vec_sigf
        if self.adapt and self.beta > 0:
            self.sum_sigf = sum_sigf
        if numpy.shape(ans_mean) == ():
            return gvar.gvar(ans_mean, sqrt(ans_var))
        else:
            return gvar.gvar(ans_mean, ans_var)

    def random_vec(Integrator self not None):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an 
        integral. The points are provided in arrays ``x[i, d]`` where 
        ``i=0...`` labels the integration points in a batch 
        (or vector) and ``d=0...`` labels direction. The corresponding
        weights assigned by |vegas| to each point are provided
        in an array ``wgt[i]``. 

        Given an |Integrator| ``integ``, presumably trained on some
        integrand, the following code would create a Monte Carlo
        estimate of the integral of a possibly different 
        (vector) integrand ``f(x)``::

            integral = 0.0
            for x, wgt in integ.random_vec():
                f_array = f(x)
                integral += numpy.dot(wgt, f_array)

        Here ``f(x)`` returns an array ``f_array[i]`` corresponding
        to the integrand values for points ``x[i, d]``. The points and
        weights yielded by the iterator are memoryview objects which
        can be converted to :mod:`numpy` arrays, if needed, using::

            x = numpy.asarray(x)
            wgt = numpy.asarray(wgt)

        """
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef double dv_y = 1. / nhcube
        cdef INT_TYPE nhcube_vec = min(self.nhcube_vec, nhcube)
        cdef INT_TYPE neval_vec
        cdef INT_TYPE hcube_base 
        cdef INT_TYPE i_start, ihcube, i
        cdef double vec_sigf
        old_defaults = self.set(adapt=False)
        for hcube_base in range(0, nhcube, nhcube_vec):
            if (hcube_base + nhcube_vec) > nhcube:
                nhcube_vec = nhcube - hcube_base 
            neval_vec, vec_sigf = self._calculate_neval_hcube(
                    hcube_base=hcube_base,
                    nhcube_vec=nhcube_vec,
                    )
            self._resize_workareas(neval_vec)
            self._generate_random_y_x_jac(
                    neval_vec=neval_vec,
                    hcube_base=hcube_base, 
                    nhcube_vec=nhcube_vec,
                    )
            i_start = 0
            for ihcube in range(nhcube_vec):
                neval_hcube = self.neval_hcube[ihcube]
                for i in range(i_start, i_start + neval_hcube):
                    self.jac[i] *= dv_y / neval_hcube
                i_start += neval_hcube
            yield (
                numpy.asarray(self.x[:neval_vec, :]), 
                numpy.asarray(self.jac[:neval_vec]),
                )
        self.set(old_defaults)

    def random(Integrator self not None):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an 
        integral. Each point ``x[d]`` is accompanied by the weight
        assigned to that point by |vegas| when estimating an integral.

        Given an |Integrator| ``integ``, presumably trained on some
        integrand, the following code would create a Monte Carlo
        estimate of the integral of a possibly different integrand ``f(x)``::

            integral = 0.0
            for x, wgt in integ.random():
                integral += wgt * f(x)

        Here ``f(x)`` returns an array ``f_array[i]`` corresponding
        to the integrand values for points ``x[i, d]``.
        """
        cdef double[:, ::1] x 
        cdef double[::1] wgt
        # numpy.ndarray[numpy.double_t, ndim=2] x
        # numpy.ndarray[numpy.double_t, ndim=1] wgt
        cdef INT_TYPE i
        for x,wgt in self.random_vec():
            for i in range(x.shape[0]):
                yield (x[i], wgt[i])

    def multi(Integrator self not None, fcn, nitn=10):
        """ Estimate multiple integrals simultaneously.

        This method estimates integrals for arrays (with any shape) of 
        integrands using the same integration points for every 
        integral. A typical application might look something
        like the following::

            def p(x):
                ... some function of x[d] ...

            def f(x):
                pp = p(x)
                return [pp, pp * x[0], pp * x[1]]

            integ = Integrator(...)

            # train the integrator on p(x)
            training = integ(p, ...)

            # compute multiple integrals
            result = integ.multi(f, nitn=20)
            ... use integral estimates result[i] for i=0, 1, 2 ...

        Here the integrator is first trained on function ``p(x)`` in 
        a normal integration step. The trained integrator is then
        applied to function ``f(x)`` which returns values for 
        three different integrands, arranged in an array. 

        The number of integration points used and the adaptations 
        are carried over from the training step. |vegas|
        does *not* adapt to ``f(x)`` in ``multi``, 
        which is why there is a training step.

        :meth:`vegas.Integrator.multi` also works for vectorized
        integrands from classes of the form::

            class fvec(vegas.VecIntegrand):
                ...
                def __call__(self, x):
                    ... x[i, d] are integration points ...
                    ... f[i, s1, s2, ...] are integrand values ...
                    return f
                ...

        ``fvec()`` creates an integrand that accepts multiple
        integration points ``x[i, d]`` and returns multiple integrand 
        values ``f[i, s1, s2, ...]`` where ``i`` labels the integration 
        point, ``d`` labels the direction, and ``s1, s2, ...`` label the
        different integrands.

        The covariance matrix for the integral estimates is determined
        from fluctuations between the ``nitn`` iterations. Taking 
        ``nitn=10`` or ``20`` usually results in error estimates 
        that are accurate to within 15--20%.

        This method requires the :mod:`gvar` module from ``lsqfit``
        (install using ``pip install lsqfit``, for example). 
        """
        cdef double[:, ::1] x 
        cdef double[::1] wgt
        cdef INT_TYPE itn
        if not have_gvar:
            raise ImportError(
                'cannot find gvar module (from lsqfit) -- needed for Integrator.multi'
                )
        if nitn <= 1:
            raise ValueError('nitn must be greater than 1, not ' + str(nitn))
        assert nitn>1, "nitn must be greater than 1"
        fcntype = getattr(fcn, 'fcntype', self.fcntype)
        results = []
        if fcntype == 'scalar':
            for itn in range(nitn):
                integral = 0.0
                for x, wgt in self.random_vec():
                    for i in range(x.shape[0]):
                        integral += numpy.asarray(fcn(x[i])) * wgt[i]
                results.append(integral)
            return gvar.dataset.avg_data(results)
        else:
            for itn in range(nitn):
                integral = 0.0
                for x, wgt in self.random_vec():
                    integral += numpy.dot(wgt, fcn(x))
                results.append(integral)
            return gvar.dataset.avg_data(results)


class reporter:
    """ Analyzer class that prints out a report, iteration
    by interation, on how vegas is doing. Parameter ngrid
    specifies how many x[i]'s to print out from the maps
    for each axis.

    :param ngrid: Number of grid nodes printed out for 
        each direction. Default is 0.
    :type ngrid: int 
    """
    def __init__(self, ngrid=0):
        self.ngrid = ngrid

    def begin(self, itn, integrator):
        self.integrator = integrator
        self.itn = itn
        if itn==0:
            print(integrator.settings())

    def end(self, itn_ans, ans):
        print "    itn %2d: %s\n all itn's: %s"%(self.itn+1, itn_ans, ans)
        print(
            '    neval = %d  neval/h-cube = %s\n    chi2/dof = %.2f  Q = %.2f' 
            % (
                self.integrator.actual_neval, 
                tuple(self.integrator.neval_hcube_range),
                ans.chi2 / ans.dof if ans.dof > 0 else 0,
                ans.Q if ans.dof > 0 else 1.,
                )
            )
        print(self.integrator.map.settings(ngrid=self.ngrid))
        print('')


################
# Classes for standarizing different types of integrands.

cdef class VecPythonIntegrand:
    # cdef object fcn
    """ Vector integrand from scalar Python integrand. 

    This class is used internally by |vegas|. It is unlikely 
    to be useful elsewhere.
    """
    def __init__(self, fcn):
        self.fcn = fcn
        self.fcntype = 'vector'

    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef INT_TYPE i 
        for i in range(nx):
            f[i] = self.fcn(x[i, :])

# preferred base class for vectorized integrands
# vectorized integrands are typically faster
# der
cdef class VecIntegrand:
    """ Base class for classes providing vectorized integrands.

    A class derived from :class:`vegas.VecInterand` will normally
    provide a ``__call__(self, x, f, nx)`` method where:

        ``x[i, d]`` is a contiguous array where ``i=0...nx-1``
        labels different integrtion points and ``d=0...`` labels
        different directions in the integration space.

        ``f[i]`` is a buffer that is filled with the 
        integrand values for points ``i=0...nx-1``.

        ``nx`` is the number of integration points.

    ``x[i, d]`` and ``f[i]`` are ``memoryview`` objects. They 
    can be repackaged inside ``__call__(x, f, nx)`` 
    as :mod:`numpy` arrays, if needed, using::

        x = numpy.asarray(x)[:nx, :]
        f = numpy.asarray(f)[:nx]

    This causes the :mod:`numpy` arrays to use the storage allocated
    internally by |vegas| for ``x`` and ``f``, which is what is 
    wanted for efficiency.

    :class:`vegas.VecIntegrand` is also used for vectorized integrands
    used in :meth:`vegas.Integrator.multi`. The derived class should
    then provice a ``__call__(self, x)`` method where again 
    ``x[i, d]`` is a contiguous array containing multiple integration 
    points, but which now returns an array ``f[i, s1, s2, ...]`` of 
    integrand values where ``s1, s2, ...`` label the different 
    integrands (the shape is arbitrary).

    Deriving from :class:`vegas.VecIntegrand` is the 
    easiest way to construct integrands in Cython, and
    gives the fastest results.
    """
    # cdef object fcntype
    def __cinit__(self, *args, **kargs):
        self.fcntype = 'vector'
































