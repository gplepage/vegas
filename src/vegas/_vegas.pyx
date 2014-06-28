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
try:
    import mpi4py
    import mpi4py.MPI 
except ImportError:
    mpi4py = None

cdef double TINY = 10 ** (sys.float_info.min_10_exp + 50)  # smallest and biggest
cdef double HUGE = 10 ** (sys.float_info.max_10_exp - 50)  # with extra headroom

# following two functions are here in case gvar (from lsqfit distribution)
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


try:
    import gvar
    have_gvar = True
except ImportError:
    have_gvar = False 

    # fake version of gvar.gvar
    # for use if gvar module not available
    class GVar(object):
        """ Poor substitute for GVar in the gvar package.

        This supports arithmetic involving GVars and numbers 
        but not arithmetic involving GVars and GVars. For
        the latter, you need to install the gvar module
        (either as part of the lsqfit distribution or 
        on its own: pip install gvar).

        This also supports log, sqrt, and exp, but not
        trig functions etc --- again install gvar if 
        these are needed.
        """
        def __init__(self, mean, sdev):
            self.mean = float(mean)
            self.sdev = abs(float(sdev))
            self.var = self.sdev ** 2
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
            # taken from gvar.GVar in gvar module (lsqfit distribution)
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
            if numpy.shape(mean) == ():
                return GVar(mean, sdev)
            else:
                mean = numpy.asarray(mean)
                var = numpy.asarray(sdev)
                assert mean.ndim == 1 and var.ndim == 2, 'vectors only'
                ans = numpy.empty(mean.shape, object)
                for i in range(len(mean)):
                    ans[i] = GVar(mean[i], var[i, i] ** 0.5)
                return ans
        def mean(self, glist):
            return numpy.array([g.mean for g in glist])            
        def var(self, glist):
            return numpy.array([g.sdev ** 2 for g in glist])
        def evalcov(self, glist):
            ans = numpy.zeros((len(glist), len(glist)), float)
            for i in range(len(glist)):
                ans[i, i] = glist[i].sdev ** 2
            return ans
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

class RAvg(gvar.GVar):
    """ Running average of Monte Carlo estimates.

    This class accumulates independent Monte Carlo 
    estimates (e.g., of an integral) and combines 
    them into a single average. It 
    is derived from :class:`gvar.GVar` (from 
    the :mod:`gvar` module if it is present) and 
    represents a Gaussian random variable.

    Different estimates are weighted by their
    inverse variances if parameter ``weight=True``;
    otherwise straight, unweighted averages are used.
    """
    def __init__(self, weighted=True):
        if weighted:
            self._v_s2 = 0.
            self._v2_s2 = 0.
            self._1_s2 = 0.
            self.weighted = True
        else:
            self._v = 0.
            self._v2 = 0.
            self._s2 = 0.
            self._n = 0
            self.weighted = False
        self.itn_results = []
        super(RAvg, self).__init__(
            *gvar.gvar(0., 0.).internaldata,
           )

    def _chi2(self):
        if len(self.itn_results) <= 1:
            return 0.0
        if self.weighted:
            return self._v2_s2 - self._v_s2 ** 2 / self._1_s2
        else:
            return (self._v2 - self.mean ** 2 * self._n) * self._n / self._s2
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
        tiny = sys.float_info.epsilon * 10 * g.mean ** 2 + TINY
        if self.weighted:
            var = g.sdev ** 2 + tiny
            self._v_s2 +=   g.mean / var
            self._v2_s2 +=  g.mean ** 2 / var
            self._1_s2 +=  1. / var
            super(RAvg, self).__init__(*gvar.gvar(
                self._v_s2 / self._1_s2,
                sqrt(1. / self._1_s2),
                ).internaldata)
        else:
            self._v += g.mean 
            self._v2 += g.mean ** 2
            self._s2 += g.var + tiny
            self._n += 1
            super(RAvg, self).__init__(*gvar.gvar(
                self._v / self._n,
                sqrt(self._s2) / self._n,
                ).internaldata)


    def summary(self):
        """ Assemble summary of independent results into a string. """
        acc = RAvg()
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

class RAvgArray(numpy.ndarray):
    """ Running average of array-valued Monte Carlo estimates.

    This class accumulates independent arrays of Monte Carlo 
    estimates (e.g., of an integral) and combines 
    them into an array of averages. It 
    is derived from :class:`numpy.ndarray`. The array
    elements are :class:`gvar.GVar`\s (from the ``gvar`` module if
    present) and represent Gaussian random variables.

    Different estimates are weighted by their
    inverse covariance matrices if parameter ``weight=True``;
    otherwise straight, unweighted averages are used.
    """
    def __new__(
        subtype, shape, weighted=True,
        dtype=object, buffer=None, offset=0, strides=None, order=None
        ):
        obj = numpy.ndarray.__new__(
            subtype, shape=shape, dtype=object, buffer=buffer, offset=offset, 
            strides=strides, order=order
            )
        if buffer is None:
            obj.flat = numpy.array(obj.size * [gvar.gvar(0,0)])
        obj.itn_results = []
        if weighted:
            obj._invcov_v = 0.
            obj._v_invcov_v = 0.
            obj._invcov = 0.
            obj.weighted = True
        else:
            obj._v = 0.
            obj._v2 = 0.
            obj._cov = 0.
            obj._n = 0
            obj.weighted = False
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.itn_results = getattr(obj, 'itn_results', [])
        if obj.weighted:
            self._invcov_v = getattr(obj, '_invcov_v', 0.0)
            self._v_invcov_v = getattr(obj, '_v_invcov_v', 0.0)
            self._invcov = getattr(obj, '_invcov', 0.0)
            self.weighted = getattr(obj, 'weighted', True)
        else:
            self._v = getattr(obj, '_v', 0.)
            self._v2 = getattr(obj, '_v2', 0.)
            self._cov = getattr(obj, '_cov', 0.)
            self._n = getattr(obj, '_n', 0.)
            self.weighted = getattr(obj, 'weighted', False)

    def _inv(self, matrix):
        " Invert matrix, with protection against singular matrices. "
        # if True:
        #     return numpy.linalg.inv(matrix)
        if not have_gvar:
            return numpy.linalg.pinv(matrix, rcond=sys.float_info.epsilon * 10)
        svd = gvar.SVD(matrix, svdcut=sys.float_info.epsilon * 10, rescale=True)
        w = svd.decomp(-1)
        return numpy.sum(
            [numpy.outer(wi, wi) for wi in reversed(svd.decomp(-1))], 
            axis=0
            )

    def _chi2(self):
        if len(self.itn_results) <= 1:
            return 0.0
        if self.weighted:
            cov = self._inv(self._invcov)
            return self._v_invcov_v - self._invcov_v.dot(cov.dot(self._invcov_v))
        else:
            invcov = self._inv(self._cov / self._n)
            return numpy.trace(   # inefficient -- fix at some point
                (self._v2 - numpy.outer(self._v, self._v) / self._n).dot(invcov)
                )
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        if len(self.itn_results) <= 1:
            return 0
        return (len(self.itn_results) - 1) * self.itn_results[0].size
    dof = property(
        _dof, None, None, 
        "Number of degrees of freedom in weighted average."
        )

    def _Q(self):
        if self.dof <= 0 or self.chi2 <= 0:
            return 1.
        return gvar.gammaQ(self.dof / 2., self.chi2 / 2.)
    Q = property(
        _Q, None, None, 
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )

    def add(self, g):
        """ Add estimate ``g`` to the running average. """
        g = numpy.asarray(g)
        self.itn_results.append(g)
        g = g.reshape((-1,))
        gmean = gvar.mean(g)
        tiny = sys.float_info.epsilon * 10
        gcov = gvar.evalcov(g) 
        gcov[numpy.diag_indices_from(gcov)] = (
            abs(gcov[numpy.diag_indices_from(gcov)])
            + tiny * gmean ** 2 + TINY
            )
        if self.weighted:
            invcov = self._inv(gcov)
            v = gvar.mean(g)
            u = invcov.dot(v)
            self._invcov += invcov
            self._invcov_v += u 
            self._v_invcov_v += v.dot(u)
            cov = self._inv(self._invcov)
            mean = cov.dot(self._invcov_v)
            self[:] = gvar.gvar(mean, cov).reshape(self.shape)
        else:
            gmean = gvar.mean(g)
            self._v2 += numpy.outer(gmean, gmean)
            self._v += gmean
            self._cov += gcov
            self._n += 1
            mean = self._v / self._n 
            cov = self._cov / (self._n ** 2)
            self[:] = gvar.gvar(mean, cov).reshape(self.shape)            
   
    def summary(self):
        """ Assemble summary of independent results into a string. """
        acc = RAvgArray(self.shape)

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
            integral = '%-15s' % res.flat[0]
            wgtavg = '%-15s' % acc.flat[0]
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
            y = numpy.random.random(self.dim)
        else:
            y = numpy.random.random((n, self.dim))
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
    the specific features of the integrand. Successive estimates (iterations)
    typically improve in accuracy until the integrator has fully
    adapted. The integrator returns the weighted average of all
    ``nitn`` estimates, together with an estimate of the statistical
    (Monte Carlo) uncertainty in that estimate of the integral. The
    result is an object of type :class:`RAvg` (which is derived
    from :class:`gvar.GVar`).

    Integrands can be array-valued, in which case ``f(x)`` 
    returns an array of values corresponding to different 
    integrands. Also |vegas| can generate integration points
    in batches for integrands built from classes
    derived from :class:`vegas.BatchIntegrand`, or integrand
    functions decorated by :func:`vegas.batchintegrand`. Batch 
    integrands are typically much faster, especially if they
    are coded in Cython.

    |Integrator|\s have a large number of parameters but the 
    only ones that most people will care about are: the
    number ``nitn`` of iterations of the |vegas| algorithm;
    the maximum number ``neval`` of integrand evaluations per
    iteration; and the damping parameter ``alpha``, which is used
    to slow down the adaptive algorithms when they would otherwise
    be unstable (e.g., with very peaky integrands). Setting parameter
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
        problems often require 10--1000 times more evaluations
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
        stratified sampling of the integral (over transformed
        variables). Smaller values limit the amount of 
        redistribution. The theoretically optimal value is 1;
        setting ``beta=0`` prevents any redistribution of 
        evaluations. The default value is 0.75.
    :type beta: float 
    :param adapt: Setting ``adapt=False`` prevents further 
        adaptation by |vegas|. Typically this would be done 
        after training the |Integrator| on an integrand, in order
        to stabilize further estimates of the integral. |vegas| uses 
        unweighted averages to combine results from different 
        iterations when ``adapt=False``. The default setting 
        is ``adapt=True``.
    :type adapt: bool
    :param nhcube_batch: The number of hypercubes (in |y| space)
        whose integration points are combined into a single
        batch to be passed to the integrand, together,
        when using |vegas| in batch mode.
        The default value is 1000. Larger values may be
        lead to faster evaluations, but at the cost of 
        more memory for internal work arrays.
    :type nhcube_batch: positive int 
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
        the number of the number of stratifications (``self.nstrat``)
        in the |y| grid is large (> 50?). It is typically 
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
        a good idea. The default setting (1e9) was chosen to 
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
    :param ran_array_generator: Function that generates 
        :mod:`numpy` arrays of random numbers distributed uniformly 
        between 0 and 1. ``ran_array_generator(shape)`` should 
        create an array whose dimensions are specified by the 
        integer-valued tuple ``shape``. The default generator
        is ``numpy.random.random``.
    """

    # Settings accessible via the constructor and Integrator.set
    defaults = dict(
        map=None,
        neval=1000,       # number of evaluations per iteration
        maxinc_axis=1000,  # number of adaptive-map increments per axis
        nhcube_batch=1000,    # number of h-cubes per batch
        max_nhcube=1e9,    # max number of h-cubes
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
        ran_array_generator=numpy.random.random,
        )

    def __init__(Integrator self not None, map, **kargs):
        # N.B. All attributes initialized automatically by cython.
        #      This is why self.set() works here.
        self.sigf = numpy.array([], float) # dummy
        self.neval_hcube_range = None
        self.last_neval = 0
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
            self.map = AdaptiveMap(map)
            self.set(args)

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
        # reset parameters
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_val = dict() 
        for k in kargs:
            if k == 'map':
                old_val[k] = self.map
                self.map = AdaptiveMap(kargs[k])
            elif k == 'nhcube_vec':
                # old name --- here for legacy reasons
                old_val['nhcube_batch'] = self.nhcube_batch
                self.nhcube_batch = kargs[k]
            elif k == 'neval':
                old_val[k] = self.neval
                self.neval = kargs[k]
            elif k == 'maxinc_axis':
                old_val[k] = self.maxinc_axis
                self.maxinc_axis = kargs[k]
            elif k == 'nhcube_batch':
                old_val[k] = self.nhcube_batch
                self.nhcube_batch = kargs[k]
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
            elif k == 'ran_array_generator':
                old_val[k] = self.ran_array_generator
                self.ran_array_generator = kargs[k]
            else:
                raise AttributeError('no attribute named "%s"' % str(k))

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
        self.nhcube = self.nstrat ** self.dim
        self.min_neval_hcube = int(neval_eff // self.nhcube)
        if self.min_neval_hcube < 2:
            self.min_neval_hcube = 2

        # allocate work arrays -- these are stored in the 
        # the Integrator so that the storage is held between
        # iterations, thereby minimizing the amount of allocating
        # that goes on
        neval_batch = self.nhcube_batch * self.min_neval_hcube
        nsigf = (
            self.nhcube_batch if self.minimize_mem else
            self.nhcube
            )
        if self.beta > 0 and len(self.sigf) != nsigf:
            if self.minimize_mem:
                self.sigf = numpy.empty(nsigf, float)
                self.sum_sigf = HUGE
            else:
                self.sigf = numpy.ones(nsigf, float)
                self.sum_sigf = nsigf
        self.neval_hcube = (
            numpy.zeros(self.nhcube_batch, int) + self.min_neval_hcube 
            )
        self.y = numpy.empty((neval_batch, self.dim), float)
        self.x = numpy.empty((neval_batch, self.dim), float)
        self.jac = numpy.empty(neval_batch, float)
        self.fdv2 = numpy.empty(neval_batch, float)
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
        ans += "                h-cubes/batch = %d\n" % self.nhcube_batch
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

    def _fill_sigf(
        Integrator self not None, fcn,  INT_TYPE hcube_base, INT_TYPE nhcube_batch
        ):
        cdef INT_TYPE i_start
        cdef INT_TYPE ihcube, hcube, tmp_hcube
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef INT_TYPE i, d
        cdef INT_TYPE neval_hcube = self.min_neval_hcube
        cdef INT_TYPE neval_batch = nhcube_batch * neval_hcube
        cdef double[:, ::1] yran = self.ran_array_generator((neval_batch, self.dim))
        cdef double dv_y = (1./self.nstrat) ** self.dim
        cdef double sum_fdv, sum_fdv2, sigf2, fdv
        cdef numpy.ndarray[numpy.double_t, ndim=1] fx
        cdef double[:, ::1] y = self.y[:neval_batch, :]
        cdef double[:, ::1] x = self.x[:neval_batch, :]
        cdef double[::1] jac = self.jac[:neval_batch]
        cdef double[::1] sigf = self.sigf
        # generate random points
        i_start = 0
        for ihcube in range(nhcube_batch):
            hcube = hcube_base + ihcube
            tmp_hcube = hcube
            for d in range(self.dim):
                y0[d] = tmp_hcube % self.nstrat
                tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
            for d in range(self.dim):
                for i in range(i_start, i_start + neval_hcube):
                    y[i, d] = (y0[d] + yran[i, d]) / self.nstrat
            i_start += neval_hcube
        self.map.map(y, x, jac, neval_batch)
        fx = fcn.training_f(numpy.asarray(x))

        # accumulate sigf for each h-cube
        i_start = 0
        for ihcube in range(nhcube_batch):
            sum_fdv = 0.0
            sum_fdv2 = 0.0
            for i in range(i_start, i_start + neval_hcube):
                fdv = fx[i] * self.jac[i] * dv_y
                sum_fdv += fdv
                sum_fdv2 += fdv ** 2
            mean = sum_fdv / neval_hcube
            sigf2 = abs(sum_fdv2 / neval_hcube - mean * mean)
            sigf[ihcube] = sigf2 ** (self.beta / 2.)

    def random_batch(
        Integrator self not None, 
        bint yield_hcube=False,
        bint yield_y=False,
        fcn = None,
        ):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an 
        integral. The points are provided in arrays ``x[i, d]`` where 
        ``i=0...`` labels the integration points in a batch 
        and ``d=0...`` labels direction. The corresponding
        weights assigned by |vegas| to each point are provided
        in an array ``wgt[i]``. 

        Optionally the integrator will also return the indices of 
        the hypercubes containing the integration points and/or the |y|-space 
        coordinates of those points::

            integ.random()  yields  x, wgt

            integ.random(yield_hcube=True) yields x, wgt, hcube 

            integ.random(yield_y=True) yields x, y, wgt

            integ.random(yield_hcube=True, yield_y=True) yields x, y, wgt, hcube
        
        The number of integration points returned by the iterator 
        corresponds to a single iteration. The number in a batch
        is controlled by parameter ``nhcube_batch``.
        """
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef double dv_y = 1. / nhcube
        cdef INT_TYPE nhcube_batch = min(self.nhcube_batch, nhcube)
        cdef INT_TYPE neval_batch
        cdef INT_TYPE hcube_base 
        cdef INT_TYPE i_start, ihcube, i, d, tmp_hcube, hcube
        cdef INT_TYPE[::1] hcube_array
        cdef double neval_sigf = (
            self.neval / 2. / self.sum_sigf 
            if self.beta > 0 and self.sum_sigf > 0 
            else HUGE
            )
        cdef INT_TYPE[::1] neval_hcube = self.neval_hcube
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef double[::1] sigf
        cdef double[:, ::1] yran
        cdef double[:, ::1] y
        cdef double[:, ::1] x
        cdef double[::1] jac
        self.last_neval = 0
        self.neval_hcube_range = numpy.zeros(2, int) + self.min_neval_hcube        
        if yield_hcube:
            hcube_array = numpy.empty(self.y.shape[0], int)
        for hcube_base in range(0, nhcube, nhcube_batch):
            if (hcube_base + nhcube_batch) > nhcube:
                nhcube_batch = nhcube - hcube_base 

            # determine number of evaluations per h-cube
            if self.beta > 0:
                if self.minimize_mem:
                    self._fill_sigf(
                        fcn=fcn, hcube_base=hcube_base, nhcube_batch=nhcube_batch,
                        )
                    sigf = self.sigf
                else:
                    sigf = self.sigf[hcube_base:]
                neval_batch = 0
                for ihcube in range(nhcube_batch):
                    neval_hcube[ihcube] = <int> (sigf[ihcube] * neval_sigf)
                    if neval_hcube[ihcube] < self.min_neval_hcube:
                        neval_hcube[ihcube] = self.min_neval_hcube
                    if neval_hcube[ihcube] > self.max_neval_hcube:
                        neval_hcube[ihcube] = self.max_neval_hcube
                    if neval_hcube[ihcube] < self.neval_hcube_range[0]:
                        self.neval_hcube_range[0] = neval_hcube[ihcube]
                    elif neval_hcube[ihcube] > self.neval_hcube_range[1]:
                        self.neval_hcube_range[1] = neval_hcube[ihcube]
                    neval_batch += neval_hcube[ihcube]
            else:
                neval_hcube[:] = self.min_neval_hcube
                neval_batch = nhcube_batch * self.min_neval_hcube
            self.last_neval += neval_batch

            # resize work arrays if needed
            if neval_batch > self.y.shape[0]:
                self.y = numpy.empty((neval_batch, self.dim), float)
                self.x = numpy.empty((neval_batch, self.dim), float)
                self.jac = numpy.empty(neval_batch, float)
                self.fdv2 = numpy.empty(neval_batch, float)
            y = self.y 
            x = self.x 
            jac = self.jac 

            # self._resize_workareas(neval_batch)
            if yield_hcube and neval_batch > hcube_array.shape[0]:
                hcube_array = numpy.empty(neval_batch, int)

            # generate random points
            yran = self.ran_array_generator((neval_batch, self.dim))
            i_start = 0
            for ihcube in range(nhcube_batch):
                hcube = hcube_base + ihcube
                tmp_hcube = hcube
                for d in range(self.dim):
                    y0[d] = tmp_hcube % self.nstrat
                    tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
                for d in range(self.dim):
                    for i in range(i_start, i_start + neval_hcube[ihcube]):
                        y[i, d] = (y0[d] + yran[i, d]) / self.nstrat
                i_start += neval_hcube[ihcube]
            self.map.map(y, x, jac, neval_batch)
            
            # compute weights and yield answers
            i_start = 0
            for ihcube in range(nhcube_batch):
                for i in range(i_start, i_start + neval_hcube[ihcube]):
                    jac[i] *= dv_y / neval_hcube[ihcube]
                    if yield_hcube:
                        hcube_array[i] = hcube_base + ihcube
                i_start += neval_hcube[ihcube]
            answer = (numpy.asarray(x[:neval_batch, :]),)
            if yield_y:
                answer += (numpy.asarray(y[:neval_batch, :]),)
            answer += (numpy.asarray(jac[:neval_batch]),)
            if yield_hcube:
                answer += (numpy.asarray(hcube_array[:neval_batch]),)
            yield answer

    # old name --- for legacy code
    random_vec = random_batch

    def random(
        Integrator self not None, bint yield_hcube=False, bint yield_y=False
        ):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an 
        integral. Each point ``x[d]`` is accompanied by the weight
        assigned to that point by |vegas| when estimating an integral.
        Optionally it will also return the index of the hypercube 
        containing the integration point and/or the |y|-space 
        coordinates::

            integ.random()  yields  x, wgt

            integ.random(yield_hcube=True) yields x, wgt, hcube 

            integ.random(yield_y=True) yields x, y, wgt

            integ.random(yield_hcube=True, yield_y=True) yields x, y, wgt, hcube
        
        The number of integration points returned by the iterator 
        corresponds to a single iteration.
        """
        cdef double[:, ::1] x 
        cdef double[::1] wgt
        cdef INT_TYPE[::1] hcube
        cdef double[:, ::1] y
        cdef INT_TYPE i
        if yield_hcube and yield_y:
            for x, y, wgt, hcube in self.random_batch(yield_hcube=True, yield_y=True):
                for i in range(x.shape[0]):
                    yield (x[i], y[i], wgt[i], hcube[i])
        elif yield_y:
            for x, y, wgt in self.random_batch(yield_y=True):
                for i in range(x.shape[0]):
                    yield (x[i], y[i], wgt[i])
        elif yield_hcube:
            for x, wgt, hcube in self.random_batch(yield_hcube=True):
                for i in range(x.shape[0]):
                    yield (x[i], wgt[i], hcube[i])
        else:
            for x,wgt in self.random_batch():
                for i in range(x.shape[0]):
                    yield (x[i], wgt[i])

    def __call__(Integrator self not None, fcn, **kargs):
        """ Integrate integrand ``fcn``.

        A typical integrand has the form, for example::

            def f(x):
                return x[0] ** 2 + x[1] ** 4

        The argument ``x[d]`` is an integration point, where 
        index ``d=0...`` represents direction within the 
        integration volume.

        Integrands can be array-valued, representing multiple 
        integrands: e.g., ::

            def f(x):
                return [x[0] ** 2, x[0] / x[1]]

        The return arrays can have any shape. Array-valued 
        integrands are useful for integrands that 
        are closely related, and can lead to 
        substantial reductions in the errors for 
        ratios or differences of the results.

        It is usually much faster to use |vegas| in batch
        mode, where integration points are presented to the 
        integrand in batches. A simple batch integrand might
        be, for example::

            @vegas.batchintegrand 
            def f(x):
                return x[:, 0] ** 2 + x[:, 1] ** 4

        where decorator ``@vegas.batchintegrand`` tells 
        |vegas| that the integrand processes integration
        points in batches. The array ``x[i, d]`` 
        represents a collection of different integration 
        points labeled by ``i=0...``. (The number of points is controlled
        |Integrator| parameter ``nhcube_batch``.) The batch index 
        is always first.

        Batch integrands can also be constructed from classes 
        derived from :class:`vegas.BatchIntegrand`.
    
        Batch mode is particularly useful (and fast) when the class 
        derived from :class:`vegas.BatchIntegrand` is coded 
        in Cython. Then loops over the integration points
        can be coded explicitly, avoiding the need to use
        :mod:`numpy`'s whole-array operators if they are not 
        well suited to the integrand.

        Any |vegas| parameter can also be reset: e.g., 
        ``self(fcn, nitn=20, neval=1e6)``.
        """
        # cdef double[::1] wgt
        cdef numpy.ndarray[numpy.double_t, ndim=2] x 
        cdef numpy.ndarray[numpy.double_t, ndim=1] wgt
        cdef numpy.ndarray[numpy.int_t, ndim=1] hcube
        cdef double[::1] sigf
        cdef double[:, ::1] y
        cdef double[::1] fdv2
        cdef numpy.ndarray _fx
        cdef double[:, ::1] fx
        cdef double[::1] wf
        cdef double[::1] sum_wf 
        cdef double[:, ::1] sum_wf2 
        cdef double[::1] mean 
        cdef double[:, ::1] var 
        cdef INT_TYPE itn, i, j, s, t, ns, neval, neval_batch
        cdef bint firstpass = True
        cdef double sum_sigf, sigf2

        if kargs:
            self.set(kargs)
        
        if isinstance(fcn, type(BatchIntegrand)):
            raise ValueError(
                'integrand given is a class, not an object -- need parentheses?'
                )

        fcntype = getattr(fcn, 'fcntype', 'scalar')
        if fcntype == 'scalar':
            fcn = _BatchIntegrand_from_NonBatch(fcn)
        
        sigf = self.sigf
        for itn in range(self.nitn):
            if self.analyzer is not None:
                self.analyzer.begin(itn, self)
            
            if not firstpass:
                mean[:] = 0.0
                var[:, :] = 0.0
            sum_sigf = 0.0

            # iterate batch-slices of integration points
            for x, y, wgt, hcube in self.random_batch(
                yield_hcube=True, yield_y=True, fcn=fcn
                ):
                fdv2 = self.fdv2        # must be inside loop
                
                # evaluate integrand at all points in x
                _fx = numpy.asarray(fcn(x))
                if firstpass:
                    # figure out integrand shape, initialize workareas
                    firstpass = False
                    integrand_shape = tuple(
                        [_fx.shape[s] for s in range(1, _fx.ndim)]
                        )
                    ns = _fx.size / _fx.shape[0]
                    wf = numpy.empty(ns, float)
                    sum_wf = numpy.empty(ns, float)
                    sum_wf2 = numpy.empty((ns, ns), float)
                    mean = numpy.zeros(ns, float)
                    var = numpy.zeros((ns, ns), float)
                    if integrand_shape == ():
                        result = RAvg(weighted=self.adapt)
                    else:
                        result = RAvgArray(
                            integrand_shape, weighted=self.adapt
                            )


                # repackage in simpler (uniform) format
                fx = _fx.reshape((x.shape[0], ns))
                
                # compute integral and variance for each h-cube
                j = 0
                for i in range(hcube[0], hcube[-1] + 1):
                    # iterate over h-cubes
                    sum_wf[:] = 0.0
                    sum_wf2[:, :] = 0.0
                    neval = 0
                    while j < len(hcube) and hcube[j] == i:
                        for s in range(ns):
                            wf[s] = wgt[j] * fx[j, s]
                            sum_wf[s] += wf[s]
                            for t in range(s + 1):
                                sum_wf2[s, t] += wf[s] * wf[t]
                        fdv2[j] = (wf[0] * self.neval_hcube[i - hcube[0]]) ** 2
                        j += 1
                        neval += 1
                    for s in range(fx.shape[1]):
                        mean[s] += sum_wf[s]
                        for t in range(s + 1):
                            var[s, t] += (
                                sum_wf2[s, t] * neval - sum_wf[s] * sum_wf[t]
                                ) / (neval - 1.)
                        if var[s, s] <= 0:
                            var[s, s] = mean[s] ** 2 * 1e-15 + TINY
                    if self.beta > 0 and self.adapt:
                        sigf2 = abs(sum_wf2[0, 0] * neval - sum_wf[0] * sum_wf[0])
                        if not self.minimize_mem:
                            sigf[i] = sigf2 ** (self.beta / 2.)
                        sum_sigf += sigf2 ** (self.beta / 2.)
                    if self.adapt_to_errors and self.adapt:
                        fdv2[j - 1] = var[0, 0]
                        self.map.add_training_data(
                            self.y[j - 1:, :], fdv2[j - 1:], 1
                            )
                if (not self.adapt_to_errors) and self.adapt and self.alpha > 0:
                    self.map.add_training_data(y, fdv2, y.shape[0])
            
            for s in range(var.shape[0]):
                for t in range(s):
                    var[t, s] = var[s, t]
            # create answer for this iteration, with correct shape
            if integrand_shape == ():
                result.add(gvar.gvar(mean[0], var[0,0] ** 0.5))
            else:
                result.add(gvar.gvar(mean, var).reshape(integrand_shape))
            
            if self.beta > 0 and self.adapt:
                self.sum_sigf = sum_sigf
            if self.alpha > 0 and self.adapt:
                self.map.adapt(alpha=self.alpha)
            if self.analyzer is not None:
                self.analyzer.end(result.itn_results[-1], result)
        return result

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
                self.integrator.last_neval, 
                tuple(self.integrator.neval_hcube_range),
                ans.chi2 / ans.dof if ans.dof > 0 else 0,
                ans.Q if ans.dof > 0 else 1.,
                )
            )
        print(self.integrator.map.settings(ngrid=self.ngrid))
        print('')


################
# Classes for standarizing the interface for integrands.


# preferred base class for batch integrands
# batch integrands are typically faster
# 
cdef class BatchIntegrand:
    """ Base class for classes providing batch integrands.

    A class derived from :class:`vegas.BatchIntegrand` will normally
    provide a ``__call__(self, x)`` method that returns an 
    array ``f`` where:

        ``x[i, d]`` is a contiguous :mod:`numpy` array where ``i=0...``
        labels different integrtion points and ``d=0...`` labels
        different directions in the integration space.

        ``f[i]`` is a contiguous array containing the integrand
        values corresponding to the integration 
        points ``x[i, :]``. ``f[i]`` is either a number, 
        for a single integrand, or an array (of any shape) 
        for multiple integrands (i.e., an 
        array-valued integrand).

    An example is ::

        import vegas 
        import numpy as np 
        
        class batchf(vegas.BatchIntegrand):
            def __call__(x):
                return np.exp(-x[:, 0] - x[:, 1])

        f = batchf()      # the integrand

    for the two-dimensional integrand :math:`\exp(-x_0 - x_1)`.

    Deriving from :class:`vegas.BatchIntegrand` is the 
    easiest way to construct integrands in Cython, and
    gives the fastest results.
    """
    # cdef object fcntype
    # cdef public object fcn
    def __cinit__(self, *args, **kargs):
        self.fcntype = 'batch'
        self.fcn = None
    def training_f(self, x):
        cdef numpy.ndarray fx = numpy.asarray(self(x))
        if fx.ndim == 1:
            return fx
        else:
            fx = fx.reshape((x.shape[0], -1))
            return fx[:, 0]
    def __call__(self, x):
        if self.fcn is None:
            raise TypeError('no __call__ method defined')
        else:
            return self.fcn(x)
        

def batchintegrand(f):
    """ Decorator for batch integrand functions.

    Applying :func:`vegas.batchintegrand` to a function ``fcn`` repackages
    the function in a format that |vegas| can understand. Appropriate 
    functions take a :mod:`numpy` array of integration points ``x[i, d]`` 
    as an argument, where ``i=0...`` labels the integration point and 
    ``d=0...`` labels direction, and return an array ``f[i]`` of 
    integrand values (or arrays of integrand values) for the corresponding 
    points. The meaning of ``fcn(x)`` is unchanged by the decorator, but 
    the type of ``fcn`` is changed.

    An example is ::

        import vegas 
        import numpy as np 

        @vegas.batchintegrand
        def f(x):
            return np.exp(-x[:, 0] - x[:, 1])

    for the two-dimensional integrand :math:`\exp(-x_0 - x_1)`.

    This decorator provides an alternative to deriving an integrand
    class from :class:`vegas.BatchIntegrand`.
    """
    ans = BatchIntegrand()
    ans.fcn = f
    return ans


cdef class _BatchIntegrand_from_NonBatch(BatchIntegrand):
    cdef object shape
    """ Batch integrand from non-batch integrand. 

    This class is used internally by |vegas|. 
    """
    def __init__(self, fcn):
        self.fcn = fcn
        self.shape = None

    def __call__(self, double[:, ::1] x):
        cdef INT_TYPE i 
        cdef numpy.ndarray f
        if self.shape is None:
            self.shape = numpy.asarray(self.fcn(x[0, :])).shape
        f = numpy.empty((x.shape[0],) + self.shape, float)
        for i in range(x.shape[0]):
            f[i] = self.fcn(x[i, :])
        return f

cdef class MPIintegrand(BatchIntegrand):
    """ Convert (batch) integrand into an MPI multiprocessor integrand. 

    Applying decorator :class:`vegas.MPIintegrand` to  a function
    repackages the function as a  batch |vegas| integrand that can
    execute in parallel on multiple processors. Appropriate  functions
    take a :mod:`numpy` array of integration points ``x[i, d]`` as an
    argument, where ``i=0...`` labels the integration point and
    ``d=0...`` labels direction, and return an array ``f[i]`` of
    integrand values (or arrays  f[i,...] of integrand values) for the
    corresponding  points.

    An example is ::

        import vegas 
        import numpy as np 

        @vegas.MPIintegrand
        def f(x):
            return np.exp(-x[:, 0] - x[:, 1])

    for the two-dimensional integrand :math:`\exp(-x_0 - x_1)`.  Of
    course, one could write ``f = vegas.MPIintegrand(f)`` instead of
    using the decorator.

    Message passing between processors uses MPI via Python  module
    :mod:`mpi4py`, which must be installed in Python.  To run an MPI
    integration code ``mpi-integral.py`` on 4 processors,  for
    example, one might execute::
        
        mpirun -np 4 python mpi-integral.py

    Executing ``python mpi-integral.py``, without the ``mpirun``, causes
    it to run on a single processor, in more or less the same  way an
    integral with a batch integrand runs.

    An object of type :class:`vegas.MPIintegrand` contains information
    about the MPI processes in the following attributes:

    .. attribute:: comm

        MPI intracommunicator --- :class:`mpi4py.MPI.Intracomm` object
        ``mpi4py.MPI.COMM_WORLD``.

    .. attribute:: nproc

        Number of processors used.

    .. attribute:: rank

        MPI rank of current process. Each process has a unique  rank,
        ranging from ``0`` to ``nproc-1``. The rank is used  to make
        different processes do different things (for example, one
        generally wants only one of the processes to report out  final
        results).

    .. attribute:: seed 

        The random number see used to reset ``numpy.random.random``
        in all the processes.

    The implementation used here has the entire integration code run
    on every processor. It is only when evaluating the integrand that
    the processes do different  things. This is efficient provided
    most of the time is spent evaluating the integrand, which, in any
    case, is the only situation where it might make sense to use multiple
    processors.

    Note that :class:`vegas.MPIintegrand` assumes that
    :class:`vegas.Integrator` is using the default random  number
    generator (``numpy.random.random``). If this  is not the case, it
    is important to seed the other random  number generator so that all
    processes use the same random numbers.

    The approach used here to make |vegas| parallel is  based on a
    strategy used by R. Horgan and Q. Mason  with the original Fortran
    version of |vegas|.
    """
    #cdef readonly object comm 
    #cdef readonly INTP_TYPE rank 
    #cdef readonly INTP_TYPE nproc 
    #cdef readonly object seed

    def __init__(self, fcn):
        if mpi4py is None:
            raise ImportError("MPIintegrand couldn't fine module mpi4py.")
        self.fcn = fcn
        self.comm = mpi4py.MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nproc = self.comm.Get_size()
        # synchronize random number generators
        if self.rank == 0:
            seed = tuple(numpy.random.randint(1, sys.maxsize, size=3))
        else:
            seed = None
        self.seed = self.comm.bcast(seed, root=0)
        numpy.random.seed(self.seed)
        self.fcn_shape = None

    def __call__(self, numpy.ndarray[numpy.double_t, ndim=2] x):
        """ Divide x into self.nproc chunks, feeding one to each processor. """
        # Note that the last chunk needs to be padded out to the same 
        # length as the others so that Allgather doesn't get upset. The 
        # size of the pad is smaller than ``self.nproc``.
        cdef numpy.ndarray results, f
        cdef INTP_TYPE nx, i0, i1
        if self.fcn_shape is None:
            # use a trial evaluation with the first x to get fcn_shape
            self.fcn_shape = numpy.shape(self.fcn(x[:1]))[1:]
        nx = x.shape[0] // self.nproc + 1
        i0 = self.rank * nx 
        i1 = min(i0 + nx, x.shape[0])
        f = numpy.empty((nx,) + self.fcn_shape, numpy.double)
        f[:(i1-i0)] = self.fcn(x[i0:i1])
        results = numpy.empty((self.nproc * nx,) + self.fcn_shape, numpy.double)
        self.comm.Allgather(f, results)
        return results[:x.shape[0]]


# legacy names
vecintegrand = batchintegrand






























