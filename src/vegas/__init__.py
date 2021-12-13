""" Introduction
--------------------
This package provides tools for estimating multidimensional
integrals numerically using an enhanced version of
the adaptive Monte Carlo |vegas| algorithm (G. P. Lepage,
J. Comput. Phys. 27(1978) 192, and J. Comput. Phys. 439(2021) 
110386).

A |vegas| code generally involves two objects, one representing
the integrand and the other representing an integration
operator for a particular multidimensional volume. A typical
code sequence for a D-dimensional integral has the structure::

    # create the integrand
    def f(x):
        ... compute the integrand at point x[d] d=0,1...D-1
        ...

    # create an integrator for volume with
    # xl0 <= x[0] <= xu0, xl1 <= x[1] <= xu1 ...
    integration_region = [[xl0, xu0], [xl1, xu1], ...]
    integrator = vegas.Integrator(integration_region)

    # do the integral and print out the result
    result = integrator(f, nitn=10, neval=10000)
    print(result)

The algorithm iteratively adapts to the integrand over
``nitn`` iterations, each of which uses at most ``neval``
integrand samples to generate a Monte Carlo estimate of
the integral. The final result is the weighted average
of the results from all iterations. Increase ``neval``
to increase the precision of the result. Typically
``nitn`` is between 10 and 20. ``neval`` can be
1000s to millions, or more, depending upon
the integrand and the precision desired.

The integrator remembers how it adapted to ``f(x)``
and uses this information as its starting point if it is reapplied
to ``f(x)`` or applied to some other function ``g(x)``.
An integrator's state can be archived for future applications
using Python's :mod:`pickle` module.

See the extensive Tutorial in the first section of the |vegas| documentation.
"""

# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-18 G. Peter Lepage.
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

from ._vegas import RAvg, RAvgArray, RAvgDict
from ._vegas import AdaptiveMap, Integrator, BatchIntegrand
from ._vegas import reporter, batchintegrand
from ._vegas import rbatchintegrand, RBatchIntegrand
from ._vegas import lbatchintegrand, LBatchIntegrand
from ._vegas import MPIintegrand

from ._version import __version__

# legacy names:
from ._vegas import vecintegrand, VecIntegrand

import gvar as _gvar
import functools
import numpy
import multiprocessing



class PDFRAvg(_gvar.GVar):
    def __init__(self, results):
        ans = results['f(p)*pdf'] / results['pdf']
        super(PDFRAvg, self).__init__(*ans.internaldata)
        self.chi2 = results.chi2 
        self.dof = results.dof 
        self.Q = results.Q
        self.summaryTrue = results.summary(True)
        self.summaryFalse = results.summary()
        self.itn_results = results.itn_results
        self.sum_neval = results.sum_neval
        self.avg_neval = results.avg_neval
        self.pdfnorm = results['pdf']
        self.weighted = results.weighted

    def summary(self, extended=False):
        if extended:
            return self.summaryTrue 
        else:
            return self.summaryFalse

class PDFRAvgArray(numpy.ndarray):
    def __new__(cls, results):
        ans = results['f(p)*pdf'] / results['pdf']
        self = numpy.array(ans).view(cls)
        self.chi2 = results.chi2 
        self.dof = results.dof 
        self.Q = results.Q
        self.summaryTrue = results.summary(True)
        self.summaryFalse = results.summary() 
        self.itn_results = results.itn_results
        self.pdfnorm = results['pdf']
        self.weighted = results.weighted
        self.sum_neval = results.sum_neval
        self.avg_neval = results.avg_neval
        return self 

    def summary(self, extended=False):
        if extended:
            return self.summaryTrue 
        else:
            return self.summaryFalse

class PDFRAvgDict(_gvar.BufferDict):
    def __init__(self, results):
        self.chi2 = results.chi2 
        self.dof = results.dof 
        self.Q = results.Q
        self.summaryTrue = results.summary(True)
        self.summaryFalse = results.summary()
        self.itn_results = results.itn_results
        self.weighted = results.weighted
        self.pdfnorm = results['pdf']
        self.sum_neval = results.sum_neval
        self.avg_neval = results.avg_neval
        results.buf[:] /= self.pdfnorm
        r = _gvar.BufferDict()
        for k in results:
            if k == 'pdf':
                continue 
            r[k[1]] = results[k]
        super(PDFRAvgDict, self).__init__(r)

    def summary(self, extended=False):
        if extended:
            return self.summaryTrue 
        else:
            return self.summaryFalse

class PDFIntegrator(Integrator):
    """ :mod:`vegas` integrator for PDF expectation values.

    Given a multi-dimensional Gaussian distribtuion ``g`` (a collection
    of :class:`gvar.GVar`\s), ``PDFIntegrator`` creates a |vegas| 
    integrator that evaluates expectation values of arbitrary 
    functions ``f(p)`` where ``p`` is a point in the parameter 
    space of the distribution. Typical usuage is illustrated by::

        import vegas
        import gvar as gv 

        g = dict()
        g['a'] = gv.gvar([10., 2.], [[1, 0.9], [0.9, 1]])
        g['b'] = gv.gvar('3.0(1)')

        expval = vegas.PDFIntegrator(g)

        def f(p):
            a = p['a']
            b = p['b']
            return a[0] + a[1] ** b

        result = expval(f, neval=1000, nitn=5)
        print('<f(p)> =', result)

    which gives the result: ``<f(p)> = 7.0428(84)``.

    ``PDFIntegrator`` evaluates integrals of both ``f(p) * pdf(p)`` 
    and ``pdf(p)``, where ``pdf(p)`` is the probability density 
    function (PDF) associated with distribution ``g``. The expectation 
    value of ``f(p)`` is the ratio of these two integrals (so 
    ``pdf(p)`` need not be normalized). Integration variables 
    are chosen to optimize the integral, and the integrator is 
    pre-adapted to ``pdf(p)`` (so it is usually unnecessary to discard 
    early iterations). The result of a ``PDFIntegrator`` integration
    has an extra attribute, ``result.pdfnorm``, which is the 
    |vegas| estimate of the integral over the PDF. 

    The default (Gaussian) PDF associated with ``g`` can be 
    replaced by an arbitrary PDF function ``pdf(p)``, allowing 
    PDFIntegrator to be used for non-Gaussian distributions. 
    In such cases, Gaussian distribution ``g`` does not affect
    the values of the integrals; it is used only 
    to optimize the integration variables, and pre-adapt the 
    integrator (and so affects the accuracy of the results).
    Ideally ``g`` is an approximation to the real 
    distribution --- for example, it could be the prior 
    in a Bayesian analysis, or the result of a least-squares 
    (or other peak-finding) analysis.

    Args:
        g : |GVar|, array of |GVar|\s, or dictionary whose values
            are |GVar|\s or arrays of |GVar|\s that specifies the
            multi-dimensional Gaussian distribution used to construct
            the probability density function. The integration 
            variables are optimized for this function.

        pdf: If specified, ``pdf(p)`` is used as the probability 
            density function rather than the Gaussian PDF 
            associated with ``g``. The Gaussian PDF is used if 
            ``pdf=None`` (default). Note that PDFs need not
            be normalized.

        adapt_to_pdf (bool): :mod:`vegas` adapts to the PDF 
            when ``adapt_to_pdf=True`` (default). :mod:`vegas` adapts 
            to ``pdf(p) * f(p)`` when calculating the expectation 
            value of ``f(p)`` if ``adapt_to_pdf=False``.

        limit (positive float): Limits the integrations to a finite
            region of size ``limit`` times the standard deviation on
            either side of the mean. This can be useful if the
            functions being integrated misbehave for large parameter
            values (e.g., ``numpy.exp`` overflows for a large range of
            arguments). Default is ``1e15``.

        scale (positive float): The integration variables are
            rescaled to emphasize parameter values of order
            ``scale`` times the standard deviation. The rescaling
            does not change the value of the integral but it
            can reduce uncertainties in the :mod:`vegas` estimate.
            Default is ``1.0``.

        svdcut (non-negative float or None): If not ``None``, replace
            correlation matrix of ``g`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``1e-15``.

    All other keyword parameters are passed on to the the underlying :class:`vegas.Integrator`. 
    """
    def __init__(self, g, pdf=None, adapt_to_pdf=True, limit=1e15, scale=1., svdcut=1e-15, **kargs):
        if isinstance(g, _gvar.PDF):
            self.g_pdf = g
        else:
            self.g_pdf = _gvar.PDF(g, svdcut=svdcut)
        self.limit = abs(limit)
        self.scale = scale
        self.adapt_to_pdf = adapt_to_pdf
        self._make_pdf(pdf)
        if kargs.get('sync_ran', True):
            # needed because of the Monte Carlo in _make_map()
            Integrator.synchronize_random()   # for mpi
        integ_map = self._make_map(self.limit / self.scale)
        super(PDFIntegrator, self).__init__(
            self.g_pdf.size * [integ_map], **kargs
            )

    def _make_pdf(self, pdf):
        self.pdf = pdf 
        if pdf is not None:
            pdftype = getattr(pdf, 'fcntype', 'scalar')
            if pdftype == 'scalar':
                self.pdf_lbatch = functools.partial(PDFIntegrator.scalar2lbatch, f=pdf)
                self.pdf_rbatch = functools.partial(PDFIntegrator.scalar2rbatch, f=pdf)
            elif pdftype == 'rbatch':
                self.pdf_lbatch = functools.partial(PDFIntegrator.rbatch2lbatch, f=pdf)
                self.pdf_rbatch = pdf 
            else:
                self.pdf_rbatch = functools.partial(PDFIntegrator.lbatch2rbatch, f=pdf)
                self.pdf_lbatch = pdf 

    def _make_map(self, limit):
        """ Make vegas grid that is adapted to the pdf. """
        ny = 2000
        y = numpy.random.uniform(0., 1., (ny,1))
        limit = numpy.arctan(limit)
        m = AdaptiveMap([[-limit, limit]], ninc=100)
        theta = numpy.empty(y.shape, float)
        jac = numpy.empty(y.shape[0], float)
        for itn in range(10):
            m.map(y, theta, jac)
            tan_theta = numpy.tan(theta[:, 0])
            x = self.scale * tan_theta
            fx = (tan_theta ** 2 + 1) * numpy.exp(-(x ** 2) / 2.)
            m.add_training_data(y, (jac * fx) ** 2)
            m.adapt(alpha=1.5)
        return numpy.array(m.grid[0])

    def __call__(self, f=None, pdf=None, adapt_to_pdf=None, **kargs):
        """ Estimate expectation value of function ``f(p)``.

        Uses module :mod:`vegas` to estimate the integral of
        ``f(p)`` multiplied by the probability density function
        associated with ``g`` (i.e., ``pdf(p)``). At the same 
        time it integrates the PDF. The ratio of the two integrals
        is the expectation value.

        Args:
            f (function): Function ``f(p)`` to integrate. Integral is
                the expectation value of the function with respect
                to the distribution. The function can return a number,
                an array of numbers, or a dictionary whose values are
                numbers or arrays of numbers. Setting ``f=None`` means
                that only the PDF is integrated. Integrals can be 
                substantially faster if ``f(p)`` (and ``pdf(p)`` if set)
                are batch functions (see :mod:`vegas` documentation).

            pdf: If specified, ``pdf(p)`` is used as the probability 
                density function rather than the Gaussian PDF 
                associated with ``g``. The Gaussian PDF is used if 
                ``pdf=None`` (default). Note that PDFs need not
                be normalized.

            adapt_to_pdf (bool): :mod:`vegas` adapts to the PDF 
                when ``adapt_to_pdf=True`` (default). :mod:`vegas` adapts 
                to ``pdf(p) * f(p)`` if ``adapt_to_pdf=False``.

        All other keyword arguments are passed on to a :mod:`vegas`
        integrator; see the :mod:`vegas` documentation for further information.
        """
        # if self.adapt_to_pdf and set(['adapt', 'alpha', 'beta']).isdisjoint(kargs):
        #     kargs['adapt'] = False
        if f is None:
            # dummy function
            @lbatchintegrand 
            def fdummy(p):
                if hasattr(p, 'keys'):
                    for k in p:
                        return numpy.ones(numpy.shape(p[k])[0], float)
                else:
                    return numpy.ones(numpy.shape(p)[0], float)
            f = fdummy
        if pdf is not None:
            self._make_pdf(pdf)
        if adapt_to_pdf is not None:
            self.adapt_to_pdf = adapt_to_pdf
        integrand = self._expval(f) 
        results = super(PDFIntegrator, self).__call__(integrand, **kargs)
        if self.ans_type == 'scalar':
            ans = PDFRAvg(results)
        elif self.ans_type == 'array':
            ans = PDFRAvgArray(results)
        else:
            ans = PDFRAvgDict(results)
        self.ans_type = None
        return ans

    def _expval(self, f):
        """ Return integrand using the tan mapping. """
        fcntype = getattr(f, 'fcntype', 'scalar')
        if fcntype == 'scalar':
            # convert to lbatch
            f = functools.partial(PDFIntegrator.scalar2lbatch, f=f)
            fcntype = 'lbatch'
        # elif fcntype == 'rbatch':
        #     # convert to lbatch
        #     f = functools.partial(PDFIntegrator.rbatch2lbatch, f=f)
        #     fcntype = 'lbatch'
        # return lbatchintegrand(functools.partial(self._f_lbatch, f=f))
        if fcntype == 'rbatch':
            return rbatchintegrand(
                functools.partial(self._f_rbatch, f=f)
                )
        else:
            return lbatchintegrand(
                functools.partial(self._f_lbatch, f=f)
                )

    @staticmethod
    def rbatch2lbatch(pl, f):
        "rbatch version of lbatch function f"
        if hasattr(pl, 'keys'):
            pr = _gvar.BufferDict()
            for k in pl:
                pr[k] = pl[k] if numpy.shape(pl[k]) == () else numpy.moveaxis(pl[k], 0, -1)
        elif numpy.shape(pl) == ():
            pr = pl 
        else:
            pr = numpy.moveaxis(pl, 0, -1)
        fpr = f(pr)
        if hasattr(fpr, 'keys'):
            fpl = _gvar.BufferDict()
            for k in fpr:
                fpl[k] = fpr[k] if numpy.shape(fpr[k]) == () else numpy.moveaxis(fpr[k], -1, 0)
        elif numpy.shape(fpr) == ():
            fpl = fpr 
        else:
            fpl = numpy.moveaxis(fpr, -1, 0)
        return fpl
    
    @staticmethod
    def lbatch2rbatch(pl, f):
        "lbatch version of rbatch function f"
        if hasattr(pl, 'keys'):
            pr = _gvar.BufferDict()
            for k in pl:
                pr[k] = pl[k] if numpy.shape(pl[k]) == () else numpy.moveaxis(pl[k], -1, 0)
        elif numpy.shape(pl) == ():
            pr = pl 
        else:
            pr = numpy.moveaxis(pl, -1, 0)
        fpr = f(pr)
        if hasattr(fpr, 'keys'):
            fpl = _gvar.BufferDict()
            for k in fpr:
                fpl[k] = fpr[k] if numpy.shape(fpr[k]) == () else numpy.moveaxis(fpr[k], 0, -1)
        elif numpy.shape(fpr) == ():
            fpl = fpr 
        else:
            fpl = numpy.moveaxis(fpr, 0, -1)
        return fpl

    @staticmethod
    def scalar2lbatch(p, f):
        " an lbatch version of fcn f "
        p_is_dict = hasattr(p, 'keys')
        if p_is_dict:
            for k in p:
                nbatch = numpy.shape(p[k])[0]
                break
            pj = _gvar.BufferDict()
        else:
            nbatch = p.shape[0]
        for j in range(nbatch):
            if p_is_dict:
                for k in p:
                    pj[k] = p[k][j]
            else:
                pj = p[j]
            fpj = f(pj)
            if j == 0:
                if hasattr(fpj, 'keys'):
                    ans_is_dict = True
                    ans = _gvar.BufferDict()
                    for k in fpj:
                        ans[k] = numpy.zeros((nbatch,) + numpy.shape(fpj[k]), float)
                else:
                    ans_is_dict = False
                    ans = numpy.zeros((nbatch,) + numpy.shape(fpj))
            if ans_is_dict:
                for k in fpj:
                    ans[k][j] = fpj[k] 
            else:
                ans[j] = fpj
        return ans

    @staticmethod
    def scalar2rbatch(p, f):
        " an rbatch version of fcn f "
        p_is_dict = hasattr(p, 'keys')
        if p_is_dict:
            for k in p:
                nbatch = numpy.shape(p[k])[-1]
                break
            pj = _gvar.BufferDict()
        else:
            nbatch = p.shape[-1]
        for j in range(nbatch):
            if p_is_dict:
                for k in p:
                    pj[k] = p[k][..., j]
            else:
                pj = p[..., j]
            fpj = f(pj)
            if j == 0:
                if hasattr(fpj, 'keys'):
                    ans_is_dict = True
                    ans = _gvar.BufferDict()
                    for k in fpj:
                        ans[k] = numpy.zeros(numpy.shape(fpj[k]) + (nbatch,), float)
                else:
                    ans_is_dict = False
                    ans = numpy.zeros(numpy.shape(fpj) + (nbatch,))
            if ans_is_dict:
                for k in fpj:
                    ans[k][..., j] = fpj[k] 
            else:
                ans[..., j] = fpj
        return ans

    def _f_rbatch(self, theta, f):
        tan_theta = numpy.tan(theta)
        x = self.scale * tan_theta
        jac = self.scale * numpy.product((tan_theta ** 2 + 1.), axis=0) 
        dp = self.g_pdf.x2dpflat(x.T).T
        p = self.g_pdf.meanflat[:, None] + dp
        if self.g_pdf.shape is None:
            parg = _gvar.BufferDict()
            for k in self.g_pdf.g:
                sl,sh = self.g_pdf.g.slice_shape(k)
                parg[k] = p[sl, :].reshape(sh + p.shape[-1:])
        else:
            parg = p.reshape(self.g_pdf.shape + p.shape[-1:])
        fparg = f(parg) 
        if self.pdf is None:
            pdf = numpy.product(numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=0)
        else:
            pdf = self.pdf_rbatch(parg) * numpy.product(self.g_pdf.pjac)
        ans = _gvar.BufferDict()
        if hasattr(fparg, 'keys'):
            self.ans_type = 'dict'
            ans['pdf'] = jac * pdf 
            for k in fparg:
                ans[('f(p)*pdf', k)] = fparg[k] * ans['pdf']
        else:
            fparg = numpy.asarray(fparg)
            if numpy.ndim(fparg) == 1:
                self.ans_type = 'scalar'
            else:
                self.ans_type = 'array'
            ans['pdf'] = jac * pdf
            fparg *= ans['pdf']
            ans['f(p)*pdf'] = fparg
        if not self.adapt_to_pdf:
            pdf = ans.pop('pdf')
            ans['pdf'] = pdf
        return ans

    def _f_lbatch(self, theta, f):
        tan_theta = numpy.tan(theta)
        x = self.scale * tan_theta
        jac = self.scale * numpy.product((tan_theta ** 2 + 1.), axis=1) 
        dp = self.g_pdf.x2dpflat(x)
        p = self.g_pdf.meanflat[None, :] + dp
        if self.g_pdf.shape is None:
            parg = _gvar.BufferDict()
            for k in self.g_pdf.g:
                sl,sh = self.g_pdf.g.slice_shape(k)
                parg[k] = p[:, sl].reshape(p.shape[:1] + sh)
        else:
            parg = p.reshape(p.shape[:1] + self.g_pdf.shape)
        fparg = f(parg) 
        if self.pdf is None:
            pdf = numpy.product(numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=1)
        else:
            pdf = self.pdf_lbatch(parg) * numpy.product(self.g_pdf.pjac)
        ans = _gvar.BufferDict()
        if hasattr(fparg, 'keys'):
            self.ans_type = 'dict'
            ans['pdf'] = jac * pdf
            for k in fparg:
                shape = numpy.shape(fparg[k])
                ans[('f(p)*pdf', k)] = fparg[k] * ans['pdf'].reshape(shape[:1] + len(shape[1:]) * (1,))
        else:
            fparg = numpy.asarray(fparg)
            if numpy.ndim(fparg) == 1:
                self.ans_type = 'scalar'
            else:
                self.ans_type = 'array'
            ans['pdf'] = jac * pdf
            shape = fparg.shape
            fparg *= ans['pdf'].reshape(shape[:1] + len(shape[1:]) * (1,))
            ans['f(p)*pdf'] = fparg
        if not self.adapt_to_pdf:
            ans_pdf = ans.pop('pdf')
            ans['pdf'] = ans_pdf
        return ans

class oldPDFIntegrator(Integrator):
    """ :mod:`vegas` integrator for PDF expectation values.

    ``PDFIntegrator(g)`` is a :mod:`vegas` integrator that evaluates
    expectation values for the multi-dimensional Gaussian distribution
    specified by with ``g``, which is a |GVar| or an array of |GVar|\s or a
    dictionary whose values are |GVar|\s or arrays of |GVar|\s.

    ``PDFIntegrator`` integrates over the entire parameter space of the
    distribution but reexpresses integrals in terms of variables
    that diagonalize ``g``'s covariance matrix and are centered at
    its mean value. This greatly facilitates integration over these
    variables using the :mod:`vegas` module, making integrals over
    10s or more of parameters feasible.

    A simple illustration of ``PDFIntegrator`` is given by the following
    code::

        import vegas
        import gvar as gv

        # multi-dimensional Gaussian distribution
        g = gv.BufferDict()
        g['a'] = gv.gvar([0., 1.], [[1., 0.9], [0.9, 1.]])
        g['b'] = gv.gvar('1(1)')

        # integrator for expectation values in distribution g
        g_expval = vegas.PDFIntegrator(g)

        # want expectation value of [fp, fp**2]
        def f_f2(p):
            fp = p['a'][0] * p['a'][1] + p['b']
            return [fp, fp ** 2]

        # adapt integrator to f_f2
        warmup = g_expval(f_f2, neval=1000, nitn=5)

        # <f_f2> in distribution g
        results = g_expval(f_f2, neval=1000, nitn=5, adapt=False)
        print(results.summary())
        print('results =', results, '\\n')

        # mean and standard deviation of f(p)'s distribution
        fmean = results[0]
        fsdev = gv.sqrt(results[1] - results[0] ** 2)
        print ('f.mean =', fmean, '   f.sdev =', fsdev)
        print ("Gaussian approx'n for f(g) =", f_f2(g)[0])

    where the ``warmup`` calls to the integrator are used to adapt it to
    the integrand, and the final results are in ``results``. Here ``neval`` is
    the (approximate) number of function calls per iteration of the
    :mod:`vegas` algorithm and ``nitn`` is the number of iterations. We
    use the integrator to calculated  the expectation value of ``fp`` and
    ``fp**2``, so we can compute the standard deviation for the
    distribution of ``fp``\s. The output from this code shows that the
    Gaussian approximation (1.0(1.4)) for the mean and standard deviation
    of the ``fp`` distribution is not particularly accurate here
    (correct value is 1.9(2.0)), because of the large uncertainties in
    ``g``::

        itn   integral        average         chi2/dof        Q
        -------------------------------------------------------
          1   1.893(38)       1.893(38)           0.00     1.00
          2   1.905(35)       1.899(26)           0.25     0.78
          3   1.854(41)       1.884(22)           0.47     0.76
          4   1.921(36)       1.893(19)           0.44     0.85
          5   1.913(37)       1.897(17)           0.35     0.94

        results = [1.897(17) 7.48(10)]

        f.mean = 1.897(17)    f.sdev = 1.969(21)
        Gaussian approx'n for f(g) = 1.0(1.4)

    In general functions being integrated can return a number, or an array of
    numbers, or a dictionary whose values are numbers or arrays of numbers.
    This allows multiple expectation values to be evaluated simultaneously.

    See the documentation with the :mod:`vegas` module for more details on its
    use, and on the attributes and methods associated with integrators.
    The example above sets ``adapt=False`` when  computing final results. This
    gives more reliable error estimates  when ``neval`` is small. Note
    that ``neval`` may need to be much larger (tens or hundreds of
    thousands) for more difficult high-dimension integrals.

    Args:
        g : |GVar|, array of |GVar|\s, or dictionary whose values
            are |GVar|\s or arrays of |GVar|\s that specifies the
            multi-dimensional Gaussian distribution used to construct
            the probability density function.

        limit (positive float): Limits the integrations to a finite
            region of size ``limit`` times the standard deviation on
            either side of the mean. This can be useful if the
            functions being integrated misbehave for large parameter
            values (e.g., ``numpy.exp`` overflows for a large range of
            arguments). Default is ``1e15``.

        scale (positive float): The integration variables are
            rescaled to emphasize parameter values of order
            ``scale`` times the standard deviation. The rescaling
            does not change the value of the integral but it
            can reduce uncertainties in the :mod:`vegas` estimate.
            Default is ``1.0``.

        svdcut (non-negative float or None): If not ``None``, replace
            covariance matrix of ``g`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``1e-15``.
    """
    def __init__(self, g, limit=1e15, scale=1., svdcut=1e-15, **kargs):
        if isinstance(g, _gvar.PDF):
            self.pdf = g
        else:
            self.pdf = _gvar.PDF(g, svdcut=svdcut)
        self.limit = abs(limit)
        self.scale = scale
        if kargs.get('sync_ran', True):
            # needed because of the Monte Carlo in _make_map()
            Integrator.synchronize_random()   # for mpi
        integ_map = self._make_map(self.limit / self.scale)
        super(PDFIntegrator, self).__init__(
            self.pdf.size * [integ_map], **kargs
            )

    def _make_map(self, limit):
        """ Make vegas grid that is adapted to the pdf. """
        ny = 2000
        y = numpy.random.uniform(0., 1., (ny,1))
        limit = numpy.arctan(limit)
        m = AdaptiveMap([[-limit, limit]], ninc=100)
        theta = numpy.empty(y.shape, float)
        jac = numpy.empty(y.shape[0], float)
        for itn in range(10):
            m.map(y, theta, jac)
            tan_theta = numpy.tan(theta[:, 0])
            x = self.scale * tan_theta
            fx = (tan_theta ** 2 + 1) * numpy.exp(-(x ** 2) / 2.)
            m.add_training_data(y, (jac * fx) ** 2)
            m.adapt(alpha=1.5)
        return numpy.array(m.grid[0])

    def __call__(self, f=None, nopdf=False, _fstd=None, **kargs):
        """ Estimate expectation value of function ``f(p)``.

        Uses module :mod:`vegas` to estimate the integral of
        ``f(p)`` multiplied by the probability density function
        associated with ``g`` (i.e., ``pdf(p)``). The probability
        density function is omitted if ``nopdf=True`` (default
        setting is ``False``).

        Args:
            f (function): Function ``f(p)`` to integrate. Integral is
                the expectation value of the function with respect
                to the distribution. The function can return a number,
                an array of numbers, or a dictionary whose values are
                numbers or arrays of numbers.

            nopdf (bool): If ``True`` drop the probability density function
                from the integrand (so no longer an expectation value).
                This is useful if one wants to use the optimized
                integrator for something other than a standard
                expectation value (e.g., an expectation value with a
                non-Gaussian PDF). Default is ``False``.

        All other keyword arguments are passed on to a :mod:`vegas`
        integrator; see the :mod:`vegas` documentation for further information.
        """
        if nopdf and f is None:
            raise ValueError('nopdf==True and f is None => no integrand')
        integrand = batchintegrand(self._expval(f, nopdf)) # fstd, nopdf))
        results = super(PDFIntegrator, self).__call__(integrand, **kargs)
        return results

    def _expval(self, f, nopdf):
        """ Return integrand using the tan mapping. """
        def ff(theta, nopdf=nopdf):
            tan_theta = numpy.tan(theta)
            x = self.scale * tan_theta
            jac = self.scale * (tan_theta ** 2 + 1.)
            if nopdf:
                pdf = jac * self.pdf.pjac[None, :]
            else:
                pdf = jac * numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi)
            dp = self.pdf.x2dpflat(x)
            parg = None
            ans = None
            fparg_is_dict = False
            # iterate through the batch
            for i, (dpi, pdfi) in enumerate(zip(dp, pdf)):
                p = self.pdf.meanflat + dpi
                if parg is None:
                    # first time only
                    if self.pdf.shape is None:
                        parg = _gvar.BufferDict(self.pdf.g, buf=p)
                    else:
                        parg = p.reshape(self.pdf.shape)
                else:
                    if parg.shape is None:
                        parg.buf = p
                    else:
                        parg.flat[:] = p
                fparg = 1. if f is None else f(parg)
                if ans is None:
                    # first time only
                    if hasattr(fparg, 'keys'):
                        fparg_is_dict = True
                        if not isinstance(fparg, _gvar.BufferDict):
                            fparg = _gvar.BufferDict(fparg)
                        ans = _gvar.BufferDict()
                        for k in fparg:
                            ans[k] = numpy.empty(
                                (len(pdf),) + fparg.slice_shape(k)[1], float
                                )
                    else:
                        if numpy.shape(fparg) == ():
                            ans = numpy.empty(len(pdf), float)
                        else:
                            ans = numpy.empty(
                                (len(pdf),) + numpy.shape(fparg), float
                                )
                if fparg_is_dict:
                    prod_pdfi = numpy.prod(pdfi)
                    for k in ans:
                        ans[k][i] = fparg[k]
                        ans[k][i] *= prod_pdfi
                else:
                    if not isinstance(fparg, numpy.ndarray):
                        fparg = numpy.asarray(fparg)
                    ans[i] = fparg * numpy.prod(pdfi)
            return ans
        return ff

class parallelintegrand(BatchIntegrand):
    """ Convert (batch) integrand into multiprocessor integrand.

    Usage::

        fparallel = vegas.parallelintegrand(fbatch, 4)

    turns batch integrand ``fbatch`` into multi-process integrand
    that spreads integrand evaluations over 4 processes. The original
    integrand ``fbatch`` needs to be defined at the top level
    of the Python script and should return a :mod:`numpy` array.

    This is *not* part of the public API.
    """
    def __init__(self, fcn, nproc=4):
        " Save integrand; create pool of nproc processes. "
        self.fcn = fcn
        self.nproc = nproc
        self.pool = multiprocessing.Pool(processes=nproc)

    def __del__(self):
        " Standard cleanup. "
        self.pool.close()
        self.pool.join()

    def __call__(self, x):
        " Divide x into self.nproc chunks, feeding one to each process. "
        nx = x.shape[0] // self.nproc + 1
        # launch evaluation of self.fcn for each chunk, in parallel
        results = self.pool.map(
            self.fcn,
            [x[i*nx : (i+1)*nx] for i in range(self.nproc)],
            1,
            )
        # convert list of results into a single numpy array
        return numpy.concatenate(results)
