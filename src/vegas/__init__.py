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
# Copyright (c) 2013-23 G. Peter Lepage.
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

try:
    import sys

    if sys.version_info >= (3, 8):
        from importlib import metadata
    else:
        import importlib_metadata as metadata
    __version__ = metadata.version('vegas')
except:
    # less precise default if fail
    __version__ = '>=5.4.1'

# legacy names:
from ._vegas import vecintegrand, VecIntegrand

import gvar as _gvar
import functools
import numpy
import pickle 

#########################################
# PDFRAvg, etc are wrappers for RAvg, etc

class PDFRAvg(_gvar.GVar):
    def __init__(self, results):
        self.results = pickle.loads(results) if isinstance(results, bytes) else results
        ans = self.results['f(p)*pdf'] / self.results['pdf']
        super(PDFRAvg, self).__init__(*ans.internaldata)

    def extend(self, pdfravg):
        """ Merge results from :class:`PDFRAvg` object ``pdfravg`` after results currently in ``self``. """
        self.results.extend(pdfravg.results)

    def __getattr__(self, k):
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def _remove_gvars(self, gvlist):
        tmp = PDFRAvg(results=self.results)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tgvar = _gvar.gvar_factory() # small cov matrix
        super(PDFRAvg, tmp).__init__(*tgvar(0,0).internaldata)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFRAvg(results = _gvar.distribute_gvars(self.results, gvlist))

    def __reduce_ex__(self, protocol):
        return (PDFRAvg, (pickle.dumps(self.results), ))

class PDFRAvgArray(numpy.ndarray):
    def __new__(cls, results):
        results = pickle.loads(results) if isinstance(results, bytes) else results
        self = numpy.array(results['f(p)*pdf'] / results['pdf']).view(cls)
        self.results = results
        return self 

    def extend(self, pdfravg):
        """ Merge results from :class:`PDFRAvgArray` object ``pdfravg`` after results currently in ``self``. """
        self.results.extend(pdfravg.results)

    def __getattr__(self, k):
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def _remove_gvars(self, gvlist):
        tmp = PDFRAvgArray(results=self.results)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tmp.flat[:] = _gvar.remove_gvars(numpy.array(tmp), gvlist)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFRAvgArray(results=_gvar.distribute_gvars(self.results, gvlist))

    def __reduce_ex__(self, protocol):
        return (PDFRAvgArray, (pickle.dumps(self.results), ))

class PDFRAvgDict(_gvar.BufferDict):
    def __init__(self, results):
        super(PDFRAvgDict, self).__init__()
        self.results = pickle.loads(results) if isinstance(results, bytes) else results
        for k in self.results:
            if k == 'pdf':
                continue 
            self[k[1]] = self.results[k]
        self.buf[:] /= self.results['pdf']

    def extend(self, pdfravg):
        """ Merge results from :class:`PDFRAvgDict` object ``pdfravg`` after results currently in ``self``. """
        self.results.extend(pdfravg.results)

    def _remove_gvars(self, gvlist):
        tmp = PDFRAvgDict(results=self.results)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tmp._buf = _gvar.remove_gvars(tmp.buf, gvlist)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFRAvgDict(results=_gvar.distribute_gvars(self.results, gvlist))


    def __getattr__(self, k):
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def __reduce_ex__(self, protocol):
        pickle.dumps(self.results)
        return (PDFRAvgDict, (pickle.dumps(self.results), ))

class PDFIntegrator(Integrator):
    """ :mod:`vegas` integrator for PDF expectation values.

    Given a multi-dimensional Gaussian distribtuion ``g`` (a collection
    of :class:`gvar.GVar`\s), ``PDFIntegrator`` creates a |vegas| 
    integrator that evaluates expectation values of arbitrary 
    functions ``f(p)`` where ``p`` is a point in the parameter 
    space of the distribution. Typical usage is illustrated by::

        import vegas
        import gvar as gv
        import numpy as np 

        g = gv.BufferDict()
        g['a'] = gv.gvar([10., 2.], [[1, 1.4], [1.4, 2]])
        g['b'] = gv.BufferDict.uniform('fb', 2.9, 3.1)

        expval = vegas.PDFIntegrator(g)

        def f(p):
            a = p['a']
            b = p['b']
            return a[0] + np.fabs(a[1]) ** b

        result = expval(f, neval=1000, nitn=5)
        print('<f(p)> =', result)

    Here dictionary ``g`` specifies a three-dimensional distribution
    where the first two variables ``g['a'][0]`` and ``g['a'][1]`` 
    are Gaussian with means 10 and 2, respectively, and covariance 
    matrix [[1, 1.4], [1.4, 2.]]. The last variable ``g['b']`` is 
    uniformly distributed on interval [2.9, 3.1]. The result
    is: ``<f(p)> = 30.233(14)``.

    ``PDFIntegrator`` evaluates integrals of both ``f(p) * pdf(p)`` 
    and ``pdf(p)``, where ``pdf(p)`` is the probability density 
    function (PDF) associated with distribution ``g``. The expectation 
    value of ``f(p)`` is the ratio of these two integrals (so 
    ``pdf(p)`` need not be normalized). Integration variables 
    are chosen to optimize the integral, and the integrator is 
    pre-adapted to ``pdf(p)`` (so it is often unnecessary to discard 
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
            arguments). Default is ``limit=100``.

        scale (positive float): The integration variables are
            rescaled to emphasize parameter values of order
            ``scale`` times the standard deviation. The rescaling
            does not change the value of the integral but it
            can reduce uncertainties in the :mod:`vegas` estimate.
            Default is ``scale=1.0``.

        svdcut (non-negative float or None): If not ``None``, replace
            correlation matrix of ``g`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``svdcut=1e-15``.

    All other keyword parameters are passed on to the the underlying :class:`vegas.Integrator`; 
    the ``uses_jac`` keyword is ignored.
    """
    def __init__(self, g, pdf=None, adapt_to_pdf=True, limit=100., scale=1., svdcut=1e-15, **kargs):
        if isinstance(g, _gvar.PDF):
            self.g_pdf = g
        elif isinstance(g, bytes):
            self.g_pdf = _gvar.loads(g)
        else:
            self.g_pdf = _gvar.PDF(g, svdcut=svdcut)
        self.limit = abs(limit)
        self.scale = scale
        self.adapt_to_pdf = adapt_to_pdf
        self._make_pdf(pdf)
        integ_map = self._make_map(self.limit / self.scale)
        if kargs and 'uses_jac' in kargs:
            kargs = dict(kargs)
            del kargs['uses_jac']
        super(PDFIntegrator, self).__init__(
            self.g_pdf.size * [integ_map], **kargs
            )
        if getattr(self, 'sync_ran'):
            # needed because of the Monte Carlo in _make_map()
            Integrator.synchronize_random()   # for mpi

    def __reduce__(self):
        kargs = dict()
        for k in Integrator.defaults:
            if Integrator.defaults[k] != getattr(self, k) and k != 'uses_jac':
                kargs[k] = getattr(self, k)
        kargs['sigf'] = numpy.array(self.sigf)
        return (
            PDFIntegrator, 
            (_gvar.dumps(self.g_pdf), self.pdf, self.adapt_to_pdf, self.limit, self.scale),
            kargs,
            )

    def __setstate__(self, kargs):
        self.set(**kargs)

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

    @staticmethod
    @lbatchintegrand 
    def _fdummy(p):
        if hasattr(p, 'keys'):
            for k in p:
                return numpy.ones(numpy.shape(p[k])[0], float)
        else:
            return numpy.ones(numpy.shape(p)[0], float)

    def __call__(self, f=None, pdf=None, adapt_to_pdf=None, save=None, saveall=None, **kargs):
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

            save (str or file or None): Writes ``results`` into pickle file specified
                by ``save`` at the end of each iteration. For example, setting
                ``save='results.pkl'`` means that the results returned by the last 
                vegas iteration can be reconstructed later using::

                    import pickle
                    with open('results.pkl', 'rb') as ifile:
                        results = pickle.load(ifile)
                
                Ignored if ``save=None`` (default).

            saveall (str or file or None): Writes ``(results, integrator)`` into pickle 
                file specified by ``saveall`` at the end of each iteration. For example, 
                setting ``saveall='allresults.pkl'`` means that the results returned by 
                the last vegas iteration, together with a clone of the (adapted) integrator, 
                can be reconstructed later using::

                    import pickle
                    with open('allresults.pkl', 'rb') as ifile:
                        results, integrator = pickle.load(ifile)
                
                Ignored if ``saveall=None`` (default).
                
        All other keyword arguments are passed on to a :mod:`vegas`
        integrator; see the :mod:`vegas` documentation for further information.
        """
        # if self.adapt_to_pdf and set(['adapt', 'alpha', 'beta']).isdisjoint(kargs):
        #     kargs['adapt'] = False
        if kargs and 'uses_jac' in kargs:
            kargs = dict(kargs)
            del kargs['uses_jac']
        if kargs:
            self.set(kargs)
        if save is not None or saveall is not None:
            self.set(analyzer=PDFAnalyzer(self, analyzer=self.analyzer, save=save, saveall=saveall))
        if f is None:
            f = PDFIntegrator._fdummy
        if pdf is not None:
            self._make_pdf(pdf)
        if adapt_to_pdf is not None:
            self.adapt_to_pdf = adapt_to_pdf
        integrand = self._expval(f) 
        results = super(PDFIntegrator, self).__call__(integrand) #*, **kargs)
        if self.ans_type == 'scalar':
            ans = PDFRAvg(results)
        elif self.ans_type == 'array':
            ans = PDFRAvgArray(results)
        else:
            ans = PDFRAvgDict(results)
        if isinstance(self.analyzer, PDFAnalyzer):
            self.set(analyzer=self.analyzer.analyzer)
        self.ans_type = None
        return ans

    def _expval(self, f):
        """ Return integrand using the tan mapping. """
        fcntype = getattr(f, 'fcntype', 'scalar')
        if fcntype == 'scalar':
            # convert to lbatch
            f = functools.partial(PDFIntegrator.scalar2lbatch, f=f)
            fcntype = 'lbatch'
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
        jac = self.scale * numpy.prod((tan_theta ** 2 + 1.), axis=0) 
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
            pdf = numpy.prod(numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=0)
        else:
            pdf = self.pdf_rbatch(parg) * numpy.prod(self.g_pdf.pjac)
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
        jac = self.scale * numpy.prod((tan_theta ** 2 + 1.), axis=1) 
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
            pdf = numpy.prod(numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=1)
        else:
            pdf = self.pdf_lbatch(parg) * numpy.prod(self.g_pdf.pjac)
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

class PDFAnalyzer(object):
    """ |vegas| analyzer for implementing ``save``, ``saveall`` keywords for :class:`PDFIntegrator` """
    def __init__(self, pdfinteg, analyzer, save=None, saveall=None):
        self.pdfinteg = pdfinteg 
        self.analyzer = analyzer 
        self.save = save 
        self.saveall = saveall
    
    def begin(self, itn, integrator):
        if self.analyzer is not None:
            self.analyzer.begin(itn, integrator)

    def end(self, itn_result, results):
        if self.analyzer is not None:
            self.analyzer.end(itn_result, results)
        if self.save is None and self.saveall is None:
            return
        if self.pdfinteg.ans_type == 'scalar':
            ans = PDFRAvg(results)
        elif self.pdfinteg.ans_type == 'array':
            ans = PDFRAvgArray(results)
        else:
            ans = PDFRAvgDict(results)
        if isinstance(self.save, str):
            with open(self.save, 'wb') as ofile:
                pickle.dump(ans, ofile)
        elif self.save is not None:
            pickle.dump(ans, self.save)
        if isinstance(self.saveall, str):
            with open(self.saveall, 'wb') as ofile:
                pickle.dump((ans,self.pdfinteg), ofile)
        elif self.saveall is not None:
            pickle.dump((ans,self.pdfinteg), self.saveall)
            

def ravg(reslist, weighted=None, rescale=None):
    """ Create running average from list of :mod:`vegas` results.

    This function is used to change how the weighted average of 
    |vegas| results is calculated. For example, the following code 
    discards the first five results (where |vegas| is still adapting) 
    and does an unweighted average of the last five::

        import vegas

        def fcn(p):
            return p[0] * p[1] * p[2] * p[3] * 16.

        itg = vegas.Integrator(4 * [[0,1]])
        r = itg(fcn)
        print(r.summary())
        ur = vegas.ravg(r.itn_results[5:], weighted=False)
        print(ur.summary())

    The unweighted average can be useful because it is unbiased. 
    The output is::
    
        itn   integral        wgt average     chi2/dof        Q
        -------------------------------------------------------
          1   1.013(19)       1.013(19)           0.00     1.00
          2   0.997(14)       1.002(11)           0.45     0.50
          3   1.021(12)       1.0112(80)          0.91     0.40
          4   0.9785(97)      0.9980(62)          2.84     0.04
          5   1.0067(85)      1.0010(50)          2.30     0.06
          6   0.9996(75)      1.0006(42)          1.85     0.10
          7   1.0020(61)      1.0010(34)          1.54     0.16
          8   1.0051(52)      1.0022(29)          1.39     0.21
          9   1.0046(47)      1.0029(24)          1.23     0.27
         10   0.9976(47)      1.0018(22)          1.21     0.28

        itn   integral        average         chi2/dof        Q
        -------------------------------------------------------
          1   0.9996(75)      0.9996(75)          0.00     1.00
          2   1.0020(61)      1.0008(48)          0.06     0.81
          3   1.0051(52)      1.0022(37)          0.19     0.83
          4   1.0046(47)      1.0028(30)          0.18     0.91
          5   0.9976(47)      1.0018(26)          0.31     0.87

    Args:
        reslist (list): List whose elements are |GVar|\s, arrays of 
            |GVar|\s, or dictionaries whose values are |GVar|\s or
            arrays of |GVar|\s. Alternatively ``reslist`` can be 
            the object returned by a call to a 
            :class:`vegas.Integrator` object (i.e, an instance of 
            any of :class:`vegas.RAvg`, :class:`vegas.RAvgArray`, 
            :class:`vegas.RAvgArray`, :class:`vegas.PDFRAvg`, 
            :class:`vegas.PDFRAvgArray`, :class:`vegas.PDFRAvgArray`).
        weighted (bool): Running average is weighted (by the inverse
            covariance matrix) if ``True``. Otherwise the 
            average is unweighted, which makes most sense if ``reslist`` 
            items were generated by :mod:`vegas` with little or no 
            adaptation (e.g., with ``adapt=False``). If ``weighted`` 
            is not specified (or is ``None``), it is set equal to 
            ``getattr(reslist, 'weighted', True)``.
        rescale: Integration results are divided by ``rescale`` 
            before taking the weighted average if 
            ``weighted=True``; otherwise ``rescale`` is ignored. 
            Setting ``rescale=True`` is equivalent to setting
            ``rescale=reslist[-1]``. If ``rescale`` is not
            specified (or is ``None``), it is set equal to 
            ``getattr(reslist, 'rescale', True)``.
    """
    for t in [PDFRAvg, PDFRAvgArray, PDFRAvgDict]:
        if isinstance(reslist, t):
            return t(ravg(reslist.itn_results, weighted=weighted, rescale=rescale))
    for t in [RAvg, RAvgArray, RAvgDict]:
        if isinstance(reslist, t):
            reslist = reslist.itn_results
    try:
        if len(reslist) < 1:
            raise ValueError('reslist empty')
    except:
        raise ValueError('improper type for reslist')
    if weighted is None:
        weighted = getattr(reslist, 'weighted', True)
    if rescale is None:
        rescale = getattr(reslist, 'rescale', reslist[-1])
    if hasattr(reslist[0], 'keys'):
        return RAvgDict(itn_results=reslist, weighted=weighted, rescale=rescale)
    try:
        shape = numpy.shape(reslist[0])
    except:
        raise ValueError('reslist[i] not GVar, array, or dictionary')
    if shape == ():
        return RAvg(itn_results=reslist, weighted=weighted)
    else:
        return RAvgArray(itn_results=reslist, weighted=weighted, rescale=rescale)

