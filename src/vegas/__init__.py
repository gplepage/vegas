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
# Copyright (c) 2013-24 G. Peter Lepage.
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
from ._vegas import reporter, VegasIntegrand, batchintegrand
from ._vegas import rbatchintegrand, RBatchIntegrand
from ._vegas import lbatchintegrand, LBatchIntegrand
from ._vegas import MPIintegrand
from ._version import __version__

# legacy names:
from ._vegas import vecintegrand, VecIntegrand

import gvar as _gvar
import functools
import numpy
import pickle 

###############################################
# PDFEV, etc PDFIntegrator expectation values


class PDFEV(_gvar.GVar):
    """ Expectation value from |PDFIntegrator|.
    
    Expectation values are returned by 
    :meth:`vegas.PDFIntegrator.__call__` and 
    :class:`vegas.PDFIntegrator.stats`::

        >>> g = gvar.gvar(['1(1)', '10(10)'])
        >>> g_ev = vegas.PDFIntegrator(g)
        >>> def f(p):
        ...     return p[0] * p[1]
        >>> print(g_ev(f))
        10.051(57)
        >>> print(g_ev.stats(f))
        10(17)

    In the first case, the quoted error is the uncertainty 
    in the :mod:`vegas` estimate of the mean of ``f(p)``. 
    In the second case, the quoted uncertainty is the 
    standard deviation evaluated with respect to the 
    Gaussian distribution associated with ``g`` (added
    in quadrature to the :mod:`vegas` error, which 
    is negligible here). 
          
    :class:`vegas.PDFEV`\s have the following attributes:

    Attributes:

        pdfnorm: Divide PDF by ``pdfnorm`` to normalize it.

        results (:class:`vegas.RAvgDict`): Results from 
            the underlying integrals.

    In addition, they have all the attributes of the :class:`vegas.RAvgDict`
    (``results``) corresponding to the underlying integrals.
    
    A :class:`vegas.PDFEV` returned by 
    ``vegas.PDFIntegrator.stats(self, f...)`` has three further attributes:

    Attributes:

        stats: An instance of :class:`gvar.PDFStatistics`
            containing statistical information about 
            the distribution of ``f(p)``.

        vegas_mean: |vegas| estimate for the mean value of 
            ``f(p)``. The uncertainties in ``vegas_mean`` 
            are the integration errors from |vegas|.

        vegas_cov: |vegas| estimate for the covariance matrix 
            of ``f(p)``. The uncertainties in ``vegas_cov`` 
            are the integration errors from |vegas|.

        vegas_sdev: |vegas| estimate for the standard deviation 
            of ``f(p)``. The uncertainties in ``vegas_sdev`` 
            are the integration errors from |vegas|.
    """
    def __init__(self, results, analyzer=None):
        self.results = pickle.loads(results) if isinstance(results, bytes) else results
        if analyzer is None:
            ans = self.results['f(p)*pdf'] / self.results['pdf']
            super(PDFEV, self).__init__(*ans.internaldata)
            self.analyzer = None
        else:
            ans, extras = analyzer(self.results)
            super(PDFEV, self).__init__(*ans.internaldata)
            for k in extras:
                setattr(self, k, extras[k])
            self.analyzer = analyzer

    def extend(self, pdfev):
        """ Merge results from :class:`PDFEV` object ``pdfev`` after results currently in ``self``. """
        self.results.extend(pdfev.results)

    def __getattr__(self, k):
        if k in ['keys']:
            raise AttributeError('no keys method')
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def _remove_gvars(self, gvlist):
        tmp = PDFEV(results=self.results, analyzer=self.analyzer)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tgvar = _gvar.gvar_factory() # small cov matrix
        super(PDFEV, tmp).__init__(*tgvar(0,0).internaldata)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFEV(
            results = _gvar.distribute_gvars(self.results, gvlist),
            analyzer=self.analyzer
            )

    def __reduce_ex__(self, protocol):
        return (PDFEV, (pickle.dumps(self.results), self.analyzer))

class PDFEVArray(numpy.ndarray):
    """ Array of expectation values from |PDFIntegrator|.
    
    Expectation values are returned by 
    :meth:`vegas.PDFIntegrator.__call__` and 
    :class:`vegas.PDFIntegrator.stats`::

        >>> g = gvar.gvar(['1(1)', '10(10)'])
        >>> g_ev = vegas.PDFIntegrator(g)
        >>> def f(p):
        ...     return [p[0], p[1], p[0] * p[1]]
        >>> print(g_ev(f))
        [0.9992(31) 10.024(29) 10.051(57)]
        >>> print(g_ev.stats(f))
        [1.0(1.0) 10(10) 10(17)]

    In the first case, the quoted errors are the uncertainties 
    in the :mod:`vegas` estimates of the means. In the second 
    case, the quoted uncertainties are the standard deviations 
    evaluated with respect to the Gaussian distribution 
    associated with ``g`` (added in quadrature to the 
    :mod:`vegas` errors, which are negligible here).
          
    :class:`vegas.PDFEVArray`\s have the following attributes:
    
    Attributes:

        pdfnorm: Divide PDF by ``pdfnorm`` to normalize it.

        results (:class:`vegas.RAvgDict`): Results from 
            the underlying integrals.

    In addition, they have all the attributes of the :class:`vegas.RAvgDict`
    (``results``) corresponding to the underlying integrals.

    A :class:`vegas.PDFEVArray` ``s`` returned by 
    ``vegas.PDFIntegrator.stats(self, f...)`` has three further 
    attributes:

    Attributes:

        stats: ``s.stats[i]`` is a :class:`gvar.PDFStatistics`
            object containing statistical information about 
            the distribution of ``f(p)[i]``.

        vegas_mean: |vegas| estimates for the mean values
            of ``f(p)``. The uncertainties in ``vegas_mean`` 
            are the integration errors from |vegas|.

        vegas_cov: |vegas| estimate for the covariance matrix
            of ``f(p)``. The uncertainties in ``vegas_cov`` 
            are the integration errors from |vegas|.

        vegas_sdev: |vegas| estimate for the standard deviation 
            of ``f(p)``. The uncertainties in ``vegas_sdev`` 
            are the integration errors from |vegas|.
    """
    def __new__(cls, results, analyzer=None):
        results = pickle.loads(results) if isinstance(results, bytes) else results
        if analyzer is None:
            self = numpy.asarray(results['f(p)*pdf'] / results['pdf']).view(cls)
            self.analyzer = None
        else:
            ans, extras = analyzer(results)
            self = numpy.asarray(ans).view(cls)
            for k in extras:
                setattr(self, k, extras[k])
            self.analyzer = analyzer 
        self.results = results
        return self 

    def extend(self, pdfev):
        """ Merge results from :class:`PDFEVArray` object ``pdfev`` after results currently in ``self``. """
        self.results.extend(pdfev.results)

    def __getattr__(self, k):
        if k in ['keys']:
            raise AttributeError('no keys method')
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def _remove_gvars(self, gvlist):
        tmp = PDFEVArray(results=self.results, analyzer=self.analyzer)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tmp.flat[:] = _gvar.remove_gvars(numpy.array(tmp), gvlist)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFEVArray(
            results=_gvar.distribute_gvars(self.results, gvlist), analyzer=self.analyzer
            )

    def __reduce_ex__(self, protocol):
        return (PDFEVArray, (pickle.dumps(self.results), self.analyzer))

class PDFEVDict(_gvar.BufferDict):
    """ Dictionary of expectation values from |PDFIntegrator|.
    
    Expectation values are returned by 
    :meth:`vegas.PDFIntegrator.__call__` and 
    :class:`vegas.PDFIntegrator.stats`::

        >>> g = gvar.gvar(['1(1)', '10(10)'])
        >>> g_ev = vegas.PDFIntegrator(g)
        >>> def f(p):
        ...     return dict(p=p, prod=p[0] * p[1])
        >>> print(g_ev(f))
        {'p': array([0.9992(31), 10.024(29)], dtype=object), 'prod': 10.051(57)}
        >>> print(g_ev.stats(f))
        {'p': array([1.0(1.0), 10(10)], dtype=object), 'prod': 10(17)}

    In the first case, the quoted errors are the uncertainties 
    in the :mod:`vegas` estimates of the means. In the second 
    case, the quoted uncertainties are the standard deviations 
    evaluated with respect to the Gaussian distribution 
    associated with ``g`` (added in quadrature to the 
    :mod:`vegas` errors, which are negligible here).
    
    :class:`vegas.PDFEVDict` objects have the following attributes:

    Attributes:

        pdfnorm: Divide PDF by ``pdfnorm`` to normalize it.

        results (:class:`vegas.RAvgDict`): Results from 
            the underlying integrals.

    In addition, they have all the attributes of the :class:`vegas.RAvgDict`
    (``results``) corresponding to the underlying integrals.

    A :class:`vegas.PDFEVDict` object ``s`` returned by 
    :meth:`vegas.PDFIntegrator.stats` has three further attributes:

    Attributes:

        stats: ``s.stats[k]`` is a :class:`gvar.PDFStatistics`
            object containing statistical information about 
            the distribution of ``f(p)[k]``.

        vegas_mean: |vegas| estimates for the mean values
             of ``f(p)``. The uncertainties in ``vegas_mean`` 
             are the integration errors from |vegas|.

        vegas_cov: |vegas| estimate for the covariance matrix
            of ``f(p)``. The uncertainties in ``vegas_cov`` 
            are the integration errors from |vegas|.

        vegas_sdev: |vegas| estimate for the standard deviation 
            of ``f(p)``. The uncertainties in ``vegas_sdev`` 
            are the integration errors from |vegas|.
    """
    def __init__(self, results, analyzer=None):
        super(PDFEVDict, self).__init__()
        self.results = pickle.loads(results) if isinstance(results, bytes) else results
        if analyzer is None:
            for k in self.results:
                if k == 'pdf':
                    continue 
                self[k[1]] = self.results[k]
            self.buf[:] /= self.results['pdf']
            self.analyzer = None
        else:
            ans, extras = analyzer(self.results)
            for k in extras:
                setattr(self, k, extras[k])
            for k in ans:
                self[k] = ans[k]
            self.analyzer = analyzer

    def extend(self, pdfev):
        """ Merge results from :class:`PDFEVDict` object ``pdfev`` after results currently in ``self``. """
        self.results.extend(pdfev.results)

    def _remove_gvars(self, gvlist):
        tmp = PDFEVDict(results=self.results, analyzer=self.analyzer)
        tmp.results = _gvar.remove_gvars(tmp.results, gvlist)
        tmp._buf = _gvar.remove_gvars(tmp.buf, gvlist)
        return tmp 

    def _distribute_gvars(self, gvlist):
        return PDFEVDict(
            results=_gvar.distribute_gvars(self.results, gvlist),
            analyzer=self.analyzer
            )

    def __getattr__(self, k):
        if k == 'pdfnorm':
            return self.results['pdf']
        return getattr(self.results, k)

    def __reduce_ex__(self, protocol):
        pickle.dumps(self.results)
        return (PDFEVDict, (pickle.dumps(self.results), self.analyzer))

class PDFIntegrator(Integrator):
    """ :mod:`vegas` integrator for PDF expectation values.

    ``PDFIntegrator(param, pdf)`` creates a |vegas| integrator that 
    evaluates expectation values of arbitrary functions ``f(p)`` with 
    respect to the probability density function  ``pdf(p)``, where 
    ``p`` is a point in the parameter space defined by ``param``. 
    
    ``param`` is a collection of :class:`gvar.GVar`\s (Gaussian random
    variables) that together define a multi-dimensional Gaussian 
    distribution with the same parameter space as the distribution 
    described by ``pdf(p)``. ``PDFIntegrator`` internally 
    re-expresses the integrals over these parameters in terms 
    of new variables that emphasize the region defined by
    ``param`` (i.e., the region where the PDF associated with 
    the ``param``'s Gaussian distribution is large).
    The new variables are also aligned with the principal axes 
    of ``param``'s correlation matrix, to facilitate integration.
    
    ``param``'s means and covariances are chosen to emphasize the 
    important regions of the ``pdf``'s  distribution (e.g., ``param`` 
    might be set equal to the prior in a Bayesian analysis). 
    ``param`` is used to define and optimize the integration variables; 
    it does not affect the values of the integrals but can have a big 
    effect on the accuracy.
    
    The Gaussian PDF associated with ``param`` is used if 
    ``pdf`` is unspecified (i.e., ``pdf=None``, which is the default).
    
    Typical usage is illustrated by the following code, where
    dictionary ``g`` specifies both the parameterization (``param``) 
    and the PDF::

        import vegas
        import gvar as gv
        import numpy as np 

        g = gv.BufferDict()
        g['a'] = gv.gvar([10., 2.], [[1, 1.4], [1.4, 2]])
        g['fb(b)'] = gv.BufferDict.uniform('fb', 2.9, 3.1)

        g_ev = vegas.PDFIntegrator(g)

        def f(p):
            a = p['a']
            b = p['b']
            return a[0] + np.fabs(a[1]) ** b

        result = g_ev(f, neval=10_000, nitn=5)
        print('<f(p)> =', result)

    Here ``g`` indicates a three-dimensional distribution
    where the first two variables ``g['a'][0]`` and ``g['a'][1]`` 
    are Gaussian with means 10 and 2, respectively, and covariance 
    matrix [[1, 1.4], [1.4, 2.]]. The last variable ``g['b']`` is 
    uniformly distributed on interval [2.9, 3.1]. The result
    is: ``<f(p)> = 30.145(83)``.

    ``PDFIntegrator`` evaluates integrals of both ``f(p) * pdf(p)`` 
    and ``pdf(p)``. The expectation value of ``f(p)`` is the ratio 
    of these two integrals (so ``pdf(p)`` need not be normalized). 
    The result of a ``PDFIntegrator`` integration
    has an extra attribute, ``result.pdfnorm``, which is the 
    |vegas| estimate of the integral over the PDF. 

    Args:
        param : A |GVar|, array of |GVar|\s, or dictionary, whose values
            are |GVar|\s or arrays of |GVar|\s, that specifies the 
            integration parameters. When parameter ``pdf=None``, the 
            PDF is set equal to the Gaussian distribution corresponding 
            to ``param``. 

        pdf: The probability density function ``pdf(p)``. 
            The PDF's parameters ``p`` have the same layout 
            as ``param`` (arrays or dictionary), with the same
            keys and/or shapes. The Gaussian PDF associated with
            ``param`` is used when ``pdf=None`` (default).
            Note that PDFs need not be normalized.

        adapt_to_pdf (bool): :mod:`vegas` adapts to the PDF 
            when ``adapt_to_pdf=True`` (default). :mod:`vegas` adapts 
            to ``pdf(p) * f(p)`` when calculating the expectation 
            value of ``f(p)`` if ``adapt_to_pdf=False``.

        limit (positive float): Integration variables are determined from 
            ``param``. ``limit`` limits the range of each variable to 
            a region of size ``limit`` times the standard deviation on 
            either side of the mean, where means and standard deviations
            are specified by ``param``. This can be useful if the
            functions being integrated misbehave for large parameter
            values (e.g., ``numpy.exp`` overflows for a large range of
            arguments). Default is ``limit=100``; results should become 
            independent of ``limit`` as it is increased.

        scale (positive float): The integration variables are
            rescaled to emphasize parameter values of order
            ``scale`` times the standard deviation measured from 
            the mean, where means and standard deviations are 
            specified by ``param``. The rescaling
            does not change the value of the integral but it
            can reduce uncertainties in the :mod:`vegas` estimate.
            Default is ``scale=1.0``.

        svdcut (non-negative float or None): If not ``None``, replace
            correlation matrix of ``param`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``svdcut=1e-12``.

    All other keyword parameters are passed on to the the underlying 
    :class:`vegas.Integrator`; the ``uses_jac`` keyword is ignored.
    """
    def __init__(self, param, pdf=None, adapt_to_pdf=True, limit=100., scale=1., svdcut=1e-15, **kargs):
        if 'g' in kargs and param is None:
            # for legacy code
            param = kargs['g']
            del kargs['g']
        if param is None:
            raise ValueError('param must be specified')
        if isinstance(param, PDFIntegrator):
            super(PDFIntegrator, self).__init__(param)
            for k in ['param_pdf', 'param_sample', 'pdf', 'adapt_to_pdf', 'limit', 'scale']:
                setattr(self, k, getattr(param, k))
            return
        elif isinstance(param, _gvar.PDF):
            self.param_pdf = param
        else:
            self.param_pdf = _gvar.PDF(param, svdcut=svdcut)
        self.param_sample = self.param_pdf.sample(mode=None)
        self.limit = abs(limit)
        self.scale = abs(scale)
        self.set(adapt_to_pdf=adapt_to_pdf, pdf=pdf)
        integ_map = self._make_map(self.limit / self.scale)
        if kargs and 'uses_jac' in kargs:
            kargs = dict(kargs)
            del kargs['uses_jac']
        super(PDFIntegrator, self).__init__(
            AdaptiveMap(self.param_pdf.size * [integ_map]), **kargs
            )
        if getattr(self, 'mpi') and getattr(self, 'sync_ran'):
            # needed because of the Monte Carlo in _make_map()
            Integrator.synchronize_random()   # for mpi only

    def __reduce__(self):
        kargs = dict()
        for k in Integrator.defaults:
            if Integrator.defaults[k] != getattr(self, k) and k != 'uses_jac':
                kargs[k] = getattr(self, k)
        kargs['sigf'] = numpy.array(self.sigf)
        return (
            PDFIntegrator, 
            (self.param_pdf, self.pdf, self.adapt_to_pdf, self.limit, self.scale),
            kargs,
            )

    def __setstate__(self, kargs):
        self.set(**kargs)

    def set(self, ka={}, **kargs):
        """ Reset default parameters in integrator.

        Usage is analogous to the constructor
        for :class:`PDFIntegrator`: for example, ::

            old_defaults = pdf_itg.set(neval=1e6, nitn=20)

        resets the default values for ``neval`` and ``nitn``
        in :class:`PDFIntegrator` ``pdf_itg``. A dictionary,
        here ``old_defaults``, is returned. It can be used
        to restore the old defaults using, for example::

            pdf_itg.set(old_defaults)
        """
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_defaults = {}
        if 'param' in kargs:
            raise ValueError("Can't reset param.")
        if 'pdf' in kargs:
            if hasattr(self, 'pdf'):
                old_defaults['pdf'] = self.pdf
            pdf = kargs['pdf']
            self.pdf = (
                pdf if pdf is None else 
                self._make_std_integrand(pdf, xsample=self.param_sample)
                )
            del kargs['pdf']
        if 'adapt_to_pdf' in kargs:
            if hasattr(self, 'adapt_to_pdf'):
                old_defaults['adapt_to_pdf'] = self.adapt_to_pdf
            self.adapt_to_pdf = kargs['adapt_to_pdf']
            del kargs['adapt_to_pdf']
        if kargs:
            old_defaults.update(super(PDFIntegrator, self).set(kargs))
        return old_defaults

    def _make_map(self, limit):
        """ Make vegas grid that is adapted to the pdf. """
        ny = 2000
        y = _gvar.RNG.random((ny,1))
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
    def _f_lbatch(theta, f, param_pdf, pdf, scale, adapt_to_pdf):
        """ Integrand for PDFIntegrator.
        
        N.B. Static method is more efficient because less to carry around
        (eg, when nproc>1).
        N.B. ``f`` has been converted to a ``VegasIntegrand`` object (as has
        ``self.pdf`` if it is defined externally.
        """
        tan_theta = numpy.tan(theta)
        chiv = scale * tan_theta   
        dp_dtheta = numpy.prod(scale * (tan_theta ** 2 + 1.), axis=1) * param_pdf.dp_dchiv
        p = param_pdf.pflat(chiv, mode='lbatch')
        if pdf is None:
            # normalized in chiv space so don't want param_pdf.dp_dchiv in jac
            pdf = numpy.prod(numpy.exp(-(chiv ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=1) / param_pdf.dp_dchiv
        else:
            pdf = numpy.prod(pdf.eval(p, jac=None), axis=1)
        if f is None:
            ans = _gvar.BufferDict(pdf=dp_dtheta * pdf)
            return ans
        fp = dp_dtheta * pdf if f is None else f.format_evalx(f.eval(p))
        ans = _gvar.BufferDict()
        if hasattr(fp, 'keys'):
            ans['pdf'] = dp_dtheta * pdf
            for k in fp:
                shape = numpy.shape(fp[k])
                ans[('f(p)*pdf', k)] = fp[k] * ans['pdf'].reshape(shape[:1] + len(shape[1:]) * (1,))
        else:
            fp = numpy.asarray(fp)
            ans['pdf'] = dp_dtheta * pdf
            shape = fp.shape
            fp *= ans['pdf'].reshape(shape[:1] + len(shape[1:]) * (1,))
            ans['f(p)*pdf'] = fp
        if not adapt_to_pdf:
            ans_pdf = ans.pop('pdf')
            ans['pdf'] = ans_pdf
        return ans

    def __call__(self, f=None, save=None, saveall=None, **kargs):
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

            save (str or file or None): Writes ``results`` into pickle 
                file specified by ``save`` at the end of each iteration. 
                For example, setting ``save='results.pkl'`` means that 
                the results returned by the last vegas iteration can be 
                reconstructed later using::

                    import pickle
                    with open('results.pkl', 'rb') as ifile:
                        results = pickle.load(ifile)
                
                Ignored if ``save=None`` (default).

            saveall (str or file or None): Writes ``(results, integrator)`` 
                into pickle file specified by ``saveall`` at the end of 
                each iteration. For example, setting ``saveall='allresults.pkl'`` 
                means that the results returned by the last vegas iteration, 
                together with a clone of the (adapted) integrator, can be 
                reconstructed later using::

                    import pickle
                    with open('allresults.pkl', 'rb') as ifile:
                        results, integrator = pickle.load(ifile)
                
                Ignored if ``saveall=None`` (default).
                
        All other keyword arguments are passed on to a :mod:`vegas`
        integrator; see the :mod:`vegas` documentation for further information.

        Returns:
            Expectation value(s) of ``f(p)`` as object of type
            :class:`vegas.PDFEV`, :class:`vegas.PDFEVArray`, or 
            :class:`vegas.PDFEVDict`.
        """
        if kargs and 'uses_jac' in kargs:
            kargs = dict(kargs)
            del kargs['uses_jac']
        if kargs:
            self.set(kargs)
        if save is not None or saveall is not None:
            self.set(analyzer=PDFAnalyzer(self, analyzer=self.analyzer, save=save, saveall=saveall))
        if f is not None:
            f = self._make_std_integrand(f, self.param_sample)
        integrand = lbatchintegrand(functools.partial(
            PDFIntegrator._f_lbatch, f=f, param_pdf=self.param_pdf, 
            pdf=self.pdf, scale=self.scale, adapt_to_pdf=self.adapt_to_pdf,
            )) 
        results = super(PDFIntegrator, self).__call__(integrand)
        if results['pdf'] == 0:
            raise RuntimeError('Integral of PDF vanishes; increase neval?')
        if f is None:
            ans = results
            ans.pdfnorm = results['pdf']
        else:
            ans = PDFIntegrator._make_ans(results)
        if isinstance(self.analyzer, PDFAnalyzer):
            self.set(analyzer=self.analyzer.analyzer)
        return ans

    @staticmethod
    def _make_ans(results):
        if 'f(p)*pdf' not in results:
            ans = PDFEVDict(results)
        elif numpy.ndim(results['f(p)*pdf']) == 0:
            ans = PDFEV(results)
        else:
            ans = PDFEVArray(results)
        return ans

    def stats(self, f=None, moments=False, histograms=False, **kargs):
        """ Statistical analysis of function ``f(p)``.

        Uses the :mod:`vegas` integrator to evaluate the expectation
        values and (co)variances of ``f(p)`` with 
        respect to the probability density function associated 
        with the :class:`PDFIntegrator`. Typical usage
        is illustrated by::

            >>> import gvar as gv
            >>> import vegas
            >>> g = gv.gvar(dict(a='1.0(5)', b='2(1)')) * gv.gvar('1.0(5)')
            >>> g_ev = vegas.PDFIntegrator(g)
            >>> g_ev(neval=10_000)    # adapt the integrator to the PDF
            >>> @vegas.rbatchintegrand
            ... def f(p):
            ...     fp = dict(a=p['a'], b=p['b'])
            ...     fp['a**2 * b'] = p['a']**2 * p['b']
            ...     return fp
            >>> r = g_ev.stats(f)
            >>> print(r)
            {'a': 1.00(71), 'b': 2.0(1.4), 'a**2 * b': 4.0(6.1)}
            >>> print(r.vegas_mean['a**2 * b'])
            3.9972(30)
            >>> print(r.vegas_cov['a**2 * b', 'a**2 * b'] ** 0.5)            
            6.073(13)

        ``g_ev.stats(f)`` returns a dictionary of |GVar|\s whose 
        means and (co)variances are calculated from integrals of 
        ``f(p) * pdf(p)`` and ``f(p)**2 * pdf(p)``, where ``pdf(p)`` 
        is the probability density function associated with ``g``.
        The means and standard deviations for each component of ``f(p)``
        are displayed by ``print(r)``. The values for the means 
        and standard deviations have uncertainties coming from the 
        integrations (|vegas| errors) but these are negligible compared 
        to the standard deviations. (The last two 
        print statements show the |vegas| results for the 
        mean and standard deviation in ``r['a**2 * b']``: 3.9972(30)
        and 6.073(13), respectively.)

        Th Gaussian approximation for the expectation value of 
        ``f(p)`` is given by ::

            >>> print(f(g))
            {'a': 1.00(71), 'b': 2.0(1.4), 'a**2 * b': 2.0(3.7)}
        
        Results for ``a`` and ``b`` agree with the results from 
        ``g_ev.stats(f)``, as expected since the distributions 
        for these quantities are (obviously) Gaussian. Results 
        for ``a**2 * b``, however, are quite different, indicating 
        a distribution that is not Gaussian.
             
        Additional statistical data are collected by setting keywords
        ``moments=True`` and/or ``histogram=True``::

            >>> r = g_ev.stats(f, moments=True, histograms=True)
            >>> for k in r:
            ...     print(10 * '-', k)
            ...     print(r.stats[k])
            ---------- a
               mean = 0.99972(23)   sdev = 0.70707(29)   skew = -0.0036(20)   ex_kurt = -0.0079(49)
               split-normal: 1.0013(14) +/- 0.70862(97)/0.71091(98)
                     median: 0.99927(62) +/- 0.7077(10)/0.7063(10)
            ---------- b
               mean = 1.99954(47)   sdev = 1.41424(72)   skew = -0.0041(28)   ex_kurt = -0.0074(65)
               split-normal: 2.0042(33) +/- 1.4162(23)/1.4224(24)
                     median: 1.9977(11) +/- 1.4162(18)/1.4115(19)
            ---------- a**2 * b
               mean = 3.9957(29)   sdev = 6.054(12)   skew = 3.048(22)   ex_kurt = 14.52(35)
               split-normal: -0.4891(25) +/- 6.9578(88)/0.519(10)
                     median: 1.7447(24) +/- 6.284(12)/2.0693(26)
            
        where the uncertainties are all |vegas| errors. Here the 
        integrator was used to calculate the first four moments 
        of the distributions for each component of ``f(p)``, from 
        which the mean, standard deviation, skewness, and excess 
        kurtosis of those distributions are calculated. As expected 
        the first two distribuitons here are clearly Gaussian, 
        but the distribution for ``a**2 * b`` is not. 
        
        The integrator also calculates histograms
        for each of the distributions and fits them to two 
        different two-sided Gaussians: one is a continuous split-normal
        distribution, and the other is centered on the median of the 
        distribution and is discontinuous there. (For more information
        see the documentation for :class:`gvar.PDFStatistics`.)
        Both models suggest large asymmetries in the distribution 
        for ``a**2 * b``. The histogram for this distribution can 
        be displayed using::
        
            >>> r.stats['a**2 * b'].plot_histogram(show=True)

        Note that |vegas| adaptation is turned off (``adapt=False``)
        by default in :meth:`PDFIntegrator.stats`. This setting 
        can be overridden by setting the ``adapt`` parameter 
        explicitly, but this is not recommended. 

        Args:
            f (callable): Statistics are calculated for the 
                components of the output from function ``f(p)``,
                where ``p`` is a point drawn from the distribution 
                specified by the ``param`` or ``pdf`` associated with the 
                :class:`PDFIntegrator`. Parameters ``p`` have
                the same structure as ``param`` (i.e., array or 
                dictionary). If ``f=None``, it is replaced by
                ``f=lbatchintegrand(lambda p:p)``.

            moments (bool): If ``True``, moments are calculated so 
                that the skewness and excess kurtosis can be determined.

            histograms (bool or dict): Setting ``histograms=True`` 
                causes histograms to be calculated for the 
                distributions associated with each component of 
                the output from ``f(p)``. Alternatively, ``histograms``
                can be set equal to a dictionary to specify the 
                the width ``binwidth`` of each histogram bin, the total 
                number ``nbin`` of bins, and/or the location ``loc``
                of each histogram: for example, ::

                    histograms=dict(
                        binwidth=0.5, nbin=12, 
                        loc=gv.gvar({
                            'a': '1.0(5)', 'b': '2(1)', 
                            'a**2 * b': '2.5(2.7)'
                            }), 
                        )

                where ``loc`` specifies the location of the center of the histogram 
                for each output quantity (e.g., ``loc['a'].mean``) and the width of 
                the bins (e.g., ``binwidth * loc['a'].sdev``). If ``loc`` is not 
                specified explicitly, it is determined from a simulation using
                values drawn from the Gaussian distribution for ``self.g``
                (or from the distribution described by ``self.pdf`` if it is specified).

            kargs (dict): Additional keywords passed on to the 
                integrator.
        
        Returns:
            Expectation value(s) of ``f(p)`` as an object of type
            :class:`vegas.PDFEV`, :class:`vegas.PDFEVArray`, 
            or :class:`vegas.PDFEVDict`.
        """
        oldsettings = {}
        if 'adapt' not in kargs:
            oldsettings['adapt'] = self.adapt
            kargs['adapt'] = False 
        
        if f is None:
            if self.param_sample.shape is None and not hasattr(self, 'extrakeys'):
                self.extrakeys = []
                for k in self.param_sample.all_keys():
                    if k not in self.param_sample:
                        self.extrakeys.append(k)
            else:
                self.extrakeys = None
            f = lbatchintegrand(functools.partial(
                PDFIntegrator.default_stats_f, jac=None, extrakeys=self.extrakeys
                ))
        f = self._make_std_integrand(f, xsample=self.param_sample)
        fpsample = f(self.param_sample)

        if histograms is not False:
            if histograms is True:
                histograms = {}
            nbin = histograms.get('nbin', 12)
            binwidth = histograms.get('binwidth', 0.5)
            histograms['nbin'] = nbin 
            histograms['binwidth'] = binwidth
            loc = histograms.get('loc', None)
            # bins = histograms.get('bins', None)
            if loc is not None:
                if hasattr(loc, 'keys'):
                    loc = _gvar.asbufferdict(loc).flat[:]
                else:
                    loc = numpy.asarray(loc).flat[:]
                mean = _gvar.mean(loc)
                sdev = _gvar.sdev(loc)
            else: 
                @lbatchintegrand
                def ff2(p):
                    if hasattr(p, 'keys'):
                        p = p.lbatch_buf 
                    else:
                        p = p.reshape(p.shape[0], -1)
                    fp = f.eval(p)
                    return dict(f=fp, f2=fp ** 2)
                oldnitn = self.nitn
                r = self(ff2, nitn=1)
                self.set(nitn=oldnitn)
                mean = _gvar.mean(r['f'])
                sdev = numpy.fabs(_gvar.mean(r['f2']) - mean * mean) **  0.5
            bins = []
            halfwidth = nbin / 2 * binwidth
            for i in range(mean.shape[0]):
                bins.append(
                    mean[i] + numpy.linspace(-halfwidth * sdev[i], halfwidth * sdev[i], nbin+1)
                    )
            histograms['bins'] = numpy.array(bins) 
        integrand = lbatchintegrand(functools.partial(
            PDFIntegrator._stats_integrand, f=f, moments=moments, histograms=histograms
            ))
        integrand = self._make_std_integrand(integrand, xsample=self.param_sample.flat[:])
        results = self(integrand, **kargs)
        analyzer = functools.partial(
            PDFIntegrator._stats_analyzer, 
            fpsample=fpsample, moments=moments, histograms=histograms
            )
        if fpsample.shape is None:
            ans = PDFEVDict(results.results, analyzer)
        elif fpsample.shape == ():
            ans = PDFEV(results.results, analyzer)
        else:
            ans = PDFEVArray(results.results, analyzer)
        if oldsettings:
            self.set(**oldsettings)
        return ans

    @staticmethod
    def _stats_analyzer(results, fpsample, moments, histograms):
        """ Create final stats results from Integrator results """
        # convert from Integrator to PDFIntegrator results
        tmp = _gvar.BufferDict()
        for k in results:
            if k == 'pdf':
                continue 
            tmp[k[1]] = results[k]
        results = tmp / results['pdf']

        # 1) mean/cov
        fp = results['fp']
        meanfp = _gvar.mean(fp)
        covfpfp = numpy.zeros(2 * meanfp.shape, dtype=object)
        fp2 = numpy.array(len(meanfp) * [None])
        fpfp = iter(results['fpfp'])
        for i in range(meanfp.shape[0]):
            for j in range(i + 1):
                if i == j:
                    fp2[i] = next(fpfp)
                    covfpfp[i, i] = fp2[i] - fp[i] ** 2
                else:
                    covfpfp[i, j] = covfpfp[j, i] = next(fpfp) - fp[i] * fp[j]
        # add vegas errors to cov and create final result
        covfpfp += _gvar.evalcov(fp)
        ans = _gvar.gvar(meanfp, _gvar.mean(covfpfp))
        if fpsample.shape is None:
            ans = _gvar.BufferDict(fpsample, buf=ans)
            mean = _gvar.BufferDict(fpsample, buf=fp)
            tcov = _gvar.evalcov(mean)
            cov = _gvar.BufferDict()
            sdev= _gvar.BufferDict()
            for k in mean:
                ksl = mean.slice(k)
                for l in mean:
                    lsl = mean.slice(l)
                    if tcov[k,l].shape == (1, 1) or tcov[k,l].shape == ():
                        cov[k, l] = covfpfp[ksl, lsl]
                    else:
                        cov[k, l] = covfpfp[ksl,lsl].reshape(tcov[k,l].shape) 
                sdev[k] = _gvar.fabs(cov[k, k]) ** 0.5           
        elif fpsample.shape == ():
            ans = ans.flat[0]
            mean = fp.flat[0]
            cov = covfpfp
            sdev = _gvar.fabs(cov) ** 0.5
        else:
            ans = ans.reshape(fpsample.shape)
            mean = fp.reshape(fpsample.shape)
            tcov = _gvar.evalcov(mean)
            cov = covfpfp.reshape(tcov.shape)
            sdev = _gvar.fabs(tcov.diagonal()).reshape(mean.shape) ** 0.5
        # 2) moments and histogram
        stats = numpy.array(fpsample.size * [None])
        for i in range(len(stats)):
            if moments:
                mom = [fp[i], fp2[i], results['fp**3'][i], results['fp**4'][i]]
            else:
                mom = [fp[i], fp2[i]] 
            if histograms:
                hist = histograms['bins'][i], results['count'][i]
            else:
                hist = None
            stats[i] = _gvar.PDFStatistics(moments=mom, histogram=hist)
        if fpsample.shape is None:
            stats = _gvar.BufferDict(ans, buf=stats)
        elif fpsample.shape == ():
            stats = stats.flat[0]
        else:
            stats = stats.reshape(ans.shape)
        return ans, dict(stats=stats, vegas_mean=mean, vegas_cov=cov, vegas_sdev=sdev)

    @staticmethod
    def _stats_integrand(p, f, moments=False, histograms=False):
        fp = f.eval(p) 
        nfp = fp.shape[1]
        nbatch = fp.shape[0]
        fpfp = []
        for i in range(fp.shape[1]):
            for j in range(i + 1):
                fpfp.append(fp[:, i] * fp[:, j])
        fpfp = numpy.array(fpfp).T
        ans = _gvar.BufferDict(fp=fp, fpfp=fpfp)
        if moments:
            ans['fp**3'] = fp ** 3
            ans['fp**4'] = fp ** 4
        if histograms:
            count = numpy.zeros((nbatch, nfp, histograms['nbin'] + 2), dtype=float)
            idx = numpy.arange(nbatch)
            for j in range(nfp):
                jdx = numpy.searchsorted(histograms['bins'][j], fp[:, j], side='right')
                count[idx, j, jdx] = 1
            ans['count'] = count
        return ans
    
    @staticmethod
    def default_stats_f(p, jac=None, extrakeys=None):
        if extrakeys is not None and extrakeys:
            for k in extrakeys:
                p[k] = p[k]
        return p

    def sample(self, nbatch, mode='rbatch'):
        """ Generate random samples from the integrator's PDF.

        Typical usage is::

            import gvar as gv 
            import numpy as np
            import vegas

            @vegas.rbatchintegrand
            def g_pdf(p):
                ans = 0
                h = 1.
                for p0 in [0.3, 0.6]:
                    ans += h * np.exp(-np.sum((p-p0)**2, axis=0)/2/.01)
                    h /= 2
                return ans

            g_param = gv.gvar([0.5, 0.5], [[.25, .2], [.2, .25]])
            g_ev = vegas.PDFIntegrator(param=g_param, pdf=g_pdf)
            
            # adapt integrator to g_pdf(p) and evaluate <p>
            g_ev(neval=4000, nitn=10)
            r = g_ev.stats()
            print('<p> =', r, '(vegas)')

            # sample g_pdf(p) 
            wgts, p_samples = g_ev.sample(nbatch=40_000)
            # evaluate mean values <p> and <cos(p0)>
            p_avg = np.sum(wgts * p_samples, axis=1)
            cosp0_avg = np.sum(wgts * np.cos(p_samples[0]))
            print('<p> =', p_avg, '(sample)')
            print('<cos(p0)> =', cosp0_avg, '(sample)')
        
        Here ``p_samples[d, i]`` is a batch of about 40,000 random samples 
        for parameter ``p[d]`` drawn from the (bimodal) distribution with 
        PDF ``g_pdf(p)``. Index ``d=0,1`` labels directions in parameter 
        space, while index ``i`` labels the sample. The samples 
        are weighted by ``wgts[i]``; the sum of all weights equals one. 
        The batch index in ``p_samples`` is the rightmost index because 
        by default ``mode='rbatch'``. (Set ``mode='lbatch'`` to move the 
        batch index to the leftmost position: ``p_samples[i, d]``.)
        The output from this script is::

            <p> = [0.40(17) 0.40(17)] (vegas)
            <p> = [0.40011804 0.39999454] (sample)
            <cos(p0)> = 0.9074221724843065 (sample)
        
        Samples are also useful for making histograms and contour 
        plots of the probability density. For example, the following 
        code uses the :mod:`corner` Python module to create histograms 
        for each parameter, and a contour plot showing 
        their joint distribution::

            import corner
            import matplotlib.pyplot as plt
            
            corner.corner(
                data=p_samples.T, weights=wgts, labels=['p[0]', 'p[1]'],
                range=[0.999, 0.999], show_titles=True, quantiles=[0.16, 0.5, 0.84],
                plot_datapoints=False, fill_contours=True, smooth=1,
                )
            plt.show()

        The output, showing the bimodal structure, is:

        .. image:: bimodal.png
            :width: 80%
        

        Args:
            nbatch (int): The integrator will return
                at least ``nbatch`` samples drawn from its PDF. The 
                actual number of samples is the smallest multiple of 
                ``self.last_neval`` that is equal to or larger than ``nbatch``.
                Results are packaged in arrays or dictionaries
                whose elements have an extra index labeling the different 
                samples in the batch. The batch index is 
                the rightmost index if ``mode='rbatch'``; it is 
                the leftmost index if ``mode`` is ``'lbatch'``. 
            mode (bool): Batch mode. Allowed 
                modes are ``'rbatch'`` or ``'lbatch'``,
                corresponding to batch indices that are on the 
                right or the left, respectively. 
                Default is ``mode='rbatch'``.
        
        Returns:
            A tuple ``(wgts,samples)`` containing samples drawn from the integrator's
            PDF, together with their weights ``wgts``. The weighted sample points 
            are distributed through parameter space with a density proportional to
            the PDF. 
            
            In general, ``samples`` is either a dictionary or an array 
            depending upon the format of |PDFIntegrator| parameter ``param``. 
            For example, if ::

                param = gv.gvar(dict(s='1.5(1)', v=['3.2(8)', '1.1(4)']))

            then ``samples['s'][i]`` is a sample for parameter ``p['s']``
            where index ``i=0,1...nbatch(approx)`` labels the sample. The 
            corresponding sample for ``p['v'][d]``, where ``d=0`` or ``1``, 
            is ``samples['v'][d, i]`` provided ``mode='rbatch'``, which 
            is the default. (Otherwise it is ``p['v'][i, d]``, for 
            ``mode='lbatch'``.) The corresponding weight for this sample
            is ``wgts[i]``.

            When ``param`` is an array, ``samples`` is an array with the same 
            shape plus an extra sample index which is either on the right 
            (``mode='rbatch'``, default) or left (``mode='lbatch'``).
        """
        neval = self.last_neval if hasattr(self, 'last_neval') else self.neval
        nit = 1 if nbatch is None else nbatch // neval 
        if nit * neval < nbatch:
            nit += 1    
        samples = []
        wgts = []
        for _ in range(nit):
            for theta, wgt in self.random_batch():
                # following code comes mostly from _f_lbatch
                tan_theta = numpy.tan(theta)
                chiv = self.scale * tan_theta
                # jac = dp_dtheta
                dp_dtheta = self.scale * numpy.prod((tan_theta ** 2 + 1.), axis=1) * self.param_pdf.dp_dchiv
                pflat = self.param_pdf.pflat(chiv, mode='lbatch')
                if self.pdf is None:
                    # normalized in chiv space so don't want param_pdf.dpdchiv in jac
                    pdf = numpy.prod(numpy.exp(-(chiv ** 2) / 2.) / numpy.sqrt(2 * numpy.pi), axis=1) / self.param_pdf.dp_dchiv
                else:
                    pdf = numpy.prod(self.pdf.eval(pflat, jac=None), axis=1)
                p = self.param_pdf._unflatten(pflat, mode='lbatch')
                wgts.append(wgt * dp_dtheta * pdf)
                samples.append(pflat)
        samples =  numpy.concatenate(samples, axis=0)
        wgts = numpy.concatenate(wgts)
        wgts /= numpy.sum(wgts)
        if mode == 'rbatch':
            samples = self.param_pdf._unflatten(samples.T, mode='rbatch')
        else:
            samples = self.param_pdf._unflatten(samples, mode='lbatch')
        return wgts, samples

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
        ans = PDFIntegrator._make_ans(results)
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
            :class:`vegas.RAvgArray`, :class:`vegas.PDFEV`, 
            :class:`vegas.PDFEVArray`, :class:`vegas.PDFEVArray`).
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
    for t in [PDFEV, PDFEVArray, PDFEVDict]:
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

