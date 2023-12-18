# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-21 G. Peter Lepage.
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

import collections
import math
import os
import pickle
import unittest
import warnings
try:
    import h5py 
except ImportError:
    h5py = None

import gvar as gv
import numpy as np

from numpy.testing import assert_allclose as np_assert_allclose
from vegas import *


class TestAdaptiveMap(unittest.TestCase):
    def setUp(self):
        gv.ranseed(123)

    def tearDown(self):
        pass

    def test_init(self):
        " AdaptiveMap(...) "
        m = AdaptiveMap(grid=[[0, 1], [2, 4]])
        np_assert_allclose(m.grid, [[0, 1], [2, 4]])
        np_assert_allclose(m.inc, [[1], [2]])
        np_assert_allclose(m.ninc, [1, 1])
        m = AdaptiveMap(grid=[[0, 1], [-2, 4]], ninc=2)
        np_assert_allclose(m.grid, [[0, 0.5, 1.], [-2., 1., 4.]])
        np_assert_allclose(m.inc, [[0.5, 0.5], [3., 3.]])
        self.assertEqual(m.dim, 2)
        np_assert_allclose(m.ninc, [2, 2])
        m = AdaptiveMap([[0, 0.4, 1], [-2, 0., 4]], ninc=4)
        np_assert_allclose(
            m.grid,
            [[0, 0.2, 0.4, 0.7, 1.], [-2., -1., 0., 2., 4.]]
            )
        np_assert_allclose(m.inc, [[0.2, 0.2, 0.3, 0.3], [1, 1, 2, 2]])
        self.assertEqual(m.dim, 2)
        np_assert_allclose(m.ninc, [4, 4])
        m = AdaptiveMap([[0,1], [2, 2.5, 4]])
        np_assert_allclose(m.ninc, [1, 2])
        np_assert_allclose(m.grid[0, :2], [0,1])
        np_assert_allclose(m.grid[1, :3], [2, 2.5, 4])
        np_assert_allclose(m.inc[0, :1], [1])
        np_assert_allclose(m.inc[1, :2], [.5, 1.5])

    def test_pickle(self):
        " pickle AdaptiveMap "
        m1 = AdaptiveMap(grid=[[0, 1, 3], [-2, 0, 6]])
        with open('test_map.p', 'wb') as ofile:
            pickle.dump(m1, ofile)
        with open('test_map.p', 'rb') as ifile:
            m2 = pickle.load(ifile)
        os.remove('test_map.p')
        np_assert_allclose(m2.grid, m1.grid)
        np_assert_allclose(m2.inc, m1.inc)

    def test_map(self):
        " map(...) "
        m = AdaptiveMap(grid=[[0, 1, 3], [-2, 0, 6]])
        # 5 y values
        y = np.array(
            [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]
            )
        x = np.empty(y.shape, float)
        jac = np.empty(y.shape[0], float)
        m.map(y, x, jac)
        np_assert_allclose(
            x,
            [[0, -2], [0.5, -1], [1, 0], [2, 3], [3, 6]]
            )
        np_assert_allclose(
            jac,
            [8, 8, 48, 48, 48]
            )
        np_assert_allclose(m(y), x)
        np_assert_allclose(m.jac(y), jac)
    
    def test_invmap(self):
        " invmap(...) "
        m = AdaptiveMap(grid=[[0., 1., 3.], [-2., 0., 6.]])
        # 5 x values
        x = np.array([[0., -2.], [0.5, -1], [1, 0], [2, 3], [3, 6]])
        ytrue = np.array(
            [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]
            )
        jactrue = np.array([8., 8, 48, 48, 48])
        y = np.empty(x.shape, float)
        jac = np.empty(x.shape[0], float)
        m.invmap(x, y, jac)
        np_assert_allclose(y, ytrue)
        np_assert_allclose(jac, jactrue)
        np_assert_allclose(m(y), x)
        np_assert_allclose(m.jac(y), jac)

    def test_region(self):
        " region(...) "
        m = AdaptiveMap(grid=[[0, 1, 3], [-2, 0, 6]])
        np_assert_allclose(m.region(0), [0, 3])
        np_assert_allclose(m.region(1), [-2, 6])
        np_assert_allclose(m.region(), [[0, 3], [-2, 6]])

    def test_settings(self):
        " settings(...) "
        m = AdaptiveMap(grid=[[0, 1, 3], [-2, 0, 6]], ninc=4)
        output = "    grid[ 0] = [ 0.   0.5  1.   2.   3. ]\n"
        output += "    grid[ 1] = [-2. -1.  0.  3.  6.]\n"
        self.assertEqual(m.settings(5).replace(' ', ''), output.replace(' ', ''))
        output = "    grid[ 0] = [ 0.5  2. ]\n"
        output += "    grid[ 1] = [-1.  3.]\n"
        self.assertEqual(m.settings(2).replace(' ', ''), output.replace(' ', ''))

    def test_adapt_to_samples(self):
        " adapt_to_samples(...) "
        m1  = AdaptiveMap([[0, 2], [0, 1]])
        x = np.random.normal(0,.1, (1000, 2))
        def F(x):
            return np.exp(-np.sum(x**2, axis=1) * 100 / 2)
        Fx = F(x)
        m1.adapt_to_samples(x, Fx, nitn=5)
        m1.adapt(ninc=2)
        np_assert_allclose(np.asarray(m1.grid), [[0., 0.071, 2.0], [0., 0.073, 1.]], rtol=0.4)

    def test_training_data_adapt(self):
        "add_training_data(...)  adapt(...) "
 
        # change ninc; no adaptation -- already adapted
        m = AdaptiveMap([[0, 2], [-1, 1]], ninc=2)
        y = np.array([[0.25, 0.25], [0.75, 0.75]])
        f = m.jac(y)
        m.add_training_data(y, f)
        m.adapt(alpha=1.5)
        np_assert_allclose(m.grid, [[0., 1., 2.], [-1, 0, 1]])

        # no adaptation -- alpha = 0
        m = AdaptiveMap([[0, 1.5, 2], [-1, 0, 1]])
        y = np.array([[0.25, 0.25], [0.75, 0.75]])
        f = m.jac(y)
        m.add_training_data(y, f)
        m.adapt(alpha=0.0)
        np_assert_allclose(m.grid, [[0, 1.5, 2], [-1, 0, 1]])

        # Adapt to functions:
        # Place y values at 2-pt Gaussian quadrature
        # abscissas so training is equivalent to
        # an integral for functions that are linear
        # or quadratic (or a superposition). This
        # simulates random y's spread uniformly
        # over space. ygauss below is for ninc=2,
        # with two abscissas per increment.
        g = 1. /3.**0.5
        ygauss = [(1-g)/4., (1+g)/4, (3-g)/4, (3+g)/4.]

        m = AdaptiveMap([[0, 2]], ninc=2)
        y = np.array([[yi] for yi in ygauss])
        def F(x):
            return x[:, 0] ** 2
        for i in range(60):
            f = F(m(y)) * m.jac(y)
            m.add_training_data(y, f)
            m.adapt(alpha=2.)
        np_assert_allclose(m.grid, [[0, 2./2.**(1./3.), 2.]])

        m = AdaptiveMap([[0, 2], [0, 4]], ninc=2)
        y = np.array([[yi, yj] for yi in ygauss for yj in ygauss])
        def F(x):
            return x[:, 0] * x[:, 1] ** 2
        for i in range(60):
            f = F(m(y)) * m.jac(y)
            m.add_training_data(y, f)
            m.adapt(alpha=2.)
        np_assert_allclose(
            m.grid,
            [[0, 2. * 2.**(-0.5), 2.], [0, 4. * 2**(-1./3.), 4.]]
            )

        # same again, with no smoothing
        m = AdaptiveMap([[0, 2], [0, 4]], ninc=2)
        y = np.array([[yi, yj] for yi in ygauss for yj in ygauss])
        def F(x):
            return x[:, 0] * x[:, 1] ** 2
        for i in range(20):
            f = F(m(y)) * m.jac(y)
            m.add_training_data(y, f)
            m.adapt(alpha=-2.)
        np_assert_allclose(
            m.grid,
            [[0, 2. * 2.**(-0.5), 2.], [0, 4. * 2**(-1./3.), 4.]]
            )
       
class TestRAvg(unittest.TestCase):
    def setUp(self):
        gv.ranseed(123)

    def tearDown(self):
        pass

    def test_all(self):
        " RWavg "
        a = RAvg()
        a.add(gv.gvar(1, 1))
        a.add(gv.gvar(2, 2))
        a.add(gv.gvar(3, 3))
        np_assert_allclose(a.mean, 1.346938775510204)
        np_assert_allclose(a.sdev, 0.8571428571428571)
        self.assertEqual(a.dof, 2)
        np_assert_allclose(a.chi2, 0.5306122448979592)
        np_assert_allclose(a.Q, 0.7669711269557102)
        self.assertEqual(str(a), '1.35(86)')
        s = [
            "itn   integral        wgt average     chi2/dof        Q",
            "-------------------------------------------------------",
            "  1   1.0(1.0)        1.0(1.0)            0.00     1.00",
            "  2   2.0(2.0)        1.20(89)            0.20     0.65",
            "  3   3.0(3.0)        1.35(86)            0.27     0.77",
            ""
            ]
        self.assertEqual(a.summary(), '\n'.join(s))

    def test_ravg_wgtd(self):
        " weighted RAvg "
        # if not have_gvar:
        #     return
        mean = np.random.uniform(-10., 10.)
        xbig = gv.gvar(mean, 1.)
        xsmall = gv.gvar(mean, 0.1)
        ravg = RAvg()
        N = 30
        for i in range(N):
            ravg.add(gv.gvar(xbig(), xbig.sdev))
            ravg.add(gv.gvar(xsmall(), xsmall.sdev))
        np_assert_allclose(
            ravg.sdev, 1/ (N * ( 1. / xbig.var + 1. / xsmall.var)) ** 0.5
            )
        self.assertLess(abs(ravg.mean - mean), 5 * ravg.sdev)
        self.assertGreater(ravg.Q, 1e-3)
        self.assertEqual(ravg.dof, 2 * N - 1)

    def test_ravg_unwgtd(self):
        " unweighted RAvg "
        # if not have_gvar:
        #     return
        mean = np.random.uniform(-10., 10.)
        x = gv.gvar(mean, 0.1)
        ravg = RAvg(weighted=False)
        N = 30
        for i in range(N):
            ravg.add(gv.gvar(x(), x.sdev))
        np_assert_allclose( ravg.sdev, x.sdev / (N ** 0.5))
        self.assertLess(abs(ravg.mean - mean), 5 * ravg.sdev)
        self.assertGreater(ravg.Q, 1e-3)
        self.assertEqual(ravg.dof, N - 1)

    def test_ravgarray_wgtd(self):
        " weighted RAvgArray "
        # if not have_gvar:
        #     return
        mean = np.random.uniform(-10., 10., (2,))
        cov = np.array([[1., 0.5], [0.5, 2.]])
        invcov = np.linalg.inv(cov)
        N = 30
        xbig = gv.gvar(mean, cov)
        rbig = gv.raniter(xbig, N)
        xsmall = gv.gvar(mean, cov / 10.)
        rsmall = gv.raniter(xsmall, N)
        ravg = RAvgArray((1, 2))
        for rb, rs in zip(rbig, rsmall):
            ravg.add([gv.gvar(rb, cov)])
            ravg.add([gv.gvar(rs, cov / 10.)])
        np_assert_allclose(gv.evalcov(ravg.flat), cov / (10. + 1.) / N)
        for i in range(2):
            self.assertLess(abs(mean[i] - ravg[0, i].mean), 5 * ravg[0, i].sdev)
        self.assertEqual(ravg.dof, 4 * N - 2)
        self.assertGreater(ravg.Q, 1e-3)

    def test_ravgarray_unwgtd(self):
        " unweighted RAvgArray "
        # if not have_gvar:
        #     return
        mean = np.random.uniform(-10., 10., (2,))
        cov = np.array([[1., 0.5], [0.5, 2.]]) / 10.
        N = 30
        x = gv.gvar(mean, cov)
        r = gv.raniter(x, N)
        ravg = RAvgArray((1, 2), weighted=False)
        for ri in r:
            ravg.add([gv.gvar(ri, cov)])
        np_assert_allclose(gv.evalcov(ravg.flat), cov / N)
        for i in range(2):
            self.assertLess(abs(mean[i] - ravg[0, i].mean), 5 * ravg[0, i].sdev)
        self.assertEqual(ravg.dof, 2 * N - 2)
        self.assertGreater(ravg.Q, 1e-3)

    def test_array(self):
        " RAvgArray "
        a = RAvgArray((1, 2))
        a.add([[gv.gvar(1, 1), gv.gvar(10,10)]])
        a.add([[gv.gvar(2, 2), gv.gvar(20,20)]])
        a.add([[gv.gvar(3, 3), gv.gvar(30,30)]])
        self.assertEqual(a.shape, (1, 2))
        np_assert_allclose(a[0, 0].mean, 1.346938775510204)
        np_assert_allclose(a[0, 0].sdev, 0.8571428571428571)
        self.assertEqual(a.dof, 4)
        np_assert_allclose(a.chi2, 2*0.5306122448979592)
        np_assert_allclose(a.Q, 0.900374555485)
        self.assertEqual(str(a[0, 0]), '1.35(86)')
        self.assertEqual(str(a[0, 1]), '13.5(8.6)')
        s = [
            "itn   integral        wgt average     chi2/dof        Q",
            "-------------------------------------------------------",
            "  1   1.0(1.0)        1.0(1.0)            0.00     1.00",
            "  2   2.0(2.0)        1.20(89)            0.20     0.82",
            "  3   3.0(3.0)        1.35(86)            0.27     0.90",
            ""
            ]
        self.assertEqual(a.summary(), '\n'.join(s))

    def test_ravgdict(self):
        " RAvgDict "
        a = RAvgDict(dict(s=1.0, a=[[2.0, 3.0]]))
        a.add(dict(s=gv.gvar(1, 1), a=[[gv.gvar(1, 1), gv.gvar(10,10)]]))
        a.add(dict(s=gv.gvar(2, 2), a=[[gv.gvar(2, 2), gv.gvar(20,20)]]))
        a.add(dict(s=gv.gvar(3, 3), a=[[gv.gvar(3, 3), gv.gvar(30,30)]]))
        self.assertEqual(a['a'].shape, (1, 2))
        np_assert_allclose(a['a'][0, 0].mean, 1.346938775510204)
        np_assert_allclose(a['a'][0, 0].sdev, 0.8571428571428571)
        self.assertEqual(str(a['a'][0, 0]), '1.35(86)')
        self.assertEqual(str(a['a'][0, 1]), '13.5(8.6)')
        np_assert_allclose(a['s'].mean, 1.346938775510204)
        np_assert_allclose(a['s'].sdev, 0.8571428571428571)
        self.assertEqual(str(a['s']), '1.35(86)')
        self.assertEqual(a.dof, 6)
        np_assert_allclose(a.chi2, 3*0.5306122448979592)
        np_assert_allclose(a.Q, 0.953162484587)
        s = [
            "itn   integral        wgt average     chi2/dof        Q",
            "-------------------------------------------------------",
            "  1   1.0(1.0)        1.0(1.0)            0.00     1.00",
            "  2   2.0(2.0)        1.20(89)            0.20     0.90",
            "  3   3.0(3.0)        1.35(86)            0.27     0.95",
            ""
            ]
        self.assertEqual(a.summary(), '\n'.join(s))

    def test_ravgdict_unwgtd(self):
        " unweighted RAvgDict "
        # scalar
        mean_s = np.random.uniform(-10., 10.)
        sdev_s = 0.1
        x_s = gv.gvar(mean_s, sdev_s)
        # array
        mean_a = np.random.uniform(-10., 10., (2,))
        cov_a = np.array([[1., 0.5], [0.5, 2.]]) / 10.
        x_a = gv.gvar(mean_a, cov_a)
        N = 30
        r_a = gv.raniter(x_a, N)
        ravg = RAvgDict(dict(scalar=1.0, array=[[2., 3.]]), weighted=False)
        for ri in r_a:
            ravg.add(dict(
                scalar=gv.gvar(x_s(), sdev_s), array=[gv.gvar(ri, cov_a)]
                ))
        np_assert_allclose( ravg['scalar'].sdev, x_s.sdev / (N ** 0.5))
        self.assertLess(
            abs(ravg['scalar'].mean - mean_s),
            5 * ravg['scalar'].sdev
            )
        np_assert_allclose(gv.evalcov(ravg['array'].flat), cov_a / N)
        for i in range(2):
            self.assertLess(
                abs(mean_a[i] - ravg['array'][0, i].mean),
                5 * ravg['array'][0, i].sdev
                )
        self.assertEqual(ravg.dof, 2 * N - 2 + N - 1)
        self.assertGreater(ravg.Q, 1e-3)

    def test_ravgdict_wgtd(self):
        " weighted RAvgDict "
        # scalar
        mean_s = np.random.uniform(-10., 10.)
        xbig_s = gv.gvar(mean_s, 1.)
        xsmall_s = gv.gvar(mean_s, 0.1)
        # array
        mean_a = np.random.uniform(-10., 10., (2,))
        cov_a = np.array([[1., 0.5], [0.5, 2.]])
        invcov = np.linalg.inv(cov_a)
        N = 30
        xbig_a = gv.gvar(mean_a, cov_a)
        rbig_a = gv.raniter(xbig_a, N)
        xsmall_a = gv.gvar(mean_a, cov_a / 10.)
        rsmall_a = gv.raniter(xsmall_a, N)

        ravg = RAvgDict(dict(scalar=1.0, array=[[2., 3.]]))
        for rb, rw in zip(rbig_a, rsmall_a):
            ravg.add(dict(
                scalar=gv.gvar(xbig_s(), 1.), array=[gv.gvar(rb, cov_a)]
                ))
            ravg.add(dict(
                scalar=gv.gvar(xsmall_s(), 0.1),
                array=[gv.gvar(rw, cov_a / 10.)]
                ))
        np_assert_allclose(
            ravg['scalar'].sdev,
            1/ (N * ( 1. / xbig_s.var + 1. / xsmall_s.var)) ** 0.5
            )
        self.assertLess(
            abs(ravg['scalar'].mean - mean_s), 5 * ravg['scalar'].sdev
            )
        np_assert_allclose(gv.evalcov(ravg['array'].flat), cov_a / (10. + 1.) / N)
        for i in range(2):
            self.assertLess(
                abs(mean_a[i] - ravg['array'][0, i].mean),
                5 * ravg['array'][0, i].sdev
                )
        self.assertEqual(ravg.dof, 4 * N - 2 + 2 * N - 1)
        self.assertGreater(ravg.Q, 0.5e-3)

    def test_ravg_pickle(self): 
        @rbatchintegrand
        def g(p):
            return p[0] ** 2 * 1.5 / 8

        @rbatchintegrand
        def ga(p):
            return [p[0] ** 2 * 1.5, 1+ p[0] ** 2 * 1.5]

        @rbatchintegrand
        def gd(p):
            return dict(x2=p[0] ** 2 * 1.5, one=[[1 + p[0] ** 2 * 1.5]])
        dim = 2
        itg = Integrator(dim * [[-1,1]], nitn=2, neval=100)
        for _g in [g, ga, gd]:
            r = itg(_g)
            def test_rx(rx):
                " rx = r ?"
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                self.assertAlmostEqual(rx.chi2, r.chi2)
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
            r1 = ravg(r.itn_results)
            test_rx(r1)
            r1a = ravg(r.itn_results, rescale=r.itn_results[0])
            test_rx(r1a)
            r1b = ravg(r.itn_results, rescale=gv.mean(r.itn_results[0]))
            test_rx(r1b)
            r2 = ravg(r)
            test_rx(r2)
            d3 = pickle.dumps(r)
            test_rx(r) # r unchanged?
            r3 = pickle.loads(d3)
            test_rx(r3)  
            d4 = gv.dumps(r)
            test_rx(r)
            r4 = gv.loads(d4)
            test_rx(r4)  
            r5 = ravg(r, weighted=False)
            self.assertNotEqual(str(r5), str(r))

    def test_save_extend(self):
        " save and saveall keywords " 
        global r
        @rbatchintegrand
        def g(p):
            return p[0] ** 2 * 1.5 / 8

        @rbatchintegrand
        def ga(p):
            return [p[0] ** 2 * 1.5, 1+ p[0] ** 2 * 1.5]

        @rbatchintegrand
        def gd(p):
            return dict(x2=p[0] ** 2 * 1.5, one=[[1 + p[0] ** 2 * 1.5]])
        dim = 2
        itg = Integrator(dim * [[-1,1]], nitn=2, neval=100)
        for _g in [g, ga, gd]:
            r = itg(_g, save='test-save.pkl')
            def test_rx(rx):
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
            with open('test-save.pkl', 'rb') as ifile:
                r1 = pickle.load(ifile)
                test_rx(r1)
        os.remove('test-save.pkl')
        for _g in [g, ga, gd]:
            r = itg(_g, saveall='test-save.pkl')
            def test_rx(rx, itgx):
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
                self.assertEqual(itgx.settings(), itg.settings())
                self.assertAlmostEqual(list(itgx.sigf), list(itg.sigf))
            with open('test-save.pkl', 'rb') as ifile:
                r1, itg1 = pickle.load(ifile)
                test_rx(r1, itg1)
                new_r = itg1(_g)
                r1.extend(new_r)
                self.assertEqual(r.nitn + new_r.nitn, r1.nitn)
                self.assertEqual(r.sum_neval + new_r.sum_neval, r1.sum_neval)
                if _g is g:
                    _r  = r
                    _r1 = r1 
                    _new_r = new_r
                else:
                    _r  = r.flat[0]
                    _r1 = r1.flat[0] 
                    _new_r = new_r.flat[0]
                self.assertGreater(_r.sdev, _r1.sdev)
                self.assertGreater(_new_r.sdev, _r1.sdev)
                self.assertAlmostEqual(1/(1/_r.sdev**2 + 1/_new_r.sdev**2)**.5, _r1.sdev, delta=0.001)
        os.remove('test-save.pkl')

    def test_rescaling(self):
        " enormous scale differences "
        # test on degenerate multi-integrand (to stress it)
        x = gv.gvar(1, 0.001)

        a = RAvgArray((2, ))
        for i in range(3):
            xx = x - x.mean + x()
            a.add([xx, 1e50 * xx])
        self.assertEqual(str(a[0] * 1e50), str(a[1]))
        self.assertEqual(str(gv.evalcorr(a).flat[:]), str(np.ones(4, float)))

        d = RAvgDict(dict(a=1.,b=2.))
        for i in range(3):
            xx = x - x.mean + x()
            d.add(dict(a=xx, b=1e50 * xx))
        self.assertEqual(str(d['a'] * 1e50), str(d['b']))
        self.assertEqual(str(gv.evalcorr(d.buf).flat[:]), str(np.ones(4, float)))
        
class TestIntegrator(unittest.TestCase):
    def setUp(self):
        gv.ranseed(123)

    def tearDown(self):
        pass

    def test_init(self):
        " Integrator "
        I = Integrator([[0.,1.],[-1.,1.]], neval=234, nitn=123, neval_frac=0.75)
        self.assertEqual(I.neval, 234)
        self.assertEqual(I.nitn, 123)
        for k in Integrator.defaults:
            if k in ['neval', 'nitn']:
                self.assertNotEqual(getattr(I,k), Integrator.defaults[k])
            elif k not in ['map']:
                self.assertEqual(getattr(I,k), Integrator.defaults[k])
        np.testing.assert_allclose([I.map.grid[0,0], I.map.grid[0, I.map.ninc[0]]], [0., 1.])
        np.testing.assert_allclose([I.map.grid[1,0], I.map.grid[1, I.map.ninc[1]]], [-1., 1.])
        self.assertEqual(list(I.map.ninc), [20, 20])
        self.assertEqual(list(I.nstrat), [5, 5])
        self.assertEqual(I.neval, 234)
    
        I = Integrator([[0.,1.],[-1.,1.]], nstrat=[1,1], neval=1000)
        self.assertEqual(list(I.map.ninc), [100, 100])
        self.assertEqual(list(I.nstrat), [1, 1])
        self.assertEqual(I.neval, 1000)
        self.assertEqual(I.min_neval_hcube, 1000)
        np.testing.assert_allclose([I.map.grid[0,0], I.map.grid[0, I.map.ninc[0]]], [0., 1.])
        np.testing.assert_allclose([I.map.grid[1,0], I.map.grid[1, I.map.ninc[1]]], [-1., 1.])
        # make sure it works
        def f(x):
            return sum(x)
        I(f)

        I = Integrator([[0., 1.], [-1., 1.]], nstrat=[10, 11], neval_frac=0.75)
        self.assertEqual(list(I.map.ninc), [80, 88])
        self.assertEqual(list(I.nstrat), [10, 11])
        self.assertEqual(I.neval, 880)
        self.assertEqual(I.min_neval_hcube, 2)
        np.testing.assert_allclose([I.map.grid[0,0], I.map.grid[0, I.map.ninc[0]]], [0., 1.])
        np.testing.assert_allclose([I.map.grid[1,0], I.map.grid[1, I.map.ninc[1]]], [-1., 1.])

    def test_settings(self):
        I = Integrator([[0.,1.],[-1.,1.]], neval=254, nitn=123, neval_frac=0.75)
        outstr = [
            "Integrator Settings:",
            "    {neval} (approx) integrand evaluations in each of 123 iterations",
            "    number of: strata/axis = [{nstrat0} {nstrat1}]",
            "               increments/axis = [{ninc0} {ninc1}]",
            "               h-cubes = {nhcube}  processors = 1",
            "               evaluations/batch >= {min_neval_batch:.2g}",
            "               {min_neval_hcube} <= evaluations/h-cube <= {max_neval_hcube:.2g}",
            "    minimize_mem = False  adapt_to_errors = False  adapt = True",
            "    accuracy: relative = 0  absolute = 0",
            "    damping: alpha = {alpha}  beta= {beta}",
            "",
            "    axis    integration limits",
            "    --------------------------",
            "       0            (0.0, 1.0)",
            "       1           (-1.0, 1.0)\n",
            ]
        outstr = ('\n'.join(outstr)).format(
            neval=I.neval, nstrat0=I.nstrat[0], nstrat1=I.nstrat[1],
            ninc0=I.map.ninc[0], ninc1=I.map.ninc[1], nhcube=I.nhcube,
            min_neval_hcube=I.min_neval_hcube, alpha=I.alpha, beta=I.beta,
            min_neval_batch=I.min_neval_batch, max_neval_hcube=float(I.max_neval_hcube),
            )
        self.assertEqual(outstr, I.settings())
        I.set(xdict=dict(x=[0.], y=0.))
        outstr = [
            "Integrator Settings:",
            "    {neval} (approx) integrand evaluations in each of 123 iterations",
            "    number of: strata/axis = [{nstrat0} {nstrat1}]",
            "               increments/axis = [{ninc0} {ninc1}]",
            "               h-cubes = {nhcube}  processors = 1",
            "               evaluations/batch >= {min_neval_batch:.2g}",
            "               {min_neval_hcube} <= evaluations/h-cube <= {max_neval_hcube:.2g}",
            "    minimize_mem = False  adapt_to_errors = False  adapt = True",
            "    accuracy: relative = 0  absolute = 0",
            "    damping: alpha = {alpha}  beta= {beta}",
            "",
            "    key/index    axis    integration limits",
            "    ---------------------------------------",
            "          x 0       0            (0.0, 1.0)",
            "            y       1           (-1.0, 1.0)\n",
            ]
        outstr = ('\n'.join(outstr)).format(
            neval=I.neval, nstrat0=I.nstrat[0], nstrat1=I.nstrat[1],
            ninc0=I.map.ninc[0], ninc1=I.map.ninc[1], nhcube=I.nhcube,
            min_neval_hcube=I.min_neval_hcube, alpha=I.alpha, beta=I.beta,
            min_neval_batch=I.min_neval_batch, max_neval_hcube=float(I.max_neval_hcube),
            )
        self.assertEqual(outstr, I.settings())

    def test_pickle(self):
        I1 = Integrator([[0.,1.],[-1.,1.]], neval=234)
        # @batchintegrand
        # def f(x):
        #     return np.sum(x**20, axis=1) * 21 / 4.
        # I1(f, nitn=5)
        with open('test_integ.p', 'wb') as ofile:
            pickle.dump(I1, ofile)
        with open('test_integ.p', 'rb') as ifile:
            I2 = pickle.load(ifile)
        os.remove('test_integ.p')
        self.assertTrue(isinstance(I2, Integrator))
        for k in Integrator.defaults:
            if k == 'map':
                np_assert_allclose(I1.map.ninc, I2.map.ninc)
                for d in range(I1.dim):
                    # protect against 0.0 entry using 1e-8
                    grid1 = np.array(I1.map.grid[d, :I1.map.ninc[d] + 1]) + 1e-8
                    grid2 = np.array(I2.map.grid[d, :I2.map.ninc[d] + 1]) + 1e-8
                    np_assert_allclose(grid1, grid2, rtol=0.01)
                    inc1 = I1.map.inc[d, :I1.map.ninc[d]]
                    inc2 = I2.map.inc[d, :I2.map.ninc[d]]
                    np_assert_allclose(inc1, inc2, rtol=0.01)
            elif k in ['ran_array_generator']:
                continue
            else:
                self.assertEqual(getattr(I1, k), getattr(I2, k))

    def test_set(self):
        " set "
        new_defaults = dict(
            map=AdaptiveMap([[1,2],[0,1]]),
            neval=100,          # number of evaluations per iteration
            maxinc_axis=100,    # number of adaptive-map increments per axis
            min_neval_batch=10,    # number of h-cubes per batch
            max_neval_hcube=1e1, # max number of evaluations per h-cube
            max_mem=100,        # memory limit
            nitn=100,           # number of iterations
            alpha=0.35,
            beta=0.25,
            adapt_to_errors=True,
            rtol=0.1,
            atol=0.2,
            analyzer=reporter(5),
            )
        I = Integrator([[1,2]])
        old_defaults = I.set(**new_defaults)
        for k in new_defaults:
            if k == 'map':
                np_assert_allclose(
                    [
                        [I.map.grid[0, 0], I.map.grid[0, -1]],
                        [I.map.grid[1, 0], I.map.grid[1, -1]]
                    ],
                    new_defaults['map'].grid)
            else:
                self.assertEqual(getattr(I,k), new_defaults[k])
        
        # test special cases (2-d)
        I = Integrator([[1,1.3,2], [0,1]], maxinc_axis=1000, neval_frac=0.75)
        I.set(nstrat=[22, 13])
        self.assertEqual(I.neval, 22 * 13 * 2 / 0.25)
        self.assertEqual(I.min_neval_hcube, 2)
        I.set(nstrat=[7,9], neval=2000)
        self.assertEqual(I.neval, 2000)
        self.assertEqual(list(I.nstrat), [7, 9])
        self.assertEqual(list(I.map.ninc), [196, 198])
        self.assertEqual(I.min_neval_hcube, 7)
        I.set(nstrat=[7,9])
        self.assertEqual(I.neval, 7 * 9 * 2 / 0.25)
        self.assertEqual(I.min_neval_hcube, 2)
        with self.assertRaises(ValueError):
            I.set(nstrat=[7,9], neval=20)
        I.set(neval=3100) 
        self.assertEqual(list(I.nstrat), [20, 19])
        self.assertEqual(I.min_neval_hcube, 2)
        with self.assertRaises(ValueError):
            I.set(nstrat=[2,3,5])
        I.set(neval=3500) 
        self.assertEqual(list(I.nstrat), [21, 20])
        self.assertEqual(I.min_neval_hcube, 2)
        with self.assertRaises(ValueError):
            I.set(nstrat=[10,12], neval=120*4)
        I.set(nstrat=[10,12], neval=120*8)
        self.assertEqual(I.neval, 120*8)
        self.assertEqual(I.min_neval_hcube, 2)
        # check alignment of the two grids
        I.set(neval=3.5e8)
        nstrat = np.array(I.nstrat)
        ninc = np.array(I.map.ninc)
        self.assertTrue(np.all(np.round(nstrat / ninc) * ninc == nstrat))
        I.set(neval=300)
        nstrat = np.array(I.nstrat)
        ninc = np.array(I.map.ninc)
        self.assertTrue(np.all(np.round(ninc / nstrat) * nstrat == ninc))
        # sigf
        I.set(sigf=[1.])
        self.assertNotEqual(len(I.sigf), 1)

    def test_volume(self):
        " integrate constants "
        def f(x):
            return 2.
        I = Integrator([[-1, 1], [0, 4]])
        r = I(f)
        np_assert_allclose(r.mean, 16, rtol=1e-6)
        self.assertTrue(r.sdev < 1e-6)
        def f(x):
            return [-1., 2.]
        I = Integrator([[-1, 1], [0, 4]])
        r = I(f)
        np_assert_allclose(r[0].mean, -8, rtol=5e-2)
        self.assertTrue(r[0].sdev < 1e-6)
        np_assert_allclose(r[1].mean, 16, rtol=5e-2)
        self.assertTrue(r[1].sdev < 1e-6)

    def test_scalar(self):
        " integrate scalar fcn "
        def f(x):
            return (math.sin(x[0]) ** 2 + math.cos(x[1]) ** 2) / math.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f, neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)

    def test_batch(self):
        " integrate batch fcn "
        @batchintegrand
        class f_batch:
            def __call__(self, x):
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / np.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f_batch(), neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)
        self.assertLess(r.sdev, 1e-3)
        class f_batch(BatchIntegrand):
            def __call__(self, x):
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / np.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f_batch(), neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)
        self.assertLess(r.sdev, 1e-3)
        class f_batch:
            def info(self):
                return 'hi'
            def __call__(self, x):
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / np.pi ** 2
        f = batchintegrand(f_batch())
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f, neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)
        self.assertLess(r.sdev, 1e-3)
        self.assertEqual(f.info(), 'hi')
        @batchintegrand
        def f_batch(x):
            return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / np.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f_batch, neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)
        self.assertLess(r.sdev, 1e-3)

    def test_VegasIntegrand(self):
        " VegasIntegrand(fcn) "
        from vegas._vegas import VegasIntegrand
        def f0(x):
            return x[0] + x[1] + x[2]
        @lbatchintegrand 
        class f1_class:
            def __call__(self, x):
                return x[:, 0] + x[:, 1] + x[:, 2]
            def __getattr__(self, name):
                raise AttributeError()
            def __setattr__(self, name, value):
                raise AttributeError()
        f1 = f1_class()
        self.assertTrue(batchintegrand is lbatchintegrand)
        self.assertTrue(BatchIntegrand is LBatchIntegrand)
        f2 = rbatchintegrand(f0)
        @lbatchintegrand
        def f3(x):
            return [[x[:, 0] + x[:, 1] + x[:, 2]]]
        @rbatchintegrand
        class f4_class:
            def __call__(self, x):
                return [[x[0] + x[1] + x[2]]]
            def __getattr__(self, name):
                raise AttributeError()
            def __setattr__(self, name, value):
                raise AttributeError()
        f4 = f4_class()
        def f5(x):
            return collections.OrderedDict([('a', [[x[0] + x[1] + x[2]]])])
        @lbatchintegrand
        def f6(x):
            return collections.OrderedDict([('a', [[x[:, 0] + x[:, 1] + x[:, 2]]])])
        f7 = rbatchintegrand(f5)
        x = numpy.random.uniform(size=(5,3))
        ans = numpy.sum(x, axis=1).reshape((5,1))
        map = AdaptiveMap(3 * [(0,1)])
        for f in [f0, f1, f2, f3, f4, f5, f6, f7]:
            fcn = VegasIntegrand(f, map, uses_jac=False, xdict=None, mpi=False)
            self.assertTrue(numpy.allclose(ans, fcn.eval(x, jac=None)))
            fans = fcn.format_result(x[0,:1], x[:1,:1]**2)
            if f in [f0, f1, f2]:
                self.assertEqual(numpy.shape(fans), ())
            if f in [f3, f4]:
                self.assertEqual(numpy.shape(fans), (1,1))
            if f in [f5, f6, f7]:
                self.assertEqual(list(fans.keys()), ['a'])
                self.assertEqual(numpy.shape(fans['a']), (1,1))

        def f8(x):
            return [[x[0]], [x[0] + x[1] + x[2]]]
        @lbatchintegrand 
        def f9(x):
            ans = numpy.empty((x.shape[0], 2, 1), float)
            ans[:, 0, 0] = x[:, 0]
            ans[:, 1, 0] = x[:, 0] + x[:, 1] + x[:, 2]
            return ans
        f10 = rbatchintegrand (f8)
        def f11(x):
            return collections.OrderedDict([('a', x[0]), ('b', [[x[0] + x[1] + x[2]]])])
        @lbatchintegrand 
        def f12(x):
            return collections.OrderedDict([('a', x[:, 0]), ('b', (x[:, 0] + x[:, 1] + x[:, 2]).reshape((-1,1,1)))])
        f13 = rbatchintegrand(f11)
        ans = numpy.empty((5,2), float)
        ans[:, 0] = x[:, 0]
        ans[:, 1] = numpy.sum(x, axis=1)
        map = AdaptiveMap(3 * [(0,1)])
        for f in [f8, f9, f10, f11, f12, f13]:
            fcn = VegasIntegrand(f, map, uses_jac=False, xdict=None, mpi=False)
            self.assertTrue(numpy.allclose(ans, fcn.eval(x, jac=None)))
            fans = fcn.format_result(x[0,:2], x[:2,:2].dot(x[:2, :2].T))
            if f in [f8, f9, f10]:
                self.assertEqual(numpy.shape(fans), (2,1))
            if f in [f11, f12, f13]:
                self.assertEqual(list(fans.keys()), ['a', 'b'])
                self.assertEqual(numpy.shape(fans['a']), ())
                self.assertEqual(numpy.shape(fans['b']), (1,1))
        
        # test xdict 
        xdict = gv.BufferDict(a=[0., 0.], b=0.)
        x = np.random.uniform(size=(5,3))
        ans = (x[:, 0] * x[:, 1] * x[:, 2]).reshape(5, 1)
        @lbatchintegrand
        def f14(xd):
            return xd['a'][:, 0] * xd['a'][:, 1] * xd['b']
        def f15(xd):        
            return xd['a'][0] * xd['a'][1] * xd['b']
        @rbatchintegrand
        def f16(xd):        
            return xd['a'][0] * xd['a'][1] * xd['b']
        map = AdaptiveMap(3 * [(0,1)])
        for f in [f14, f15, f16]:
            fcn = VegasIntegrand(f, map, uses_jac=False, xdict=xdict, mpi=False)
            self.assertTrue(numpy.allclose(ans, fcn.eval(x, jac=None)))

        
    @unittest.skipIf(h5py is None,"missing h5py => minimize_mem=True not available")
    def test_minimize_mem(self):
        " test minimize_mem=True mode "
        # test 1
        I = Integrator([[0.,1.],[-1.,1.]], neval=1300, neval_frac=0.75, minimize_mem=True)
        self.assertEqual(list(I.map.ninc), [130, 120])
        self.assertEqual(list(I.nstrat), [13, 12])
        self.assertEqual(I.neval, 1300)
        self.assertEqual(I.min_neval_hcube, 2)
        np.testing.assert_allclose([I.map.grid[0,0], I.map.grid[0, I.map.ninc[0]]], [0., 1.])
        np.testing.assert_allclose([I.map.grid[1,0], I.map.grid[1, I.map.ninc[1]]], [-1., 1.])
        # test 2
        gv.ranseed(1)
        @batchintegrand
        class f_batch:
            def __call__(self, x):
                f = np.empty(x.shape[0], float)
                for i in range(f.shape[0]):
                    f[i] = (
                        math.sin(x[i, 0]) ** 2 + math.cos(x[i, 1]) ** 2
                        ) / math.pi ** 2
                return f
        I = Integrator(
            [[0, math.pi],
            [-math.pi/2., math.pi/2.]],
            minimize_mem=True,
            )
        r = I(f_batch(), neval=10000)
        self.assertTrue(abs(r.mean - 1.) < 5 * r.sdev)
        self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.sdev < 1e-3)
        gv.ranseed(1)
        I2 = Integrator(
            [[0, math.pi],
            [-math.pi/2., math.pi/2.]],
            minimize_mem=False,
            )
        r2 = I2(f_batch(), neval=10000)
        self.assertEqual(r2.summary(), r.summary())

    def test_scalar_exception(self):
        " integrate scalar fcn "
        def f(x):
            return (math.sin(x[0]) ** 2 + math.cos(x[1]) ** 2) / math.pi ** 2 / 0.0
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        with self.assertRaises(ZeroDivisionError):
            I(f, neval=100)

    def test_batch_exception(self):
        " integrate batch fcn "
        @batchintegrand
        class f_batch:
            def __call__(self, x):
                f = 1/0.
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / f
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        with self.assertRaises(ZeroDivisionError):
            I(f_batch(), neval=100)

    def test_batch_b0(self):
        " integrate batch fcn beta=0 "
        @batchintegrand
        class f_batch:
            def __call__(self, x):
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / math.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]], beta=0.0)
        r = I(f_batch(), neval=10000)
        self.assertTrue(abs(r.mean - 1.) < 5 * r.sdev)
        self.assertGreater(r.Q, 0.5e-3)
        self.assertGreater(1e-3, r.sdev)

    def test_adapt_to_errors(self):
        " use adapt_to_errors "
        def f(x):
            return (math.sin(x[0]) ** 2 + math.cos(x[1]) ** 2) / math.pi ** 2
        I = Integrator(
            [[0, math.pi], [-math.pi/2., math.pi/2.]],
            adapt_to_errors=True,
            )
        r = I(f, neval=10000)
        self.assertTrue(abs(r.mean - 1.) < 5 * r.sdev)
        self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.sdev < 1e-3)

    def test_adapt_to_errors_b0(self):
        " use adapt_to_errors with beta=0 "
        def f(x):
            return (math.sin(x[0]) ** 2 + math.cos(x[1]) ** 2) / math.pi ** 2
        I = Integrator(
            [[0, math.pi], [-math.pi/2., math.pi/2.]],
            adapt_to_errors=True,
            beta=0.0,
            )
        r = I(f, neval=10000)
        self.assertTrue(abs(r.mean - 1.) < 5 * r.sdev)
        self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.sdev < 1e-3)

    def test_random_batch(self):
        " random_batch "
        def f(x):
            dx2 = 0.
            for d in range(4):
                dx2 += (x[d] - 0.5) ** 2
            return math.exp(-100. * dx2)
        def fv(x):
            dx2 = 0.
            for d in range(4):
                dx2 += (x[:, d] - 0.5) ** 2
            return np.exp(-100. * dx2)
        integ = Integrator(4 * [[0, 1]])
        warmup = integ(f, nitn=10, neval=1000)
        result = integ(f, nitn=1, neval=1000, adapt=False)
        integral = 0.0
        for x, wgt in integ.random_batch():
            integral += wgt.dot(fv(x))
        self.assertLess(abs(result.mean-integral), 5 * result.sdev)

    def test_random(self):
        " random "
        def f(x):
            return x[0] ** 2 + x[1] ** 3
        integ = Integrator(2 * [[0, 2]])
        warmup = integ(f, nitn=10, neval=100)
        result = integ(f, nitn=1, neval=100, adapt=False)
        integral = 0.0
        for x, wgt in integ.random():
            integral += wgt * f(x)
        self.assertLess(abs(result.mean-integral), 5 * result.sdev)

    def test_multi(self):
        " multi-integrand "
        def f_s(x):
            dx2 = 0.
            for d in range(4):
                dx2 += (x[d] - 0.5) ** 2
            return math.exp(-100. * dx2)
        def f_multi_s(x):
            dx2 = 0.
            for d in range(4):
                dx2 += (x[d] - 0.5) ** 2
            f = math.exp(-100. * dx2)
            return [[f, f * x[0]]]
        @batchintegrand
        class f_multi_v:
            def __call__(self, x):
                x = np.asarray(x)
                f = np.empty((x.shape[0], 1, 2), float)
                dx2 = 0.
                for d in range(4):
                    dx2 += (x[:, d] - 0.5) ** 2
                f[:, 0, 0] = np.exp(-100. * dx2)
                f[:, 0, 1] = x[:, 0] * f[:, 0, 0]
                return f
        I = Integrator(4 * [[0, 1]])
        warmup = I(f_s, neval=1000, nitn=10)
        for r in [I(f_multi_v(), nitn=10), I(f_multi_s, nitn=10)]:
            ratio = r[0, 1] / r[0, 0]
            self.assertLess(abs(ratio.mean - 0.5), 5 * ratio.sdev)
            self.assertLess(ratio.sdev, 1e-2)

    def test_multi_corrd(self):
        " correlated multi-integrand "
        def f(x):
            f1 = np.sin(x[0]) * x[1]
            f2 = np.cos(x[1]) * x[0]
            fs = f1 + f2
            return [fs, f1, f2]
        integ = Integrator([(0,1), (0,1)])
        integ(f,neval=1e3, nitn=5)
        rs, r1, r2 = integ(f, neval=1e3, nitn=5)
        diff = rs - r1 - r2 
        r12 = r1 + r2
        self.assertLess(diff.mean / r12.mean, 1e-7)
        self.assertLess(diff.sdev / r12.mean, 1e-7)
        
    def test_adaptive(self):
        " adaptive? "
        def f(x):
            dx2 = 0
            for i in range(4):
                dx2 += (x[i] - 0.5) ** 2
            return math.exp(-dx2 * 100.) * 1013.2118364296088
        I = Integrator(4 * [[0, 1]])
        r0 = I(f, neval=10000, nitn=10)
        r1 = I(f, neval=10000, nitn=10)
        self.assertTrue(r0.itn_results[0].sdev / 30 > r1.itn_results[-1].sdev)
        self.assertTrue(r0.itn_results[0].sdev < 1.)
        self.assertTrue(r1.itn_results[-1].sdev < 0.01)
        self.assertTrue(r1.Q > 1e-3)

    def test_dictintegrand(self):
        " dictionary-valued integrand "
        def f(x):
            return dict(a=x[0] + x[1], b=[[x[0] ** 2 * 3., x[1] ** 3 * 4.]])
        I = Integrator(2 * [[0, 1]])
        r = I(f, neval=1000)
        self.assertTrue(abs(r['a'].mean - 1.) < 5. * r['a'].sdev)
        self.assertTrue(abs(r['b'][0, 0].mean - 1.) < 5. * r['b'][0, 0].sdev)
        self.assertTrue(abs(r['b'][0, 1].mean - 1.) < 5. * r['b'][0, 1].sdev)
        self.assertTrue(r['a'].sdev < 1e-2)
        self.assertTrue(r['b'][0, 0].sdev < 1e-2)
        self.assertTrue(r['b'][0, 1].sdev < 1e-2)
        # self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.dof == 27)
        @batchintegrand
        def f(x):
            ans = dict(
                a=np.empty(x.shape[0], float),
                b=np.empty((x.shape[0], 1, 2), float)
                )
            ans['a'] = x[:, 0] + x[:, 1]
            ans['b'][:, 0, 0] = x[:, 0] ** 2 * 3.
            ans['b'][:, 0, 1] = x[:, 1] ** 3 * 4.
            return ans
        I = Integrator(2 * [[0, 1]])
        r = I(f, neval=1000)
        self.assertTrue(abs(r['a'].mean - 1.) < 5. * r['a'].sdev)
        self.assertTrue(abs(r['b'][0, 0].mean - 1.) < 5. * r['b'][0, 0].sdev)
        self.assertTrue(abs(r['b'][0, 1].mean - 1.) < 5. * r['b'][0, 1].sdev)
        self.assertTrue(r['a'].sdev < 1e-2)
        self.assertTrue(r['b'][0, 0].sdev < 1e-2)
        self.assertTrue(r['b'][0, 1].sdev < 1e-2)
        # self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.dof == 27)

    def test_tol(self):
        " test rtol, atol stopping conditions "
        def f(x):
            return 10 * np.exp(-100. * x[0]) * 100.
        def f_array(x):
            return [f(x), f(x)]
        def f_dict(x):
            return dict(a=f(x), b=f(x))
        for args, nitn in [
            (dict(), 2),
            (dict(rtol=0.5), 1),
            (dict(rtol=0.0001), 2),
            (dict(atol=0.5 * 10), 1),
            (dict(atol=0.0001 * 10), 2),
            ]:
            for fcn in [f, f_array, f_dict]:
                I = Integrator([[0,1.]], neval=1000, nitn=2, **args)
                result = I(f)
                self.assertEqual(result.nitn, nitn)

    def test_constant(self):
        " integral of a constant "
        @batchintegrand
        def g(x):
            return np.ones(x.shape[0], float)
        # test weighted results
        integ = Integrator([(0,1)],)
        integ(g, nitn=1, neval=1e2, neval_frac=0.5)
        res = integ(g, nitn=4, neval=1e2)
        self.assertAlmostEqual(res.itn_results[0].sdev / res.sdev, 2.0)
        # test unweighted results
        res = integ(g, nitn=4, neval=1e2, adapt=False)
        self.assertAlmostEqual(res.itn_results[0].sdev / res.sdev, 2.0)
        # zero
        @batchintegrand
        def g(x):
            return np.zeros(x.shape[0], float)
        # test weighted results
        integ = Integrator([(0,1)],)
        res = integ(g, nitn=4, neval=1e2)
        self.assertEqual(res.mean, 0.0)

    def test_uses_jac(self):
        " uses_jac=True "
        integ = Integrator(2 * [[0, 2.]])
        def test(f, mode):
            r = integ(f, nitn=1, neval=10, uses_jac=True)
            if mode == 'array':
                ans = r[0, 0]
            elif mode == 'dict':
                ans = r['a']
            else:
                ans = r
            self.assertAlmostEqual(ans.mean, 1.)
        
        def f(x, jac):
            return 1. / np.prod(jac)
        test(f, 'scalar')
        def f(x, jac):
            return [[1. / np.prod(jac)]]
        test(f, 'array')
        def f(x, jac):
            return dict(a=1. / np.prod(jac))
        test(f, 'dict')

        @rbatchintegrand
        def f(x, jac):
            return 1. / np.prod(jac, axis=0)
        test(f, 'scalar')
        @rbatchintegrand
        def f(x, jac):
            return [[1. / np.prod(jac, axis=0)]]
        test(f, 'array')
        @rbatchintegrand
        def f(x, jac):
            return dict(a=1. / np.prod(jac, axis=0))
        test(f, 'dict')

        @batchintegrand
        def f(x, jac):
            return 1. / np.prod(jac, axis=-1)
        test(f, 'scalar')
        @batchintegrand
        def f(x, jac):
            return [[1. / np.prod(jac, axis=-1)]]
        test(f, 'array')
        @batchintegrand
        def f(x, jac):
            return dict(a=1. / np.prod(jac, axis=-1))
        test(f, 'dict')

    @unittest.skipIf(h5py is None,"missing h5py => minimize_mem=True not available")
    def test_mem(self):
        " max_mem "
        def f(x): return np.prod(x)
        with self.assertRaises(MemoryError):
            I = Integrator(3 * [(0,1)], max_mem=10)
        I = Integrator(3 * [(0,1)], max_mem=10, minimize_mem=True)
        with self.assertRaises(MemoryError):
            I(f, neval=1e4)

class test_PDFIntegrator(unittest.TestCase):
    def test_nobatch(self):
        # g is vector
        g = gv.gvar([1., 2.], [[1, .1], [.1, 4]])
        gev = PDFIntegrator(g, adapt=False)
        def f(p):
            return p[0] + p[1]
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r.mean - sum(g).mean) < 5 * r.sdev)
        ff = str(r)
        def f(p):
            ff = p[0] + p[1]
            ff2 = ff * ff
            return [[ff, ff, ff2], [ff2, ff, ff2 * ff]]
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r[0,0].mean - sum(g).mean) < 5 * r[0,0].sdev)
        var = r[1, 0] - r[0, 0] ** 2
        self.assertTrue(abs(var.mean - sum(g).var) < 5. * var.sdev)
        ff2 = str(r[0, 2])
        ff3 = str(r[1, 2])
        self.assertTrue(ff == str(r[0,0]) == str(r[0, 1]) == str(r[1, 1]))
        self.assertTrue(ff2 == str(r[1,0]))
        self.assertTrue(ff != ff2 != ff3 != ff)
        self.assertTrue(r.shape == (2,3))
        diff3 = r[1,2] - 3 * r[1,0] * r[0,0] + 2 * r[0,0]**3
        self.assertTrue(abs(diff3.mean) < 5. * diff3.sdev)
        def f(p):
            ff = p[0] + p[1]
            ff2 = ff * ff
            ans = collections.OrderedDict()
            ans[0] = ff
            ans[1] = [[ff, ff, ff2], [ff2, ff, ff2 * ff]]
            return ans
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(ff == str(r[0]) == str(r[1][0,0]) == str(r[1][0,1]) == str(r[1][1,1]) )
        self.assertTrue(ff2 == str(r[1][0,2]) == str(r[1][1,0]))
        self.assertTrue(ff3 == str(r[1][1,2]))
        self.assertTrue(r[1].shape == (2,3))

    def test_rbatch(self):
        # g is vector
        g = gv.gvar([1., 2.], [[1, .1], [.1, 4]])
        gev = PDFIntegrator(g, adapt=False)
        @rbatchintegrand
        def f(p):
            return p[0] + p[1]
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r.mean - sum(g).mean) < 5 * r.sdev)
        ff = str(r)
        @rbatchintegrand
        def f(p):
            ff = p[0] + p[1]
            ff2 = ff * ff
            return [[ff, ff, ff2], [ff2, ff, ff2 * ff]]
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r[0,0].mean - sum(g).mean) < 5 * r[0,0].sdev)
        var = r[1, 0] - r[0, 0] ** 2
        self.assertTrue(abs(var.mean - sum(g).var) < 5. * var.sdev)
        ff2 = str(r[0, 2])
        ff3 = str(r[1, 2])
        self.assertTrue(ff == str(r[0,0]) == str(r[0, 1]) == str(r[1, 1]))
        self.assertTrue(ff2 == str(r[1,0]))
        self.assertTrue(ff != ff2 != ff3 != ff)
        self.assertTrue(r.shape == (2,3))
        diff3 = r[1,2] - 3 * r[1,0] * r[0,0] + 2 * r[0,0]**3
        self.assertTrue(abs(diff3.mean) < 5. * diff3.sdev)
        @rbatchintegrand
        def f(p):
            ff = p[0] + p[1]
            ff2 = ff * ff
            ans = collections.OrderedDict()
            ans[0] = ff
            ans[1] = [[ff, ff, ff2], [ff2, ff, ff2 * ff]]
            return ans
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(ff == str(r[0]) == str(r[1][0,0]) == str(r[1][0,1]) == str(r[1][1,1]) )
        self.assertTrue(ff2 == str(r[1][0,2]) == str(r[1][1,0]))
        self.assertTrue(ff3 == str(r[1][1,2]))
        self.assertTrue(r[1].shape == (2,3))

    def test_lbatch(self):
        # g is vector
        g = gv.gvar([1., 2.], [[1, .1], [.1, 4]])
        gev = PDFIntegrator(g, adapt=False) 
        # does not work with alpha=0,beta=0 (weighted avg bad for very highly correlated answers)
        @lbatchintegrand
        def f(p):
            return p[:, 0] + p[:, 1]
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r.mean - sum(g).mean) < 5 * r.sdev)
        ff = str(r)
        # array
        @lbatchintegrand
        def f(p):
            ff = p[:, 0] + p[:, 1]
            ff2 = ff * ff
            ans = np.zeros(p.shape[:1] + (2,3), float)
            ans[:, 0, 0] = ff 
            ans[:, 0, 1] = ff 
            ans[:, 0, 2] = ff2 
            ans[:, 1, 0] = ff2 
            ans[:, 1, 1] = ff 
            ans[:, 1, 2] = ff * ff2
            return ans
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(abs(r[0,0].mean - sum(g).mean) < 5 * r[0,0].sdev)
        var = r[1, 0] - r[0, 0] ** 2
        self.assertTrue(abs(var.mean - sum(g).var) < 5. * var.sdev)
        ff2 = str(r[0, 2])
        ff3 = str(r[1, 2])
        self.assertTrue(ff == str(r[0,0]) == str(r[0, 1]) == str(r[1, 1]))
        self.assertTrue(ff2 == str(r[1,0]))
        self.assertTrue(ff != ff2 != ff3 != ff)
        self.assertTrue(r.shape == (2,3))
        diff3 = r[1,2] - 3 * r[1,0] * r[0,0] + 2 * r[0,0]**3
        self.assertTrue(abs(diff3.mean) < 5. * diff3.sdev)
        # dict
        @lbatchintegrand
        def f(p):
            ff = p[:, 0] + p[:, 1]
            ff2 = ff * ff
            ans = collections.OrderedDict()
            ans[0] = ff
            ans[1] = np.zeros(p.shape[:1] + (2,3), float)
            ans[1][:, 0, 0] = ff 
            ans[1][:, 0, 1] = ff 
            ans[1][:, 0, 2] = ff2 
            ans[1][:, 1, 0] = ff2 
            ans[1][:, 1, 1] = ff 
            ans[1][:, 1, 2] = ff * ff2
            return ans
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertTrue(ff == str(r[0]) == str(r[1][0,0]) == str(r[1][0,1]) == str(r[1][1,1]) )
        self.assertTrue(ff2 == str(r[1][0,2]) == str(r[1][1,0]))
        self.assertTrue(ff3 == str(r[1][1,2]))
        self.assertTrue(r[1].shape == (2,3))

    def test_scalar_pdf(self):
        g = gv.gvar(1, 2)
        gev = PDFIntegrator(g, alpha=0, beta=0)
        @rbatchintegrand
        def f(p):
            ff = p
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - g.mean), 5 * r.sdev)
        @lbatchintegrand
        def f(p):
            ff = p
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - g.mean), 5 * r.sdev)

    def test_array_pdf(self):
        g = gv.gvar([1., 2.], [[1, .1], [.1, 4]])
        gev = PDFIntegrator(g, alpha=0, beta=0)
        @rbatchintegrand
        def f(p):
            ff = np.sum(p, axis=0)
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - sum(g).mean), 5 * r.sdev)
        @lbatchintegrand
        def f(p):
            ff = np.sum(p, axis=1)
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - sum(g).mean), 5 * r.sdev)

    def test_dict_pdf(self):
        g = dict(a=gv.gvar(1, 1), b=gv.gvar([1., 2.], [[1, .1], [.1, 4]]))
        gev = PDFIntegrator(g, alpha=0, beta=0)
        def f(p):
            ff = np.sum(p['b'], axis=0)
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - sum(g['b']).mean), 5 * r.sdev)
        @rbatchintegrand
        def f(p):
            ff = np.sum(p['b'], axis=0)
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - sum(g['b']).mean), 5 * r.sdev)
        @lbatchintegrand
        def f(p):
            ff = np.sum(p['b'], axis=1)
            return ff 
        gv.ranseed(1)
        r = gev(f, nitn=1)
        self.assertLess(abs(r.mean - sum(g['b']).mean), 5 * r.sdev)

    def test_change_pdf(self):
        gv.ranseed(12345)
        g = gv.gvar(1, 2)
        gev = PDFIntegrator(g, alpha=0, beta=0)
        # shift peak
        @rbatchintegrand
        def f(p):
            return [p, p**2] 
        def pdf(p):
            return gv.exp(-(p - g.mean - 0.5 * g.sdev) ** 2 / 8) / np.sqrt(2 * np.pi * g.var)
        r = gev(f, pdf=pdf, nitn=2, adapt=True)
        self.assertLess(abs(r[0].mean - g.mean - 0.5 * g.sdev), 10 * r[0].sdev)

        gv.ranseed(12345)
        g = gv.gvar(1, 2)
        gev = PDFIntegrator(g, alpha=0, beta=0)
        # shift peak
        @lbatchintegrand
        def f(p):
            return np.moveaxis([p, p**2], -1, 0)
        def pdf(p):
            return gv.exp(-(p - g.mean - 0.5 * g.sdev) ** 2 / 8) / np.sqrt(2 * np.pi * g.var)
        r = gev(f, pdf=pdf, nitn=2, adapt=True)
        self.assertLess(abs(r[0].mean - g.mean - 0.5 * g.sdev), 10 * r[0].sdev)

    def test_adapt_to_pdf(self):
        gv.ranseed(1)
        g = gv.gvar([1., 2.], [[1, .1], [.1, 4]])
        gev = PDFIntegrator(g, alpha=0, beta=0, adapt_to_pdf=False)
        @rbatchintegrand
        def f(p):
            return p[0] + p[1]
        gv.ranseed(1)
        r = gev(f, nitn=2)
        self.assertLess(abs(r.mean - sum(g).mean), 5 * r.sdev)

    def test_limit_scale(self):
        " integrator(...,limit=..,scale=..) "
        g = gv.gvar(1,0.1)
        for scale in [1., 2.]:
            integ = PDFIntegrator(g, limit=1., scale=scale)
            norm = integ(neval=1000, nitn=5).pdfnorm
            self.assertLess(
                abs(norm.mean - 0.682689492137), 5 * norm.sdev
                )
            integ = PDFIntegrator(g, limit=2., scale=scale)
            norm = integ(neval=1000, nitn=5).pdfnorm
            self.assertLess(
                abs(norm.mean - 0.954499736104), 5 * norm.sdev
                )

    def test_no_f(self):
        for g in [gv.gvar(1,1), gv.gvar([2*['1(1)']]), gv.gvar(dict(a='1(1)', b=[2*['2(2)']]))]:
            gev = PDFIntegrator(g)
            norm = gev(nitn=1).pdfnorm 
            self.assertLess(abs(norm.mean-1), 5 * norm.sdev)

    def test_histogram(self):
        x = gv.gvar([5., 3.], [[4., 0.2], [0.2, 1.]])
        xsum = x[0] + x[1]
        integ = PDFIntegrator(x)
        hist = gv.PDFHistogram(xsum, nbin=40, binwidth=0.2)
        integ(neval=1000, nitn=5)
        def fhist(x):
            return hist.count(x[0] + x[1])
        r = integ(fhist, neval=1000, nitn=5, adapt=False)
        bins, prob, stat, norm = hist.analyze(r)
        self.assertLess(abs(gv.mean(np.sum(prob)) - 1.), 5. * gv.sdev(np.sum(prob)))
        self.assertLess(abs(stat.mean.mean - xsum.mean), 5. * stat.mean.sdev)
        self.assertLess(abs(stat.sdev.mean - xsum.sdev), 5. * stat.sdev.sdev)
        self.assertLess(abs(stat.skew.mean), 5. * stat.skew.sdev)
        self.assertLess(abs(stat.ex_kurt.mean), 5. * stat.ex_kurt.sdev)

    def test_ravg(self): 
        @rbatchintegrand
        def g(p):
            return p[0] ** 2 * 1.5 / 8

        @rbatchintegrand
        def ga(p):
            return [p[0] ** 2 * 1.5, 1+ p[0] ** 2 * 1.5]

        @rbatchintegrand
        def gd(p):
            return dict(x2=p[0] ** 2 * 1.5, one=1 + p[0] ** 2 * 1.5)
        dim = 2
        gg = gv.gvar(['1(2)', '2(1)'])
        eval = PDFIntegrator(gg, nitn=2, neval=100)
        for _g in [g, ga, gd]:
            r = eval(_g)
            def test_rx(rx):
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                self.assertAlmostEqual(rx.chi2, r.chi2)
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
            r1 = ravg(r)
            test_rx(r1)
            d3 = pickle.dumps(r)
            test_rx(r) # unchanged?
            r3 = pickle.loads(d3)
            test_rx(r3)  
            d4 = gv.dumps(r)
            test_rx(r) # unchanged?
            r4 = gv.loads(d4)
            test_rx(r4)  
            r5 = ravg(r, weighted=False)
            self.assertNotEqual(str(r5), str(r))

    def test_save_extend(self): 
        "save, saveall keywords in PDFIntegrator (checks pickle too)"
        @rbatchintegrand
        def g(p):
            return p[0] ** 2 * 1.5 / 8

        @rbatchintegrand
        def ga(p):
            return [p[0] ** 2 * 1.5, 1+ p[1] ** 2 * 1.5]

        @rbatchintegrand
        def gd(p):
            return dict(x2=p[0] ** 2 * 1.5, one=1 + p[1] ** 2 * 1.5)
        dim = 2
        gg = gv.gvar(['1(2)', '2(1)'])
        eval = PDFIntegrator(gg, nitn=2, neval=100)
        for _g in [g, ga, gd]:
            r = eval(_g, save='test-pdfsave.pkl')
            def test_rx(rx):
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                self.assertAlmostEqual(rx.chi2, r.chi2)
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
            with open('test-pdfsave.pkl', 'rb') as ifile:
                r1 = pickle.load(ifile)
                test_rx(r1)
        os.remove('test-pdfsave.pkl')
        eval = PDFIntegrator(gg, nitn=2, neval=100)
        for _g in [g, ga, gd]:
            r = eval(_g,  saveall='test-pdfsave.pkl')
            def test_rx(rx, evalx):
                self.assertEqual(str(rx), str(r))
                self.assertEqual(rx.summary(), r.summary())
                self.assertAlmostEqual(rx.chi2, r.chi2)
                self.assertEqual(evalx.settings(), eval.settings())
                self.assertAlmostEqual(list(evalx.sigf), list(eval.sigf))
                if _g is not g:
                    self.assertEqual(str(gv.evalcorr(rx.flat[:])), str(gv.evalcorr(r.flat[:])))
            with open('test-pdfsave.pkl', 'rb') as ifile:
                (r1, eval1) = pickle.load(ifile)
                test_rx(r1, eval1)
                new_r = eval1(_g)
                r1.extend(new_r)
                self.assertEqual(r.nitn + new_r.nitn, r1.nitn)
                self.assertEqual(r.sum_neval + new_r.sum_neval, r1.sum_neval)
                self.assertGreater(r.pdfnorm.sdev, r1.pdfnorm.sdev)
                self.assertGreater(new_r.pdfnorm.sdev, r1.pdfnorm.sdev)
                self.assertAlmostEqual(1/(1/r.pdfnorm.sdev**2 + 1/new_r.pdfnorm.sdev**2)**.5, r1.pdfnorm.sdev, delta=0.001)
        os.remove('test-pdfsave.pkl')

if __name__ == '__main__':
    unittest.main()