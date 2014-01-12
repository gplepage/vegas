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

from vegas import * 
import math
import pickle
import numpy as np
from numpy.testing import assert_allclose as np_assert_allclose
import unittest
import warnings

if have_gvar:
    import gvar as gv

class TestAdaptiveMap(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        " AdaptiveMap(...) "
        m = AdaptiveMap(grid=[[0, 1], [2, 4]])
        np_assert_allclose(m.grid, [[0, 1], [2, 4]])
        np_assert_allclose(m.inc, [[1], [2]])
        m = AdaptiveMap(grid=[[0, 1], [-2, 4]], ninc=2)
        np_assert_allclose(m.grid, [[0, 0.5, 1.], [-2., 1., 4.]])
        np_assert_allclose(m.inc, [[0.5, 0.5], [3., 3.]])
        self.assertEqual(m.dim, 2)
        self.assertEqual(m.ninc, 2)
        m = AdaptiveMap([[0, 0.4, 1], [-2, 0., 4]], ninc=4)
        np_assert_allclose(
            m.grid, 
            [[0, 0.2, 0.4, 0.7, 1.], [-2., -1., 0., 2., 4.]]
            )
        np_assert_allclose(m.inc, [[0.2, 0.2, 0.3, 0.3], [1, 1, 2, 2]])
        self.assertEqual(m.dim, 2)
        self.assertEqual(m.ninc, 4)

    def test_pickle(self):
        " pickle AdaptiveMap "
        m1 = AdaptiveMap(grid=[[0, 1, 3], [-2, 0, 6]])
        with open('test_map.p', 'wb') as ofile:
            pickle.dump(m1, ofile)
        with open('test_map.p', 'rb') as ifile:
            m2 = pickle.load(ifile)
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
        self.assertEqual(m.settings(5), output)
        output = "    grid[ 0] = [ 0.5  2. ]\n"
        output += "    grid[ 1] = [-1.  3.]\n"
        self.assertEqual(m.settings(2), output)

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
        pass 

    def tearDown(self):
        pass 

    def test_all(self):
        " RWavg "
        a = RAvg()
        a.add(gvar.gvar(1, 1))
        a.add(gvar.gvar(2, 2))
        a.add(gvar.gvar(3, 3))
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
        if not have_gvar:
            return
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
        if not have_gvar:
            return
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
        if not have_gvar:
            return
        mean = np.random.uniform(-10., 10., (2,))
        cov = np.array([[1., 0.5], [0.5, 2.]])
        invcov = np.linalg.inv(cov)
        N = 30
        xbig = gv.gvar(mean, cov)
        rbig = gv.raniter(xbig, N)
        xsmall = gv.gvar(mean, cov / 10.)
        rsmall = gv.raniter(xsmall, N)
        ravg = RAvgArray(2)
        for rb, rs in zip(rbig, rsmall):
            ravg.add(gv.gvar(rb, cov))
            ravg.add(gv.gvar(rs, cov / 10.))
        np_assert_allclose(gv.evalcov(ravg), cov / (10. + 1.) / N)
        for i in range(2):
            self.assertLess(abs(mean[i] - ravg[i].mean), 5 * ravg[i].sdev)
        self.assertEqual(ravg.dof, 4 * N - 2)
        self.assertGreater(ravg.Q, 1e-3)

    def test_ravgarray_unwgtd(self):
        " unweighted RAvgArray "
        if not have_gvar:
            return
        gv.ranseed((1,2))
        mean = np.random.uniform(-10., 10., (2,))
        cov = np.array([[1., 0.5], [0.5, 2.]]) / 10.
        N = 30
        x = gv.gvar(mean, cov)
        r = gv.raniter(x, N)
        ravg = RAvgArray(2, weighted=False)
        for ri in r: 
            ravg.add(gv.gvar(ri, cov))
        np_assert_allclose(gv.evalcov(ravg), cov / N)
        for i in range(2):
            self.assertLess(abs(mean[i] - ravg[i].mean), 5 * ravg[i].sdev)
        self.assertEqual(ravg.dof, 2 * N - 2)
        self.assertGreater(ravg.Q, 1e-3)

    def test_array(self):
        " RAvgArray "
        a = RAvgArray((2,))
        a.add([gvar.gvar(1, 1), gvar.gvar(10,10)])
        a.add([gvar.gvar(2, 2), gvar.gvar(20,20)])
        a.add([gvar.gvar(3, 3), gvar.gvar(30,30)])
        self.assertEqual(a.shape, (2,))
        np_assert_allclose(a[0].mean, 1.346938775510204)
        np_assert_allclose(a[0].sdev, 0.8571428571428571)
        self.assertEqual(a.dof, 4)
        np_assert_allclose(a.chi2, 2*0.5306122448979592)
        np_assert_allclose(a.Q, 0.900374555485)
        self.assertEqual(str(a[0]), '1.35(86)')
        self.assertEqual(str(a[1]), '13.5(8.6)')
        s = [
            "itn   integral        wgt average     chi2/dof        Q",
            "-------------------------------------------------------",
            "  1   1.0(1.0)        1.0(1.0)            0.00     1.00",
            "  2   2.0(2.0)        1.20(89)            0.20     0.82",
            "  3   3.0(3.0)        1.35(86)            0.27     0.90",
            ""
            ]
        self.assertEqual(a.summary(), '\n'.join(s))

class TestIntegrator(unittest.TestCase):
    def setUp(self):
        pass 

    def tearDown(self):
        pass 

    def test_have_gvar(self):
        " have gvar module? "
        if not have_gvar:
            warnings.warn(
                "no gvar module -- for better results try: pip install gvar"
                )

    def test_init(self):
        " Integrator "
        I = Integrator([[0.,1.],[-1.,1.]], neval=234, nitn=123)
        self.assertEqual(I.neval, 234)
        self.assertEqual(I.nitn, 123)
        for k in Integrator.defaults:
            if k in ['neval', 'nitn']:
                self.assertNotEqual(getattr(I,k), Integrator.defaults[k])
            elif k not in ['map']:
                self.assertEqual(getattr(I,k), Integrator.defaults[k])
        np.testing.assert_allclose([I.map.grid[0,0], I.map.grid[0, -1]], [0., 1.])
        np.testing.assert_allclose([I.map.grid[1,0], I.map.grid[1, -1]], [-1., 1.])
        lines = [
            'Integrator Settings:',
            '    234 (max) integrand evaluations in each of 123 iterations',
            '    number of:  strata/axis = 7  increments/axis = 21',
            '                h-cubes = 49  evaluations/h-cube = 2 (min)',
            '                h-cubes/batch = 1000',
            '    minimize_mem = False',           
            '    adapt_to_errors = False',
            '    damping parameters: alpha = 0.5  beta= 0.75',
            '    limits: h-cubes < 1e+09  evaluations/h-cube < 1e+07',
            '    accuracy: relative = 0  absolute accuracy = 0',
            '',
            '    axis 0 covers (0.0, 1.0)',
            '    axis 1 covers (-1.0, 1.0)',
            '',
            ]
        for i, l in enumerate(I.settings().split('\n')):
            self.assertEqual(l, lines[i])
        I = Integrator([[0.,1.],[-1.,1.]], max_nhcube=1, minimize_mem=True)
        lines = [
            'Integrator Settings:',
            '    1000 (max) integrand evaluations in each of 10 iterations',
            '    number of:  strata/axis = 15  increments/axis = 90',
            '                h-cubes = 225  evaluations/h-cube = 2 (min)',
            '                h-cubes/batch = 1000',
            '    minimize_mem = True',
            '    adapt_to_errors = False',
            '    damping parameters: alpha = 0.5  beta= 0.75',
            '    limits: h-cubes < 1  evaluations/h-cube < 1e+07',
            '    accuracy: relative = 0  absolute accuracy = 0',
            '',
            '    axis 0 covers (0.0, 1.0)',
            '    axis 1 covers (-1.0, 1.0)',
            '',
            ]
        for i, l in enumerate(I.settings().split('\n')):
            self.assertEqual(l, lines[i])

    def test_pickle(self):
        I1 = Integrator([[0.,1.],[-1.,1.]], neval=234, nitn=123)
        with open('test_integ.p', 'wb') as ofile:
            pickle.dump(I1, ofile)
        with open('test_integ.p', 'rb') as ifile:
            I2 = pickle.load(ifile)
        assert isinstance(I2, Integrator)
        for k in Integrator.defaults:
            if k == 'map':
                np_assert_allclose(I1.map.grid, I2.map.grid)
                np_assert_allclose(I1.map.inc, I2.map.inc)
            else:
                self.assertEqual(getattr(I1, k), getattr(I2, k))

    def test_set(self):
        " set "
        new_defaults = dict(
            map=AdaptiveMap([[1,2],[0,1]]),
            neval=100,       # number of evaluations per iteration
            maxinc_axis=100,  # number of adaptive-map increments per axis
            nhcube_batch=10,    # number of h-cubes per batch
            max_nhcube=5e2,    # max number of h-cubes
            max_neval_hcube=1e1, # max number of evaluations per h-cube
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
        class f_batch(BatchIntegrand):
            def __call__(self, x):
                f = np.empty(x.shape[0], float)
                for i in range(f.shape[0]):
                    f[i] = (
                        math.sin(x[i, 0]) ** 2 + math.cos(x[i, 1]) ** 2
                        ) / math.pi ** 2
                return f
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        r = I(f_batch(), neval=10000)
        self.assertLess(abs(r.mean - 1.), 5 * r.sdev)
        self.assertGreater(r.Q, 1e-3)
        self.assertLess(r.sdev, 1e-3)

    def test_min_sigf(self):
        " test minimize_mem=True mode "
        class f_batch(BatchIntegrand):
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

    def test_scalar_exception(self):
        " integrate scalar fcn "
        def f(x):
            return (math.sin(x[0]) ** 2 + math.cos(x[1]) ** 2) / math.pi ** 2 / 0.0
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        with self.assertRaises(ZeroDivisionError):
            I(f, neval=100)

    def test_batch_exception(self):
        " integrate batch fcn "
        class f_batch(BatchIntegrand):
            def __call__(self, x):
                f = 1/0.
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / f
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]])
        with self.assertRaises(ZeroDivisionError):
            I(f_batch(), neval=100)

    def test_batch_b0(self):
        " integrate batch fcn beta=0 "
        class f_batch(BatchIntegrand):
            def __call__(self, x):
                return (np.sin(x[:, 0]) ** 2 + np.cos(x[:, 1]) ** 2) / math.pi ** 2
                # for i in range(nx):
                #     f[i] = (
                #         math.sin(x[i, 0]) ** 2 + math.cos(x[i, 1]) ** 2
                #         ) / math.pi ** 2
        I = Integrator([[0, math.pi], [-math.pi/2., math.pi/2.]], beta=0.0)
        r = I(f_batch(), neval=10000)
        self.assertTrue(abs(r.mean - 1.) < 5 * r.sdev)
        self.assertTrue(r.Q > 1e-3)
        self.assertTrue(r.sdev < 1e-3)

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
            return [f, f * x[0]]
        class f_multi_v(BatchIntegrand):
            def __call__(self, x):
                x = np.asarray(x)
                f = np.empty((x.shape[0], 2), float)
                dx2 = 0.
                for d in range(4):
                    dx2 += (x[:, d] - 0.5) ** 2
                f[:, 0] = np.exp(-100. * dx2)
                f[:, 1] = x[:, 0] * f[:, 0]
                return f
        I = Integrator(4 * [[0, 1]])
        warmup = I(f_s, neval=1000, nitn=10)
        if have_gvar:
            for r in [I(f_multi_v(), nitn=10), I(f_multi_s, nitn=10)]:
                ratio = r[1] / r[0]
                self.assertLess(abs(ratio.mean - 0.5), 5 * ratio.sdev)
                self.assertLess(ratio.sdev, 1e-2)
                
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


class Testgvar(unittest.TestCase):
    """ tests gvar and GVar since might be the vegas substitutes if no gvar """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        x = gvar.gvar(1., 3.)
        self.assertEqual(x.mean, 1.)
        self.assertEqual(x.sdev, 3.)

    def test_math(self):
        " test math involving GVars "
        x = gvar.gvar(2., 5.)
        cases = [
            (x + 10., 12., 5.),
            (10 + x, 12. , 5.),
            (+x, 2., 5.),
            (x - 6., -4., 5.),
            (6 - x, 4., 5.),
            (-x, -2., 5.),
            (3 * x, 6., 15.),
            (x * 3., 6., 15.),
            (x / 4., 0.5, 1.25),
            (10. / x, 5., 12.5),
            (x ** 2, 4., 20.),
            (2 ** x, 4., math.log(2) * 20.),
            (np.log(x), math.log(2.), 2.5),
            (np.exp(x), math.exp(2.), math.exp(2.) * 5.),
            (np.exp(np.log(x)), 2., 5.),
            (np.sqrt(x), math.sqrt(2.), math.sqrt(2.) * 1.25),
            (np.sqrt(x ** 2), 2., 5.),
            ]
        for y, ymean, ysdev in cases:
            np_assert_allclose(y.mean, ymean)
            np_assert_allclose(y.sdev, ysdev)

    def test_gammaQ(self):
        " gammaQ(a, x) "
        cases = [
            (2.371, 5.243, 0.05371580082389009, 0.9266599665892222),
            (20.12, 20.3, 0.4544782602230986, 0.4864172139106905),
            (100.1, 105.2, 0.29649013488390663, 0.6818457585776236),
            (1004., 1006., 0.4706659307021259, 0.5209695379094582),
            ]
        for a, x, gax, gxa in cases:
            np_assert_allclose(gax, gvar.gammaQ(a, x), rtol=0.01)
            np_assert_allclose(gxa, gvar.gammaQ(x, a), rtol=0.01)

if __name__ == '__main__':
    unittest.main()