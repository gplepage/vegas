import pyximport; pyximport.install()    

from vegas import * 
import numpy as np
import lsqfit
import unittest

class TestAdaptiveMap(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        " AdaptiveMap(...) (also make_uniform and adapt)"
        m = AdaptiveMap(grid=[[1., 3.]], alpha=1.5)
        np.testing.assert_allclose(m.grid, [[1., 3.]])
        np.testing.assert_allclose(m.inc, [[2.]])
        self.assertEqual(m.alpha, 1.5)

        m = AdaptiveMap(grid=[[1., 3.]], ninc=4, alpha=1.5)
        np.testing.assert_allclose(m.grid, [[1., 1.5, 2., 2.5, 3.]])
        np.testing.assert_allclose(m.inc, [[0.5, 0.5, 0.5, 0.5]])
        self.assertEqual(m.alpha, 1.5)

        m = AdaptiveMap(grid=[[1., 1.25, 3.]], ninc=2)
        np.testing.assert_allclose(m.grid, [[1., 1.25, 3.]])
        np.testing.assert_allclose(m.inc, [[0.25, 1.75]])

        m = AdaptiveMap(grid=[[1., 3., 7.]], ninc=8)
        np.testing.assert_allclose(
            m.grid, 
            [[1, 1.5, 2., 2.5, 3., 4., 5., 6., 7.]]
            )
        np.testing.assert_allclose(
            m.inc, 
            [[0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1.]]
            )

        m = AdaptiveMap(grid=[[-2., 3., 7.]], ninc=8)
        np.testing.assert_allclose(
            m.grid, 
            [[-2., -0.75, 0.5, 1.75, 3., 4., 5., 6., 7.]]
            )
        np.testing.assert_allclose(
            m.inc, 
            [[1.25, 1.25, 1.25, 1.25, 1., 1., 1., 1.]]
            )
        m = AdaptiveMap(grid=[[-2., 7., 3.]], ninc=8)
        np.testing.assert_allclose(
            m.grid, 
            [[-2., -0.75, 0.5, 1.75, 3., 4., 5., 6., 7.]]
            )
        np.testing.assert_allclose(
            m.inc, 
            [[1.25, 1.25, 1.25, 1.25, 1., 1., 1., 1.]]
            )

    def test_map(self):
        " map(y) "
        m = AdaptiveMap(grid=[[1., 1.25, 3.]], ninc=2)
        y = np.array([[0., 0.25, 0.5, 0.75, 1.0]])
        x = np.empty(y.shape, float)
        jac = np.empty(y.shape[1], float)
        m.map(y, x, jac)
        np.testing.assert_allclose(x, [[1., 1.125, 1.25, 2.125, 3.]])
        np.testing.assert_allclose(jac, [0.5, 0.5, 3.5, 3.5, 3.5])
        np.testing.assert_allclose(m(y), x)
        np.testing.assert_allclose(m.jac(y), jac)

        np.testing.assert_allclose(m([0.6]), 1.6)
        np.testing.assert_allclose(m.jac([0.6]), 3.5)
        np.testing.assert_allclose(m([0.1]), 1.05)
        np.testing.assert_allclose(m.jac([0.1]), 0.5)


    def test_training(self):
        # no adaptation
        m = AdaptiveMap([[0.,2.]], ninc=2, alpha=1.5)
        y = np.array([[0.25, 0.75]])
        f = m.jac(y)
        m.accumulate_training_data(y, f)
        m.adapt()
        np.testing.assert_allclose(m.grid, [[0., 1., 2.]])
        np.testing.assert_allclose(m.inc, [[1., 1.]])
        
        # no adaptation
        m = AdaptiveMap([[0.,2.]], ninc=2, alpha=0.0)
        y = np.array([[0.25, 0.75]])
        f = m(y)[0] * m.jac(y)
        m.accumulate_training_data(y, f)
        m.adapt()
        np.testing.assert_allclose(m.grid, [[0., 1., 2.]])
        np.testing.assert_allclose(m.inc, [[1., 1.]])

        # adapt to linear function
        m = AdaptiveMap([[0.,2.]], ninc=2, alpha=3.)
        y1 = np.array([[0.25, 0.75]]) - .125
        y2 = np.array([[0.25, 0.75]]) + .125
        for i in range(30):
            m.accumulate_training_data(y1, m(y1)[0] * m.jac(y1))
            m.accumulate_training_data(y2, m(y2)[0] * m.jac(y2))
            m.adapt()
            f = m(y)[0] * m.jac(y)
        np.testing.assert_allclose(m.grid, [[0., 2**0.5, 2.]])
        np.testing.assert_allclose(f[0], f[1])

        # adapt to linear function with no smoothing
        m = AdaptiveMap([[0.,2.]], ninc=2, alpha=-3.)
        y = np.array([[0.25, 0.75]])
        for i in range(15):
            m.accumulate_training_data(y, f)
            m.adapt()
            f = m(y)[0] * m.jac(y)
        np.testing.assert_allclose(m.grid, [[0., 2**0.5, 2.]])
        np.testing.assert_allclose(f[0], f[1])

class TestIntegrator(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        I = Integrator([[0.,1.],[-1.,1.]], neval=234, nitn=123)
        self.assertEqual(I.neval, 234)
        self.assertEqual(I.nitn, 123)
        for k in Integrator.defaults:
            if k in ['neval', 'nitn']:
                self.assertNotEqual(getattr(I,k), Integrator.defaults[k])
            else:
                self.assertEqual(getattr(I,k), Integrator.defaults[k])
        np.testing.assert_allclose(I.map.grid[0], [0., 1.])
        np.testing.assert_allclose(I.map.grid[1], [-1., 1.])
        I._prep_integration()
        print '\n', I.status()

    def test_integrator(self):
        region = [[0., 1.], [-1.,1.]]
        I = Integrator(region, neval=50, nitn=10, redistribute=False, alpha=1.5)
        def fcn(x):
            return (x[1] ** 2 * 3. / 2. + x[0]) / 2.
        # fcn = test_integrand(region=region, x0=[0.], ampl=[1.], sig=[0.1])
        print '\n', I.status()
        print I(fcn), I.last_neval #, lsqfit.wavg.chi2/lsqfit.wavg.dof, lsqfit.wavg.Q


if __name__ == '__main__':
    unittest.main()