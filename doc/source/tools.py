import vegas
import numpy as np 
from math import erf, gamma
import scipy.linalg as linalg
import gvar as gv 

class Gaussians(vegas.BatchIntegrand):
    def __init__(self, g):
        self.g = []
        for r, s in g:
            norm = len(g)
            for ri in r:
                norm *= np.sqrt(np.pi) / 2. / s * (erf(s - ri * s) - erf(-ri * s))
            self.g.append((np.array(r)[None,:], s, 1. / norm))
    
    def samples(self, n):
        n = n // len(self.g)
        dim = len(self.g[0][0][0])
        x = np.empty((n * len(self.g), dim), float)
        for i, (r, s, norm) in enumerate(self.g):
            r = r[0]
            for d in range(dim):
                x[i * n:(i + 1)*n, d] = np.random.normal(loc=r[d], scale=1. / s / 2**.5, size=n)
        return x

    def __call__(self, x):
        # return np.ones(x.shape[0], dtype=float)
        ans = 0.0
        for r, s, norm in self.g:
            ans += norm * np.exp(-np.sum((x-r) ** 2, axis=1) * s ** 2)
        return ans 

class HGaussian(vegas.BatchIntegrand):
    def __init__(self, r, s=1):
        dim = len(r)
        self.invH = linalg.invhilbert(dim)
        # self.invH = self.invH @ self.invH
        self.s = s
        self.r = np.array(r)
        self.r.shape = (1,-1)
        self.norm = ( s / np.sqrt(np.pi)) ** dim * np.sqrt(linalg.det(self.invH))

    def samples(self, n):
        dim = self.r.shape[1]
        H = linalg.hilbert(dim) / (2 * self.s**2)
        x = gv.gvar(self.r[0], H)
        print(gv.evalcorr(x))
        return np.array([rx for rx in gv.raniter(x, n)])

    def __call__(self, x):
        dx = x - self.r 
        invH_dx = self.invH.dot(dx.T).T 
        dx_invH_dx = np.sum(dx * invH_dx, axis=1) *  self.s ** 2
        return np.exp(-dx_invH_dx) * self.norm

class Exponentials(vegas.BatchIntegrand):
    def __init__(self, g):
        self.g = []
        for r, s in g:
            norm = len(g)
            dim = len(r)
            norm = len(g) / (s ** dim) * gamma(dim + 1) * np.pi ** (dim / 2) / gamma(dim / 2 + 1)
            self.g.append((np.array(r)[None,:], s, 1. / norm))
    
    def __call__(self, x):
        # return np.ones(x.shape[0], dtype=float)
        ans = 0.0
        for r, s, norm in self.g:
            ans += norm * np.exp(-np.sum((x-r) ** 2, axis=1)**0.5 * s)
        return ans 

class Balls(vegas.BatchIntegrand):
    def __init__(self, g):
        self.g = []
        for r, s in g:
            norm = len(g)
            dim = len(r)
            radius = 1. / s
            norm = len(g) * (np.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1))
            self.g.append((np.array(r)[None,:], radius, 1. / norm))
    
    def __call__(self, x):
        # return np.ones(x.shape[0], dtype=float)
        ans = np.zeros(x.shape[0], dtype=float)
        for r, radius, norm in self.g:
            dx2 = np.sum((x - r) ** 2, axis=1)
            idx = dx2 <= radius ** 2
            ans[idx] += norm 
        return ans 

