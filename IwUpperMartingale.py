from abc import ABCMeta, abstractmethod

class BaseMartingale(metaclass=ABCMeta):
    """ Return current bet """
    @property
    @abstractmethod
    def curlam(self):
        pass

    """ Return (smallest feasible beta, current beta being bet against, and associated wealth of current beta) """
    @property
    @abstractmethod
    def curbeta(self):
        pass

    @abstractmethod
    def addobs(self, x, q, l):
        pass

    

        
    
class GridMartingale(BaseMartingale, metaclass=ABCMeta):
    @abstractmethod
    def xi(self, x, q, l, betas):
        pass

    @property
    @abstractmethod
    def ximin(self):
        pass

    @property
    def curlam(self):
        return self._curlam

    @property
    def curbeta(self):
        return ( self._betas[self._curbetaindex],
                 self._betas[max(0, self._curbetaindex-1)],
                 self._stats[max(0, self._curbetaindex-1), 0],
               )

    # assumptions: beta_min = 0, beta_max = 1
    def __init__(self, *, n_betas, alpha, stop_comp=True):
        from collections import defaultdict
        from math import log
        import numpy as np

        super().__init__()

        assert n_betas == int(n_betas) and n_betas > 1
        assert 0 < alpha < 1

        self._n_betas = n_betas
        self._alpha = alpha

        self._betas = np.linspace(0, 1, n_betas)

        # Column 0: sum logwealth = sum log1p(curlam * xi)
        # Column 1: sum xi
        # Column 2: sum xi**2
        self._stats = np.zeros((n_betas, 3))

        self._curbetaindex = n_betas - 1     # smallest beta that is proven safe
        self._curlam = 0
        self._maxlam = (1/2) * (1 / abs(self.ximin))
        self._thres = -log(alpha)
        self._stop_comp = True


    def addobs(self, x, q, l):
        if self._curbetaindex > 0 or not self._stop_comp:
            import numpy as np

            xibetas = np.array(self.xi(x, q, l, self._betas))
            self._stats[:,0] += np.log1p(self._curlam * xibetas)
            self._stats[:,1] += xibetas
            self._stats[:,2] += xibetas**2

            if self._stats[self._curbetaindex - 1, 0] >= self._thres:
                self._curbetaindex = np.argmax(self._stats[:,0] >= self._thres)
                self._betas = self._betas[:self._curbetaindex+1]
                self._stats = self._stats[:self._curbetaindex+1,:]

        if self._curbetaindex > 0 or not self._stop_comp:
            ftlnum = self._stats[self._curbetaindex - 1, 1]
            ftldenom = self._stats[self._curbetaindex - 1, 1] + self._stats[self._curbetaindex - 1, 2]
            self._curlam = 0 if ftlnum <= 0 else min(self._maxlam, ftlnum / ftldenom)
        else:
            self._curlam = 0


class FullIwUpperMartingale(GridMartingale):
    def xi(self, x, q, l, betas):
        return (l / q) * (self._theta - self._rho(x, betas))

    @property
    def ximin(self):
        return (self._theta - 1) / self._q_min

    def addobs(self, x, q, l):
        if l == 1:
            super().addobs(x, q, l)

    # assumption: rho_max == 1
    def __init__(self, rho, q_min, theta, *args, **kwargs):
        self._rho = rho
        self._q_min = q_min
        self._theta = theta

        super().__init__(*args, **kwargs)
        
class DummyMartingale(FullIwUpperMartingale):
    """Class that always outputs a singular beta"""
    def __init__(self, init_beta, *args, **kwargs):
        import numpy as np

        super().__init__(*args, n_betas=2, stop_comp=False, **kwargs)
        self._betas = np.array([init_beta, 1])
        self._curbetaindex = 1
        self._thres = np.inf

        
class PartialIwUpperMartingale(GridMartingale):
    def xi(self, x, q, l, betas):
        return self._theta - (l / q) * self._rho(x, betas)

    @property
    def ximin(self):
        return self._theta - (1 / self._q_min)

    def addobs(self, x, q, l):
        super().addobs(x, q, l)

    # assumption: rho_max == 1
    def __init__(self, rho, q_min, theta, *args, **kwargs):
        self._rho = rho
        self._q_min = q_min
        self._theta = theta

        super().__init__(*args, **kwargs)


class ShiftedIwUpperMartingale(GridMartingale):
    def xi(self, x, q, l, betas):
        return self._theta - 1 + (l / q) * (1 - self._rho(x, betas))

    @property
    def ximin(self):
        return self._theta - 1

    def addobs(self, x, q, l):
        super().addobs(x, q, l)

    # assumption: rho_max == 1
    def __init__(self, rho, theta, *args, **kwargs):
        self._rho = rho
        self._theta = theta

        super().__init__(*args, **kwargs)


class FullyObservedUpperMartingale(GridMartingale):
    def xi(self, x, q, l, betas):
        return self._theta - self._rho(x, betas)

    @property
    def ximin(self):
        return self._theta - 1

    def addobs(self, x, q, l):
        super().addobs(x, q, l)

    # assumption: rho_max == 1
    def __init__(self, rho, theta, *args, **kwargs):
        self._rho = rho
        self._theta = theta

        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    def test_once(gen, n_max):
        import numpy as np
        # p is the true probability of a positive
        # y is the realized outcome
        # our decision is 1_{beta <= p}

        def rho(x, betas):
            (p, y) = x
            return np.array(betas <= p) * (1 - y)

        q_min = 1/10
        fullcs = FullIwUpperMartingale(rho=rho, q_min=q_min, theta=0.05, n_betas=100, alpha=0.05)
        partcs = PartialIwUpperMartingale(rho=rho, q_min=q_min, theta=0.05, n_betas=100, alpha=0.05)
        shiftcs = ShiftedIwUpperMartingale(rho=rho, theta=0.05, n_betas=100, alpha=0.05)

        fullrv, partrv, shiftrv = [], [], []

        for n in range(n_max):
            p = round(gen.uniform(), 2)
            y = 1 if gen.uniform() <= p else 0
            q = gen.uniform(low=q_min, high=1)
            l = gen.binomial(1, q)
            fullcs.addobs(x=(p, y), q=q, l=l)
            fullrv.append(fullcs.curbeta[0])
            partcs.addobs(x=(p, y), q=q, l=l)
            partrv.append(partcs.curbeta[0])
            shiftcs.addobs(x=(p, y), q=q, l=l)
            shiftrv.append(shiftcs.curbeta[0])

        return fullrv, partrv, shiftrv

    def test():
        import numpy as np
        gen = np.random.default_rng(2)
        fullcurve, partcurve, shiftcurve = test_once(gen, 50_000)
        truefp = lambda z: (100 - int(z)) * (101 - int(z)) / 20_000

        fulltruefpbetas = np.array([ truefp(100*beta) for beta in fullcurve ])
        assert np.all(fulltruefpbetas <= 0.05)
        assert fullcurve[-1] < 0.725, (fullcurve[-1], fulltruefpbetas[-1])
        assert fulltruefpbetas[-1] > 0.045
        print('first test pass')

        parttruefpbetas = np.array([ truefp(100*beta) for beta in partcurve ])
        assert np.all(parttruefpbetas <= 0.05)
        assert partcurve[-1] < 0.725, (partcurve[-1], parttruefpbetas[-1])
        assert parttruefpbetas[-1] > 0.045
        print('second test pass')

        shifttruefpbetas = np.array([ truefp(100*beta) for beta in shiftcurve ])
        assert np.all(shifttruefpbetas <= 0.05)
        assert shiftcurve[-1] < 0.775, (shiftcurve[-1], shifttruefpbetas[-1])
        assert shifttruefpbetas[-1] >= 0.03, shifttruefpbetas[-1]
        print('third test pass')

    test()
