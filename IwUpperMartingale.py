from abc import ABCMeta, abstractmethod
from math import log

import numpy as np
import torch


class BaseMartingale(metaclass=ABCMeta):
    """Return current bet."""

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
    def addobs(self, x, q, l, cv):
        pass


class GridMartingale(BaseMartingale, metaclass=ABCMeta):

    @abstractmethod
    def xi(self, x, q, l, betas, cv):
        pass

    @abstractmethod
    def ximin(self, betas):
        pass

    @property
    def curlam(self):
        return self._curlam[max(0, self._curbetaindex - 1)]

    @property
    def curbeta(self):
        return (
            self._betas[self._curbetaindex],
            self._betas[max(0, self._curbetaindex - 1)],
            self._stats[max(0, self._curbetaindex - 1), 0],
        )

    def _maxlam(self, betas):
        ximin_betas = self.ximin(betas)

        return (1 / 2) * (1 / np.abs(ximin_betas))

    def update_q_min(self, q_min):
        self._q_min = q_min

    # assumptions: beta_min = 0, beta_max = 1
    def __init__(self, *, n_betas, alpha, n_beta_min=0., stop_comp=True):

        super().__init__()

        assert n_betas == int(n_betas) and n_betas > 1
        assert 0 < alpha < 1

        self._n_betas = n_betas
        self._alpha = alpha

        self._betas = np.linspace(n_beta_min, 1., n_betas)
        # Column 0: sum logwealth = sum log1p(curlam * xi)
        # Column 1: sum xi
        # Column 2: sum xi**2
        self._stats = np.zeros((n_betas, 3))

        self._curbetaindex = n_betas - 1  # smallest beta that is proven safe
        self._curlam = np.zeros(n_betas)
        self._thres = -log(alpha)
        self._stop_comp = True

    def addobs(self, x, q, l, cv_predictor):
        if self._curbetaindex > 0 or not self._stop_comp:

            if cv_predictor is not None:
                cvs = torch.tensor(
                    [cv_predictor.predict(x[0], beta) for beta in self._betas])
            else:
                cvs = torch.zeros(self._betas.shape)
            xibetas = np.array(self.xi(x, q, l, self._betas, cvs)).reshape(-1)
            self._stats[:, 0] += np.log1p(self._curlam * xibetas)
            self._stats[:, 1] += xibetas
            self._stats[:, 2] += xibetas**2

            if self._stats[self._curbetaindex - 1, 0] >= self._thres:
                self._curbetaindex = np.argmax(self._stats[:,
                                                           0] >= self._thres)
                self._betas = self._betas[:self._curbetaindex + 1]
                self._stats = self._stats[:self._curbetaindex + 1, :]

        if self._curbetaindex > 0 or not self._stop_comp:
            ftlnum = self._stats[:, 1]
            ftldenom = self._stats[:, 1] + self._stats[:, 2]
            # ftldenom = self._stats[:, 2]

            relevant_betas = self._betas
            max_lams = self._maxlam(relevant_betas)

            lams = np.divide(ftlnum,
                             ftldenom,
                             out=np.zeros_like(ftlnum),
                             where=ftldenom != 0)

            self._curlam = np.clip(lams, 0, max_lams)

        else:
            self._curlam = np.zeros_like(self._betas)


class FullIwUpperMartingale(GridMartingale):

    def xi(self, x, q, l, betas, cv):
        return (l / q) * (self._theta - self._rho(x, betas) - cv) + cv

    def ximin(self, betas):
        return (self._theta - 1) / self._q_min

    def addobs(self, x, q, l, cv):
        if l == 1:
            super().addobs(x, q, l, cv)

    # assumption: rho_max == 1
    def __init__(self, rho, q_min, theta, *args, **kwargs):
        self._rho = rho
        self._q_min = q_min
        self._theta = theta

        super().__init__(*args, **kwargs)


class DummyMartingale(FullIwUpperMartingale):
    """Class that always outputs a singular beta."""

    def __init__(self, init_beta, *args, **kwargs):

        super().__init__(*args, n_betas=2, stop_comp=False, **kwargs)
        self._betas = np.array([init_beta, 1])
        self._curbetaindex = 1
        self._thres = np.inf


class PartialIwUpperMartingale(GridMartingale):

    def ximin(self, betas):
        with torch.no_grad():
            torch_arr = torch.as_tensor(betas) if not isinstance(
                betas, np.ndarray) else torch.tensor(betas)
            cv_min = self._cv_predictor.min_val(
                torch_arr) if self._cv_predictor is not None else 0
        return self._theta + (
            (1 / self._q_min) - 1) * cv_min - (1 / self._q_min)

    def xi(self, x, q, l, betas, cv):
        """Black box insertion of cv_predictor.

        Note that OnlineMinimax calls self._xi as the loss function when
        updating the Player. So this should update corectly for (1) if
        we use the LabelingPolicy in the Player itself (2) if we use
        another cv_predictor that we update through backprop.
        """
        torch_arr = torch.as_tensor(betas) if not isinstance(
            betas, np.ndarray) else torch.tensor(betas)
        risk_r = self._rho(x, torch_arr)
        xi = self._theta + (l / q) * (cv - risk_r) - cv
        assert q >= self._q_min, (q, self._q_min, f'cv type: {type(cv), cv}')
        if isinstance(xi, torch.Tensor):
            test_xi = xi.detach().numpy()
        else:
            test_xi = xi
        assert np.all(test_xi >= self.ximin(betas)), (
            f'xi: {xi}', f'test_xi: {test_xi}',
            f'self.ximin(betas): {self.ximin(betas)}', f'risk_r: {risk_r}',
            f'cv: {cv}', f'self._q_min: {self._q_min}', f'q: {q}', f'l: {l}',
            f'self._cv_predictor.min_val(betas): {self._cv_predictor.min_val(betas)}',
            f'betas: {betas}')
        return xi

    def addobs(self, x, q, l, cv):
        super().addobs(x, q, l, cv)

    # assumption: rho_max == 1
    def __init__(self, rho, q_min, theta, *args, cv_predictor=None, **kwargs):
        self._rho = rho
        self._q_min = q_min
        self._theta = theta
        self._cv_predictor = cv_predictor

        super().__init__(*args, **kwargs)


class ShiftedIwUpperMartingale(GridMartingale):

    def xi(self, x, q, l, betas, cv):
        self._xi_state = self._theta - 1 + (l / q) * (1 - self._rho(x, betas) -
                                                      cv) + cv
        return self._xi_state

    def ximin(self, betas):
        return self._theta - 1

    def addobs(self, x, q, l, cv):
        super().addobs(x, q, l, cv)

    # assumption: rho_max == 1
    def __init__(self, rho, theta, *args, **kwargs):
        self._rho = rho
        self._theta = theta
        self._q_min = 0

        super().__init__(*args, **kwargs)


class FullyObservedUpperMartingale(GridMartingale):

    def xi(self, x, q, l, betas, cv):
        torch_arr = torch.as_tensor(betas) if not isinstance(
            betas, np.ndarray) else torch.tensor(betas)
        return self._theta - self._rho(x, torch_arr)

    def ximin(self, betas):
        return self._theta - 1

    def addobs(self, x, q, l, cv):
        super().addobs(x, q, l, cv)

    # assumption: rho_max == 1
    def __init__(self, rho, theta, *args, **kwargs):
        self._rho = rho
        self._theta = theta

        super().__init__(*args, **kwargs)


if __name__ == '__main__':

    def test_once(gen, n_max):

        # p is the true probability of a positive
        # y is the realized outcome
        # our decision is 1_{beta <= p}

        def rho(x, betas):
            (p, y) = x
            return np.array(betas <= p) * (1 - y)

        q_min = 1 / 10
        fullcs = FullIwUpperMartingale(rho=rho,
                                       q_min=q_min,
                                       theta=0.05,
                                       n_betas=100,
                                       alpha=0.05)
        partcs = PartialIwUpperMartingale(rho=rho,
                                          q_min=q_min,
                                          theta=0.05,
                                          n_betas=100,
                                          alpha=0.05)
        shiftcs = ShiftedIwUpperMartingale(rho=rho,
                                           theta=0.05,
                                           n_betas=100,
                                           alpha=0.05)

        fullrv, partrv, shiftrv = [], [], []

        for n in range(n_max):
            p = round(gen.uniform(), 2)
            y = 1 if gen.uniform() <= p else 0
            q = gen.uniform(low=q_min, high=1)
            l = gen.binomial(1, q)
            fullcs.addobs(x=(p, y), q=q, l=l, cv=0)
            fullrv.append(fullcs.curbeta[0])
            partcs.addobs(x=(p, y), q=q, l=l, cv=0)
            partrv.append(partcs.curbeta[0])
            shiftcs.addobs(x=(p, y), q=q, l=l, cv=0)
            shiftrv.append(shiftcs.curbeta[0])

        return fullrv, partrv, shiftrv

    def test():
        import numpy as np
        gen = np.random.default_rng(2)
        fullcurve, partcurve, shiftcurve = test_once(gen, 50_000)
        truefp = lambda z: (100 - int(z)) * (101 - int(z)) / 20_000

        fulltruefpbetas = np.array([truefp(100 * beta) for beta in fullcurve])
        assert np.all(fulltruefpbetas <= 0.05)
        assert fullcurve[-1] < 0.725, (fullcurve[-1], fulltruefpbetas[-1])
        assert fulltruefpbetas[-1] > 0.045
        print('first test pass')

        parttruefpbetas = np.array([truefp(100 * beta) for beta in partcurve])
        assert np.all(parttruefpbetas <= 0.05)
        assert partcurve[-1] < 0.725, (partcurve[-1], parttruefpbetas[-1])
        assert parttruefpbetas[-1] > 0.045
        print('second test pass')

        shifttruefpbetas = np.array(
            [truefp(100 * beta) for beta in shiftcurve])
        assert np.all(shifttruefpbetas <= 0.05)
        assert shiftcurve[-1] < 0.775, (shiftcurve[-1], shifttruefpbetas[-1])
        assert shifttruefpbetas[-1] >= 0.03, shifttruefpbetas[-1]
        print('third test pass')

    test()
