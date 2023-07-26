class IwUpperMartingale(object):
    # assumptions: rho_max = 1, beta_min = 0, beta_max = 1
    def __init__(self, *, rho, theta, q_min, n_betas, alpha):
        from collections import defaultdict
        from math import log
        import numpy as np
        
        super().__init__()
        
        assert 0 <= theta <= 1
        assert q_min > 0
        assert n_betas == int(n_betas) and n_betas > 1
        assert 0 < alpha < 1
        
        self._rho = rho
        self._theta = theta
        self._q_min = q_min
        self._n_betas = n_betas
        self._alpha = alpha
        
        self._betas = np.linspace(0, 1, n_betas)
        
        # Column 0: logwealth
        # Column 1: sum (theta - rho_i) / q_i
        # Column 2: sum ((theta - rho_i) / q_i)^2
        self._stats = np.zeros((n_betas, 3))
        
        self._curbetaindex = n_betas - 1     # smallest beta that is proven safe
        self._curlam = 0
        self._maxlam = q_min/2
        self._thres = -log(alpha)

    # only call this if you actually sample, otherwise no update
    def addobs(self, qi, xi):
        if self._curbetaindex > 0:
            import numpy as np

            xibetas = (self._theta - self._rho(xi, self._betas)) / qi
            self._stats[:,0] += np.log1p(self._curlam * xibetas)
            self._stats[:,1] += xibetas
            self._stats[:,2] += xibetas**2

            if self._stats[self._curbetaindex - 1, 0] >= self._thres:
                self._curbetaindex = np.argmax(self._stats[:,0] >= self._thres)
                self._betas = self._betas[:self._curbetaindex+1]
                self._stats = self._stats[:self._curbetaindex+1,:]

        if self._curbetaindex > 0:
            ftlnum = self._stats[self._curbetaindex - 1, 1]
            ftldenom = self._stats[self._curbetaindex - 1, 1] + self._stats[self._curbetaindex - 1, 2]
            self._curlam = 0 if ftlnum <= 0 else min(self._maxlam, ftlnum / ftldenom)
        else:
            self._curlam = 0
    
    def wealth_fn(self, qi, xi, beta):
        return torch.log1p(self._curlam * (self._theta - qi * self._rho(xi, beta)))

    def getbeta(self):
        return self._betas[self._curbetaindex], self._stats[self._curbetaindex, 0]
    
    def getlam(self):
        return self._curlam

if __name__ == '__main__':
    def test_once(gen, n_max):
        import numpy as np
        # p is the true probability of a positive 
        # y is the realized outcome
        # our decision is 1_{beta <= p}
        
        def rho(x, betas):
            (p, y) = x
            return np.array(betas <= p) * (1 - y)
        
        cs = IwUpperMartingale(rho=rho, theta=0.05, q_min=1/10, n_betas=100, alpha=0.05)
            
        rv = []
            
        for n in range(n_max):
            p = round(gen.uniform(), 2)
            y = 1 if gen.uniform() <= p else 0
            qi = gen.uniform(low=0.1, high=1)
            li = gen.binomial(1, qi)
            if li:
                cs.addobs(qi=qi, xi=(p, y))
            rv.append(cs.getbeta()[0])
            
        return rv
            
    def test():
        import numpy as np
        gen = np.random.default_rng(2)
        curve = test_once(gen, 50000)
        truefp = lambda z: (100 - int(z)) * (101 - int(z)) / 20_000
        truefpbetas = np.array([ truefp(100*beta) for beta in curve ])
        assert np.all(truefpbetas <= 0.05)
        assert curve[-1] < 0.725
        assert truefpbetas[-1] > 0.045
        print('test pass')
        
    test()      
