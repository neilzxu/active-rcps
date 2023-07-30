class BestResponse(object):
    def __init__(self, *, max_dual):
        super().__init__()
        self._max_dual = max_dual
        
    def predict(self, peek):
        return self._max_dual if peek > 0 else 0
    
    def update(self, reward):
        pass

# best response seems better (?)
class BeTheLeader(object):
    def __init__(self, *, max_dual):
        super().__init__()
        self._max_dual = max_dual
        self._tot_reward = 0
        
    def predict(self, peek):
        return self._max_dual if self._tot_reward + peek > 0 else 0
    
    def update(self, reward):
        self._tot_reward += reward

class OnlineMinimax(object):
    def __init__(self, *, primal_player, dual_player):
        super().__init__()
        
        self._primal_player = primal_player
        self._dual_player = dual_player
        
    def addobs(self, x):
        pred, cons = self._primal_player.predict(x)
        d = self._dual_player.predict(peek=-cons)
        self._dual_player.update(-cons)
        self._primal_player.update(x, pred, dual=d)


if __name__ == '__main__':
    from LogisticRegression import LogisticRegressor

    class OnlinePenalizedLogisticRegression(object):
        import torch
        
        def __init__(self, *, dim, obj_fn, cons_fn):
            import torch
            
            super().__init__()
            
            self.regressor = LogisticRegressor(in_features=1, out_features=dim)
            self.obj_fn = obj_fn
            self.cons_fn = cons_fn
            self.opt = torch.optim.Adam(self.regressor.parameters(), lr=5e-1)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:(1000+t)**(-0.5))
            
        def predict(self, x):
            import torch
            
            pred = self.regressor(torch.zeros((1,1))).squeeze(0)
            return pred, self.cons_fn(x, pred).item()
            
        def update(self, x, pred, dual):
            import torch
            
            self.opt.zero_grad()
            loss = -self.obj_fn(x, pred) - dual*self.cons_fn(x, pred)
            loss.mean().backward()
            self.opt.step()
            self.scheduler.step()

    def test_deterministic_once(gen):    
        # maximize x^\top c
        # subject to 0 <= x <= 1
        #            b^\top x <= 1
        
        import numpy as np
        import scipy.optimize as so
        import torch
            
        b = gen.uniform(low=0.5, high=1, size=3)
        c = gen.uniform(size=3)
        res = so.linprog(-c, 
                         A_ub = np.array([b]),
                         b_ub = np.array([1]),
                         bounds=[(0, 1) for _ in range(3)])
        assert res.success
        xstar = res.x
        
        def obj_fn(z, pred):
            return z.dot(pred)
        
        def cons_fn(z, pred):
            return 1 - torch.Tensor(b).dot(pred)
        
        primal_player = OnlinePenalizedLogisticRegression(dim=3, obj_fn=obj_fn, cons_fn=cons_fn)
        dual_player = BestResponse(max_dual=np.max(1/b))
        minimax = OnlineMinimax(primal_player=primal_player, dual_player=dual_player)
    
        print(f'xstar = {xstar}')
        ctorch = torch.Tensor(c)
        for n in range(16385):
            minimax.addobs(ctorch)
            if n & (n - 1) == 0: 
                with torch.no_grad():
                    print(n, primal_player.predict(ctorch), dual_player.predict(peek=0))
        
        with torch.no_grad():
            assert np.allclose(xstar, primal_player.predict(ctorch)[0].numpy(), atol=5e-2), (xstar, primal_player.predict(ctorch)[0].numpy())
    
    def test_stochastic_once(gen):    
        # maximize x^\top c
        # subject to 0 <= x <= 1
        #            b^\top x <= 1
        
        import numpy as np
        import scipy.optimize as so
        import torch
            
        b = gen.uniform(low=0.5, high=1, size=3)
        c = gen.uniform(size=3)
        res = so.linprog(-c, 
                         A_ub = np.array([b]),
                         b_ub = np.array([1]),
                         bounds=[(0, 1) for _ in range(3)])
        assert res.success
        xstar = res.x
        
        def obj_fn(z, pred):
            return z.dot(pred)
        
        def cons_fn(z, pred):
            return 1 - torch.Tensor(b).dot(pred)
        
        primal_player = OnlinePenalizedLogisticRegression(dim=3, obj_fn=obj_fn, cons_fn=cons_fn)
        dual_player = BestResponse(max_dual=np.max(1/b))
        minimax = OnlineMinimax(primal_player=primal_player, dual_player=dual_player)
    
        print(f'xstar = {xstar}')
        for n in range(16385):
            ctorch = torch.tensor(gen.lognormal(mean=np.log(c)-1/2, sigma=1)).float()
            minimax.addobs(ctorch)
            if n & (n - 1) == 0: 
                with torch.no_grad():
                    print(n, primal_player.predict(ctorch), dual_player.predict(peek=0))
        
        with torch.no_grad():
            assert np.allclose(xstar, primal_player.predict(ctorch)[0].numpy(), atol=5e-2), (xstar, primal_player.predict(ctorch)[0].numpy())
    
    def test():
        import numpy as np
        import torch
        
        gen = np.random.default_rng(45)
        torch.manual_seed(1)
        test_deterministic_once(gen)
        test_stochastic_once(gen)
        print('test pass')
        
    test()
