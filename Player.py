class Player(object):
    def __init__(self, *, iwmart):
        self._iwmart = iwmart
        self._suml = 0
        
    def predict(self, PXY):
        import torch
        (_, X), _ = PXY
       
        n = X.shape[0]
        self._suml += n
        return (torch.ones(n), torch.ones(n)), torch.zeros(n)
    
    def update(self, PXY, QL, dual):
        import torch
        (P, X), Y = PXY
        Q, L = QL

        with torch.no_grad():
            for pj, xj, yj, qj, lj in zip(P, X, Y, torch.ones(Y.shape[0]), torch.ones(Y.shape[0])):
                self._iwmart.addobs(x=((pj, xj), yj), q=qj.item(), l=lj)
                self._suml += Y.shape[0]


class LabellingPolicyPrimalPlayer(Player):
    def __init__(self, *, policy, q_min, target_rate, theta, rho, opt, sched, iwmart):
        assert 0 <= q_min < target_rate 
        
        super().__init__(iwmart=iwmart)
        self._policy = policy
        self._q_min = q_min
        self._target_rate = target_rate
        self._theta = theta
        self._rho = rho
        self._opt = opt
        self._sched = sched
        
    def predict(self, PXY):
        import torch
        
        (_, X), _ = PXY
       
        Q = self._q_min + (1 - self._q_min) * self._policy(X).squeeze(1)
        with torch.no_grad():
            cons = (self._target_rate - torch.mean(Q)).item()
            L = torch.bernoulli(Q)
            self._suml += torch.sum(L).item()

        return (Q, L), cons
        
    def update(self, PXY, QL, dual):
        import torch
        
        (P, X), Y = PXY
        Q, L = QL
        
        # dual penalty
        loss = -dual * (self._target_rate - torch.mean(Q))
                    
        # policy gradient
        curlam = self._iwmart.curlam
        curbeta = self._iwmart.curbeta[1]
        f = torch.log1p(curlam * self._iwmart.xi(PXY, Q, L, curbeta))
        stopf = f.detach()
        probL = (L == 1).float() * Q + (L == 0).float() * (1 - Q)
        loss -= (torch.log(probL) * stopf + f).mean()
        
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        if self._sched is not None:
            self._sched.step()
        
        # update betting martingale
        super().update(PXY, QL, dual)