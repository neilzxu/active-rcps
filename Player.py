import torch

from cocob import COCOB
from IwUpperMartingale import PartialIwUpperMartingale, ShiftedIwUpperMartingale


class Player(object):

    def __init__(self, *, iwmart):
        self._iwmart = iwmart
        self._suml = 0

    def predict(self, PXY):
        import torch
        (_, X), _ = PXY

        n = X.shape[0]
        self._suml += n
        return (torch.ones(n), torch.ones(n), None), torch.zeros(n)

    def update(self, PXY, QL, dual):
        import torch
        (P, X), Y = PXY
        Q, L, cv_predictor = QL

        with torch.no_grad():
            # for pj, xj, yj, qj, lj in zip(P, X, Y, torch.ones(Y.shape[0]),
            #                               torch.ones(Y.shape[0])):
            self._iwmart.addobs(x=((P, X), Y),
                                q=Q.item(),
                                l=L,
                                cv=cv_predictor)


class LabellingPolicyPrimalPlayer(Player):

    def __init__(self,
                 *,
                 q_min,
                 target_rate,
                 theta,
                 rho,
                 opt,
                 sched,
                 iwmart,
                 policy=None,
                 cv_predictor=None):
        assert 0 <= q_min <= target_rate

        super().__init__(iwmart=iwmart)
        self._policy = policy
        if policy is None:
            self._q_min = target_rate
        else:
            self._q_min = q_min
        self._target_rate = target_rate
        self._theta = theta
        self._rho = rho
        self._opt = opt
        self._sched = sched
        self._cv_predictor = cv_predictor
        self._opts = [self._opt] if self._opt is not None else []
        # if self._cv_predictor is not None:
        #     self._opts.append(self._cv_predictor._opt)
        #
    def _cv_predict(self, PX, betas):
        return self._cv_predictor(
            PX, betas) if self._cv_predictor is not None else 0

    def predict(self, PXY):

        (P, X), _ = PXY

        features = (P, X, self._iwmart.curbeta[1])
        if self._policy is None:
            Q = torch.tensor([self._target_rate])
        else:
            Q = self._q_min + (1 - self._q_min) * self._policy(features)

        CV = self._cv_predict((P, X), self._iwmart.curbeta[1])

        if isinstance(self._iwmart, ShiftedIwUpperMartingale):
            assert CV == 0

        with torch.no_grad():
            cons = (self._target_rate - torch.mean(Q)).item()
            L = torch.bernoulli(Q)
            self._suml += torch.sum(L).item()

        return (Q, L, CV), cons

    def update(self, PXY, QL, dual):

        (P, X), Y = PXY
        Q, L, CV = QL
        # if self._policy is not None:
        #     q_min = self._q_min + (
        #         (1 - self._q_min) * self._policy.min(self._iwmart.curbeta[1]))
        #  assert torch.all(Q >= self._iwmart._q_min)
        if isinstance(self._iwmart, ShiftedIwUpperMartingale):
            assert CV == 0

        # dual penalty
        loss = -dual * (self._target_rate - torch.mean(Q))

        # get updated best beta and associated lambda
        curlam = self._iwmart.curlam
        torch_lam = torch.as_tensor(curlam)
        curbeta = self._iwmart.curbeta[1]
        # get updated CV value corresponding to new curbeta

        xi = self._iwmart.xi(PXY, Q, L, curbeta, CV)

        # policy gradient
        f = torch.log1p(torch_lam * xi)
        stopf = f.detach()
        probL = (L == 1.).float() * Q + (L == 0.).float() * (1 - Q)
        prim_loss_vec = torch.log(probL) * stopf + f
        loss -= prim_loss_vec.mean()

        for opt in self._opts:
            opt.zero_grad()
        if len(self._opts) > 0:
            loss.backward()
        for opt in self._opts:
            opt.step()
        if self._sched is not None:
            self._sched.step()

        super().update(PXY, (Q, L, self._cv_predictor), dual)
        # update betting martingale
        if self._cv_predictor is not None:
            self._cv_predictor.update(PXY)

        if self._policy is not None:
            new_q_min = self._q_min + (
                (1 - self._q_min) * self._policy.min(self._iwmart.curbeta[1]))
            self._iwmart.update_q_min(max(new_q_min - 1e-3, 0))


class RiskPredictorPolicyPrimalPlayer(Player):

    def __init__(self,
                 *,
                 var_predictor,
                 q_min,
                 target_rate,
                 iwmart,
                 cv_predictor=None):
        assert 0 <= q_min <= target_rate

        super().__init__(iwmart=iwmart)
        self._var_predictor = var_predictor
        self._cv_predictor = cv_predictor
        self._q_min = q_min if self._var_predictor is not None else target_rate
        self._target_rate = target_rate
        self._norm_const = torch.nn.Parameter(torch.zeros(1),
                                              requires_grad=True)
        self._norm_opt = COCOB([self._norm_const])

    def _get_norm_const(self):
        return torch.exp(self._norm_const)

    def _cv_predict(self, PX, betas):
        return self._cv_predictor(
            PX, betas) if self._cv_predictor is not None else 0

    def predict(self, PXY):
        (P, X), _ = PXY

        # features = (P, X, self._iwmart.curbeta[1])
        # Play greedily and predict sd of current beta not safe
        if self._var_predictor is not None:
            var = self._var_predictor((P, X), self._iwmart.curbeta[1])

            # Use exp so it's always nonnegative
            Q = torch.clip((var**0.5) / self._get_norm_const(), self._q_min, 1)
            assert Q >= self._iwmart._q_min
        else:
            Q = torch.tensor([self._target_rate])
        CV = self._cv_predict((P, X), self._iwmart.curbeta[1])

        with torch.no_grad():
            cons = (self._target_rate - torch.mean(Q)).item()
            L = torch.bernoulli(Q)
            self._suml += torch.sum(L).item()

        return (Q, L, CV), cons

    def update(self, PXY, QL, dual):

        (P, X), Y = PXY
        Q, L, CV = QL

        # dual penalty
        loss = (1 / self._get_norm_const()) - dual * (self._target_rate -
                                                      torch.mean(Q))

        self._norm_opt.zero_grad()
        loss.backward()
        self._norm_opt.step()

        super().update(PXY, (Q, L, self._cv_predictor), dual)
        if L:
            if self._var_predictor is not None:
                self._var_predictor.update(PXY)
            if self._cv_predictor is not None:
                self._cv_predictor.update(PXY)

        # new q_min is based on what the next beta to bet against is
        if self._var_predictor is not None:
            new_q_min = max(((
                (self._var_predictor.min_val(self._iwmart.curbeta[1]))**0.5) /
                             self._get_norm_const()).item(), self._q_min)
            # update betting martingale
            self._iwmart.update_q_min(max(new_q_min - 1e-3, 0))
