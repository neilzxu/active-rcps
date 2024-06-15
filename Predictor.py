from cocob import COCOB
import numpy as np
import torch

from Features import digitize
from Risk import TopBetaCoverage, NegRecall


class NegRecallPredictor(torch.nn.Module):

    def __init__(self, **kwargs):
        super(NegRecallPredictor, self).__init__()
        self._mode = mode
        self._risk_fn = NegRecall(**kwargs)

    def forward(self, PX, betas):
        P, X = PX
        mask = self._risk_fn.beta_coverage_set(P, betas)
        est_trues = P.sum(axis=1)
        est_fn = (P * (~mask)).sum(axis=1)
        est_trues[
            est_trues ==
            0.] = 1.  # if denom is 0, then numerator is 0, so this avoids 0 / 0
        est_fnp = est_fn / est_trues
        return est_fnp

    def predict(self, PX, betas):
        with torch.no_grad():
            return self.forward(PX, betas)

    def min_val(self, betas):
        with torch.no_grad():
            return 0 - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return (1. - betas) + 1e-3

    def update(self, PXY):
        pass


class ClassRiskPredictor(torch.nn.Module):

    def __init__(self, mode='mean', **kwargs):
        super(ClassRiskPredictor, self).__init__()
        self._mode = mode
        self._risk_fn = TopBetaCoverage(**kwargs)

    def forward(self, PX, betas):
        with torch.no_grad():
            P, X = PX
            if isinstance(betas, torch.Tensor) or isinstance(
                    betas, np.ndarray):
                mask = torch.concat([
                    self._risk_fn.beta_coverage_set(P, beta) for beta in betas
                ],
                                    dim=0)
            else:
                mask = self._risk_fn.beta_coverage_set(P, betas)
            est_risk_mean = ((~mask) * (P * self._risk_fn.torch_w)).sum(axis=1)

            if self._mode == 'mean':
                est_risk = est_risk_mean
            else:
                wrong_est = ((~mask) * (P * (
                    (self._risk_fn.torch_w - est_risk_mean.reshape(-1, 1))**2))
                             ).sum(axis=1)
                right_est = (mask * P *
                             ((est_risk_mean.reshape(-1, 1) - 0)**2)).sum(
                                 axis=1)
                est_var = wrong_est + right_est
                if self._mode == 'var':
                    est_risk = est_var
                else:
                    assert self._mode == 'mse'
                    est_risk = (est_risk_mean**2) + est_var
            return est_risk

    def predict(self, PX, betas):
        return self.forward(PX, betas)

    def min_val(self, betas):
        with torch.no_grad():
            return np.full(
                betas.shape,
                0.)  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return np.full(betas.shape, 1.)

    def update(self, PXY):
        pass


class FalsePositivePredictor(torch.nn.Module):

    def __init__(self, mode='mean', **kwargs):
        super(FalsePositivePredictor, self).__init__()
        self._mode = mode

    def forward(self, PX, betas):
        with torch.no_grad():
            # assume P is single dimensional obj
            P, X = PX
            P = P[0, 0].item()
            res = torch.full(betas.shape, 1 - P) * (
                betas <= P) if self._mode == 'mean' else torch.full(
                    betas.shape, P * (1 - P)) * (betas <= P)
            return res

    def predict(self, PX, betas):
        return self.forward(PX, betas)

    def min_val(self, betas):
        with torch.no_grad():
            return 0  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return 1.

    def update(self, PXY):
        pass


class LinearRiskPredictor(torch.nn.Module):
    """Predicts some kind of expected risk for different betas."""

    def __init__(self,
                 beta_knots,
                 risk_knots,
                 featurize_fn,
                 risk_fn,
                 pretr_fn,
                 n_beta_min,
                 n_betas,
                 iwmart=None,
                 n_beta_max=1,
                 add_pretrain=False):
        super(LinearRiskPredictor, self).__init__()
        self._risk_fn = risk_fn
        self._featurize_fn = featurize_fn
        self._risk_knots, self._beta_knots = risk_knots, beta_knots
        self._in_features = len(risk_knots) * (len(beta_knots) + 1)
        self._linear = torch.nn.Linear(in_features=self._in_features,
                                       out_features=1,
                                       bias=True)

        # Since our output is bounded on [0, 1] we set the intial value to be 0.5
        self._linear.weight.data.fill_(0)
        self._linear.bias.data.fill_(0)
        self._beta_maxes = torch.full((len(beta_knots), ), 1.)
        self._beta_mins = torch.full((len(beta_knots), ), 0.)

        self._pretr_fn = pretr_fn
        self._add_pretrain = add_pretrain

        self._opt = COCOB(self._linear.parameters())
        self._iwmart = iwmart

        self._betas = torch.linspace(n_beta_min, n_beta_max, n_betas)

    def _forward(self, PX, betas):
        P, _ = PX
        pretrain_guesses = self._pretr_fn(PX, betas)
        if isinstance(betas, float) or isinstance(betas, np.float64):
            Xs = self._featurize_fn(pretrain_guesses.reshape(1, -1), betas)
        else:
            Xs = torch.concatenate([
                self._featurize_fn(guess.reshape(1, -1), beta)
                for guess, beta in zip(pretrain_guesses, betas)
            ])
        res = self._linear(Xs) + (pretrain_guesses
                                  if self._add_pretrain else 0)
        res = torch.clamp(res, 0, 1)
        return res

    def forward(self, PX, betas):
        # disable backprop from wealth loss --- backprop only from update
        with torch.no_grad():
            return self._forward(PX, betas)

    def predict(self, PX, betas):
        return self.forward(PX, betas)

    def _digitize_betas(self, betas):
        if isinstance(betas, float):
            return digitize(betas, self._beta_knots)
        if isinstance(betas, np.float64):
            return digitize(betas, self._beta_knots)
        return torch.as_tensor(
            [digitize(beta, self._beta_knots) for beta in betas])

    def min_val(self, betas):
        with torch.no_grad():
            return np.clip(
                self._digitize_betas(betas).dot(self._beta_mins).numpy(), 0, 1)

    def max_val(self, betas):
        with torch.no_grad():
            return np.clip(
                self._digitize_betas(betas).dot(self._beta_maxes).numpy(), 0,
                1)

    def update(self, PXY):
        """Compute risk for each value of beta considered in the CS and update
        regressor based on the average."""
        (P, X), Y = PXY
        betas = self._betas if self._iwmart is None else torch.tensor(
            self._iwmart._betas)
        risks = self._risk_fn(PXY, betas)

        # Minimize against the squared risk, since we're
        # trying to estimate variance
        loss = (risks.detach() - self._forward((P, X), betas)).square()
        self._opt.zero_grad()
        loss.sum().backward()
        self._opt.step()

        ## Update max and min, assuming features are some kind of one shot shape
        reshape_w = self._linear.weight.data.reshape(
            len(self._beta_knots) + 1,
            len(self._risk_knots))  # (beta_knots + 1, risk_knots)

        y_knots = (reshape_w[-1, :].reshape(1, -1) + reshape_w[:-1, :])
        y_knots += (
            self._linear.bias.data +
            (self._beta_knots. if self._add_pretrain else 0))
        self._beta_mins = torch.min(y_knots, dim=1).values
        self._beta_maxes = torch.max(y_knots, dim=1).values


class ConstantPredictor(torch.nn.Module):

    def __init__(self,
                 risk_fn,
                 n_beta_min,
                 n_betas,
                 n_beta_max=1,
                 bias_init=1.):
        super(ConstantPredictor, self).__init__()
        self._risk_fn = risk_fn
        self._bias = torch.nn.Parameter(torch.full((n_betas, ), bias_init),
                                        requires_grad=True)
        self._betas = torch.linspace(n_beta_min, n_beta_max, n_betas)
        self._opt = COCOB([self._bias])

    def _calc(self, betas):
        idxs = torch.bucketize(betas, self._betas, right=False)
        return self._bias[idxs]

    def forward(self, PX, betas):
        # disable backprop from wealth loss --- backprop only from update
        with torch.no_grad():
            res = torch.clamp(self._calc(betas), 0, 1)
            assert torch.all(res >= self.min_val(betas))
            assert torch.all(res <= self.max_val(betas))
            return res

    def predict(self, PX, betas):
        return self.forward(PX, betas)

    def min_val(self, betas):
        with torch.no_grad():
            return torch.clamp(self._calc(betas), 0, 1).numpy(
            ) - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return torch.clamp(self._calc(betas), 0, 1).numpy(
            ) + 1e-3  # add small constant to ensure its always larger

    def update(self, PXY):
        """Compute risk for each value of beta considered in the CS and update
        regressor based on the average."""
        (P, _), Y = PXY
        risks = self._risk_fn(PXY, self._betas)

        loss = (risks.detach() - self._bias).square()
        self._opt.zero_grad()
        loss.sum().backward()
        self._opt.step()


class FixedConstantPredictor(torch.nn.Module):

    def __init__(self, constant):
        super(FixedConstantPredictor, self).__init__()
        self._constant = torch.as_tensor(constant)

    def forward(self, PX, betas):
        # disable backprop from wealth loss --- backprop only from update
        with torch.no_grad():
            return self._constant

    def predict(self, PX, betas):
        return self.forward(PX, betas)

    def min_val(self, betas):
        with torch.no_grad():
            return self._constant.numpy(
            ) - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return self._constant.numpy(
            ) + 1e-3  # add small constant to ensure its always larger

    def update(self, PXY):
        """Compute risk for each value of beta considered in the CS and update
        regressor based on the average."""
        pass
