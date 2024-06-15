import torch
import numpy as np
from Risk import TopBetaCoverage, NegRecall


class DummyRegressor(torch.nn.Module):

    def __init__(self, in_features, out_classes, base_rate):
        super().__init__()
        assert 0 <= base_rate <= 1
        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=out_classes,
                                      bias=True)
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(base_rate)
        self._base_rate = base_rate

    def predict(self, features):
        _, X, _ = features
        with torch.no_grad():
            return self.linear(X)

    def forward(self, features):
        res = self.predict(features)
        return res

    def parameters(self):
        return []

    def min(self, beta):
        return self._base_rate


class NegRecallRegressor(torch.nn.Module):

    def __init__(self, **kwargs):
        super(NegRecallRegressor, self).__init__()
        self._risk_fn = NegRecall(**kwargs)
        self._scale = torch.tensor([1.], requires_grad=True)

    def forward(self, features):
        P, X, beta = features
        mask = self._risk_fn.beta_coverage_set(P, beta)
        est_trues = P.sum(axis=1)
        est_fn = (P * (~mask)).sum(axis=1)
        est_trues[
            est_trues ==
            0.] = 1.  # if denom is 0, then numerator is 0, so this avoids 0 / 0
        est_fnp = est_fn / est_trues

        # we only have an estimate of the risk, not the squared risk
        return est_fnp * self._scale

    def predict(self, features):
        with torch.no_grad():
            return self.forward(features)

    def min(self, beta):
        with torch.no_grad():
            return 0 - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return (1. - betas) + 1e-3


class CVWrapperRegressor(torch.nn.Module):

    def __init__(self, cv_predictor):
        super(CVWrapperRegressor, self).__init__()
        self._cv_predictor = cv_predictor

    def forward(self, features):
        P, X, beta = features
        cv_risk = self._cv_predictor((P, X), beta)
        worst_case_var = (1 - cv_risk) * cv_risk
        return worst_case_var

    def predict(self, features):
        with torch.no_grad():
            return self.forward(features)

    def min(self, beta):
        with torch.no_grad():
            return self._cv_predictor.min_val(beta) * (
                1 - self._cv_predictor.min_val(beta))


class SqrtRiskRegressor(torch.nn.Module):

    def __init__(self,
                 learning=False,
                 in_features=None,
                 out_classes=None,
                 bias=True,
                 **kwargs):
        super(SqrtRiskRegressor, self).__init__()
        self._scale = torch.tensor([0.], requires_grad=True)
        self._learning = learning
        if learning:
            assert in_features is not None and out_classes is not None
            self._linear = torch.nn.Linear(in_features=in_features,
                                           out_features=out_classes,
                                           bias=bias)
            self._linear.weight.data.fill_(0)
            self._linear.bias.data.fill_(0)
            self._combo = torch.tensor([1.], requires_grad=True)

    def _get_est_risk(self, P, beta):
        return torch.sqrt(P)

    def forward(self, features):
        P, X, beta = features
        est_risk = self._get_est_risk(P, beta)
        if self._learning:
            weight = torch.special.sigmoid(self._combo)
            res = torch.special.sigmoid(
                self._linear(X)) * weight + (1 - weight) * est_risk
        else:
            res = torch.clamp(est_risk * self._scale, 0, 1)
        return res

    def predict(self, features):
        with torch.no_grad():
            return self.forward(features)

    def min(self, beta):
        with torch.no_grad():
            return 0 - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return (1. - betas) + 1e-3


class ClassRiskRegressor(torch.nn.Module):

    def __init__(self,
                 learning=False,
                 in_features=None,
                 out_classes=None,
                 bias=True,
                 **kwargs):
        super(ClassRiskRegressor, self).__init__()
        self._risk_fn = TopBetaCoverage(**kwargs)
        self._scale = torch.tensor([0.], requires_grad=True)
        self._learning = learning
        if learning:
            assert in_features is not None and out_classes is not None
            self._linear = torch.nn.Linear(in_features=in_features,
                                           out_features=out_classes,
                                           bias=bias)
            self._linear.weight.data.fill_(0)
            self._linear.bias.data.fill_(0)
            self._combo = torch.tensor([1.], requires_grad=True)

    def _get_est_risk(self, P, beta):
        mask = self._risk_fn.beta_coverage_set(P, beta)
        est_risk = ((~mask) *
                    (P * torch.square(self._risk_fn.torch_w))).sum(axis=1)
        return torch.sqrt(est_risk)

    def forward(self, features):
        P, X, beta = features
        est_risk = self._get_est_risk(P, beta)
        if self._learning:
            weight = torch.special.sigmoid(self._combo)
            res = torch.special.sigmoid(
                self._linear(X)) * weight + (1 - weight) * est_risk
        else:
            res = torch.clamp(est_risk * self._scale, 0, 1)
        return res

    def predict(self, features):
        with torch.no_grad():
            return self.forward(features)

    def min(self, beta):
        with torch.no_grad():
            return 0 - 1e-3  # add small constant to ensure it's always smaller

    def max_val(self, betas):
        with torch.no_grad():
            return (1. - betas) + 1e-3


# class ConstantPredictor(torch.nn.Module):
#
#     def __init__(self,
#                  risk_fn,
#


class MultilabelRegressor(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_classes,
                 bias=True,
                 base_rate=None,
                 in_dim=None,
                 min_type='bins'):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=out_classes,
                                      bias=bias)
        if in_dim is not None:
            self._in_row, self._in_col = in_dim
        else:
            self._in_row = self._in_col = None
        if bias and base_rate is not None:
            bias_init = torch.special.logit(torch.Tensor([base_rate])).item()
            # print(f'Bias init {bias_init}')
            self.linear.weight.data.fill_(0)
            self.linear.bias.data.fill_(bias_init)
        self._min_type = min_type

    def pre_logits(self, X):
        res = self.linear(X)
        # assert torch.all(~torch.isnan(res)), (self.linear.weight.data,
        #                                       self.linear.bias.data)
        return res

    def predict(self, features):
        _, X, _ = features
        return torch.special.expit(self.pre_logits(X))

    def forward(self, features):
        _, X, _ = features
        res = torch.special.expit(self.pre_logits(X))
        return res

    def _min(self):
        if self._min_type == 'bins':
            with torch.no_grad():
                # this is b/c we know that every feature induces a convex combination of coefficients + bias. otherwise this isn't right
                if self._in_row is not None:
                    reshaped = self.linear.weight.data.reshape(
                        self._in_row, self._in_col)
                    col_mins = reshaped[:(-1), :].min(dim=0).values
                    # print(torch.special.expit(col_mins + self.linear.bias.data))
                    actual_mins = col_mins + reshaped[-1, :]
                    c_mins = torch.special.expit(actual_mins +
                                                 self.linear.bias.data)
                    actual_min = actual_mins.min()
                    res = torch.special.expit(actual_min +
                                              self.linear.bias.data)
                    return res, c_mins
                else:
                    return 0, 0
        # elif self._min_type == 'ml_prob':
        #     return torch.minimum(self.linear.weight.data, 0) +

    def min(self, beta):
        # ensure that the float -> double conversion only makes something smaller
        # return np.nextafter(self._min()[0].item(), float('-inf'))
        if self._in_row is not None:
            return self._min()[0].item() - (1e-3)
        else:
            return 0 - 1e-3


class MulticlassRegressor(torch.nn.Module):

    def __init__(self, in_features, out_classes, bias=True):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=out_classes,
                                      bias=bias)

    def pre_logits(self, X):
        return self.log_softmax(self.linear(X))

    def predict(self, X):
        return torch.exp(self.pre_logits(X))

    def forward(self, X):
        return torch.exp(self.pre_logits(X))
