import torch


class DummyRegressor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def predict(self, X):
        return torch.ones(X.shape[0])

    def forward(self, X):
        return torch.ones(X.shape[0])


class MultilabelRegressor(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_classes,
                 bias=True,
                 init_weight=None,
                 init_bias=None,
                 base_rate=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=out_classes,
                                      bias=bias)
        if init_weight is not None:
            self.linear.weight.data_ = init_weight
        if bias and init_bias is not None:
            self.linear.bias.data_ = init_bias
        if base_rate is not None:
            self.linear.weight.data_ = torch.zeros((out_classes, in_features))
            self.linear.bias.data_ = torch.logit(torch.Tensor([base_rate]))

    def constant_init(self, pconst):
        if self.linear.bias is not None:
            self.linear.bias.fill_(
                torch.special.logit(torch.Tensor([pconst])).item())
        self.linear.weight.fill_(0)

    def pre_logits(self, X):
        return self.linear(X)

    def predict(self, X):
        return torch.special.expit(self.pre_logits(X))

    def forward(self, X):
        return torch.special.expit(self.pre_logits(X))


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
