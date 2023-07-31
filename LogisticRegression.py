import torch

class MultilabelRegressor(torch.nn.Module):
    def __init__(self, in_features, out_classes, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_classes, bias=bias)

    def pre_logits(self, X):
        return self.linear(X)

    def predict(self, X):
        return torch.special.expit(self.pre_logits(X))

    def forward(self, X):
        return torch.special.expit(self.pre_logits(X))

class MulticlassRegressor(torch.nn.Module):
    def __init__(self, in_features, out_classes, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_classes, bias=bias)

    def pre_logits(self, X):
        return self.log_softmax(self.linear(X))

    def predict(self, X):
        return torch.exp(self.pre_logits(X))

    def forward(self, X):
        return torch.exp(self.pre_logits(X))
