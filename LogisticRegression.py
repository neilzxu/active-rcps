import torch

class LogisticRegressor(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear  = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.sigmoid = torch.nn.Sigmoid()

    def pre_logits(self, X):
        return self.linear(X)

    def forward(self, X):
        return self.sigmoid(self.pre_logits(X))
    def predict(self, X):
        return self.sigmoid(self.pre_logits(X))
